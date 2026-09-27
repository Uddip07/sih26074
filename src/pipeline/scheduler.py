"""
Daily live refresh: issue today's forecast (today + 7 days) once per day, automatically.

Runs inside the dashboard server as a background thread (started on app startup), so the
dashboard always opens on the current day's forecast. It can also be run on its own, e.g. from
Windows Task Scheduler / cron:

    python -m src.pipeline.scheduler --once

Each check:
1. does nothing if today's live issue already exists, or it is earlier than
   ``operational.refresh_hour_ist`` (the 00 UTC ECMWF run is on Open-Meteo by ~05:30 IST);
2. refreshes preliminary CHIRPS (for the antecedent-rain feature; failures are non-fatal);
3. fetches the live ECMWF IFS forecast and runs the downscaler (``predict.run(source="live")``),
   which also publishes the issue to the dashboard.

A failed attempt (network, API quota) is retried at the next check.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from datetime import date, datetime

from src.common.config import Config, load_config
from src.common.logging_utils import get_logger

log = get_logger("pipeline.scheduler")
CHECK_EVERY_S = 15 * 60
_lock = threading.Lock()


def has_live_issue(cfg: Config, day: date) -> bool:
    meta = cfg.paths.forecasts / day.isoformat() / "meta.json"
    if not meta.exists():
        return False
    try:
        return json.loads(meta.read_text(encoding="utf-8")).get("source") == "live"
    except (OSError, ValueError):
        return False


def refresh_if_due(cfg: Config, force: bool = False) -> str:
    """Returns 'exists', 'too_early', 'issued' or 'failed: <reason>'."""
    today = date.today()
    if not force and has_live_issue(cfg, today):
        return "exists"
    if not force and datetime.now().hour < int(cfg["operational"].get("refresh_hour_ist", 6)):
        return "too_early"
    if not _lock.acquire(blocking=False):
        return "running"
    try:
        try:
            from src.ingest.chirps import update_prelim

            update_prelim(cfg)
        except Exception as exc:  # noqa: BLE001 - antecedent rain is optional at inference
            log.warning("CHIRPS prelim refresh skipped: %s", exc)
        from src.pipeline.predict import run

        meta = run(cfg, today, "live")["meta"]
        try:  # score every earlier issue whose observations have now arrived
            from src.pipeline import verify

            verify.run(cfg)
        except Exception as exc:  # noqa: BLE001
            log.warning("verification skipped: %s", exc)
        log.info("live issue %s published (%d GPs, leads %s)", today, meta["n_gps"], meta["leads"])
        return "issued"
    except Exception as exc:  # noqa: BLE001 - keep the server alive, retry next check
        log.warning("live refresh for %s failed, will retry: %s", today, exc)
        return f"failed: {exc}"
    finally:
        _lock.release()


def start_background(cfg: Config, on_issue=None) -> threading.Thread:
    def loop():
        while True:
            if refresh_if_due(cfg) == "issued" and on_issue:
                on_issue()
            time.sleep(CHECK_EVERY_S)

    t = threading.Thread(target=loop, name="live-refresh", daemon=True)
    t.start()
    return t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--once", action="store_true", help="single check, then exit")
    ap.add_argument("--force", action="store_true", help="re-issue today even if it exists")
    a = ap.parse_args()
    c = load_config(a.config) if a.config else load_config()
    if a.once or a.force:
        print(refresh_if_due(c, force=a.force))
    else:
        start_background(c).join()
