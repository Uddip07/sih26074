"""
Daily live refresh: issue today's forecast (today + 7 days) once per day, automatically.

Runs inside the dashboard server as a background thread (started on app startup), so the
dashboard always opens on the current day's forecast. It can also be run on its own, e.g. from
Windows Task Scheduler / cron:

    python -m src.pipeline.scheduler --once

Each check:
1. does nothing if today's live issue already exists, or it is earlier than
   ``operational.refresh_hour_local`` in the district's timezone;
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
from datetime import date

from src.common.config import Config, load_config
from src.common.logging_utils import get_logger

log = get_logger("pipeline.scheduler")
_lock = threading.Lock()


def has_live_issue(cfg: Config, day: date) -> bool:
    meta = cfg.paths.forecasts / day.isoformat() / "meta.json"
    if not meta.exists():
        return False
    try:
        return json.loads(meta.read_text(encoding="utf-8")).get("source") == "live"
    except (OSError, ValueError):
        return False


def refresh_ndvi_if_stale(cfg: Config, today: date) -> bool:
    import pandas as pd

    p = cfg.paths.interim / "gp_ndvi.parquet"
    newest = pd.read_parquet(p, columns=["available_date"])["available_date"].max().date() if p.exists() else None
    if newest is not None and (today - newest).days <= int(cfg["ndvi"]["refresh_when_older_than_days"]):
        return False
    from src.ingest import ndvi

    log.info("NDVI newest usable composite %s is stale: refreshing", newest)
    ndvi.build(cfg)
    return True


def refresh_if_due(cfg: Config, force: bool = False) -> str:
    """Returns 'exists', 'too_early', 'issued' or 'failed: <reason>'."""
    today = cfg.today()
    if not force and has_live_issue(cfg, today):
        return "exists"
    if not force and cfg.now().hour < int(cfg["operational"]["refresh_hour_local"]):
        return "too_early"
    if not _lock.acquire(blocking=False):
        return "running"
    try:
        try:
            from src.ingest.chirps import update_prelim

            update_prelim(cfg)
        except Exception as exc:  # noqa: BLE001 - reported via the issue's input_completeness
            log.warning("CHIRPS prelim refresh failed: %s", exc)
        try:
            refresh_ndvi_if_stale(cfg, today)
        except Exception as exc:  # noqa: BLE001 - reported via the issue's input_completeness
            log.warning("NDVI refresh failed: %s", exc)
        from src.pipeline.predict import run

        meta = run(cfg, today, "live")["meta"]
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
            time.sleep(60 * int(cfg["operational"]["refresh_check_minutes"]))

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
