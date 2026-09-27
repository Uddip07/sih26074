"""
Polite, resumable HTTP helpers.

* ``get_json`` retries on transient errors and honours HTTP 429 (rate limit)
  with exponential back-off, so long Open-Meteo downloads survive quota windows.
* ``download_file`` streams to a temp file and renames atomically.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import requests

from src.common.logging_utils import get_logger

log = get_logger("http")


def _net() -> dict[str, Any]:
    from src.common.config import SETTINGS_FILE, _read_yaml

    return _read_yaml(SETTINGS_FILE)["network"]


NET = _net()
USER_AGENT = NET["user_agent"]

_session = requests.Session()
_session.headers.update({"User-Agent": USER_AGENT})


class RateLimited(RuntimeError):
    """Raised when the server keeps refusing after the maximum wait (or on a daily quota)."""


def get_json(url: str, params: dict[str, Any] | None = None, timeout: int | None = None) -> Any:
    timeout = timeout or NET["timeout_s"]
    wait = float(NET["backoff_start_s"])
    waited_for_limit = 0.0
    attempt = 0  # counts only genuine failures; rate-limit waits are budgeted separately
    while True:
        try:
            resp = _session.get(url, params=params, timeout=timeout)
        except requests.RequestException as exc:
            attempt += 1
            if attempt >= NET["max_retries"]:
                raise RuntimeError(f"giving up on {url} after {attempt} failed attempts: {exc}") from exc
            log.warning("request error (%s), attempt %d/%d", exc, attempt, NET["max_retries"])
            time.sleep(wait)
            wait = min(wait * 2, NET["backoff_max_s"])
            continue
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 429:
            reason = resp.text[:200]
            if "aily" in reason:  # daily quota: the caller decides how long to wait (resumable)
                raise RateLimited(reason)
            sleep_s = NET["rate_limit_minutely_wait_s"] if "inute" in reason else NET["rate_limit_hourly_wait_s"]
            if waited_for_limit + sleep_s > NET["rate_limit_max_wait_s"]:
                raise RateLimited(reason)
            log.warning("rate limited (%s) - sleeping %ss", reason.strip(), sleep_s)
            time.sleep(sleep_s)
            waited_for_limit += sleep_s
            continue
        if 500 <= resp.status_code < 600:
            attempt += 1
            if attempt >= NET["max_retries"]:
                raise RuntimeError(f"giving up on {url}: HTTP {resp.status_code} after {attempt} attempts")
            log.warning("server error %s, attempt %d/%d", resp.status_code, attempt, NET["max_retries"])
            time.sleep(wait)
            wait = min(wait * 2, NET["backoff_max_s"])
            continue
        raise RuntimeError(f"HTTP {resp.status_code} for {resp.url}: {resp.text[:300]}")


def patiently(fn, label: str, patient: bool = True):
    """Run ``fn()`` for long, resumable archive downloads: wait out daily quotas and retry transient
    failures, both per ``config/settings.yaml`` network policy."""
    transient = 0
    while True:
        try:
            return fn()
        except RateLimited as exc:
            if not patient:
                raise
            log.warning("quota exhausted (%s); sleeping %ds then resuming", exc, NET["daily_quota_wait_s"])
            time.sleep(NET["daily_quota_wait_s"])
        except RuntimeError as exc:  # network/DNS/server trouble: back off, retry
            transient += 1
            if transient > NET["node_max_retries"]:
                raise
            log.warning("transient failure on %s (%s); retry %d/%d in %ds", label, exc, transient,
                        NET["node_max_retries"], NET["node_retry_wait_s"])
            time.sleep(NET["node_retry_wait_s"])


def download_file(url: str, dest: Path, timeout: int | None = None, min_bytes: int = 1) -> Path:
    timeout = timeout or NET["download_timeout_s"]
    n = NET["download_attempts"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size >= min_bytes:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, n + 1):
        try:
            with _session.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            if tmp.stat().st_size < min_bytes:
                raise RuntimeError(f"download too small: {tmp.stat().st_size} B")
            tmp.replace(dest)
            return dest
        except Exception as exc:  # noqa: BLE001 - retry any transport failure
            log.warning("download %s failed (%s), attempt %d/%d", url, exc, attempt, n)
            time.sleep(NET["backoff_start_s"] * attempt)
    raise RuntimeError(f"could not download {url}")
