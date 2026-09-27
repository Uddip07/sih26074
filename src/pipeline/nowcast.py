"""
Day-0 short-range precipitation layer (feature F30).

**What it is, honestly:** the next 6 hours of precipitation from the Open-Meteo forecast API
(15-minute resolution, ``best_match`` model blend) at each block's representative point. It is a
**short-range NWP nowcast, not a radar or satellite nowcast.**

Radar (IMD Doppler mosaics) and INSAT-3D/3DR rainfall estimates are the operational nowcast
sources, but they require MOSDAC / IMD credentials. ``RADAR_ADAPTER`` documents where such a
feed plugs in: any callable returning ``{block_lgd: {"mm_total": ...}}`` can replace ``_nwp_nowcast``.

Results are cached for 15 minutes to respect the API quota.
"""

from __future__ import annotations

import time

from src.common.config import Config, load_config
from src.common.http import get_json
from src.common.logging_utils import get_logger

log = get_logger("pipeline.nowcast")
URL = "https://api.open-meteo.com/v1/forecast"
_CACHE: dict = {"t": 0.0, "data": None}
TTL_S = 900
RADAR_ADAPTER = None  # e.g. a function reading IMD radar / INSAT-3D rain rates when credentials exist


def _nwp_nowcast(cfg: Config) -> dict:
    from src.models.train import block_points

    pts = block_points(cfg)
    js = get_json(URL, params={
        "latitude": ",".join(f"{x:.4f}" for x in pts["latitude"]),
        "longitude": ",".join(f"{x:.4f}" for x in pts["longitude"]),
        "minutely_15": "precipitation", "forecast_minutely_15": int(cfg["nowcast"]["steps_15min"]), "timezone": cfg["forecast"]["timezone"],
    })
    js = js if isinstance(js, list) else [js]
    blocks = {}
    for (_, p), res in zip(pts.iterrows(), js):
        m = res["minutely_15"]
        raw = m["precipitation"]
        vals = [v for v in raw if v is not None]  # missing steps stay missing, never 0 mm
        blocks[str(int(p["block_lgd"]))] = {
            "mm_total": round(sum(vals), 1) if vals else None, "max_15min_mm": round(max(vals), 1) if vals else None,
            "hours": len(raw) / 4, "missing_steps": len(raw) - len(vals),
            "times": m["time"], "series_mm": [None if v is None else round(v, 2) for v in raw]}
    return blocks


def nowcast(cfg: Config | None = None) -> dict:
    cfg = cfg or load_config()
    if _CACHE["data"] is not None and time.time() - _CACHE["t"] < TTL_S:
        return _CACHE["data"]
    try:
        blocks = RADAR_ADAPTER(cfg) if RADAR_ADAPTER else _nwp_nowcast(cfg)
        data = {"source": "radar/satellite adapter" if RADAR_ADAPTER else
                "Open-Meteo 15-minute NWP (best_match) - short-range NWP nowcast, not radar",
                "horizon_hours": 6, "blocks": blocks, "generated_epoch": int(time.time())}
    except Exception as exc:  # noqa: BLE001
        log.warning("nowcast unavailable: %s", exc)
        data = {"source": "unavailable", "error": str(exc), "blocks": {}}
    _CACHE.update(t=time.time(), data=data)
    return data
