"""
Real block-level NWP forecasts (Feature F1).

Operational GKMS block forecasts are produced from coarse NWP guidance
(IMD GFS / NCMRWF NCUM / ECMWF). Their archives are not public, so we use the
next-best open equivalent:

* **Model:** ECMWF IFS open data at 0.25° (~27 km). This is the same class of
  coarse global NWP that feeds district and block bulletins.
* **Archive with true lead times:** the Open-Meteo *Previous Runs API* keeps,
  for every valid hour, the value forecast 1, 2, ... 5 days earlier
  (``<var>_previous_dayN``). So "lead day N" for valid date D is the forecast
  issued N days before D. It is a genuine forecast, with genuine forecast error.
* **Block value:** each block gets the **area-weighted mean** of the 0.25° cells
  it overlaps (spec section 3: "NWP grid resampled to block polygons").

Daily aggregation (IST calendar day, Asia/Kolkata):
rain = sum, tmax = max, tmin = min, rh = mean, wind = max of the hourly 10 m wind (km/h).

Outputs
-------
``data/<d>/raw/forecast/nodes/<lat>_<lon>.parquet``: per grid node cache (resumable)
``data/<d>/interim/nwp_grid_daily.parquet``: node x valid_date x lead
``data/<d>/interim/block_forecasts.parquet``: block x valid_date x lead (issue_date = valid - lead)
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import manifest
from src.common.config import Config, load_config
from src.common.geo import Grid, area_weights
from src.common.http import get_json, patiently
from src.common.logging_utils import get_logger
from src.ingest.boundaries import load_blocks, load_panchayats

log = get_logger("ingest.nwp")

PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY_VARS = {
    "precipitation": "rain",
    "temperature_2m": "temp",
    "relative_humidity_2m": "rh",
    "wind_speed_10m": "wind",
}
VARIABLES = ["rain", "tmax", "tmin", "rh", "wind"]


def forecast_grid(cfg: Config) -> tuple[Grid, pd.DataFrame]:
    """0.25° grid nodes whose cells overlap any block, plus block area weights."""
    res = float(cfg["forecast"]["grid_resolution_deg"])
    grid = Grid.covering(cfg.bbox, res)
    blocks = load_blocks(cfg)
    w = area_weights(blocks, "block_lgd", grid, cfg.metric_crs)
    return grid, w


def _daily_from_hourly(times: list[str], hourly: dict[str, list], lead: int) -> pd.DataFrame:
    df = pd.DataFrame({"time": pd.to_datetime(times)})
    for om, short in HOURLY_VARS.items():
        key = f"{om}_previous_day{lead}"
        df[short] = pd.to_numeric(pd.Series(hourly.get(key, [None] * len(times))), errors="coerce")
    df["valid_date"] = df["time"].dt.normalize()
    g = df.groupby("valid_date")
    out = pd.DataFrame({
        "rain": g["rain"].sum(min_count=20),      # need >= 20 of 24 hours
        "tmax": g["temp"].max(),
        "tmin": g["temp"].min(),
        "rh": g["rh"].mean(),
        "wind": g["wind"].max(),
        "n_hours": g["temp"].count(),
    }).reset_index()
    out.loc[out["n_hours"] < 20, ["rain", "tmax", "tmin", "rh", "wind"]] = np.nan
    out["lead_day"] = lead
    return out.drop(columns="n_hours")


def _date_chunks(start: str, end: str, days: int) -> list[tuple[str, str]]:
    s, e = date.fromisoformat(start), date.fromisoformat(end)
    out = []
    while s <= e:
        c = min(e, s + timedelta(days=days - 1))
        out.append((s.isoformat(), c.isoformat()))
        s = c + timedelta(days=1)
    return out


def fetch_node(cfg: Config, lat: float, lon: float, cache_dir: Path) -> Path:
    """Download lead 1..5 archive for one node over the configured period (resumable)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{lat:.2f}_{lon:.2f}.parquet"
    if out.exists():
        return out
    leads = cfg.lead_days
    hourly = ",".join(f"{v}_previous_day{k}" for v in HOURLY_VARS for k in leads)
    frames = []
    for s, e in _date_chunks(cfg["period"]["start"], cfg["period"]["end"], 280):
        js = get_json(PREVIOUS_RUNS_URL, params={
            "latitude": lat, "longitude": lon, "hourly": hourly,
            "start_date": s, "end_date": e, "models": cfg["forecast"]["model"],
            "timezone": cfg["forecast"]["timezone"], "wind_speed_unit": "kmh",
        })
        h = js["hourly"]
        for k in leads:
            frames.append(_daily_from_hourly(h["time"], h, k))
    df = pd.concat(frames, ignore_index=True)
    df["node_lat"], df["node_lon"] = lat, lon
    df["om_lat"], df["om_lon"] = js.get("latitude"), js.get("longitude")
    df["om_elevation"] = js.get("elevation")
    tmp = out.with_suffix(".part")
    df.to_parquet(tmp, index=False)
    tmp.replace(out)
    return out


def fetch_archive(cfg: Config | None = None, patient: bool = True) -> Path:
    """Download all nodes (resumable) and build node- and block-level daily tables."""
    cfg = cfg or load_config()
    paths = cfg.paths.ensure()
    cache = paths.raw / "forecast" / "nodes"
    cache.mkdir(parents=True, exist_ok=True)
    grid, w = forecast_grid(cfg)
    nodes = w[["i", "j", "lat", "lon"]].drop_duplicates().sort_values(["lat", "lon"]).to_numpy()
    log.info("NWP archive: %d grid nodes (%.2f°) x %s..%s x leads %s",
             len(nodes), grid.res, cfg["period"]["start"], cfg["period"]["end"], cfg.lead_days)
    for n, (_, _, lat, lon) in enumerate(nodes, 1):
        patiently(lambda lat=lat, lon=lon: fetch_node(cfg, float(lat), float(lon), cache), f"node {float(lat):.2f},{float(lon):.2f}", patient)
        if n % 5 == 0 or n == len(nodes):
            log.info("  nodes done: %d/%d", n, len(nodes))
    return build_tables(cfg)


def build_tables(cfg: Config | None = None) -> Path:
    cfg = cfg or load_config()
    paths = cfg.paths
    cache = paths.raw / "forecast" / "nodes"
    grid, w = forecast_grid(cfg)
    files = sorted(cache.glob("*.parquet"))
    node_df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    node_df = node_df.rename(columns={"node_lat": "lat", "node_lon": "lon"})
    node_df.to_parquet(paths.interim / "nwp_grid_daily.parquet", index=False)

    # Block area-weighted means (NaN-aware renormalisation)
    blocks = load_blocks(cfg)[["block_lgd", "block_name"]]
    merged = w.merge(node_df, on=["lat", "lon"], how="inner")
    parts = []
    for var in VARIABLES:
        m = merged[["block_lgd", "valid_date", "lead_day", "weight", var]].copy()
        ok = m[var].notna()
        m["wv"] = np.where(ok, m[var] * m["weight"], 0.0)
        m["wok"] = np.where(ok, m["weight"], 0.0)
        g = m.groupby(["block_lgd", "valid_date", "lead_day"])[["wv", "wok"]].sum()
        parts.append((g["wv"] / g["wok"].where(g["wok"] > 0.5)).rename(var))
    blk = pd.concat(parts, axis=1).reset_index()
    blk = blk.merge(blocks, on="block_lgd", how="left")
    blk["issue_date"] = blk["valid_date"] - pd.to_timedelta(blk["lead_day"], unit="D")
    blk = blk[["block_lgd", "block_name", "issue_date", "valid_date", "lead_day", *VARIABLES]]
    blk[VARIABLES] = blk[VARIABLES].round(3)
    out = paths.interim / "block_forecasts.parquet"
    blk.sort_values(["valid_date", "lead_day", "block_lgd"]).to_parquet(out, index=False)

    # Also keep the GP-level bilinear-equivalent reference (area-weighted from the
    # same 0.25° grid). Used only as an *upper-reference baseline*: operational
    # GKMS users have block values, not the grid.
    gps = load_panchayats(cfg, modelled_only=True)
    wg = area_weights(gps, "gp_code", grid, cfg.metric_crs)
    mg = wg.merge(node_df, on=["lat", "lon"], how="inner")
    parts = []
    for var in VARIABLES:
        m = mg[["gp_code", "valid_date", "lead_day", "weight", var]].copy()
        ok = m[var].notna()
        m["wv"] = np.where(ok, m[var] * m["weight"], 0.0)
        m["wok"] = np.where(ok, m["weight"], 0.0)
        g = m.groupby(["gp_code", "valid_date", "lead_day"])[["wv", "wok"]].sum()
        parts.append((g["wv"] / g["wok"].where(g["wok"] > 0.5)).rename(f"nwpgrid_{var}"))
    gp_ref = pd.concat(parts, axis=1).reset_index()
    gp_ref.to_parquet(paths.interim / "gp_nwp_grid_reference.parquet", index=False)

    manifest.register(paths.data / "manifest.json", "block_forecasts", out,
                      source=f"Open-Meteo Previous Runs API, model={cfg['forecast']['model']} (ECMWF IFS open data 0.25°)",
                      licence="CC-BY 4.0 (Open-Meteo); ECMWF open data licence (CC-BY 4.0)",
                      produced_by="src.ingest.nwp_forecast",
                      extra={"rows": int(len(blk)), "grid_nodes": int(len(files)),
                             "period": [cfg["period"]["start"], cfg["period"]["end"]]})
    log.info("block forecasts: %d rows, %d blocks, %d valid dates, leads %s",
             len(blk), blk["block_lgd"].nunique(), blk["valid_date"].nunique(), sorted(blk["lead_day"].unique()))
    return out


# ---------------------------------------------------------------------------
# Live operational forecast (used by the inference pipeline, F5)
# ---------------------------------------------------------------------------
def fetch_live(cfg: Config | None = None, issue_date: date | None = None) -> pd.DataFrame:
    """
    Today's (or ``issue_date``'s, within the last ~3 months) ECMWF IFS 0.25° forecast for the
    operational horizon ``cfg.live_lead_days`` (today + 7 days), aggregated to blocks exactly like
    the training archive.
    """
    cfg = cfg or load_config()
    issue = issue_date or cfg.today()
    is_today = issue == cfg.today()
    grid, w = forecast_grid(cfg)
    nodes = w[["lat", "lon"]].drop_duplicates().sort_values(["lat", "lon"]).reset_index(drop=True)
    leads = cfg.live_lead_days
    start = issue + timedelta(days=min(leads))
    end = issue + timedelta(days=max(leads))
    frames = []
    # Open-Meteo accepts comma-separated coordinate lists
    for chunk_start in range(0, len(nodes), 50):
        sub = nodes.iloc[chunk_start:chunk_start + 50]
        params = {
            "latitude": ",".join(f"{x:.2f}" for x in sub["lat"]),
            "longitude": ",".join(f"{x:.2f}" for x in sub["lon"]),
            "hourly": ",".join(HOURLY_VARS),
            "models": cfg["operational"]["live_forecast_model"],
            "timezone": cfg["forecast"]["timezone"], "wind_speed_unit": "kmh",
        }
        if is_today:
            params["forecast_days"] = max(leads) + 1
        else:  # past issue: reconstruct from the previous-runs archive
            params.update({"start_date": start.isoformat(), "end_date": end.isoformat()})
        url = FORECAST_URL if is_today else PREVIOUS_RUNS_URL
        if not is_today:
            params["hourly"] = ",".join(f"{v}_previous_day{k}" for v in HOURLY_VARS for k in leads)
        js = get_json(url, params=params)
        js = js if isinstance(js, list) else [js]
        for (_, node), res in zip(sub.iterrows(), js):
            h = res["hourly"]
            if is_today:
                renamed = {f"{k}_previous_day0": v for k, v in h.items() if k != "time"}
                d = _daily_from_hourly(h["time"], renamed, 0)
                d["lead_day"] = (d["valid_date"] - pd.Timestamp(issue)).dt.days
            else:
                d = pd.concat([_daily_from_hourly(h["time"], h, k) for k in leads])
                d = d[(d["valid_date"] - pd.to_timedelta(d["lead_day"], "D")) == pd.Timestamp(issue)]
            d["lat"], d["lon"] = node["lat"], node["lon"]
            frames.append(d)
    node_df = pd.concat(frames, ignore_index=True)
    node_df = node_df[node_df["lead_day"].isin(leads)]
    merged = w.merge(node_df, on=["lat", "lon"])
    parts = []
    for var in VARIABLES:
        m = merged.assign(wv=merged[var] * merged["weight"])
        parts.append(m.groupby(["block_lgd", "valid_date", "lead_day"])["wv"].sum().rename(var))
    blk = pd.concat(parts, axis=1).reset_index()
    blk = blk.merge(load_blocks(cfg)[["block_lgd", "block_name"]], on="block_lgd")
    blk["issue_date"] = pd.Timestamp(issue)
    return blk[["block_lgd", "block_name", "issue_date", "valid_date", "lead_day", *VARIABLES]]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--tables-only", action="store_true")
    a = ap.parse_args()
    if a.live:
        print(fetch_live().head(20))
    elif a.tables_only:
        build_tables()
    else:
        fetch_archive()
