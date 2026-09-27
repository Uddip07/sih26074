"""
Panchayat-scale temperature / humidity / wind ground truth: ERA5-Land, 0.1° (F2, F3).

ERA5-Land (ECMWF/Copernicus) is a land-surface reanalysis at 0.1° (~9 km). It
is forced by observations through ERA5, and it is the finest *complete, gap-free*
daily Tmax/Tmin/RH/wind record freely available over India. Station networks
(IMD AWS, Maharashtra Mahavedh) are the preferred truth when they can be
obtained. ``src/ingest/stations.py`` can ingest them for independent validation.

Resolution comparison: ERA5-Land cell ≈ 100 km², Pune block ≈ 1,100 km², so a
block spans about 11 cells. That is enough spatial detail to learn the terrain
effects (lapse rate, valley cold pooling, exposure) the downscaling model needs.

Retrieved through the Open-Meteo Historical Weather API (``models=era5_land``).
Each 0.1° node is cached separately, so downloads resume after rate-limit pauses.

Outputs
-------
``data/<d>/raw/era5land/nodes/<lat>_<lon>.parquet``
``data/<d>/interim/gp_met_obs.parquet``: gp_code, date, tmax_obs, tmin_obs, rh_obs, wind_obs, ...
``data/<d>/interim/block_met_obs.parquet``
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.common import manifest
from src.common.config import Config, load_config
from src.common.geo import Grid, area_weights
from src.common.http import get_json, patiently
from src.common.logging_utils import get_logger
from src.ingest.boundaries import load_blocks, load_panchayats

log = get_logger("ingest.era5land")

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DAILY = {
    "temperature_2m_max": "tmax_obs",
    "temperature_2m_min": "tmin_obs",
    "relative_humidity_2m_mean": "rh_obs",
    "relative_humidity_2m_max": "rhmax_obs",
    "relative_humidity_2m_min": "rhmin_obs",
    "wind_speed_10m_max": "wind_obs",
    "precipitation_sum": "era5l_rain",
    "et0_fao_evapotranspiration": "et0_obs",
    "shortwave_radiation_sum": "srad_obs",
}
OBS_VARS = list(DAILY.values())
# Open-Meteo's ERA5-Land archive serves temperature/humidity only; wind, precipitation, ET0 and
# radiation come back null. They are taken from ERA5 (0.25°) instead - coarser, and documented as such.
ERA5_DAILY = {
    "wind_speed_10m_max": "wind_obs",
    "precipitation_sum": "era5_rain",
    "et0_fao_evapotranspiration": "et0_obs",
    "shortwave_radiation_sum": "srad_obs",
}
LAND_ONLY = ["tmax_obs", "tmin_obs", "rh_obs", "rhmax_obs", "rhmin_obs"]


def era5_grid(cfg: Config) -> tuple[Grid, pd.DataFrame, pd.DataFrame]:
    grid = Grid.covering(cfg.bbox, float(cfg["ground_truth"]["temperature_humidity_wind"]["resolution_deg"]))
    gps = load_panchayats(cfg, modelled_only=True)
    blocks = load_blocks(cfg)
    wg = area_weights(gps, "gp_code", grid, cfg.metric_crs)
    wb = area_weights(blocks, "block_lgd", grid, cfg.metric_crs)
    return grid, wg, wb


def fetch_node(cfg: Config, lat: float, lon: float, cache: Path, model: str = "era5_land",
               daily: dict | None = None) -> Path:
    daily = daily or DAILY
    cache.mkdir(parents=True, exist_ok=True)
    out = cache / f"{lat:.2f}_{lon:.2f}.parquet"
    if out.exists():
        return out
    js = get_json(ARCHIVE_URL, params={
        "latitude": lat, "longitude": lon, "models": model,
        "start_date": cfg["period"]["start"], "end_date": cfg["period"]["end"],
        "daily": ",".join(daily), "timezone": cfg["forecast"]["timezone"], "wind_speed_unit": "kmh",
    })
    d = js["daily"]
    df = pd.DataFrame({"date": pd.to_datetime(d["time"])})
    for k, v in daily.items():
        df[v] = pd.to_numeric(pd.Series(d.get(k)), errors="coerce").astype("float32")
    df["lat"], df["lon"] = lat, lon
    df["om_lat"], df["om_lon"], df["om_elevation"] = js.get("latitude"), js.get("longitude"), js.get("elevation")
    tmp = out.with_suffix(".part")
    df.to_parquet(tmp, index=False)
    tmp.replace(out)
    return out


def _weighted(merged: pd.DataFrame, id_col: str, variables: list[str]) -> pd.DataFrame:
    parts = []
    for var in variables:
        ok = merged[var].notna()
        m = pd.DataFrame({id_col: merged[id_col], "date": merged["date"],
                          "wv": np.where(ok, merged[var] * merged["weight"], 0.0),
                          "wok": np.where(ok, merged["weight"], 0.0)})
        g = m.groupby([id_col, "date"])[["wv", "wok"]].sum()
        parts.append((g["wv"] / g["wok"].where(g["wok"] > 0.5)).astype("float32").rename(var))
    return pd.concat(parts, axis=1).reset_index()


def build(cfg: Config | None = None, patient: bool = True) -> None:
    cfg = cfg or load_config()
    paths = cfg.paths.ensure()
    cache = paths.raw / "era5land" / "nodes"
    grid, wg, wb = era5_grid(cfg)
    nodes = pd.concat([wg[["lat", "lon"]], wb[["lat", "lon"]]]).drop_duplicates().sort_values(["lat", "lon"])
    log.info("ERA5-Land: %d nodes at %.2f° for %s..%s", len(nodes), grid.res, cfg["period"]["start"], cfg["period"]["end"])
    _fetch_all(cfg, nodes, cache, patient, "era5_land", DAILY)

    # ERA5 (0.25°) supplement for wind / precipitation / ET0 / radiation
    from src.common.geo import Grid as _G
    g25 = _G.covering(cfg.bbox, 0.25)
    gps = load_panchayats(cfg, modelled_only=True)
    wg25 = area_weights(gps, "gp_code", g25, cfg.metric_crs)
    wb25 = area_weights(load_blocks(cfg), "block_lgd", g25, cfg.metric_crs)
    nodes25 = pd.concat([wg25[["lat", "lon"]], wb25[["lat", "lon"]]]).drop_duplicates().sort_values(["lat", "lon"])
    cache25 = paths.raw / "era5" / "nodes"
    log.info("ERA5 supplement: %d nodes at 0.25° for wind/precipitation/ET0/radiation", len(nodes25))
    _fetch_all(cfg, nodes25, cache25, patient, "era5", ERA5_DAILY)

    node_df = pd.concat([pd.read_parquet(f) for f in sorted(cache.glob("*.parquet"))], ignore_index=True)
    gp = _weighted(wg.merge(node_df, on=["lat", "lon"]), "gp_code", LAND_ONLY)
    blk = _weighted(wb.merge(node_df, on=["lat", "lon"]), "block_lgd", LAND_ONLY)
    n25 = pd.concat([pd.read_parquet(f) for f in sorted(cache25.glob("*.parquet"))], ignore_index=True)
    sup_vars = list(ERA5_DAILY.values())
    gp = gp.merge(_weighted(wg25.merge(n25, on=["lat", "lon"]), "gp_code", sup_vars), on=["gp_code", "date"], how="left")
    blk = blk.merge(_weighted(wb25.merge(n25, on=["lat", "lon"]), "block_lgd", sup_vars), on=["block_lgd", "date"],
                    how="left")
    OBS = LAND_ONLY + sup_vars
    gp.to_parquet(paths.interim / "gp_met_obs.parquet", index=False)
    blk.to_parquet(paths.interim / "block_met_obs.parquet", index=False)
    for name, fn in [("gp_met_obs", "gp_met_obs.parquet"), ("block_met_obs", "block_met_obs.parquet")]:
        manifest.register(paths.data / "manifest.json", name, paths.interim / fn,
                          source="ERA5-Land 0.1° (T, RH) + ERA5 0.25° (wind, precipitation, ET0, radiation) via Open-Meteo",
                          licence="Copernicus licence (free use with attribution); Open-Meteo CC-BY 4.0",
                          produced_by="src.ingest.era5land", extra={"nodes": int(len(nodes))})
    log.info("GP met obs: %d rows; missing %%: %s", len(gp), (gp[OBS].isna().mean() * 100).round(2).to_dict())


def _fetch_all(cfg, nodes, cache, patient, model, daily) -> None:
    for n, (lat, lon) in enumerate(nodes.itertuples(index=False), 1):
        patiently(lambda lat=lat, lon=lon: fetch_node(cfg, float(lat), float(lon), cache, model, daily), f"node {float(lat):.2f},{float(lon):.2f}", patient)
        if n % 10 == 0 or n == len(nodes):
            log.info("  nodes done %d/%d", n, len(nodes))



if __name__ == "__main__":
    build()
