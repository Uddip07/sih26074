"""
Real station observations (IMD AWS, Maharashtra Mahavedh, KVK, rain gauges): **validation only**.

These networks are not openly downloadable, so this adapter reads CSV files the DAMU/KVK
places in ``data/<district>/raw/stations/``. It never fabricates data and never trains on it:
station values are compared with the downscaled forecast of the Gram Panchayat that contains
each station.

CSV columns (one row per station-day; unknown extra columns ignored):
    station_id, station_name, latitude, longitude, date (YYYY-MM-DD, IST day), rain_mm,
    [tmax_c], [tmin_c], [rh_pct]
Rain may use the IMD 08:30-08:30 convention: set ``day_convention: imd0830`` in the district
config under ``ground_truth.stations`` and the date is shifted -1 day to align with IST calendar days.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

from src.common.config import Config, load_config
from src.common.logging_utils import get_logger
from src.ingest.boundaries import load_panchayats

log = get_logger("ingest.stations")
REQUIRED = {"station_id", "latitude", "longitude", "date", "rain_mm"}


def load(cfg: Config | None = None) -> pd.DataFrame | None:
    cfg = cfg or load_config()
    d = cfg.paths.raw / "stations"
    files = sorted(Path(d).glob("*.csv")) if d.exists() else []
    if not files:
        return None
    frames = []
    for f in files:
        df = pd.read_csv(f)
        df.columns = [c.strip().lower() for c in df.columns]
        missing = REQUIRED - set(df.columns)
        if missing:
            raise ValueError(f"{f.name}: missing columns {sorted(missing)}")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    if (cfg.get("ground_truth.stations.day_convention") or "") == "imd0830":
        df["date"] = df["date"] - pd.Timedelta(days=1)
    for c in ("rain_mm", "tmax_c", "tmin_c", "rh_pct"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df.loc[(df["rain_mm"] < 0) | (df["rain_mm"] > 600), "rain_mm"] = float("nan")  # QC: impossible values
    pts = gpd.GeoDataFrame(df[["station_id", "latitude", "longitude"]].drop_duplicates("station_id"),
                           geometry=gpd.points_from_xy(df.drop_duplicates("station_id")["longitude"],
                                                       df.drop_duplicates("station_id")["latitude"]), crs=4326)
    gps = load_panchayats(cfg, modelled_only=True)[["gp_code", "geometry"]]
    j = gpd.sjoin(pts, gps, how="left", predicate="within")[["station_id", "gp_code"]]
    df = df.merge(j, on="station_id", how="left")
    outside = df["gp_code"].isna()
    if outside.any():
        log.warning("%d station rows fall outside modelled GPs and are ignored", int(outside.sum()))
    df = df[~outside]
    log.info("stations: %d stations, %d station-days", df["station_id"].nunique(), len(df))
    return df.rename(columns={"date": "valid_date"})
