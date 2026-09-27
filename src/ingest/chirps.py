"""
Panchayat-scale rainfall ground truth: CHIRPS v2.0 daily, 0.05° (~5.5 km) (Feature F2).

CHIRPS (Climate Hazards Group InfraRed Precipitation with Station data, UCSB)
blends satellite rainfall with rain-gauge observations. At 0.05° it is the
finest-resolution *observation-based* daily rainfall that is freely available
for India. Its pixels (~30 km²) are about 35x smaller than a Pune block
(median ~1,100 km²) and comparable to a Gram Panchayat (median ~8 km², mean 10.6 km²).

We read only the district window from CHC's Cloud-Optimised GeoTIFFs
(HTTP range requests), so no global files are downloaded.

Also provides **CHPclim v2** monthly rainfall climatology (0.05°). It is used for
the leakage-free ``clim_ratio`` feature (GP normal / block normal).

Outputs
-------
``data/<d>/raw/chirps/chirps_<product>_<start>_<end>.nc``: (time, lat, lon) cube
``data/<d>/raw/chirps/chpclim_monthly.nc``: (month, lat, lon)
``data/<d>/interim/gp_rain_obs.parquet``: gp_code, date, rain_obs (area-weighted)
``data/<d>/interim/block_rain_obs.parquet``: block_lgd, date, rain_obs
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.windows import from_bounds

from src.common import manifest
from src.common.config import Config, load_config
from src.common.geo import Grid, area_weights, stack_to_polygons, weight_matrix
from src.common.logging_utils import get_logger
from src.ingest.boundaries import load_blocks, load_panchayats

log = get_logger("ingest.chirps")

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "3")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "60")

BASE = "https://data.chc.ucsb.edu/products"
URLS = {
    "final": BASE + "/CHIRPS-2.0/global_daily/cogs/p05/{y}/chirps-v2.0.{y}.{m:02d}.{d:02d}.cog",
    # COGs are published with a lag; the gzipped GeoTIFF of the same final product is the fallback
    "final_gz": BASE + "/CHIRPS-2.0/global_daily/tifs/p05/{y}/chirps-v2.0.{y}.{m:02d}.{d:02d}.tif.gz",
    "prelim": BASE + "/CHIRPS-2.0/prelim/global_daily/tifs/p05/{y}/chirps-v2.0.{y}.{m:02d}.{d:02d}.tif.gz",
}
CHPCLIM_URL = BASE + "/CHPclim/v2/monthly_9090/CHPclim2.90-90.{m:02d}.tif"
SOURCE = "CHIRPS v2.0 daily p05 (Funk et al. 2015), Climate Hazards Center UCSB"
LICENCE = "Public domain (CHC) - cite Funk et al. 2015, doi:10.1038/sdata.2015.66"


def _read_window(url: str, bbox: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the bbox window; return (values[lat_asc, lon_asc], lats_asc, lons_asc).

    COGs are range-read over HTTP. Gzipped GeoTIFFs cannot be range-read, and GDAL's
    streaming /vsigzip/ reader has no timeout, so they are downloaded with a timeout,
    decompressed into memory and read from a MemoryFile.
    """
    if url.endswith(".gz"):
        import gzip

        import requests
        from rasterio.io import MemoryFile

        last = None
        for _ in range(4):
            try:
                r = requests.get(url, timeout=(20, 120))
                if r.status_code == 404:
                    raise FileNotFoundError(url)
                r.raise_for_status()
                with MemoryFile(gzip.decompress(r.content)) as mf, mf.open() as src:
                    win = from_bounds(*bbox, transform=src.transform).round_offsets().round_lengths()
                    arr = src.read(1, window=win).astype("float32")
                    t = src.window_transform(win)
                    nodata = src.nodata
                break
            except FileNotFoundError:
                raise
            except Exception as exc:  # noqa: BLE001
                last = exc
        else:
            raise RuntimeError(f"{url}: {last}")
    else:
        for attempt in range(4):
            try:
                with rasterio.open("/vsicurl/" + url) as src:
                    win = from_bounds(*bbox, transform=src.transform).round_offsets().round_lengths()
                    arr = src.read(1, window=win).astype("float32")
                    t = src.window_transform(win)
                    nodata = src.nodata
                break
            except rasterio.errors.RasterioIOError:
                if attempt == 3:
                    raise
    if nodata is not None:
        arr[arr == nodata] = np.nan
    arr[arr < 0] = np.nan
    nrow, ncol = arr.shape
    lons = t.c + t.a * (np.arange(ncol) + 0.5)
    lats = t.f + t.e * (np.arange(nrow) + 0.5)
    order = np.argsort(lats)
    return arr[order, :], np.round(lats[order], 4), np.round(lons, 4)


def fetch_daily_cube(cfg: Config, start: str, end: str, product: str = "final", workers: int = 12) -> xr.DataArray:
    out = cfg.paths.raw / "chirps" / f"chirps_{product}_{start}_{end}.nc"
    if out.exists():
        return xr.open_dataarray(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    days = pd.date_range(start, end, freq="D")
    results: dict[pd.Timestamp, np.ndarray] = {}
    lats = lons = None
    failed: list[str] = []

    def job(d: pd.Timestamp):
        try:
            return d, _read_window(URLS[product].format(y=d.year, m=d.month, d=d.day), cfg.bbox)
        except Exception:  # noqa: BLE001
            if product != "final":
                raise
            return d, _read_window(URLS["final_gz"].format(y=d.year, m=d.month, d=d.day), cfg.bbox)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(job, d): d for d in days}
        for k, f in enumerate(as_completed(futs), 1):
            d = futs[f]
            try:
                _, (arr, la, lo) = f.result()
                results[d] = arr
                lats, lons = la, lo
            except Exception as exc:  # noqa: BLE001
                failed.append(f"{d.date()}: {exc}")
            if k % 100 == 0:
                log.info("  CHIRPS %s: %d/%d days", product, k, len(days))
    if failed:
        log.warning("%d CHIRPS days failed (kept as missing): %s", len(failed), failed[:5])
    cube = np.full((len(days), len(lats), len(lons)), np.nan, dtype="float32")
    for n, d in enumerate(days):
        if d in results:
            cube[n] = results[d]
    da = xr.DataArray(cube, dims=("time", "lat", "lon"), coords={"time": days, "lat": lats, "lon": lons},
                      name="rain", attrs={"units": "mm/day", "source": SOURCE, "product": product})
    da.to_netcdf(out, encoding={"rain": {"zlib": True, "complevel": 4}})
    manifest.register(cfg.paths.data / "manifest.json", f"chirps_{product}_cube", out, source=SOURCE,
                      licence=LICENCE, produced_by="src.ingest.chirps",
                      extra={"days": len(days), "missing_days": len(failed), "shape": list(cube.shape)})
    return da


def fetch_climatology(cfg: Config) -> xr.DataArray:
    out = cfg.paths.raw / "chirps" / "chpclim_monthly.nc"
    if out.exists():
        return xr.open_dataarray(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stack, lats, lons = [], None, None
    for m in range(1, 13):
        arr, lats, lons = _read_window(CHPCLIM_URL.format(m=m), cfg.bbox)
        stack.append(arr)
    da = xr.DataArray(np.stack(stack), dims=("month", "lat", "lon"),
                      coords={"month": np.arange(1, 13), "lat": lats, "lon": lons}, name="clim",
                      attrs={"units": "mm/month", "source": "CHPclim v2 (Funk et al. 2015) 1981-2010 normals"})
    da.to_netcdf(out)
    manifest.register(cfg.paths.data / "manifest.json", "chpclim_monthly", out,
                      source="CHPclim v2 monthly climatology, CHC UCSB", licence=LICENCE, produced_by="src.ingest.chirps")
    return da


def grid_of(da: xr.DataArray) -> Grid:
    lats = da["lat"].to_numpy()
    return Grid(lats, da["lon"].to_numpy(), float(np.round(np.median(np.diff(lats)), 4)))


def to_polygons(cfg: Config, da: xr.DataArray, value_name: str = "rain_obs") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Area-weighted GP and block series from a (time, lat, lon) cube."""
    grid = grid_of(da)
    gps = load_panchayats(cfg, modelled_only=True)
    blocks = load_blocks(cfg)
    out = []
    for polys, id_col in ((gps, "gp_code"), (blocks, "block_lgd")):
        w = area_weights(polys, id_col, grid, cfg.metric_crs)
        ids = list(polys[id_col])
        W = weight_matrix(w, id_col, ids, len(grid.lats), len(grid.lons))
        vals = stack_to_polygons(da.to_numpy(), W)
        df = pd.DataFrame(vals, index=pd.DatetimeIndex(da["time"].to_numpy(), name="date"), columns=ids)
        df = df.stack(future_stack=True).rename(value_name).reset_index().rename(columns={"level_1": id_col})
        df[value_name] = df[value_name].astype("float32").round(3)
        out.append(df)
    return out[0], out[1]


def climatology_to_polygons(cfg: Config, clim: xr.DataArray) -> tuple[pd.DataFrame, pd.DataFrame]:
    grid = grid_of(clim)
    gps = load_panchayats(cfg, modelled_only=True)
    blocks = load_blocks(cfg)
    res = []
    for polys, id_col in ((gps, "gp_code"), (blocks, "block_lgd")):
        w = area_weights(polys, id_col, grid, cfg.metric_crs)
        ids = list(polys[id_col])
        W = weight_matrix(w, id_col, ids, len(grid.lats), len(grid.lons))
        vals = stack_to_polygons(clim.to_numpy(), W)  # (12, n)
        df = pd.DataFrame(vals.T, index=pd.Index(ids, name=id_col), columns=[f"clim_m{m:02d}" for m in range(1, 13)])
        df["clim_annual"] = df.sum(axis=1)
        df["clim_jjas"] = df[["clim_m06", "clim_m07", "clim_m08", "clim_m09"]].sum(axis=1)
        res.append(df.reset_index())
    return res[0], res[1]


def update_prelim(cfg: Config) -> None:
    """
    Preliminary CHIRPS (~2-day latency) extends the truth beyond the final archive. It is used
    ONLY for rolling operational verification (F29) and the latency-honest antecedent-rain
    feature at inference time, never for training.
    """
    end = cfg["period"]["end"]
    p_start = (date.fromisoformat(end) + timedelta(days=1)).isoformat()
    p_end = (date.today() - timedelta(days=2)).isoformat()
    if p_start > p_end:
        return
    for old in (cfg.paths.raw / "chirps").glob("chirps_prelim_*.nc"):
        if old.name != f"chirps_prelim_{p_start}_{p_end}.nc":
            old.unlink()
    pda = fetch_daily_cube(cfg, p_start, p_end, "prelim", workers=6)
    pgp, pblk = to_polygons(cfg, pda)
    pgp = pgp.dropna()
    pgp.to_parquet(cfg.paths.interim / "gp_rain_obs_prelim.parquet", index=False)
    pblk.dropna().to_parquet(cfg.paths.interim / "block_rain_obs_prelim.parquet", index=False)
    log.info("CHIRPS prelim: %d GP-days (%s..%s)", len(pgp), p_start, str(pgp["date"].max().date()) if len(pgp) else "-")


def build(cfg: Config | None = None, include_prelim: bool = True) -> None:
    cfg = cfg or load_config()
    paths = cfg.paths.ensure()
    start, end = cfg["period"]["start"], cfg["period"]["end"]
    log.info("CHIRPS final %s..%s over bbox %s", start, end, cfg.bbox)
    da = fetch_daily_cube(cfg, start, end, "final")
    gp, blk = to_polygons(cfg, da)
    gp.to_parquet(paths.interim / "gp_rain_obs.parquet", index=False)
    blk.to_parquet(paths.interim / "block_rain_obs.parquet", index=False)
    log.info("GP rain obs: %d rows, %.2f%% missing", len(gp), 100 * gp["rain_obs"].isna().mean())

    if include_prelim:
        update_prelim(cfg)

    clim = fetch_climatology(cfg)
    gpc, blc = climatology_to_polygons(cfg, clim)
    gpc.to_parquet(paths.interim / "gp_climatology.parquet", index=False)
    blc.to_parquet(paths.interim / "block_climatology.parquet", index=False)
    for name, fn in [("gp_rain_obs", "gp_rain_obs.parquet"), ("gp_climatology", "gp_climatology.parquet")]:
        manifest.register(paths.data / "manifest.json", name, paths.interim / fn, source=SOURCE, licence=LICENCE,
                          produced_by="src.ingest.chirps")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-prelim", action="store_true")
    build(include_prelim=not ap.parse_args().no_prelim)
