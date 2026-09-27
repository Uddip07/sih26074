"""
Terrain covariates from the Copernicus DEM GLO-30 (30 m) (audit 3.3, 3.7).

Why these features: rainfall over Pune is controlled by the Western Ghats.
Moist south-west monsoon air rises over the escarpment, giving 3,000–6,000 mm
per year at the crest (Velhe, Mulshi, Maval), and descends into a rain shadow
(400–600 mm per year at Baramati, Indapur, Daund). Temperature follows elevation
through the lapse rate, and valleys pool cold air at night. We therefore
describe every Gram Panchayat by:

========================  ===================================================
elev_mean/std/min/max/p90 elevation distribution inside the GP (m)
slope_mean                mean slope (degrees), Horn's method on UTM 30 m
aspect_sin / aspect_cos   mean unit vector of slope aspect (north = cos 1)
windward_exposure         mean of sin(slope)·cos(aspect − flow_from): > 0 means
                          the GP's slopes face the monsoon flow (orographic uplift)
tpi_300m / tpi_2km        topographic position index: ridge (+) vs valley (−)
crest_dist_km             signed distance to the Western Ghats crest line,
                          traced from the DEM (+ = east/leeward, − = west)
upwind_barrier_m          highest terrain within 40 km upwind (along the monsoon
                          flow) minus the GP's own elevation: rain-shadow depth
========================  ===================================================

Rasters are cached under ``data/<d>/raw/dem`` (git-ignored). Features are
written to ``data/<d>/interim/gp_terrain.parquet`` (and ``block_terrain.parquet``).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject
from scipy import ndimage
from shapely.geometry import LineString

from src.common import manifest
from src.common.config import Config, load_config
from src.common.http import download_file
from src.common.logging_utils import get_logger
from src.ingest.boundaries import load_blocks, load_panchayats

log = get_logger("ingest.terrain")

COP_URL = ("https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM/"
           "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM.tif")
SOURCE = "Copernicus DEM GLO-30 (ESA / Airbus), AWS Open Data Registry"
LICENCE = "Copernicus DEM licence: free, worldwide, with attribution ((c) DLR e.V. 2010-2014 and (c) Airbus 2014-2018, provided under COPERNICUS by the EU and ESA)"
RES_M = 30.0


def download_tiles(cfg: Config) -> list[Path]:
    west, south, east, north = cfg.bbox
    tiles = []
    for lat in range(int(np.floor(south)), int(np.floor(north)) + 1):
        for lon in range(int(np.floor(west)), int(np.floor(east)) + 1):
            url = COP_URL.format(ns="N" if lat >= 0 else "S", lat=abs(lat), ew="E" if lon >= 0 else "W", lon=abs(lon))
            tiles.append((url, cfg.paths.raw / "dem" / "tiles" / Path(url).name))
    log.info("Copernicus GLO-30: %d tiles", len(tiles))
    with ThreadPoolExecutor(max_workers=4) as ex:
        return list(ex.map(lambda t: download_file(t[0], t[1], min_bytes=100_000), tiles))


def build_dem(cfg: Config) -> Path:
    out = cfg.paths.raw / "dem" / "dem_utm30.tif"
    if out.exists():
        return out
    tiles = download_tiles(cfg)
    srcs = [rasterio.open(p) for p in tiles]
    mosaic, transform = merge(srcs, bounds=cfg.bbox)
    crs = srcs[0].crs
    for s in srcs:
        s.close()
    mosaic = mosaic[0].astype("float32")
    h, w = mosaic.shape
    dst_t, dw, dh = calculate_default_transform(crs, cfg.metric_crs, w, h,
                                                *rasterio.transform.array_bounds(h, w, transform), resolution=RES_M)
    dst = np.full((dh, dw), np.nan, dtype="float32")
    reproject(mosaic, dst, src_transform=transform, src_crs=crs, dst_transform=dst_t, dst_crs=cfg.metric_crs,
              resampling=Resampling.bilinear, src_nodata=-32767, dst_nodata=np.nan)
    profile = {"driver": "GTiff", "height": dh, "width": dw, "count": 1, "dtype": "float32", "crs": cfg.metric_crs,
               "transform": dst_t, "nodata": np.nan, "compress": "deflate", "predictor": 3, "tiled": True}
    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out, "w", **profile) as dst_f:
        dst_f.write(dst, 1)
    manifest.register(cfg.paths.data / "manifest.json", "dem_utm30", out, SOURCE, LICENCE, "src.ingest.terrain",
                      extra={"tiles": [p.name for p in tiles], "resolution_m": RES_M})
    return out


def _slope_aspect(z: np.ndarray, res: float) -> tuple[np.ndarray, np.ndarray]:
    """Horn (1981) 3x3 slope (deg) and aspect (deg clockwise from north, direction the slope faces)."""
    zp = np.pad(z, 1, mode="edge")
    a, b, c = zp[:-2, :-2], zp[:-2, 1:-1], zp[:-2, 2:]
    d, f = zp[1:-1, :-2], zp[1:-1, 2:]
    g, h, i = zp[2:, :-2], zp[2:, 1:-1], zp[2:, 2:]
    dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * res)          # east-positive
    dzdy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * res)          # south-positive (row index increases south)
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    # downslope direction vector = -(gradient); north component = +dzdy (since dzdy is south-positive)
    aspect = (np.degrees(np.arctan2(-dzdx, dzdy)) + 360.0) % 360.0
    aspect[np.hypot(dzdx, dzdy) < 1e-6] = np.nan
    return slope.astype("float32"), aspect.astype("float32")


def _tpi(z: np.ndarray, radius_px: int) -> np.ndarray:
    ok = np.isfinite(z)
    zf = np.where(ok, z, 0.0)
    size = 2 * radius_px + 1
    s = ndimage.uniform_filter(zf, size=size, mode="nearest")
    n = ndimage.uniform_filter(ok.astype("float32"), size=size, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        return (z - s / n).astype("float32")


def _zonal(labels: np.ndarray, n: int, arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-label mean, std and count (label 0 = background), ignoring NaN."""
    lab = labels.ravel()
    v = arr.ravel()
    ok = (lab > 0) & np.isfinite(v)
    lab, v = lab[ok], v[ok].astype("float64")
    cnt = np.bincount(lab, minlength=n + 1)
    s1 = np.bincount(lab, weights=v, minlength=n + 1)
    s2 = np.bincount(lab, weights=v * v, minlength=n + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / cnt
        std = np.sqrt(np.maximum(s2 / cnt - mean ** 2, 0))
    return mean[1:], std[1:], cnt[1:]


def _crest_line(dem_path: Path, cfg: Config) -> LineString:
    """Trace the Western Ghats crest: per 0.01° latitude row, the longitude of the smoothed elevation maximum."""
    lo_min, lo_max = cfg["static"]["ghat_crest_search_lon"]
    with rasterio.open(dem_path) as src:
        scale = 33  # ~1 km pixels
        z = src.read(1, out_shape=(src.height // scale, src.width // scale), resampling=Resampling.average)
        t = src.transform * src.transform.scale(src.width / z.shape[1], src.height / z.shape[0])
        crs = src.crs
    import pyproj

    to_ll = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    rows, cols = np.indices(z.shape)
    xs, ys = rasterio.transform.xy(t, rows.ravel(), cols.ravel())
    lon, lat = to_ll.transform(np.asarray(xs), np.asarray(ys))
    lon = lon.reshape(z.shape)
    lat = lat.reshape(z.shape)
    zs = ndimage.uniform_filter(np.nan_to_num(z, nan=0.0), size=5)
    pts = []
    for r in range(z.shape[0]):
        mask = (lon[r] >= lo_min) & (lon[r] <= lo_max)
        if mask.sum() < 5:
            continue
        c = np.argmax(np.where(mask, zs[r], -1e9))
        pts.append((lat[r, c], lon[r, c]))
    pts_df = pd.DataFrame(pts, columns=["lat", "lon"]).sort_values("lat")
    pts_df["lon"] = pts_df["lon"].rolling(15, center=True, min_periods=1).median()
    return LineString(list(zip(pts_df["lon"], pts_df["lat"])))


def build(cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    paths = cfg.paths.ensure()
    dem_path = build_dem(cfg)
    gps = load_panchayats(cfg, modelled_only=False)
    blocks = load_blocks(cfg)

    with rasterio.open(dem_path) as src:
        z = src.read(1)
        transform, crs, shape = src.transform, src.crs, src.shape
    log.info("DEM %s px (%.0f m); elevation range %.0f..%.0f m", shape, RES_M, np.nanmin(z), np.nanmax(z))

    slope, aspect = _slope_aspect(z, RES_M)
    flow = np.radians(float(cfg["static"]["monsoon_flow_from_deg"]))
    asp = np.radians(aspect)
    exposure = np.where(np.isfinite(asp), np.sin(np.radians(slope)) * np.cos(asp - flow), 0.0).astype("float32")
    a_sin = np.where(np.isfinite(asp), np.sin(asp), np.nan).astype("float32")
    a_cos = np.where(np.isfinite(asp), np.cos(asp), np.nan).astype("float32")
    tpi300 = _tpi(z, 10)
    tpi2k = _tpi(z, 67)

    def zonal_table(polys, id_col: str) -> pd.DataFrame:
        pm = polys.to_crs(crs)
        labels = rasterize(((g, k + 1) for k, g in enumerate(pm.geometry)), out_shape=shape, transform=transform,
                           fill=0, dtype="int32")
        n = len(pm)
        e_mean, e_std, cnt = _zonal(labels, n, z)
        df = pd.DataFrame({id_col: polys[id_col].to_numpy(), "elev_mean": e_mean, "elev_std": e_std, "dem_pixels": cnt})
        # min / max / p90 via groupby (exact)
        lab = labels.ravel()
        ok = (lab > 0) & np.isfinite(z.ravel())
        s = pd.Series(z.ravel()[ok], index=lab[ok])
        g = s.groupby(level=0)
        df["elev_min"] = g.min().reindex(range(1, n + 1)).to_numpy()
        df["elev_max"] = g.max().reindex(range(1, n + 1)).to_numpy()
        df["elev_p90"] = g.quantile(0.9).reindex(range(1, n + 1)).to_numpy()
        for name, arr in [("slope_mean", slope), ("aspect_sin", a_sin), ("aspect_cos", a_cos),
                          ("windward_exposure", exposure), ("tpi_300m", tpi300), ("tpi_2km", tpi2k)]:
            df[name] = _zonal(labels, n, arr)[0]
        # Tiny polygons (< 1 pixel centre inside) fall back to the value at the representative point
        small = df["dem_pixels"] == 0
        if small.any():
            rp = pm.geometry.representative_point()
            rows_cols = [rasterio.transform.rowcol(transform, p.x, p.y) for p in rp[small]]
            for col, arr in [("elev_mean", z), ("slope_mean", slope), ("windward_exposure", exposure)]:
                df.loc[small, col] = [arr[r, c] for r, c in rows_cols]
        return df

    gp_t = zonal_table(gps, "gp_code")
    blk_t = zonal_table(blocks, "block_lgd")

    # --- crest distance and upwind barrier (1 km DEM) ------------------------------------
    crest = _crest_line(dem_path, cfg)
    import geopandas as gpd

    crest_m = gpd.GeoSeries([crest], crs="EPSG:4326").to_crs(crs).iloc[0]
    rp_ll = gps.to_crs(crs).geometry.representative_point()
    rp_geo = gpd.GeoSeries(rp_ll, crs=crs).to_crs("EPSG:4326")
    crest_lon_at = np.interp(rp_geo.y.to_numpy(), np.array(crest.xy[1]), np.array(crest.xy[0]))
    sign = np.where(rp_geo.x.to_numpy() >= crest_lon_at, 1.0, -1.0)
    gp_t["crest_dist_km"] = (sign * np.array([crest_m.distance(p) for p in rp_ll]) / 1000.0).round(2)

    flow_dir = np.radians(float(cfg["static"]["monsoon_flow_from_deg"]))
    ux, uy = np.sin(flow_dir), np.cos(flow_dir)  # unit vector pointing *towards* the upwind side
    zs = ndimage.uniform_filter(np.nan_to_num(z, nan=0.0), size=33)
    barrier = []
    for p, e in zip(rp_ll, gp_t["elev_mean"]):
        best = -np.inf
        for d_km in range(1, 41):
            x, y = p.x + ux * d_km * 1000, p.y + uy * d_km * 1000
            r, c = rasterio.transform.rowcol(transform, x, y)
            if 0 <= r < shape[0] and 0 <= c < shape[1]:
                best = max(best, float(zs[r, c]))
        barrier.append(max(0.0, best - e) if np.isfinite(best) else 0.0)
    gp_t["upwind_barrier_m"] = np.round(barrier, 1)

    for df in (gp_t, blk_t):
        for c in df.columns:
            if df[c].dtype.kind == "f":
                df[c] = df[c].round(4)
    gp_t.to_parquet(paths.interim / "gp_terrain.parquet", index=False)
    blk_t.to_parquet(paths.interim / "block_terrain.parquet", index=False)
    gpd.GeoDataFrame({"name": ["western_ghats_crest"]}, geometry=[crest], crs="EPSG:4326").to_file(
        paths.interim / "ghat_crest.geojson", driver="GeoJSON")
    manifest.register(paths.data / "manifest.json", "gp_terrain", paths.interim / "gp_terrain.parquet",
                      SOURCE, LICENCE, "src.ingest.terrain")
    log.info("terrain features for %d GPs; elev_mean %.0f..%.0f m; crest_dist %.1f..%.1f km",
             len(gp_t), gp_t["elev_mean"].min(), gp_t["elev_mean"].max(),
             gp_t["crest_dist_km"].min(), gp_t["crest_dist_km"].max())
    return gp_t


if __name__ == "__main__":
    print(build().describe().T)
