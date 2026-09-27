"""
Land use / land cover fractions from ESA WorldCover 10 m 2021 v200 (audit 3.4).

The previous implementation downloaded only tile N18E072, which misses the
south of the district (< 18°N) and the east (> 75°E). It silently gave every
uncovered GP 60/25/5/10 %, and it reassigned "bare" pixels 60/40 to cropland/
built-up with an arbitrary rule. This module:

* reads **every** 3°x3° tile intersecting the district bbox (windowed COG reads),
* reports the **exact** class fractions (no reassignment, no fallback values),
* fails loudly if any GP has less than 99 % valid-pixel coverage.

Classes: 10 tree, 20 shrub, 30 grass, 40 crop, 50 built-up, 60 bare/sparse,
80 permanent water, 90 herbaceous wetland (70 snow, 95 mangrove, 100 moss are
absent in Pune but are handled).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.merge import merge
from rasterio.windows import from_bounds

from src.common import manifest
from src.common.config import Config, load_config
from src.common.logging_utils import get_logger
from src.ingest.boundaries import load_blocks, load_panchayats

log = get_logger("ingest.landcover")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")

WC_URL = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
SOURCE = "ESA WorldCover 10 m 2021 v200 (Zanaga et al. 2022)"
LICENCE = "CC-BY 4.0 (c) ESA WorldCover project 2021"
CLASSES = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "builtup", 60: "bare",
           70: "snow", 80: "water", 90: "wetland", 95: "mangrove", 100: "moss"}


def tiles_for(bbox: tuple[float, float, float, float]) -> list[str]:
    west, south, east, north = bbox
    out = []
    for lat in range(int(np.floor(south / 3) * 3), int(np.floor(north / 3) * 3) + 1, 3):
        for lon in range(int(np.floor(west / 3) * 3), int(np.floor(east / 3) * 3) + 1, 3):
            out.append(f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}{'E' if lon >= 0 else 'W'}{abs(lon):03d}")
    return out


def build_mosaic(cfg: Config) -> Path:
    out = cfg.paths.raw / "lulc" / "worldcover_2021_clip.tif"
    if out.exists():
        return out
    srcs = []
    for t in tiles_for(cfg.bbox):
        local = cfg.paths.raw / "lulc" / f"ESA_WorldCover_10m_2021_v200_{t}_Map.tif"
        srcs.append(rasterio.open(local if local.exists() else "/vsicurl/" + WC_URL.format(tile=t)))
    log.info("WorldCover tiles: %s", [Path(s.name).name for s in srcs])
    mosaic, transform = merge(srcs, bounds=cfg.bbox, nodata=0)
    crs = srcs[0].crs
    for s in srcs:
        s.close()
    prof = {"driver": "GTiff", "dtype": "uint8", "count": 1, "height": mosaic.shape[1], "width": mosaic.shape[2],
            "crs": crs, "transform": transform, "nodata": 0, "compress": "deflate", "tiled": True,
            "blockxsize": 512, "blockysize": 512}
    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out, "w", **prof) as dst:
        dst.write(mosaic)
    manifest.register(cfg.paths.data / "manifest.json", "worldcover_clip", out, SOURCE, LICENCE,
                      "src.ingest.landcover", extra={"tiles": tiles_for(cfg.bbox)})
    return out


def fractions(mosaic: Path, polys, id_col: str) -> pd.DataFrame:
    rows = []
    with rasterio.open(mosaic) as src:
        polys = polys.to_crs(src.crs)
        for pid, geom in zip(polys[id_col], polys.geometry):
            win = from_bounds(*geom.bounds, transform=src.transform).round_offsets().round_lengths()
            arr = src.read(1, window=win)
            inside = ~geometry_mask([geom], out_shape=arr.shape, transform=src.window_transform(win), all_touched=False)
            if inside.sum() == 0:  # sliver polygon: take touched pixels
                inside = ~geometry_mask([geom], out_shape=arr.shape, transform=src.window_transform(win), all_touched=True)
            vals = arr[inside]
            n_all = vals.size
            vals = vals[vals > 0]
            counts = np.bincount(vals, minlength=101)
            tot = max(1, vals.size)
            rec = {id_col: pid, "lulc_valid_frac": round(vals.size / max(1, n_all), 4), "lulc_pixels": int(n_all)}
            for code, name in CLASSES.items():
                rec[f"lc_{name}_pct"] = round(100.0 * counts[code] / tot, 3)
            rows.append(rec)
    df = pd.DataFrame(rows)
    # Agronomically meaningful aggregates (no reassignment of classes)
    df["lc_natural_veg_pct"] = df[["lc_tree_pct", "lc_shrub_pct", "lc_grass_pct"]].sum(axis=1).round(3)
    df["lc_water_all_pct"] = df[["lc_water_pct", "lc_wetland_pct"]].sum(axis=1).round(3)
    return df


def build(cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    paths = cfg.paths.ensure()
    mosaic = build_mosaic(cfg)
    gps = load_panchayats(cfg)
    gp = fractions(mosaic, gps, "gp_code")
    blk = fractions(mosaic, load_blocks(cfg), "block_lgd")
    bad = gp[gp["lulc_valid_frac"] < 0.99]
    if len(bad):
        raise RuntimeError(f"{len(bad)} GPs have <99% WorldCover coverage: {bad[['gp_code', 'lulc_valid_frac']].head()}")
    gp.to_parquet(paths.interim / "gp_landcover.parquet", index=False)
    blk.to_parquet(paths.interim / "block_landcover.parquet", index=False)
    manifest.register(paths.data / "manifest.json", "gp_landcover", paths.interim / "gp_landcover.parquet",
                      SOURCE, LICENCE, "src.ingest.landcover")
    log.info("land cover for %d GPs; district mean crop %.1f%% tree %.1f%% built %.1f%% water %.1f%%",
             len(gp), gp["lc_crop_pct"].mean(), gp["lc_tree_pct"].mean(), gp["lc_builtup_pct"].mean(),
             gp["lc_water_all_pct"].mean())
    return gp


if __name__ == "__main__":
    print(build().describe().T)
