"""
Soil covariates from ISRIC SoilGrids 2.0 (250 m) (feature list 11.4, used by F19/F20).

Per GP, mean over 0–30 cm (depth-weighted 0–5, 5–15, 15–30 cm):
* ``soil_clay_pct``, ``soil_sand_pct``, ``soil_soc_gkg`` (soil organic carbon)
* ``soil_awc_mm``: plant-available water capacity of the top 30 cm (mm). It is
  derived with the **Saxton & Rawls (2006)** pedotransfer functions from sand,
  clay and organic matter (field capacity at -33 kPa minus wilting point at
  -1500 kPa), because SoilGrids ``latest`` publishes no water-retention layers.

AWC and texture drive the irrigation (soil water budget) and waterlogging
advisories: heavy black cotton soils (Vertisols, high clay) of the Deccan plateau
pond water much faster than the lateritic soils of the Ghats.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

from src.common import manifest
from src.common.config import Config, load_config
from src.common.logging_utils import get_logger
from src.ingest.boundaries import load_panchayats

log = get_logger("ingest.soil")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "6")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "5")

URL = "https://files.isric.org/soilgrids/latest/data/{v}/{v}_{d}_mean.vrt"
DEPTHS = {"0-5cm": 5.0, "5-15cm": 10.0, "15-30cm": 15.0}
# SoilGrids stores integers: clay/sand in g/kg (/10 -> %), soc in dg/kg (/10 -> g/kg)
SCALE = {"clay": 0.1, "sand": 0.1, "soc": 0.1}
SOURCE = "ISRIC SoilGrids 2.0 (Poggio et al. 2021), 250 m"
LICENCE = "CC-BY 4.0 ISRIC - World Soil Information"


def saxton_rawls_awc(sand_pct: np.ndarray, clay_pct: np.ndarray, om_pct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Saxton & Rawls (2006) eqs. 1-2: volumetric water content (m3/m3) at -1500 and -33 kPa."""
    S, C, OM = sand_pct / 100.0, clay_pct / 100.0, np.clip(om_pct, 0, 8)
    t1500 = -0.024 * S + 0.487 * C + 0.006 * OM + 0.005 * S * OM - 0.013 * C * OM + 0.068 * S * C + 0.031
    wp = t1500 + (0.14 * t1500 - 0.02)
    t33 = -0.251 * S + 0.195 * C + 0.011 * OM + 0.006 * S * OM - 0.027 * C * OM + 0.452 * S * C + 0.299
    fc = t33 + (1.283 * t33 ** 2 - 0.374 * t33 - 0.015)
    return wp, fc


WCS = ("https://maps.isric.org/mapserv?map=/map/{v}.map&SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage"
       "&COVERAGEID={v}_{d}_mean&FORMAT=image/tiff&SUBSET=long({w},{e})&SUBSET=lat({s},{n})"
       "&SUBSETTINGCRS=http://www.opengis.net/def/crs/EPSG/0/4326&OUTPUTCRS=http://www.opengis.net/def/crs/EPSG/0/4326")


def _read(var: str, depth: str, bbox4326, cache_dir):
    """Fetch one SoilGrids layer for the bbox through the ISRIC WCS (robust small GeoTIFF),
    falling back to windowed reads of the public VRT/COG mosaic."""
    w, s_, e, n = bbox4326
    dest = cache_dir / f"{var}_{depth}.tif"
    if not dest.exists():
        from src.common.http import download_file

        try:
            download_file(WCS.format(v=var, d=depth, w=w, e=e, s=s_, n=n), dest, min_bytes=10_000)
        except RuntimeError:
            for attempt in range(6):
                try:
                    with rasterio.open("/vsicurl/" + URL.format(v=var, d=depth)) as src:
                        b = transform_bounds("EPSG:4326", src.crs, *bbox4326, densify_pts=41)
                        win = from_bounds(*b, transform=src.transform).round_offsets().round_lengths()
                        arr = src.read(1, window=win).astype("float32")
                        prof = src.profile | {"driver": "GTiff", "height": arr.shape[0], "width": arr.shape[1],
                                              "transform": src.window_transform(win), "dtype": "float32"}
                    with rasterio.open(dest, "w", **prof) as dst:
                        dst.write(arr, 1)
                    break
                except rasterio.errors.RasterioIOError as exc:
                    log.warning("SoilGrids %s %s attempt %d failed: %s", var, depth, attempt + 1, exc)
                    time.sleep(15 * (attempt + 1))
            else:
                raise RuntimeError(f"SoilGrids {var} {depth} unavailable")
    with rasterio.open(dest) as src:
        arr = src.read(1).astype("float32")
        nod = src.nodata
        arr[arr <= 0] = np.nan  # 0 = SoilGrids no-data / masked (urban, water)
        if nod is not None:
            arr[arr == nod] = np.nan
        return arr * SCALE[var], src.transform, src.crs


def build(cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    paths = cfg.paths.ensure()
    cache = paths.raw / "soil"
    cache.mkdir(parents=True, exist_ok=True)
    gps = load_panchayats(cfg)
    layers: dict[str, np.ndarray] = {}
    for var in ("clay", "sand", "soc"):
        for d in DEPTHS:
            arr, transform, crs = _read(var, d, cfg.bbox, cache)
            layers[f"{var}_{d}"] = arr

    tot = sum(DEPTHS.values())
    prof = {v: sum(layers[f"{v}_{d}"] * w for d, w in DEPTHS.items()) / tot for v in ("clay", "sand", "soc")}
    om_pct = prof["soc"] / 10.0 * 1.724  # g/kg -> % carbon -> % organic matter (van Bemmelen)
    wp, fc = saxton_rawls_awc(prof["sand"], prof["clay"], om_pct)
    awc_mm = np.clip(fc - wp, 0, 0.35) * 300.0  # top 300 mm

    gm = gps.to_crs(crs)
    shape = prof["clay"].shape
    labels = rasterize(((g, k + 1) for k, g in enumerate(gm.geometry)), out_shape=shape, transform=transform,
                       fill=0, all_touched=True, dtype="int32")
    n = len(gm)
    out = {"gp_code": gps["gp_code"].to_numpy()}
    for name, arr in [("soil_clay_pct", prof["clay"]), ("soil_sand_pct", prof["sand"]),
                      ("soil_soc_gkg", prof["soc"]), ("soil_awc_mm", awc_mm)]:
        lab = labels.ravel()
        v = arr.ravel()
        ok = (lab > 0) & np.isfinite(v)
        s = np.bincount(lab[ok], weights=v[ok], minlength=n + 1)[1:]
        c = np.bincount(lab[ok], minlength=n + 1)[1:]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[name] = np.round(s / c, 3)
    df = pd.DataFrame(out)
    # Urban GPs can be SoilGrids-masked; fill from the nearest GP with data (documented, not a constant)
    miss = df["soil_clay_pct"].isna()
    if miss.any():
        from scipy.spatial import cKDTree

        xy = np.c_[gps["longitude"], gps["latitude"]]
        tree = cKDTree(xy[~miss.to_numpy()])
        _, idx = tree.query(xy[miss.to_numpy()])
        src_rows = df[~miss].iloc[idx]
        for c in ["soil_clay_pct", "soil_sand_pct", "soil_soc_gkg", "soil_awc_mm"]:
            df.loc[miss, c] = src_rows[c].to_numpy()
        log.warning("%d GPs had no SoilGrids pixels; filled from nearest GP", int(miss.sum()))
    df.to_parquet(paths.interim / "gp_soil.parquet", index=False)
    manifest.register(paths.data / "manifest.json", "gp_soil", paths.interim / "gp_soil.parquet", SOURCE, LICENCE,
                      "src.ingest.soil", extra={"pedotransfer": "Saxton & Rawls 2006", "nearest_filled": int(miss.sum())})
    log.info("soil: clay %.1f±%.1f %%, AWC %.0f±%.0f mm (0-30 cm)", df["soil_clay_pct"].mean(),
             df["soil_clay_pct"].std(), df["soil_awc_mm"].mean(), df["soil_awc_mm"].std())
    return df


if __name__ == "__main__":
    print(build().describe().T)
