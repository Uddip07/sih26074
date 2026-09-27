"""
Static (time-invariant) Gram Panchayat covariates. One row per modelled GP.

Merges terrain, land cover, hydrology, soil, climatology and (if built)
vegetation tables. It also derives **within-block contrasts**
(``elev_diff = GP elevation - block mean elevation``, climatology ratios, ...).
Downscaling is about how a GP differs from its block, so these contrasts are
the most informative inputs.

Nothing is imputed with made-up constants. If an optional source is missing,
its columns are dropped and the omission is logged and recorded in the feature
report.
"""

from __future__ import annotations

import json

import pandas as pd

from src.common.config import Config, load_config
from src.common.logging_utils import get_logger
from src.ingest.boundaries import load_panchayats

log = get_logger("features.static")

REQUIRED = ["gp_terrain", "gp_landcover", "gp_hydro", "gp_climatology"]
OPTIONAL = ["gp_soil", "gp_ndvi_static"]


def build_static(cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    it = cfg.paths.interim
    gps = load_panchayats(cfg, modelled_only=True).drop(columns="geometry")
    base = gps[["gp_code", "gp_name", "block_name", "block_lgd", "latitude", "longitude", "area_km2"]].copy()

    report = {"required": {}, "optional": {}}
    for name in REQUIRED:
        f = it / f"{name}.parquet"
        if not f.exists():
            raise FileNotFoundError(f"required static table missing: {f} - run its ingest stage first")
        t = pd.read_parquet(f)
        t["gp_code"] = t["gp_code"].astype(str)
        base = base.merge(t, on="gp_code", how="left")
        report["required"][name] = list(t.columns.drop("gp_code"))
    for name in OPTIONAL:
        f = it / f"{name}.parquet"
        if f.exists():
            t = pd.read_parquet(f)
            t["gp_code"] = t["gp_code"].astype(str)
            base = base.merge(t, on="gp_code", how="left")
            report["optional"][name] = list(t.columns.drop("gp_code"))
        else:
            log.warning("optional static table %s not found - its features are excluded", name)
            report["optional"][name] = None

    # --- block-level context and within-block contrasts -----------------------------------
    bt = pd.read_parquet(it / "block_terrain.parquet")[["block_lgd", "elev_mean", "elev_max", "slope_mean"]]
    bt = bt.rename(columns={"elev_mean": "blk_elev_mean", "elev_max": "blk_elev_max", "slope_mean": "blk_slope_mean"})
    base = base.merge(bt, on="block_lgd", how="left")
    base["elev_diff"] = base["elev_mean"] - base["blk_elev_mean"]
    base["elev_max_diff"] = base["elev_max"] - base["blk_elev_mean"]

    bc = pd.read_parquet(it / "block_climatology.parquet")
    bc = bc.rename(columns={c: f"blk_{c}" for c in bc.columns if c != "block_lgd"})
    base = base.merge(bc, on="block_lgd", how="left")
    base["clim_annual_gp"] = base["clim_annual"]
    base["clim_ratio_annual"] = base["clim_annual"] / base["blk_clim_annual"]

    num = base.select_dtypes("number").columns
    na = base[num].isna().sum()
    if na.any():
        raise ValueError(f"static features contain NaN: {na[na > 0].to_dict()}")
    base = base.sort_values("gp_code").reset_index(drop=True)
    base.to_parquet(it / "gp_static.parquet", index=False)
    report["n_gps"] = int(len(base))
    report["columns"] = list(base.columns)
    (cfg.paths.reports / "static_features.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("static table: %d GPs x %d columns", len(base), base.shape[1])
    return base


if __name__ == "__main__":
    print(build_static().describe().T.to_string())
