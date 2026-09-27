"""
Web publishing (audit 6.8): geometry is published **once**, and values **per issue**.

The old dashboard shipped a 2.9 MB GeoJSON with one hard-coded day baked into the
properties. Now:

* ``outputs/<d>/web/geo/panchayats.geojson``: simplified (topology-preserving) GP polygons
  carrying only ids and names, cached by the browser. The same folder holds blocks, non-GP
  areas and the Ghat crest line.
* ``outputs/<d>/web/issues/<issue>.json``: compact arrays per GP (5 leads): point forecast,
  P10/P90, exceedance probabilities, block value, severity, SHAP families. About 1 MB,
  gzip-served.
* ``outputs/<d>/web/issues/index.json``: available issue dates, for the date picker.
"""

from __future__ import annotations

import json
from datetime import date

import geopandas as gpd
import numpy as np
import pandas as pd

from src.common.config import Config, load_config
from src.common.logging_utils import get_logger

log = get_logger("pipeline.publish")


def publish_geometry(cfg: Config | None = None, force: bool = False) -> None:
    cfg = cfg or load_config()
    geo = cfg.paths.web / "geo"
    geo.mkdir(parents=True, exist_ok=True)
    if (geo / "panchayats.geojson").exists() and not force:
        return
    it = cfg.paths.interim
    gps = gpd.read_parquet(it / "panchayats.parquet")
    static = pd.read_parquet(it / "gp_static.parquet")
    static["gp_code"] = static["gp_code"].astype(str)
    keep = ["gp_code", "elev_mean", "slope_mean", "lc_crop_pct", "lc_tree_pct", "clim_annual_gp", "soil_clay_pct",
            "crest_dist_km", "area_km2"]
    gps = gps.merge(static[[c for c in keep if c in static]].drop(columns=["area_km2"], errors="ignore"),
                    on="gp_code", how="left")
    for name, gdf, cols in [
        ("panchayats", gps, ["gp_code", "gp_name", "block_name", "block_lgd", "modelled", "area_km2", "elev_mean",
                             "slope_mean", "lc_crop_pct", "lc_tree_pct", "clim_annual_gp", "soil_clay_pct",
                             "crest_dist_km"]),
        ("blocks", gpd.read_parquet(it / "blocks.parquet"), ["block_name", "block_lgd", "area_km2", "n_gps", "modelled"]),
        ("non_gp_areas", gpd.read_parquet(it / "non_gp_areas.parquet"), ["lgd_block_name", "area_km2"]),
    ]:
        g = gdf[[c for c in cols if c in gdf] + ["geometry"]].copy()
        g["geometry"] = g.to_crs(cfg.metric_crs).geometry.simplify(cfg.dashboard["geometry"]["simplify_m"],
                                                                   preserve_topology=True).to_crs(4326)
        for c in g.columns:
            if g[c].dtype.kind == "f":
                g[c] = g[c].round(2)
        g.to_file(geo / f"{name}.geojson", driver="GeoJSON", COORDINATE_PRECISION=5)
    urban_areas(cfg, gps).to_file(geo / "urban_areas.geojson", driver="GeoJSON", COORDINATE_PRECISION=5)
    crest = it / "ghat_crest.geojson"
    if crest.exists():
        gpd.read_file(crest).to_file(geo / "ghat_crest.geojson", driver="GeoJSON")
    log.info("geometry published: %s", {p.name: f"{p.stat().st_size / 1e6:.2f} MB" for p in geo.glob("*.geojson")})


def urban_areas(cfg: Config, gps: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Everything inside the district's blocks that is not a modelled Gram Panchayat: municipal
    corporations (Pune, Pimpri-Chinchwad), municipal councils, cantonments and the Pune City block.
    They have no GP, so the dashboard shows their block forecast there, clearly labelled."""
    blocks = gpd.read_parquet(cfg.paths.interim / "blocks.parquet").to_crs(cfg.metric_crs)
    gp_union = gps[gps["modelled"].astype(bool)].to_crs(cfg.metric_crs).buffer(1).union_all()
    parts = blocks[["block_name", "block_lgd", "geometry"]].copy()
    parts["geometry"] = parts.geometry.difference(gp_union)
    parts = parts[~parts.geometry.is_empty].explode(index_parts=False)
    parts = parts[parts.geometry.area > cfg.dashboard["geometry"]["min_area_km2"] * 1e6]  # digitising slivers
    parts["geometry"] = parts.geometry.simplify(cfg.dashboard["geometry"]["simplify_m"], preserve_topology=True)
    parts["area_km2"] = (parts.geometry.area / 1e6).round(2)
    log.info("urban / non-GP areas: %d polygons, %.0f km²", len(parts), parts["area_km2"].sum())
    return parts.to_crs(4326).reset_index(drop=True)


def _arr(x) -> list:
    return [None if not np.isfinite(v) else round(float(v), 2) for v in np.asarray(x, float)]


def publish_issue(cfg: Config | None = None, issue: date | None = None) -> None:
    cfg = cfg or load_config()
    publish_geometry(cfg)
    d = cfg.paths.forecasts / issue.isoformat()
    gp = pd.read_parquet(d / "gp_forecast.parquet").sort_values(["gp_code", "lead_day"])
    adv = json.loads((d / "advisories.json").read_text(encoding="utf-8"))
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    fields = {"r": "rain_pred", "r10": "rain_q10", "r90": "rain_q90", "p25": "rain_p_ge_2p5", "p156": "rain_p_ge_15p6",
              "p645": "rain_p_ge_64p5", "rm": "rain_pred_mass", "tx": "tmax_pred", "tn": "tmin_pred", "rh": "rh_pred",
              "w": "wind_pred", "fr": "fc_rain", "ftx": "fc_tmax", "ftn": "fc_tmin", "frh": "fc_rh", "fw": "fc_wind",
              "tx10": "tmax_q10", "tx90": "tmax_q90"}
    shap_cols = [c for c in gp.columns if c.startswith("shap_rain_")]
    payload = {"issue_date": issue.isoformat(), "meta": meta, "leads": sorted(int(x) for x in gp["lead_day"].unique()),
               "valid_dates": sorted(str(x.date()) for x in gp["valid_date"].unique()), "gp": {}, "blocks": {}}
    for code, g in gp.groupby("gp_code"):
        rec = {k: _arr(g[c]) for k, c in fields.items() if c in g}
        a = adv[code]  # every downscaled GP has an advisory record; a gap is a pipeline error
        rec["sev"] = a["overall"]
        rec["n_adv"] = sum(1 for x in a["advisories"] if x["severity"] != "green")
        rec["rules"] = sorted({x["rule"].split(":")[0] for x in a["advisories"] if x["severity"] != "green"})
        if shap_cols:
            rec["shap_rain"] = {c.replace("shap_rain_", ""): _arr(g[c]) for c in shap_cols}
        payload["gp"][code] = rec
    blk = pd.read_parquet(d / "block_forecast.parquet").sort_values(["block_lgd", "lead_day"])
    for lgd, g in blk.groupby("block_lgd"):
        payload["blocks"][str(lgd)] = {"r": _arr(g["rain"]), "tx": _arr(g["tmax"]), "tn": _arr(g["tmin"]),
                                       "rh": _arr(g["rh"]), "w": _arr(g["wind"])}
    out = cfg.paths.web / "issues"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{issue.isoformat()}.json").write_text(json.dumps(payload, separators=(",", ":"), default=str),
                                                   encoding="utf-8")
    rebuild_index(cfg)


def rebuild_index(cfg: Config) -> list[str]:
    out = cfg.paths.web / "issues"
    issues = sorted([p.stem for p in out.glob("20*.json")], reverse=True)
    (out / "index.json").write_text(json.dumps({"issues": issues}), encoding="utf-8")
    return issues


if __name__ == "__main__":
    publish_geometry(force=True)
