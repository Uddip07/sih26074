"""
Administrative boundaries: extraction, cleaning, GP -> block join and QA.

Source: LGD (Local Government Directory) polygons redistributed by
urbanmorph/geodata (``LGD_Blocks.parquet``, ``LGD_panchayats.parquet``).

The raw Pune panchayat layer has 1,545 polygons, and they are not all GPs:
* 195 polygons have an **empty ``gp_code``**. These are municipal/cantonment/
  forest areas that do not belong to any Gram Panchayat. They are kept as
  ``non_gp_areas``: they are shown on the map and used for area checks, but no
  panchayat forecast is issued for them.
* Some GPs are split into several polygons with the same ``gp_code``
  (multi-part). These are **dissolved** into one geometry per GP.

Block assignment uses the **largest-area overlap** with the LGD block polygons
(not the centroid, which can fall outside a crescent-shaped GP). It is
cross-checked against the LGD ``block_name`` attribute, and every disagreement
is written to the QA report (spec section 5, step 1: "validate it manually").
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd

from src.common import manifest
from src.common.config import Config, load_config
from src.common.logging_utils import get_logger

log = get_logger("ingest.boundaries")

LGD_PANCHAYATS_URL = "https://pub-0429b8e3b5a946e69ea007df844a6f1c.r2.dev/admin/panchayats/LGD_panchayats.parquet"


def extract_from_lgd(cfg: Config) -> None:
    """
    Extract block and panchayat polygons for ``cfg.district.district_lgd`` from the
    national LGD parquet files (used when onboarding a new district).
    Needs DuckDB with the httpfs + spatial extensions (downloaded on first use).
    """
    import duckdb

    dist_lgd = int(cfg["district"]["district_lgd"])
    out_p = cfg.resolve(cfg["boundaries"]["source_panchayats"])
    out_b = cfg.resolve(cfg["boundaries"]["source_blocks"])
    out_p.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs; INSTALL spatial; LOAD spatial;")
    if not out_p.exists():
        log.info("Extracting panchayats for district LGD %s ...", dist_lgd)
        con.execute(f"""
            COPY (SELECT OBJECTID,
                         COALESCE(gpcode, gp_code) AS gp_code,
                         COALESCE(gpname, gp_name) AS gp_name,
                         COALESCE(blkname, block_name) AS block_name,
                         COALESCE(blklgdcode, blk_lgdcod) AS block_lgd,
                         dtname AS district_name, dt_lgd, stname AS state_name, st_lgd,
                         SHAPE_Area, geometry
                  FROM '{LGD_PANCHAYATS_URL}' WHERE dt_lgd = {dist_lgd})
            TO '{out_p.as_posix()}' WITH (FORMAT GDAL, DRIVER 'GeoJSON', LAYER_CREATION_OPTIONS 'RFC7946=YES')
        """)
    if not out_b.exists():
        local_blocks = cfg.paths.shared_raw / "lgd" / "LGD_Blocks.parquet"
        if not local_blocks.exists():
            from src.common.http import download_file

            download_file(cfg["boundaries"]["lgd_blocks_parquet_url"], local_blocks, min_bytes=10_000_000)
        src = local_blocks
        blocks = gpd.read_parquet(src)
        blocks = blocks[blocks["dist_lgd"] == dist_lgd]
        blocks.to_file(out_b, driver="GeoJSON")
    log.info("Boundaries available: %s, %s", out_p, out_b)


def _clean_text(s: object) -> str:
    return " ".join(str(s or "").split()).strip()


def build(cfg: Config | None = None) -> dict:
    cfg = cfg or load_config()
    paths = cfg.paths.ensure()
    metric = cfg.metric_crs
    src_p = cfg.resolve(cfg["boundaries"]["source_panchayats"])
    src_b = cfg.resolve(cfg["boundaries"]["source_blocks"])
    if not src_p.exists() or not src_b.exists():
        extract_from_lgd(cfg)

    raw = gpd.read_file(src_p).to_crs("EPSG:4326")
    blocks = gpd.read_file(src_b).to_crs("EPSG:4326")
    blocks["block_name"] = blocks["block_name"].map(_clean_text).str.upper()
    blocks["block_lgd"] = blocks["block_lgd"].astype(int)
    blocks = blocks[["block_name", "block_lgd", "geometry"]].reset_index(drop=True)
    blocks["geometry"] = blocks.geometry.make_valid()

    raw["gp_code"] = raw["gp_code"].map(_clean_text)
    raw["gp_name"] = raw["gp_name"].map(_clean_text)
    raw["geometry"] = raw.geometry.make_valid()
    n_raw = len(raw)

    # --- 1. split off non-GP polygons ------------------------------------------------
    is_gp = raw["gp_code"].str.fullmatch(r"\d+")
    non_gp = raw[~is_gp].copy()
    gp_raw = raw[is_gp].copy()

    # --- 2. dissolve multi-part GPs ---------------------------------------------------
    multipart_codes = gp_raw.loc[gp_raw["gp_code"].duplicated(keep=False), "gp_code"].unique()
    gps = gp_raw.dissolve(by="gp_code", aggfunc={"gp_name": "first", "block_name": "first"}).reset_index()
    gps = gps.rename(columns={"block_name": "lgd_block_name"})
    gps["lgd_block_name"] = gps["lgd_block_name"].map(_clean_text).str.upper()

    # --- 3. largest-overlap block assignment ------------------------------------------
    gps_m = gps.to_crs(metric)
    blocks_m = blocks.to_crs(metric)
    gps["area_km2"] = (gps_m.geometry.area / 1e6).round(4)
    inter = gpd.overlay(gps_m[["gp_code", "geometry"]], blocks_m, how="intersection", keep_geom_type=True)
    inter["a"] = inter.geometry.area
    inter = inter.sort_values("a", ascending=False)
    best = inter.drop_duplicates("gp_code").set_index("gp_code")
    tot = inter.groupby("gp_code")["a"].sum()
    gps["block_name"] = gps["gp_code"].map(best["block_name"])
    gps["block_lgd"] = gps["gp_code"].map(best["block_lgd"])
    gps["block_overlap_frac"] = gps["gp_code"].map((best["a"] / tot).round(4))
    gps["assignment_method"] = "largest_overlap"
    # GPs that fall entirely outside every block polygon (digitisation gaps)
    lost = gps["block_name"].isna()
    if lost.any():
        near = gpd.sjoin_nearest(gps_m.loc[lost, ["gp_code", "geometry"]], blocks_m, how="left")
        near = near.drop_duplicates("gp_code").set_index("gp_code")
        gps.loc[lost, "block_name"] = gps.loc[lost, "gp_code"].map(near["block_name"])
        gps.loc[lost, "block_lgd"] = gps.loc[lost, "gp_code"].map(near["block_lgd"])
        gps.loc[lost, "assignment_method"] = "nearest"
        gps.loc[lost, "block_overlap_frac"] = 0.0
    gps["block_lgd"] = gps["block_lgd"].astype(int)
    gps["attribute_agrees"] = gps["block_name"] == gps["lgd_block_name"]

    cen = gps_m.geometry.representative_point().to_crs("EPSG:4326")
    gps["latitude"] = cen.y.round(5).to_numpy()
    gps["longitude"] = cen.x.round(5).to_numpy()
    excluded = set(cfg["district"].get("exclude_blocks", []))
    gps["modelled"] = ~gps["block_name"].isin(excluded)

    gps = gps[["gp_code", "gp_name", "block_name", "block_lgd", "lgd_block_name", "attribute_agrees",
               "block_overlap_frac", "assignment_method", "area_km2", "latitude", "longitude", "modelled",
               "geometry"]].sort_values(["block_name", "gp_name"]).reset_index(drop=True)

    blocks["area_km2"] = (blocks_m.geometry.area / 1e6).round(2)
    blocks["n_gps"] = blocks["block_name"].map(gps.groupby("block_name").size()).fillna(0).astype(int)
    blocks["modelled"] = ~blocks["block_name"].isin(excluded)

    non_gp = non_gp[["OBJECTID", "block_name", "geometry"]].rename(columns={"block_name": "lgd_block_name"})
    non_gp["area_km2"] = (non_gp.to_crs(metric).geometry.area / 1e6).round(4)

    # --- 4. write outputs ---------------------------------------------------------------
    out_dir = paths.interim
    gps.to_parquet(out_dir / "panchayats.parquet")
    blocks.to_parquet(out_dir / "blocks.parquet")
    non_gp.to_parquet(out_dir / "non_gp_areas.parquet")
    lookup = gps.drop(columns="geometry")
    lookup.to_csv(out_dir / "panchayat_block_lookup.csv", index=False)

    mism = lookup[~lookup["attribute_agrees"]]
    qa = {
        "raw_polygons": int(n_raw),
        "non_gp_polygons": int(len(non_gp)),
        "non_gp_area_km2": round(float(non_gp["area_km2"].sum()), 2),
        "multipart_gp_codes": int(len(multipart_codes)),
        "gram_panchayats": int(len(gps)),
        "modelled_gram_panchayats": int(gps["modelled"].sum()),
        "blocks": int(len(blocks)),
        "gps_per_block": {k: int(v) for k, v in gps.groupby("block_name").size().items()},
        "attribute_disagreements": int(len(mism)),
        "attribute_disagreement_examples": mism[["gp_code", "gp_name", "lgd_block_name", "block_name",
                                                 "block_overlap_frac"]].head(25).to_dict("records"),
        "assigned_by_nearest": int((gps["assignment_method"] == "nearest").sum()),
        "gp_area_km2": {k: round(float(v), 3) for k, v in gps["area_km2"].describe().items()},
        "gp_plus_nongp_area_km2": round(float(gps["area_km2"].sum() + non_gp["area_km2"].sum()), 1),
        "block_area_km2": round(float(blocks["area_km2"].sum()), 1),
    }
    (paths.reports).mkdir(parents=True, exist_ok=True)
    (paths.reports / "boundary_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    _qa_map(gps, blocks, non_gp, paths.reports / "boundary_qa_map.png", cfg)

    mpath = paths.data / "manifest.json"
    for name, fn in [("panchayats", "panchayats.parquet"), ("blocks", "blocks.parquet"),
                     ("non_gp_areas", "non_gp_areas.parquet")]:
        manifest.register(mpath, name, out_dir / fn, source="LGD via urbanmorph/geodata",
                          licence="Government of India Open Data (GODL)", produced_by="src.ingest.boundaries")
    log.info("GPs: %d (modelled %d) | non-GP polygons: %d | multipart dissolved: %d | attr. disagreements: %d",
             qa["gram_panchayats"], qa["modelled_gram_panchayats"], qa["non_gp_polygons"],
             qa["multipart_gp_codes"], qa["attribute_disagreements"])
    return qa


def _qa_map(gps: gpd.GeoDataFrame, blocks: gpd.GeoDataFrame, non_gp: gpd.GeoDataFrame, out: Path, cfg: Config) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 10), dpi=130)
    names = sorted(blocks["block_name"].unique())
    cmap = plt.get_cmap("tab20", len(names))
    colour = {n: cmap(k) for k, n in enumerate(names)}
    gps.plot(ax=ax, color=gps["block_name"].map(colour), edgecolor="white", linewidth=0.15)
    non_gp.plot(ax=ax, color="#cccccc", edgecolor="none")
    bad = gps[~gps["attribute_agrees"]]
    if len(bad):
        bad.plot(ax=ax, facecolor="none", edgecolor="red", linewidth=1.0)
    blocks.boundary.plot(ax=ax, color="black", linewidth=0.9)
    for _, b in blocks.iterrows():
        p = b.geometry.representative_point()
        ax.annotate(f"{b['block_name']}\n{int(b['n_gps'])} GPs", (p.x, p.y), ha="center", fontsize=7, weight="bold")
    ax.set_title(f"{cfg.district_name}: Gram Panchayat -> block assignment QA\n"
                 f"colour = assigned block, grey = non-GP area, red outline = LGD attribute disagrees", fontsize=10)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def load_panchayats(cfg: Config | None = None, modelled_only: bool = False) -> gpd.GeoDataFrame:
    cfg = cfg or load_config()
    gdf = gpd.read_parquet(cfg.paths.interim / "panchayats.parquet")
    return gdf[gdf["modelled"]].reset_index(drop=True) if modelled_only else gdf


def load_blocks(cfg: Config | None = None, modelled_only: bool = False) -> gpd.GeoDataFrame:
    cfg = cfg or load_config()
    gdf = gpd.read_parquet(cfg.paths.interim / "blocks.parquet")
    return gdf[gdf["modelled"]].reset_index(drop=True) if modelled_only else gdf


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, default=str)[:3000])
