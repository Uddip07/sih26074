"""
Distance / hydrology covariates (audit 3.5).

* ``coast_dist_km``: distance to the Arabian Sea coastline (Natural Earth 10 m).
  Downloaded automatically; the old code needed a manually unzipped, untracked folder.
* ``river_dist_km`` and ``river_density_km_km2``: OpenStreetMap ``waterway=river``
  lines (Overpass API). Natural Earth rivers are far too sparse at GP scale, which
  explains the old implausible 63 km maximum.
* ``reservoir_dist_km``: distance to the nearest water body of at least 1 km²
  (Ujani, Khadakwasla, Panshet, Mulshi, Pawana, Bhatghar, Dimbhe, ...), traced
  from the ESA WorldCover permanent-water class.

Distances are measured from each GP's representative point, in UTM 43N.
"""

from __future__ import annotations

import json
import time
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.enums import Resampling
from scipy import ndimage
from shapely.geometry import LineString
from shapely.ops import unary_union

from src.common import manifest
from src.common.config import Config, load_config
from src.common.http import USER_AGENT, download_file
from src.common.logging_utils import get_logger
from src.ingest.boundaries import load_panchayats
from src.ingest.landcover import build_mosaic

log = get_logger("ingest.hydro")

NE_COAST = "https://naciscdn.org/naturalearth/10m/physical/ne_10m_coastline.zip"
HYDRORIVERS = "https://data.hydrosheds.org/file/HydroRIVERS/HydroRIVERS_v10_as_shp.zip"
OVERPASS = ["https://overpass.kumi.systems/api/interpreter", "https://overpass-api.de/api/interpreter",
            "https://maps.mail.ru/osm/tools/overpass/api/interpreter"]


def coastline(cfg: Config) -> gpd.GeoDataFrame:
    z = download_file(NE_COAST, cfg.paths.shared_raw / "natural_earth" / "ne_10m_coastline.zip", min_bytes=1_000_000)
    d = z.parent / "ne_10m_coastline"
    if not d.exists():
        with zipfile.ZipFile(z) as f:
            f.extractall(d)
    gdf = gpd.read_file(d / "ne_10m_coastline.shp")
    return gdf.cx[68:78, 8:24]


def hydrorivers(cfg: Config, min_order: int = 2) -> gpd.GeoDataFrame:
    """HydroRIVERS v1.0 (Lehner & Grill 2013) reaches in the district bbox, Strahler order >= ``min_order``."""
    z = download_file(HYDRORIVERS, cfg.paths.shared_raw / "hydrosheds" / "HydroRIVERS_v10_as_shp.zip",
                      min_bytes=50_000_000)
    d = z.parent / "HydroRIVERS_v10_as_shp"
    if not d.exists():
        with zipfile.ZipFile(z) as f:
            f.extractall(d)
    shp = next(d.rglob("HydroRIVERS_v10_as.shp"))
    w, s, e, n = cfg.bbox
    gdf = gpd.read_file(shp, bbox=(w - 0.2, s - 0.2, e + 0.2, n + 0.2))
    gdf = gdf[gdf["ORD_STRA"] >= min_order][["HYRIV_ID", "ORD_STRA", "UPLAND_SKM", "DIS_AV_CMS", "geometry"]]
    return gdf.to_crs("EPSG:4326")


def osm_rivers(cfg: Config) -> gpd.GeoDataFrame:
    out = cfg.paths.raw / "hydro" / "osm_rivers.geojson"
    if out.exists():
        return gpd.read_file(out)
    w, s, e, n = cfg.bbox
    w, s, e, n = w - 0.2, s - 0.2, e + 0.2, n + 0.2
    # Query in 0.6° tiles: small responses survive busy public Overpass servers
    lat_edges = np.arange(s, n + 1e-9, 0.6).tolist() + [n]
    lon_edges = np.arange(w, e + 1e-9, 0.6).tolist() + [e]
    elements: dict[int, dict] = {}
    for a, b in zip(lat_edges[:-1], lat_edges[1:]):
        for c, d in zip(lon_edges[:-1], lon_edges[1:]):
            if b - a < 1e-6 or d - c < 1e-6:
                continue
            q = f'[out:json][timeout:120];way["waterway"="river"]({a},{c},{b},{d});out geom;'
            js = None
            for attempt in range(6):
                url = OVERPASS[attempt % len(OVERPASS)]
                try:
                    r = requests.post(url, data={"data": q}, headers={"User-Agent": USER_AGENT}, timeout=180)
                    if r.status_code == 200 and r.text.lstrip().startswith("{"):
                        js = r.json()
                        break
                    log.warning("overpass %s -> HTTP %s", url, r.status_code)
                except requests.RequestException as exc:
                    log.warning("overpass %s failed: %s", url, str(exc)[:120])
                time.sleep(10 * (attempt + 1))
            if js is None:
                raise RuntimeError(f"Overpass failed for tile {a:.1f},{c:.1f}")
            for el in js.get("elements", []):
                elements[el["id"]] = el
    rows = []
    for el in elements.values():
        g = el.get("geometry") or []
        if len(g) >= 2:
            rows.append({"osm_id": el["id"], "name": el.get("tags", {}).get("name", ""),
                         "geometry": LineString([(p["lon"], p["lat"]) for p in g])})
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    out.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out, driver="GeoJSON")
    manifest.register(cfg.paths.data / "manifest.json", "osm_rivers", out, "OpenStreetMap contributors via Overpass API",
                      "ODbL 1.0", "src.ingest.hydro", extra={"features": len(gdf)})
    return gdf


def reservoir_distance_raster(cfg: Config):
    """Distance (m) to water bodies >= 1 km² on a ~100 m grid derived from WorldCover."""
    mosaic = build_mosaic(cfg)
    with rasterio.open(mosaic) as src:
        f = 10
        wc = src.read(1, out_shape=(src.height // f, src.width // f), resampling=Resampling.mode)
        t = src.transform * src.transform.scale(src.width / wc.shape[1], src.height / wc.shape[0])
    water = wc == 80
    lab, n = ndimage.label(water)
    lat_c = (cfg.bbox[1] + cfg.bbox[3]) / 2
    px_w = abs(t.a) * 111_320 * np.cos(np.radians(lat_c))
    px_h = abs(t.e) * 110_574
    sizes = ndimage.sum(water, lab, index=np.arange(1, n + 1)) * px_w * px_h / 1e6
    keep = np.isin(lab, np.where(sizes >= 1.0)[0] + 1)
    dist = ndimage.distance_transform_edt(~keep, sampling=(px_h, px_w))
    return dist, t, int((sizes >= 1.0).sum())


def build(cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    paths = cfg.paths.ensure()
    gps = load_panchayats(cfg)
    metric = cfg.metric_crs
    gm = gps.to_crs(metric)
    rp = gm.geometry.representative_point()

    coast = unary_union(coastline(cfg).to_crs(metric).geometry)
    # HydroRIVERS is the primary (stable, versioned, has stream order); OSM is used when
    # its cached extract exists or HydroRIVERS cannot be fetched.
    try:
        rivers = hydrorivers(cfg).to_crs(metric)
        river_source = "HydroRIVERS v1.0 (Strahler order >= 2)"
    except Exception as exc:  # noqa: BLE001
        log.warning("HydroRIVERS unavailable (%s) - falling back to OpenStreetMap", exc)
        rivers = osm_rivers(cfg).to_crs(metric)
        river_source = "OpenStreetMap waterway=river"
    river_u = unary_union(rivers.geometry)

    dist, t, n_res = reservoir_distance_raster(cfg)
    rp_ll = gpd.GeoSeries(rp, crs=metric).to_crs("EPSG:4326")
    rows, cols = rasterio.transform.rowcol(t, rp_ll.x.to_numpy(), rp_ll.y.to_numpy())
    rows = np.clip(np.asarray(rows), 0, dist.shape[0] - 1)
    cols = np.clip(np.asarray(cols), 0, dist.shape[1] - 1)

    inter_len = gpd.overlay(gm[["gp_code", "geometry"]], rivers[["geometry"]], how="intersection", keep_geom_type=False)
    inter_len["len_km"] = inter_len.geometry.length / 1000
    rlen = inter_len.groupby("gp_code")["len_km"].sum()

    df = pd.DataFrame({
        "gp_code": gps["gp_code"],
        "coast_dist_km": (rp.distance(coast) / 1000).round(2).to_numpy(),
        "river_dist_km": (rp.distance(river_u) / 1000).round(3).to_numpy(),
        "river_density_km_km2": (gps["gp_code"].map(rlen).fillna(0) / gps["area_km2"]).round(4).to_numpy(),
        "reservoir_dist_km": (dist[rows, cols] / 1000).round(3),
    })
    df.to_parquet(paths.interim / "gp_hydro.parquet", index=False)
    manifest.register(paths.data / "manifest.json", "gp_hydro", paths.interim / "gp_hydro.parquet",
                      f"Natural Earth 10m coastline; {river_source}; ESA WorldCover water",
                      "Public domain / HydroSHEDS licence (free, attribution) or ODbL / CC-BY 4.0", "src.ingest.hydro",
                      extra={"osm_river_ways": int(len(rivers)), "reservoirs_ge_1km2": n_res})
    log.info("hydro: %d river ways, %d reservoirs >=1 km2; coast %.0f-%.0f km; river dist max %.1f km",
             len(rivers), n_res, df["coast_dist_km"].min(), df["coast_dist_km"].max(), df["river_dist_km"].max())
    (paths.reports / "hydro_summary.json").write_text(json.dumps(df.describe().round(3).to_dict()), encoding="utf-8")
    return df


if __name__ == "__main__":
    print(build().describe().T)
