"""
Distance Feature Calculation Module
Computes distance from each panchayat centroid to nearest coastline and major rivers/waterbodies.
Matches Section 5, Step 5 and Section 6 of block-to-panchayat-downscaling-spec.md.
"""

import os
import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union
from typing import Optional

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PHYSICAL_DIR = os.path.join(BASE_DIR, "10m_physical")
COASTLINE_SHP = os.path.join(PHYSICAL_DIR, "ne_10m_coastline.shp")
RIVERS_SHP = os.path.join(PHYSICAL_DIR, "ne_10m_rivers_lake_centerlines.shp")
LAKES_SHP = os.path.join(PHYSICAL_DIR, "ne_10m_lakes.shp")
OUTPUT_CSV = os.path.join(BASE_DIR, "data", "interim", "panchayat_distances.csv")

UTM_CRS = "EPSG:32643"  # UTM Zone 43N (meters)


def compute_panchayat_distances(
    panchayats_gdf: gpd.GeoDataFrame,
    coastline_path: str = COASTLINE_SHP,
    rivers_path: str = RIVERS_SHP,
    lakes_path: str = LAKES_SHP,
    output_path: Optional[str] = OUTPUT_CSV
) -> pd.DataFrame:
    """
    Compute geodesic distance (in kilometers) from each panchayat centroid
    to the Arabian Sea coastline and nearest major river/water body.
    """
    if output_path and os.path.exists(output_path):
        df = pd.read_csv(output_path)
        print(f"Loaded existing distance features: {len(df)} rows from {output_path}")
        return df

    print("Computing geodesic distances to coastline and river networks in UTM metric CRS...")

    # Project panchayat centroids to UTM 43N
    panchayats_utm = panchayats_gdf.to_crs(UTM_CRS)
    centroids_utm = panchayats_utm.geometry.centroid

    # 1. Arabian Sea Coastline
    coast_gdf = gpd.read_file(coastline_path)
    # Clip to Indian West Coast bounding box for fast spatial processing
    coast_west = coast_gdf.cx[71.5:74.5, 14.0:21.0]
    if len(coast_west) == 0:
        coast_west = coast_gdf.cx[68.0:80.0, 8.0:25.0]
    coast_utm = coast_west.to_crs(UTM_CRS)
    coast_geom = unary_union(coast_utm.geometry)

    # 2. Major River Networks and Lakes (Bhima, Ghod, Nira, Indrayani, Pawana, Mula-Mutha, Kukadi)
    rivers_gdf = gpd.read_file(rivers_path)
    rivers_regional = rivers_gdf.cx[72.5:76.0, 16.5:20.5]
    if len(rivers_regional) == 0:
        rivers_regional = rivers_gdf.cx[70.0:80.0, 15.0:22.0]
    rivers_utm = rivers_regional.to_crs(UTM_CRS)

    water_geoms = [rivers_utm.geometry]
    if os.path.exists(lakes_path):
        lakes_gdf = gpd.read_file(lakes_path)
        lakes_reg = lakes_gdf.cx[72.5:76.0, 16.5:20.5]
        if len(lakes_reg) > 0:
            water_geoms.append(lakes_reg.to_crs(UTM_CRS).geometry)

    water_union = unary_union(pd.concat(water_geoms))

    # Compute distances in meters, convert to kilometers
    dist_coast_km = centroids_utm.distance(coast_geom) / 1000.0
    dist_water_km = centroids_utm.distance(water_union) / 1000.0

    res_df = pd.DataFrame({
        "gp_code": panchayats_gdf["gp_code"].values,
        "gp_name": panchayats_gdf["gp_name"].values,
        "dist_to_coast_km": dist_coast_km.round(2).values,
        "dist_to_water_km": dist_water_km.round(2).values
    })

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        res_df.to_csv(output_path, index=False)
        print(f"Saved distance features to {output_path}")

    return res_df


if __name__ == "__main__":
    from src.ingest.fetch_boundaries import load_panchayats
    gdf = load_panchayats()
    df = compute_panchayat_distances(gdf)
    print(df.head(10))
    print("\nSummary statistics:")
    print(df[["dist_to_coast_km", "dist_to_water_km"]].describe())
