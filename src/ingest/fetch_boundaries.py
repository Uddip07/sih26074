"""
Boundary Ingestion & Spatial Join Module
Extracts and links Block and Panchayat boundaries from urbanmorph/geodata LGD layers.
Matches Section 5, Step 1 of block-to-panchayat-downscaling-spec.md.
"""

import os
import geopandas as gpd
import pandas as pd
from typing import Tuple

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "boundaries"))
BLOCKS_PATH = os.path.join(DATA_DIR, "pune_blocks.geojson")
PANCHAYATS_PATH = os.path.join(DATA_DIR, "pune_panchayats.geojson")
LOOKUP_PATH = os.path.join(DATA_DIR, "panchayat_block_lookup.csv")

# Pilot Region Constants
PILOT_DISTRICT_NAME = "Pune"
PILOT_DISTRICT_LGD = 490
PILOT_STATE_NAME = "MAHARASHTRA"
PILOT_STATE_LGD = 27
UTM_CRS = "EPSG:32643"  # UTM Zone 43N for Western India


def load_blocks(path: str = BLOCKS_PATH) -> gpd.GeoDataFrame:
    """Load official LGD block polygons for the pilot district."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Blocks file not found at {path}. Run extraction first.")
    gdf = gpd.read_file(path)
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    return gdf


def load_panchayats(path: str = PANCHAYATS_PATH) -> gpd.GeoDataFrame:
    """Load official LGD panchayat polygons for the pilot district."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Panchayats file not found at {path}. Run extraction first.")
    gdf = gpd.read_file(path)
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    return gdf


def build_boundary_lookup(
    blocks_path: str = BLOCKS_PATH,
    panchayats_path: str = PANCHAYATS_PATH,
    output_lookup_path: str = LOOKUP_PATH
) -> Tuple[pd.DataFrame, gpd.GeoDataFrame]:
    """
    Perform spatial join: panchayat centroids within parent block polygons.
    Produces the authoritative panchayat_id -> block_id lookup table.
    """
    os.makedirs(os.path.dirname(output_lookup_path), exist_ok=True)
    blocks = load_blocks(blocks_path)
    panchayats = load_panchayats(panchayats_path)

    # Project to metric UTM for accurate geometric centroid calculation
    panchayats_utm = panchayats.to_crs(UTM_CRS)
    blocks_utm = blocks.to_crs(UTM_CRS)

    panchayat_centroids = panchayats.copy()
    # Centroids in UTM reprojected back to WGS84
    panchayat_centroids["geometry"] = panchayats_utm.geometry.centroid.to_crs("EPSG:4326")

    # Spatial join: centroid within block polygon
    joined = gpd.sjoin(
        panchayat_centroids,
        blocks[["block_name", "block_lgd", "geometry"]],
        how="left",
        predicate="within"
    )

    # For any boundary edge centroids, resolve via nearest block
    unmatched_mask = joined["block_name_right"].isna()
    if unmatched_mask.any():
        unmatched_idx = joined[unmatched_mask].index
        nearest = gpd.sjoin_nearest(
            panchayat_centroids.loc[unmatched_idx],
            blocks[["block_name", "block_lgd", "geometry"]],
            how="left"
        )
        for idx, row in nearest.iterrows():
            joined.loc[idx, "block_name_right"] = row["block_name_right"]
            joined.loc[idx, "block_lgd_right"] = row["block_lgd_right"]

    joined["assigned_block_name"] = joined["block_name_right"]
    joined["assigned_block_lgd"] = joined["block_lgd_right"].astype(int)

    lookup_cols = [
        "gp_code",
        "gp_name",
        "assigned_block_name",
        "assigned_block_lgd",
        "district_name",
        "state_name"
    ]
    available_cols = [c for c in lookup_cols if c in joined.columns]
    lookup_df = joined[available_cols].drop_duplicates(subset=["gp_code"])

    lookup_df.to_csv(output_lookup_path, index=False)
    print(f"Panchayat-to-Block lookup created: {len(lookup_df)} entries saved to {output_lookup_path}")
    return lookup_df, joined


if __name__ == "__main__":
    df, joined = build_boundary_lookup()
    print("Block distribution:")
    print(df["assigned_block_name"].value_counts())
