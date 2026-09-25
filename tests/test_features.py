"""
Test Static Features & Master Feature Table Assembly
"""

import os
import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import pytest
import numpy as np
import pandas as pd
from src.features.build_feature_table import load_static_covariates


def test_static_covariates_integrity():
    static_df = load_static_covariates()
    assert len(static_df) >= 1000

    # Required columns
    required_cols = [
        "gp_code", "gp_name", "assigned_block_name",
        "latitude", "longitude",
        "elevation_mean", "elevation_std", "slope_mean",
        "landuse_cropland_pct", "landuse_forest_pct", "landuse_water_pct", "landuse_builtup_pct",
        "dist_to_coast_km", "dist_to_water_km"
    ]
    for c in required_cols:
        assert c in static_df.columns, f"Missing required column {c}"

    # Elevation and slope bounds
    assert (static_df["elevation_mean"] >= 450.0).all()
    assert (static_df["elevation_mean"] <= 1450.0).all()
    assert (static_df["slope_mean"] > 0.0).all()

    # LULC sums to 100% (within tolerance)
    lulc_sum = (
        static_df["landuse_cropland_pct"] +
        static_df["landuse_forest_pct"] +
        static_df["landuse_water_pct"] +
        static_df["landuse_builtup_pct"]
    )
    assert np.allclose(lulc_sum, 100.0, atol=0.2), "LULC fractions must sum to 100%"

    # Distances are positive
    assert (static_df["dist_to_coast_km"] > 0).all()
    assert (static_df["dist_to_water_km"] >= 0).all()


def test_processed_splits():
    train_path = os.path.join(BASE_DIR, "data", "processed", "train_test_splits", "train.parquet")
    test_path = os.path.join(BASE_DIR, "data", "processed", "train_test_splits", "test.parquet")

    assert os.path.exists(train_path), "train.parquet missing"
    assert os.path.exists(test_path), "test.parquet missing"

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    assert len(train_df) > 0
    assert len(test_df) > 0

    # Ensure no spatial leakage (panchayat overlap) between train and test
    train_pids = set(train_df["panchayat_id"].unique())
    test_pids = set(test_df["panchayat_id"].unique())
    overlap = train_pids.intersection(test_pids)
    assert len(overlap) == 0, f"Spatial leakage detected! Overlapping panchayats: {len(overlap)}"
