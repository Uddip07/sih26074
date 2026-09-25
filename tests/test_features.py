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
from src.features.build_feature_table import load_static_covariates, assemble_feature_table


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


def test_ground_truth_source_population(tmp_path):
    """
    Confirm ground_truth_source column is populated with 'imerg' when matching IMERG
    records exist and falls back to 'synthetic_orographic' otherwise.
    """
    static_df = load_static_covariates()
    test_gp = str(static_df["gp_code"].iloc[0])

    # Find the first available date in forecast data
    fc_path = os.path.join(BASE_DIR, "data", "raw", "forecast", "block_forecasts.parquet")
    fc_df = pd.read_parquet(fc_path)
    test_date = pd.to_datetime(fc_df["date"].iloc[0]).strftime("%Y-%m-%d")

    # Mock IMERG CSV with 1-2 rows
    mock_imerg_csv = tmp_path / "mock_panchayat_imerg.csv"
    mock_data = pd.DataFrame([
        {"gp_code": test_gp, "date": test_date, "imerg_rainfall_mm": 55.5},
        {"gp_code": "999999", "date": test_date, "imerg_rainfall_mm": 12.0}
    ])
    mock_data.to_csv(mock_imerg_csv, index=False)

    temp_out = str(tmp_path / "test_features.parquet")
    df = assemble_feature_table(sample_dates=True, output_path=temp_out, imerg_path=str(mock_imerg_csv))

    assert "ground_truth_source" in df.columns, "ground_truth_source column must be present in feature table"
    
    # Check matching row receives 'imerg' and exact rainfall
    matching = df[(df["panchayat_id"].astype(str) == test_gp) & (pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d") == test_date)]
    if len(matching) > 0:
        assert (matching["ground_truth_source"] == "imerg").all()
        assert (matching["ground_truth_rainfall"] == 55.5).all()

    # Check non-matching row receives 'synthetic_orographic'
    non_matching = df[df["panchayat_id"].astype(str) != test_gp]
    assert (non_matching["ground_truth_source"] == "synthetic_orographic").all()


def test_dem_elevation_ranges():
    """
    Spot-check that panchayat DEM statistics match the real physical topography of Pune:
    Deccan plains ~490-620m, Western Ghats crest reaching >950-1000m+.
    """
    dem_path = os.path.join(BASE_DIR, "data", "interim", "panchayat_dem_stats.csv")
    assert os.path.exists(dem_path), "panchayat_dem_stats.csv must exist"
    dem_df = pd.read_csv(dem_path)
    assert len(dem_df) >= 1000

    assert {"gp_code", "elevation_mean", "elevation_std", "slope_mean"}.issubset(dem_df.columns)
    
    # Minimum elevation across Pune eastern plains (Indapur / Ujani dam) is ~480-500m
    assert dem_df["elevation_mean"].min() >= 470.0, "Plains elevation should not be below 470m"
    # Maximum elevation along Western Ghats ridge (Torna / Rajgad / Bhimashankar) is >1000m
    assert dem_df["elevation_mean"].max() >= 950.0, "Crest elevation should exceed 950m"
    # Terrain slopes must be positive
    assert (dem_df["slope_mean"] > 0.0).all()


def test_lulc_fractions_sum_and_bounds():
    """
    Confirm real LULC percentage rows sum to 100% within tolerance,
    and all fractions are within [0, 100].
    """
    lulc_path = os.path.join(BASE_DIR, "data", "interim", "panchayat_lulc.csv")
    assert os.path.exists(lulc_path), "panchayat_lulc.csv must exist"
    lulc_df = pd.read_csv(lulc_path)
    assert len(lulc_df) >= 1000

    cols = ["landuse_cropland_pct", "landuse_forest_pct", "landuse_water_pct", "landuse_builtup_pct"]
    for c in cols:
        assert (lulc_df[c] >= 0.0).all(), f"{c} contains negative values"
        assert (lulc_df[c] <= 100.0).all(), f"{c} exceeds 100%"

    lulc_sum = lulc_df[cols].sum(axis=1)
    assert np.allclose(lulc_sum, 100.0, atol=0.2), "All LULC percentage rows must sum to ~100%"
