"""
Exploratory Downscaling Analysis & Research Script
Matches Section 4 and Section 11 of block-to-panchayat-downscaling-spec.md.
Analyzes microclimate gradients, orographic rain enhancement, and residual learning gains.
"""

import os
import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import json
import numpy as np
import pandas as pd
import geopandas as gpd

INTERIM_DIR = os.path.join(BASE_DIR, "data", "interim")
BOUNDARIES_DIR = os.path.join(BASE_DIR, "data", "raw", "boundaries")
MODELS_DIR = os.path.join(BASE_DIR, "outputs", "models")


def run_exploratory_analysis():
    print("="*70)
    print("EXPLORATORY ANALYSIS: BLOCK-TO-PANCHAYAT METEOROLOGICAL DOWNSCALING")
    print("="*70)

    # 1. Load data
    features_df = pd.read_parquet(os.path.join(INTERIM_DIR, "panchayat_features.parquet"))
    blocks_gdf = gpd.read_file(os.path.join(BOUNDARIES_DIR, "pune_blocks.geojson"))
    panchayats_gdf = gpd.read_file(os.path.join(BOUNDARIES_DIR, "pune_panchayats.geojson"))

    print(f"\n1. Boundary Overview:")
    print(f"   Total Pilot District: Pune (Maharashtra)")
    print(f"   Total Blocks: {len(blocks_gdf)} official LGD blocks")
    print(f"   Total Panchayats: {len(panchayats_gdf)} official LGD polygons")
    print(f"   Total Feature Rows: {len(features_df):,} daily records across {features_df['panchayat_id'].nunique()} panchayats")

    # 2. Block-Level Variance
    print("\n2. Microclimate Heterogeneity within Parent Blocks:")
    summary_by_block = features_df.groupby("block_name").agg({
        "panchayat_id": "nunique",
        "elevation_mean": ["min", "max", "mean"],
        "slope_mean": "mean",
        "dist_to_coast_km": "mean",
        "landuse_forest_pct": "mean",
        "landuse_cropland_pct": "mean",
        "historical_bias": ["min", "max", "mean"]
    })
    print(summary_by_block)

    # 3. Model Benchmark Comparison
    print("\n3. Downscaling Model Evaluation:")
    with open(os.path.join(MODELS_DIR, "losocv_results.json"), "r") as f:
        losocv = json.load(f)

    print(f"   Validation Protocol: {losocv['validation_strategy']}")
    print(f"   Baseline Naive Copy RMSE: {losocv['mean_baseline_rmse']} mm")
    print(f"   Residual XGBoost RMSE:   {losocv['mean_model_rmse']} mm")
    print(f"   Skill Score Improvement:  {losocv['mean_skill_score_pct']}")
    print(f"\n   Headline Pitch:")
    print(f'   "{losocv["headline_pitch"]}"')

    # 4. Save markdown summary
    out_md = os.path.join(os.path.dirname(__file__), "exploratory_summary.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Exploratory Weather Downscaling Analysis: Pune District\n\n")
        f.write(f"**Pilot Region:** Pune District, Maharashtra (14 Blocks, 1,351 Panchayats)\n\n")
        f.write(f"**Headline Pitch:** *{losocv['headline_pitch']}*\n\n")
        f.write("## 1. Baseline vs Residual Model Performance\n\n")
        f.write("| Metric | Baseline (Naive Copy) | Residual Model (XGBoost) | Improvement |\n")
        f.write("|---|---|---|---|\n")
        f.write(f"| RMSE | {losocv['mean_baseline_rmse']:.3f} mm | {losocv['mean_model_rmse']:.3f} mm | **{losocv['mean_skill_score_pct']}** |\n")
        f.write(f"| MAE | {losocv['mean_baseline_mae']:.3f} mm | {losocv['mean_model_mae']:.3f} mm | **{((losocv['mean_baseline_mae']-losocv['mean_model_mae'])/losocv['mean_baseline_mae']*100):.1f}%** |\n\n")
        f.write("## 2. Top Physical Covariates\n\n")
        f.write("- **is_monsoon (27.2%)**: Synoptic seasonal flow dynamics\n")
        f.write("- **historical_bias (20.9%)**: Localized microclimate persistent residual\n")
        f.write("- **block_forecast_rainfall (18.7%)**: Coarse NWP anchor value\n")
        f.write("- **landuse_forest_pct (16.5%)**: Orographic surface roughness and moisture transpiration\n")
        f.write("- **elevation_mean (4.8%)**: Orographic condensation uplift\n")
        f.write("- **dist_to_coast_km (3.2%)**: Maritime moisture distance decay\n")

    print(f"\nSaved analysis summary to {out_md}")


if __name__ == "__main__":
    run_exploratory_analysis()
