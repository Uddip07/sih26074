"""
Block Forecast Ingestion & Aggregation Module
Generates historical and live block-level meteorological forecasts matching IMD GKMS / DAMU cadence.
Section 1 & Section 3 of block-to-panchayat-downscaling-spec.md:
Each block polygon has ONE single forecast value that is assigned identically to all its sub-panchayats.
"""

import os
import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Optional, Dict, Any, List

DATA_DIR = os.path.join(BASE_DIR, "data", "raw", "forecast")
OUTPUT_PARQUET = os.path.join(DATA_DIR, "block_forecasts.parquet")


def generate_block_historical_forecasts(
    ground_truth_df: pd.DataFrame,
    output_path: Optional[str] = OUTPUT_PARQUET
) -> pd.DataFrame:
    """
    Generate the block-level baseline forecasts matching operational GKMS bulletins.
    In operational practice, the block forecast represents the coarse 12km NWP model
    (IMD-NCUM / GFS) resampled to block polygons, having low spatial resolution and smoothing out
    panchayat-scale microclimatic extremes.
    """
    if output_path and os.path.exists(output_path):
        df = pd.read_parquet(output_path)
        print(f"Loaded existing block forecasts: {len(df)} rows from {output_path}")
        return df

    print(f"Generating operational block forecasts from {len(ground_truth_df)} station records...")

    # Group observations by (block_name, date) to calculate block-wide coarse forecast
    # We simulate the operational NWP forecast error by adding typical numerical weather model bias:
    # 1. Rainfall underprediction of extreme orographic peaks in Ghats
    # 2. Smoothing of microclimatic spatial gradients
    records = []
    grouped = ground_truth_df.groupby(["block_name", "date"])

    np.random.seed(42)

    for (block_name, date), grp in grouped:
        true_rain = grp["ground_truth_rainfall"].mean()
        true_tmax = grp["ground_truth_tmax"].mean()
        true_tmin = grp["ground_truth_tmin"].mean()
        true_rh = grp["ground_truth_rh"].mean()
        true_wind = grp["ground_truth_wind_kmh"].mean()
        block_lgd = grp["block_lgd"].iloc[0]

        # Operational NWP 12km grid characteristic:
        # Tends to smooth orographic rain (e.g. overpredicts in rain shadow, underpredicts in heavy Ghats)
        date_dt = pd.to_datetime(date)
        is_monsoon = date_dt.month in [6, 7, 8, 9]

        if is_monsoon:
            # Coarse NWP error: dampens sharp Ghat crest peaks by ~15-25% and smears onto dry plains
            if block_name in ["VELHE", "MULSHI", "MAVAL"]:
                rain_forecast = true_rain * 0.78 + np.random.normal(0, 2.0)
            elif block_name in ["BARAMATI", "INDAPUR", "DAUND"]:
                rain_forecast = true_rain * 1.18 + np.random.normal(0, 1.5)
            else:
                rain_forecast = true_rain * 0.95 + np.random.normal(0, 1.2)
        else:
            rain_forecast = true_rain * 0.92 + np.random.normal(0, 0.4)

        rain_forecast = max(0.0, round(float(rain_forecast), 1))
        tmax_forecast = round(float(true_tmax + np.random.normal(0, 0.8)), 1)
        tmin_forecast = round(float(true_tmin + np.random.normal(0, 0.7)), 1)
        rh_forecast = min(100.0, max(20.0, round(float(true_rh + np.random.normal(0, 3.0)), 1)))
        wind_forecast = max(0.0, round(float(true_wind + np.random.normal(0, 1.5)), 1))

        records.append({
            "block_name": block_name,
            "block_lgd": block_lgd,
            "date": pd.to_datetime(date),
            "block_forecast_rainfall": rain_forecast,
            "block_forecast_tmax": tmax_forecast,
            "block_forecast_tmin": tmin_forecast,
            "block_forecast_rh": rh_forecast,
            "block_forecast_wind_kmh": wind_forecast
        })

    df = pd.DataFrame(records)
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_parquet(output_path, index=False)
        print(f"Saved block forecasts: {len(df)} rows to {output_path}")

    return df


if __name__ == "__main__":
    gt_path = os.path.join(BASE_DIR, "data", "raw", "ground_truth", "aws_station_observations.parquet")
    if os.path.exists(gt_path):
        gt = pd.read_parquet(gt_path)
        fc = generate_block_historical_forecasts(gt)
        print(fc.head(10))
        print("\nBlock forecast summary:")
        print(fc.describe())
