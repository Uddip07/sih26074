"""
Baseline Model: Naive Spatial Copy
Section 7.1 of block-to-panchayat-downscaling-spec.md:
Every panchayat inherits its parent block's forecast value unchanged.
Serves as the rigorous benchmark that all residual models must beat.
"""

import os
import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple


def compute_csi(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 2.5) -> float:
    """
    Critical Success Index (CSI / Threat Score) at specified meteorological threshold:
    CSI = hits / (hits + misses + false_alarms)
    """
    hits = np.sum((y_true >= threshold) & (y_pred >= threshold))
    misses = np.sum((y_true >= threshold) & (y_pred < threshold))
    false_alarms = np.sum((y_true < threshold) & (y_pred >= threshold))
    denom = hits + misses + false_alarms
    return float(hits / denom) if denom > 0 else 1.0


def evaluate_baseline(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluate naive block-copy baseline on the given dataset.
    Predictions are identically the parent block forecast values.
    """
    y_true = df["ground_truth_rainfall"].values
    y_pred = df["block_forecast_rainfall"].values

    errors = y_pred - y_true
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))
    csi_2_5 = compute_csi(y_true, y_pred, threshold=2.5)
    csi_10 = compute_csi(y_true, y_pred, threshold=10.0)
    csi_25 = compute_csi(y_true, y_pred, threshold=25.0)

    # Temperature metrics
    t_true = df["ground_truth_tmax"].values
    t_pred = df["block_forecast_tmax"].values
    t_rmse = float(np.sqrt(np.mean((t_pred - t_true) ** 2)))
    t_mae = float(np.mean(np.abs(t_pred - t_true)))

    metrics = {
        "model_name": "Baseline (Naive Spatial Copy)",
        "rainfall_rmse": round(rmse, 3),
        "rainfall_mae": round(mae, 3),
        "csi_2_5mm": round(csi_2_5, 4),
        "csi_10mm": round(csi_10, 4),
        "csi_25mm": round(csi_25, 4),
        "tmax_rmse": round(t_rmse, 3),
        "tmax_mae": round(t_mae, 3),
        "n_samples": len(df)
    }
    return metrics


if __name__ == "__main__":
    test_path = os.path.join(BASE_DIR, "data", "processed", "train_test_splits", "test.parquet")
    test_df = pd.read_parquet(test_path)
    metrics = evaluate_baseline(test_df)
    print("Baseline Evaluation on Held-Out Test Panchayats:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
