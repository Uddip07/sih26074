"""
Comprehensive Evaluation & Leave-One-Station-Out Cross-Validation Module
Section 7.3 and Section 8 of block-to-panchayat-downscaling-spec.md:
Leave-One-Station-Out Cross-Validation (LOSOCV) simulates the operational case:
predicting weather for un-gauged panchayats with no ground stations.
"""

import os
import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from src.models.residual_xgboost import DownscalingResidualModel
from src.models.baseline_copy import compute_csi

MODELS_DIR = os.path.join(BASE_DIR, "outputs", "models")
INTERIM_PATH = os.path.join(BASE_DIR, "data", "interim", "panchayat_features.parquet")


def run_leave_one_station_out_cv(
    data_path: str = INTERIM_PATH,
    n_folds: int = 14
) -> Dict[str, Any]:
    """
    Execute Leave-One-Station-Out Cross-Validation (LOSOCV).
    For each fold, hold out all data for one block/station zone, train on the other 13 blocks,
    and predict on the held-out zone.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("Executing Leave-One-Station-Out Cross-Validation (LOSOCV)...")

    df = pd.read_parquet(data_path)
    blocks = df["block_name"].unique()
    print(f"Total blocks for LOSOCV folds: {len(blocks)}")

    fold_results = []

    for fold_idx, held_out_block in enumerate(blocks, 1):
        test_mask = df["block_name"] == held_out_block
        train_mask = ~test_mask

        train_fold = df[train_mask]
        test_fold = df[test_mask]

        # Baseline
        y_true = test_fold["ground_truth_rainfall"].values
        y_baseline = test_fold["block_forecast_rainfall"].values
        base_rmse = float(np.sqrt(np.mean((y_baseline - y_true) ** 2)))
        base_mae = float(np.mean(np.abs(y_baseline - y_true)))
        base_csi = compute_csi(y_true, y_baseline, 2.5)

        # Train residual model on remaining blocks
        model = DownscalingResidualModel(model_type="xgboost")
        model.fit(train_fold, train_fold["residual_rainfall"])

        # Predict on held-out station/block
        y_pred = model.predict_downscaled(test_fold)
        model_rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        model_mae = float(np.mean(np.abs(y_pred - y_true)))
        model_csi = compute_csi(y_true, y_pred, 2.5)

        skill_score = (1.0 - (model_rmse / base_rmse)) * 100.0

        fold_res = {
            "fold": fold_idx,
            "held_out_block": held_out_block,
            "n_panchayats": test_fold["panchayat_id"].nunique(),
            "n_samples": len(test_fold),
            "baseline_rmse": round(base_rmse, 3),
            "model_rmse": round(model_rmse, 3),
            "skill_score_pct": round(skill_score, 2),
            "baseline_mae": round(base_mae, 3),
            "model_mae": round(model_mae, 3),
            "csi_improvement": round(model_csi - base_csi, 4)
        }
        fold_results.append(fold_res)
        print(f"  Fold {fold_idx:2d}/{len(blocks)} [{held_out_block:10s}]: Base RMSE = {base_rmse:.2f} -> Model RMSE = {model_rmse:.2f} (Skill = +{skill_score:.1f}%)")

    results_df = pd.DataFrame(fold_results)
    results_csv = os.path.join(MODELS_DIR, "losocv_summary.csv")
    results_df.to_csv(results_csv, index=False)

    avg_base_rmse = float(results_df["baseline_rmse"].mean())
    avg_model_rmse = float(results_df["model_rmse"].mean())
    avg_skill = float(results_df["skill_score_pct"].mean())
    avg_base_mae = float(results_df["baseline_mae"].mean())
    avg_model_mae = float(results_df["model_mae"].mean())

    summary = {
        "validation_strategy": "Leave-One-Station-Out Cross-Validation (LOSOCV)",
        "total_folds": len(blocks),
        "mean_baseline_rmse": round(avg_base_rmse, 3),
        "mean_model_rmse": round(avg_model_rmse, 3),
        "mean_skill_score_pct": f"{avg_skill:.2f}%",
        "mean_baseline_mae": round(avg_base_mae, 3),
        "mean_model_mae": round(avg_model_mae, 3),
        "headline_pitch": f"{avg_skill:.1f}% RMSE reduction over naive block-value copy, validated via leave-one-station-out cross-validation.",
        "fold_breakdown": fold_results
    }

    results_json = os.path.join(MODELS_DIR, "losocv_results.json")
    with open(results_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*75)
    print("LEAVE-ONE-STATION-OUT CROSS-VALIDATION SUMMARY")
    print("="*75)
    print(f"Average Baseline RMSE:      {avg_base_rmse:.3f} mm")
    print(f"Average Model RMSE:         {avg_model_rmse:.3f} mm")
    print(f"Average Skill Score:        +{avg_skill:.2f}%")
    print(f"Average Baseline MAE:       {avg_base_mae:.3f} mm")
    print(f"Average Model MAE:          {avg_model_mae:.3f} mm")
    print(f"\nHeadline Pitch Statement:")
    print(f'"{summary["headline_pitch"]}"')
    print("="*75)

    return summary


if __name__ == "__main__":
    run_leave_one_station_out_cv()
