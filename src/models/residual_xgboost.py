"""
Residual Learning Model (XGBoost & LightGBM)
Section 7.2 of block-to-panchayat-downscaling-spec.md:
target = ground_truth - block_forecast_value
prediction = block_forecast_value + model.predict(features)
Physical post-processing constraint: prediction = max(0.0, prediction)
"""

import os
import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import json
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, Any, Tuple, List, Optional
from src.models.baseline_copy import compute_csi

MODELS_DIR = os.path.join(BASE_DIR, "outputs", "models")
PREDICTIONS_DIR = os.path.join(BASE_DIR, "outputs", "predictions")

FEATURE_COLUMNS = [
    "block_forecast_rainfall",
    "elevation_mean",
    "elevation_std",
    "slope_mean",
    "landuse_cropland_pct",
    "landuse_forest_pct",
    "landuse_water_pct",
    "landuse_builtup_pct",
    "dist_to_coast_km",
    "dist_to_water_km",
    "historical_bias",
    "latitude",
    "longitude",
    "day_of_year",
    "is_monsoon"
]


class DownscalingResidualModel:
    """
    Two-stage Residual Downscaling Model:
    Learns microclimatic correction residual: ground_truth - block_forecast.
    Applies orographic and terrain covariates to disaggregate block forecasts to panchayat polygons.
    """

    def __init__(self, model_type: str = "xgboost", params: Optional[Dict[str, Any]] = None):
        self.model_type = model_type.lower()
        self.feature_names = FEATURE_COLUMNS
        self.model = None

        if params is None:
            if self.model_type == "xgboost":
                self.params = {
                    "n_estimators": 400,
                    "max_depth": 6,
                    "learning_rate": 0.04,
                    "subsample": 0.85,
                    "colsample_bytree": 0.85,
                    "min_child_weight": 3,
                    "random_state": 42,
                    "n_jobs": -1
                }
            else:
                self.params = {
                    "n_estimators": 400,
                    "max_depth": 6,
                    "learning_rate": 0.04,
                    "subsample": 0.85,
                    "colsample_bytree": 0.85,
                    "min_child_samples": 20,
                    "random_state": 42,
                    "n_jobs": -1
                }
        else:
            self.params = params

    def fit(self, X: pd.DataFrame, y_residual: pd.Series):
        """Fit the residual regressor."""
        X_mat = X[self.feature_names].copy()
        if "is_monsoon" in X_mat.columns:
            X_mat["is_monsoon"] = X_mat["is_monsoon"].astype(int)

        if self.model_type == "xgboost":
            self.model = xgb.XGBRegressor(**self.params)
            self.model.fit(X_mat, y_residual)
        else:
            self.model = lgb.LGBMRegressor(**self.params)
            self.model.fit(X_mat, y_residual)
        return self

    def predict_residual(self, X: pd.DataFrame) -> np.ndarray:
        """Predict microclimate residual adjustment."""
        X_mat = X[self.feature_names].copy()
        if "is_monsoon" in X_mat.columns:
            X_mat["is_monsoon"] = X_mat["is_monsoon"].astype(int)
        return self.model.predict(X_mat)

    def predict_downscaled(self, X: pd.DataFrame) -> np.ndarray:
        """
        Produce final downscaled panchayat rainfall prediction:
        prediction = max(0.0, block_forecast + residual)
        """
        predicted_residuals = self.predict_residual(X)
        block_forecast = X["block_forecast_rainfall"].values
        downscaled = np.maximum(0.0, block_forecast + predicted_residuals)
        return np.round(downscaled, 2)

    def get_feature_importances(self) -> Dict[str, float]:
        """Return normalized feature importances."""
        if self.model is None:
            return {}
        importances = self.model.feature_importances_
        norm_imp = importances / np.sum(importances)
        return {feat: round(float(imp), 4) for feat, imp in zip(self.feature_names, norm_imp)}

    def save(self, filepath: str):
        """Save model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if self.model_type == "xgboost":
            self.model.save_model(filepath)
        else:
            self.model.booster_.save_model(filepath)
        print(f"Saved {self.model_type} model to {filepath}")

    def load(self, filepath: str):
        """Load model from disk."""
        if self.model_type == "xgboost":
            self.model = xgb.XGBRegressor()
            self.model.load_model(filepath)
        else:
            self.model = lgb.Booster(model_file=filepath)
        return self


def train_and_evaluate(
    train_path: str = os.path.join(BASE_DIR, "data", "processed", "train_test_splits", "train.parquet"),
    test_path: str = os.path.join(BASE_DIR, "data", "processed", "train_test_splits", "test.parquet")
) -> Dict[str, Any]:
    """
    Train residual model on spatial training panchayats and evaluate on held-out test panchayats.
    Matches Section 7 & 8 evaluation criteria.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)

    print("Loading train/test datasets...")
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    print(f"Train samples: {len(train_df)} ({train_df['panchayat_id'].nunique()} panchayats)")
    print(f"Test samples: {len(test_df)} ({test_df['panchayat_id'].nunique()} held-out panchayats)")

    # 1. Baseline Benchmark on Test Set
    y_test_true = test_df["ground_truth_rainfall"].values
    y_test_baseline = test_df["block_forecast_rainfall"].values
    baseline_rmse = float(np.sqrt(np.mean((y_test_baseline - y_test_true) ** 2)))
    baseline_mae = float(np.mean(np.abs(y_test_baseline - y_test_true)))
    baseline_csi = compute_csi(y_test_true, y_test_baseline, 2.5)

    # 2. Train XGBoost Residual Model
    print("\nTraining XGBoost Residual Model...")
    xgb_model = DownscalingResidualModel(model_type="xgboost")
    xgb_model.fit(train_df, train_df["residual_rainfall"])

    # Predictions on Test Set
    test_preds_xgb = xgb_model.predict_downscaled(test_df)
    model_rmse = float(np.sqrt(np.mean((test_preds_xgb - y_test_true) ** 2)))
    model_mae = float(np.mean(np.abs(test_preds_xgb - y_test_true)))
    model_csi = compute_csi(y_test_true, test_preds_xgb, 2.5)
    model_csi_10 = compute_csi(y_test_true, test_preds_xgb, 10.0)
    model_csi_25 = compute_csi(y_test_true, test_preds_xgb, 25.0)

    # Skill Score = 1 - (RMSE_model / RMSE_baseline)
    skill_score = 1.0 - (model_rmse / baseline_rmse)
    skill_score_pct = skill_score * 100.0

    # Save models and predictions
    xgb_model_path = os.path.join(MODELS_DIR, "residual_xgboost.json")
    xgb_model.save(xgb_model_path)

    # Save predictions
    results_df = test_df[[
        "panchayat_id", "panchayat_name", "block_name", "date",
        "block_forecast_rainfall", "ground_truth_rainfall"
    ]].copy()
    results_df["downscaled_prediction_rainfall"] = test_preds_xgb
    results_df["predicted_residual"] = np.round(test_preds_xgb - results_df["block_forecast_rainfall"], 2)
    results_df["baseline_error"] = np.round(np.abs(results_df["block_forecast_rainfall"] - results_df["ground_truth_rainfall"]), 2)
    results_df["model_error"] = np.round(np.abs(results_df["downscaled_prediction_rainfall"] - results_df["ground_truth_rainfall"]), 2)

    pred_out = os.path.join(PREDICTIONS_DIR, "test_predictions.parquet")
    results_df.to_parquet(pred_out, index=False)

    importances = xgb_model.get_feature_importances()
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)

    summary = {
        "baseline_rmse": round(baseline_rmse, 3),
        "baseline_mae": round(baseline_mae, 3),
        "baseline_csi_2_5mm": round(baseline_csi, 4),
        "model_rmse": round(model_rmse, 3),
        "model_mae": round(model_mae, 3),
        "model_csi_2_5mm": round(model_csi, 4),
        "model_csi_10mm": round(model_csi_10, 4),
        "model_csi_25mm": round(model_csi_25, 4),
        "skill_score": round(skill_score, 4),
        "skill_score_pct": f"{skill_score_pct:.2f}%",
        "headline_pitch": f"{skill_score_pct:.1f}% RMSE reduction over naive block-value copy, validated via held-out panchayat validation.",
        "top_features": sorted_imp[:6]
    }

    # Save summary report
    with open(os.path.join(MODELS_DIR, "evaluation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*70)
    print("RESIDUAL DOWNSCALING MODEL EVALUATION REPORT")
    print("="*70)
    print(f"Baseline Naive Copy RMSE:   {baseline_rmse:.3f} mm")
    print(f"Residual Model RMSE:        {model_rmse:.3f} mm")
    print(f"Skill Score (% RMSE reduction): {skill_score_pct:.2f}%")
    print(f"Baseline MAE:               {baseline_mae:.3f} mm")
    print(f"Residual Model MAE:         {model_mae:.3f} mm")
    print(f"CSI @ 2.5mm:                Baseline = {baseline_csi:.4f} -> Model = {model_csi:.4f}")
    print(f"\nHeadline Pitch:")
    print(f'"{summary["headline_pitch"]}"')
    print("\nTop Contributing Features:")
    for feat, imp in sorted_imp[:6]:
        print(f"  - {feat}: {imp*100:.1f}%")
    print("="*70)

    return summary


if __name__ == "__main__":
    train_and_evaluate()
