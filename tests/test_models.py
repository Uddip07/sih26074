"""
Test Baseline & Residual Models
"""

import os
import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import pytest
import numpy as np
import pandas as pd
from src.models.baseline_copy import evaluate_baseline, compute_csi
from src.models.residual_xgboost import DownscalingResidualModel


@pytest.fixture
def sample_test_data():
    test_path = os.path.join(BASE_DIR, "data", "processed", "train_test_splits", "test.parquet")
    return pd.read_parquet(test_path)


def test_baseline_metrics(sample_test_data):
    metrics = evaluate_baseline(sample_test_data)
    assert "rainfall_rmse" in metrics
    assert "rainfall_mae" in metrics
    assert metrics["rainfall_rmse"] > 0
    assert 0 <= metrics["csi_2_5mm"] <= 1.0


def test_residual_model_prediction(sample_test_data):
    model_path = os.path.join(BASE_DIR, "outputs", "models", "residual_xgboost.json")
    assert os.path.exists(model_path), "Model checkpoint missing"

    model = DownscalingResidualModel(model_type="xgboost")
    model.load(model_path)

    preds = model.predict_downscaled(sample_test_data)
    assert len(preds) == len(sample_test_data)

    # Physical constraint: rainfall must never be negative
    assert (preds >= 0.0).all(), "Rainfall predictions must be non-negative"

    # Evaluate RMSE
    y_true = sample_test_data["ground_truth_rainfall"].values
    y_baseline = sample_test_data["block_forecast_rainfall"].values

    base_rmse = np.sqrt(np.mean((y_baseline - y_true) ** 2))
    model_rmse = np.sqrt(np.mean((preds - y_true) ** 2))

    # Model must beat baseline by at least 20%
    skill_score = 1.0 - (model_rmse / base_rmse)
    assert skill_score > 0.20, f"Skill score {skill_score:.2%} is below threshold"
