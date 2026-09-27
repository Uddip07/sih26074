"""Audit 1.5 / 1.6: no target leakage through hist_bias, and disjoint spatio-temporal splits."""

import numpy as np
import pandas as pd

from src.features.bias_encoder import HistoricalBiasEncoder
from src.features.dataset import add_antecedent
from src.models.train import calibration_days, holdout_gps, split_masks


def test_oof_bias_does_not_use_own_rows(panel):
    r = (panel["rain_obs"] - panel["fc_rain"]).to_numpy()
    enc = HistoricalBiasEncoder(n_time_folds=5)
    oof = enc.fit_transform_oof(panel, r)
    # Perturb the targets of one time fold: that fold's OOF values must not change
    dates = np.unique(panel["valid_date"])
    fold0 = np.isin(panel["valid_date"], np.array_split(dates, 5)[0])
    r2 = r.copy()
    r2[fold0] += 100.0
    oof2 = HistoricalBiasEncoder(n_time_folds=5).fit_transform_oof(panel, r2)
    assert np.allclose(oof[fold0], oof2[fold0])
    assert not np.allclose(oof[~fold0], oof2[~fold0])


def test_bias_recovers_signal_and_unseen_gp_uses_neighbours(panel):
    seen = panel[panel["gp_code"] != "7"]
    r = (seen["rain_obs"] - seen["fc_rain"]).to_numpy()
    enc = HistoricalBiasEncoder().fit(seen, r)
    got = enc.transform(seen).astype(float)
    corr = np.corrcoef(got, seen["true_bias"])[0, 1]
    assert corr > 0.8
    unseen = panel[panel["gp_code"] == "7"]
    v = enc.transform(unseen)
    assert np.isfinite(v).all()  # IDW fallback, never the GP's own labels
    # the unseen GP's own residuals were never passed to fit
    assert "7" not in set(enc.table_["gp_code"])


def test_split_masks_disjoint(cfg, panel):
    df = panel.copy()
    df["valid_date"] = pd.date_range("2025-06-01", periods=len(df) // 30, freq="D").repeat(30)[: len(df)]
    static = df.drop_duplicates("gp_code")[["gp_code", "block_lgd"]]
    hold = holdout_gps(static, 0.2, 1)
    cfg.raw["validation"]["test_start"] = "2025-11-01"
    m = split_masks(df, cfg, hold)
    assert not (m["train"] & m["calib"]).any() and not (m["train"] & m["test"]).any()
    assert not (m["calib"] & m["test"]).any()
    assert not df.loc[m["train"] | m["calib"], "gp_code"].isin(hold).any()
    assert df.loc[m["test"], "valid_date"].min() >= pd.Timestamp("2025-11-01")
    cal, buf = calibration_days(df["valid_date"], pd.Timestamp("2025-11-01"))
    assert not df.loc[m["train"], "valid_date"].isin(cal | buf).any()
    cfg.raw["validation"]["test_start"] = "2026-01-01"


def test_antecedent_window_is_latency_honest():
    obs = pd.DataFrame({"gp_code": "A", "valid_date": pd.date_range("2025-07-01", periods=30), "rain_obs": 0.0})
    obs.loc[obs["valid_date"] == "2025-07-20", "rain_obs"] = 100.0  # rain on day 20 only
    q = pd.DataFrame({"gp_code": ["A"] * 3, "issue_date": pd.to_datetime(["2025-07-22", "2025-07-23", "2025-07-29"])})
    out = add_antecedent(q, obs).set_index("issue_date")["ante_obs7_lag3"]
    assert out.loc["2025-07-22"] == 0.0     # day-20 rain not yet available (issue-3 = 19th)
    assert out.loc["2025-07-23"] == 100.0   # window [14, 20]
    assert out.loc["2025-07-29"] == 100.0   # window [20, 26]
