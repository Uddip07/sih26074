import numpy as np
import pandas as pd
import pytest

from src.models.downscaler import Downscaler, VariableDownscaler
from src.models.postprocess import block_area_weights, conservation_report, conserve

PARAMS = {"n_estimators": 60, "max_depth": 3, "learning_rate": 0.2, "tree_method": "hist", "n_jobs": 2,
          "random_state": 0}


@pytest.fixture
def fitted(panel):
    feats = ["fc_rain", "elev_diff", "is_monsoon", "hist_bias"]
    tr = panel[panel["valid_date"] < "2024-11-01"]
    ca = panel[(panel["valid_date"] >= "2024-11-01") & (panel["valid_date"] < "2024-12-01")]
    m = VariableDownscaler(var="rain", target="rain_obs", forecast="fc_rain", features=feats, params=PARAMS,
                           thresholds=[2.5], residual_space="linear", lower=0.0, early_stopping_rounds=10)
    return m.fit(tr, ca), panel[panel["valid_date"] >= "2024-12-01"]


def test_model_beats_copy_on_learnable_signal(fitted):
    m, te = fitted
    p = m.predict(te)
    rmse_m = np.sqrt(np.mean((p["rain_pred"] - te["rain_obs"]) ** 2))
    rmse_c = np.sqrt(np.mean((te["fc_rain"] - te["rain_obs"]) ** 2))
    assert rmse_m < 0.8 * rmse_c


def test_physical_constraints_and_quantile_order(fitted):
    m, te = fitted
    p = m.predict(te)
    assert (p["rain_pred"] >= 0).all()
    assert (p["rain_q10"] <= p["rain_q50"] + 1e-6).all() and (p["rain_q50"] <= p["rain_q90"] + 1e-6).all()
    assert p["rain_p_ge_2p5"].between(0, 1).all()


def test_contributions_sum_to_prediction(fitted):
    m, te = fitted
    sub = te.head(200)
    c = m.contributions(sub, {"fc_rain": "forecast", "elev_diff": "terrain", "is_monsoon": "season"})
    resid = m.predict(sub, with_uncertainty=False)["rain_pred"].to_numpy() - sub["fc_rain"].to_numpy()
    total = c.sum(axis=1).to_numpy()
    unclipped = resid > 1e-6  # clipping at 0 breaks additivity only where rain_pred == 0
    assert np.allclose(total[unclipped], resid[unclipped], atol=1e-3)


def test_save_load_roundtrip(fitted, tmp_path):
    m, te = fitted
    ds = Downscaler({"rain": m}, {"kind": "test"})
    ds.save(tmp_path / "m.joblib")
    ds2 = Downscaler.load(tmp_path / "m.joblib")
    assert np.allclose(ds.predict(te.head(50))["rain_pred"], ds2.predict(te.head(50))["rain_pred"])


def test_mass_conservation_exact(two_blocks_four_gps):
    static = pd.DataFrame(two_blocks_four_gps.drop(columns="geometry"))
    w = block_area_weights(static)
    df = pd.DataFrame({"gp_code": ["1", "2", "3", "4"], "block_lgd": [10, 10, 20, 20],
                       "issue_date": pd.Timestamp("2025-07-01"), "lead_day": 1,
                       "fc_rain": [10.0, 10.0, 4.0, 4.0], "rain_pred": [20.0, 5.0, 1.0, 9.0]})
    df["rm"] = conserve(df, "rain", "rain_pred", "fc_rain", w)
    df["w"] = df["gp_code"].map(w)
    for _, g in df.groupby("block_lgd"):
        assert (g["rm"] * g["w"]).sum() / g["w"].sum() == pytest.approx(g["fc_rain"].iloc[0])
    assert (df["rm"] >= 0).all()
    rep = conservation_report(df.assign(rain_pred=df["rm"]), "rain_pred", "fc_rain", w)
    assert rep["mean_abs_block_departure"] == pytest.approx(0, abs=1e-9)
    df["tm"] = conserve(df.assign(t=df["rain_pred"]), "tmax", "t", "fc_rain", w)
    assert (df.groupby("block_lgd").apply(lambda g: (g["tm"] * g["w"]).sum() / g["w"].sum(), include_groups=False)
            .to_numpy() == pytest.approx([10.0, 4.0]))
