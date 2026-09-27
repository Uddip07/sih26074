import numpy as np
import pandas as pd
import pytest

from src.common.geo import Grid, apply_weights, area_weights, stack_to_polygons, weight_matrix
from src.models import metrics as M


def test_grid_covering_aligned_and_covers():
    g = Grid.covering((73.2, 17.8, 75.3, 19.5), 0.25)
    assert np.allclose(np.mod(g.lats / 0.25, 1), 0) and np.allclose(np.mod(g.lons / 0.25, 1), 0)
    assert g.lats.min() - 0.125 <= 17.8 and g.lats.max() + 0.125 >= 19.5
    g5 = Grid.covering((73.2, 17.8, 75.3, 19.5), 0.05, origin=0.025)
    assert np.allclose(np.mod((g5.lons - 0.025) / 0.05 + 1e-9, 1), 0, atol=1e-6)


def test_area_weights_sum_to_one_and_are_metric(two_blocks_four_gps):
    g = Grid.covering((72.9, 17.9, 73.5, 18.3), 0.1)
    w = area_weights(two_blocks_four_gps, "gp_code", g, "EPSG:32643")
    s = w.groupby("gp_code")["weight"].sum()
    assert np.allclose(s, 1.0)
    # GP "1" is 0.05° wide and lies in a single 0.1° column
    assert w[w["gp_code"] == "1"]["j"].nunique() == 1


def test_stack_to_polygons_nan_aware(two_blocks_four_gps):
    g = Grid.covering((72.9, 17.9, 73.5, 18.3), 0.1)
    w = area_weights(two_blocks_four_gps, "gp_code", g, "EPSG:32643")
    ids = list(two_blocks_four_gps["gp_code"])
    W = weight_matrix(w, "gp_code", ids, len(g.lats), len(g.lons))
    field = np.full((1, len(g.lats), len(g.lons)), 5.0)
    out = stack_to_polygons(field, W)
    assert np.allclose(out, 5.0)
    field[0, :, :] = np.nan
    assert np.isnan(stack_to_polygons(field, W)).all()
    # apply_weights agrees with the sparse path
    f2 = np.arange(len(g.lats) * len(g.lons), dtype=float).reshape(len(g.lats), len(g.lons))
    a = apply_weights(f2, w, "gp_code").reindex(ids).to_numpy()
    b = stack_to_polygons(f2[None], W)[0]
    assert np.allclose(a, b)


def test_csi_is_nan_without_events():
    y = np.zeros(50)
    p = np.zeros(50)
    c = M.categorical(y, p, 2.5)
    assert np.isnan(c["csi"]) and np.isnan(c["pod"]) and np.isnan(c["ets"])


def test_categorical_known_table():
    y = np.array([5, 5, 5, 0, 0, 0, 0, 0])
    p = np.array([5, 5, 0, 5, 0, 0, 0, 0])
    c = M.categorical(y, p, 2.5)
    assert (c["hits"], c["misses"], c["false_alarms"], c["correct_neg"]) == (2, 1, 1, 4)
    assert c["pod"] == pytest.approx(2 / 3) and c["far"] == pytest.approx(1 / 3) and c["csi"] == pytest.approx(0.5)
    ar = 3 * 3 / 8
    assert c["ets"] == pytest.approx((2 - ar) / (4 - ar))


def test_continuous_and_skill():
    y = np.array([0, 1, 2, 3.0])
    c = M.continuous(y, y + 1)
    assert c["rmse"] == pytest.approx(1) and c["bias"] == pytest.approx(1)
    assert M.skill(0.5, 1.0) == pytest.approx(0.5)
    assert np.isnan(M.skill(1.0, 0.0))


def test_bootstrap_ci_brackets_point():
    rng = np.random.default_rng(1)
    days = np.repeat(pd.date_range("2025-01-01", periods=120), 10)
    y = rng.gamma(1, 5, len(days))
    df = pd.DataFrame({"d": days, "y": y, "p": y + rng.normal(0, 1, len(y)), "r": y + rng.normal(0, 2, len(y))})
    b = M.bootstrap_skill(df, "y", "p", "r", day_col="d", n_boot=200)
    assert b["ci_low"] <= b["skill"] <= b["ci_high"] and b["skill"] > 0


def test_rain_class_imd():
    assert list(M.rain_class(np.array([0, 1, 10, 30, 80, 150, 250]))) == [
        "no rain", "very light", "light", "moderate", "heavy", "very heavy", "extremely heavy"]
