"""
Shared fixtures. Unit tests use small synthetic inputs built here (test fixtures, never
used by the pipeline). Integration tests read real pipeline outputs and skip when absent.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from src.common.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture
def two_blocks_four_gps():
    """Two 0.2°x0.2° blocks, each split into two GPs of unequal area."""
    gps = gpd.GeoDataFrame({
        "gp_code": ["1", "2", "3", "4"], "block_lgd": [10, 10, 20, 20],
        "geometry": [box(73.0, 18.0, 73.05, 18.2), box(73.05, 18.0, 73.2, 18.2),
                     box(73.2, 18.0, 73.3, 18.2), box(73.3, 18.0, 73.4, 18.2)]}, crs="EPSG:4326")
    gps["area_km2"] = gps.to_crs(32643).area / 1e6
    return gps


@pytest.fixture
def panel():
    """Synthetic (GP x date) panel with a known per-GP bias, for encoder/model tests."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-06-01", periods=200, freq="D")
    rows = []
    for g in range(30):
        lat, lon = 18 + (g % 6) * 0.05, 73.5 + (g // 6) * 0.05
        bias = 2.0 * np.sin(g)  # persistent GP effect
        for d in dates:
            fc = max(0.0, rng.gamma(0.8, 6.0))
            rows.append({"gp_code": str(g), "block_lgd": g // 10, "latitude": lat, "longitude": lon,
                         "valid_date": d, "issue_date": d - pd.Timedelta(days=1), "lead_day": 1,
                         "is_monsoon": int(d.month in (6, 7, 8, 9)), "elev_diff": bias * 50,
                         "fc_rain": fc, "true_bias": bias})
    df = pd.DataFrame(rows)
    df["rain_obs"] = np.clip(df["fc_rain"] + df["true_bias"] + rng.normal(0, 1, len(df)), 0, None)
    return df
