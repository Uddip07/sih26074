"""
Test IMD Binary Grid Parser & Writer
"""

import os
import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import pytest
import numpy as np
from src.ingest.imd_binary import (
    IMD_META,
    is_leap_year,
    get_days_in_year,
    get_grid_coordinates,
    write_imd_binary,
    read_imd_binary,
    extract_point_timeseries,
    extract_bbox_grid
)


def test_leap_years():
    assert is_leap_year(2020) is True
    assert is_leap_year(2024) is True
    assert is_leap_year(2025) is False
    assert is_leap_year(2026) is False
    assert get_days_in_year(2020) == 366
    assert get_days_in_year(2026) == 365


def test_imd_rainfall_roundtrip(tmp_path):
    m = IMD_META["rain"]
    year = 2026
    ndays = 365
    shape = (ndays, m["nrows"], m["ncols"])
    
    # Generate test array
    np.random.seed(42)
    test_arr = np.random.uniform(0.0, 100.0, size=shape).astype(np.float32)
    test_arr[0, 5, 5] = np.nan  # test missing value handling

    fpath = str(tmp_path / "test_rain_2026.grd")
    write_imd_binary(test_arr, fpath, "rain", year)
    assert os.path.exists(fpath)
    assert os.path.getsize(fpath) == m["ncols"] * m["nrows"] * ndays * 4

    read_back = read_imd_binary(fpath, "rain", year)
    assert read_back.shape == shape
    assert np.isnan(read_back[0, 5, 5])
    assert np.allclose(read_back[0, 0, :10], test_arr[0, 0, :10], atol=1e-4)


def test_imd_temperature_roundtrip(tmp_path):
    m = IMD_META["tmax"]
    year = 2026
    ndays = 365
    shape = (ndays, m["nrows"], m["ncols"])

    test_arr = np.random.uniform(15.0, 45.0, size=shape).astype(np.float32)
    fpath = str(tmp_path / "test_tmax_2026.grd")
    write_imd_binary(test_arr, fpath, "tmax", year)

    read_back = read_imd_binary(fpath, "tmax", year)
    assert read_back.shape == shape
    assert np.allclose(read_back[0, 0, :5], test_arr[0, 0, :5], atol=1e-4)


def test_extract_point_and_bbox():
    m = IMD_META["rain"]
    test_arr = np.ones((365, m["nrows"], m["ncols"]), dtype=np.float32) * 12.5

    # Extract point at Pune (lat 18.52, lon 73.85)
    ts = extract_point_timeseries(test_arr, 18.52, 73.85, "rain", 2026)
    assert len(ts) == 365
    assert "date" in ts.columns
    assert "rain" in ts.columns
    assert np.isclose(ts["rain"].iloc[0], 12.5)

    # Extract bounding box
    sub_arr, lats, lons = extract_bbox_grid(test_arr, 18.0, 19.0, 73.0, 74.0, "rain")
    assert sub_arr.ndim == 3
    assert len(lats) > 0
    assert len(lons) > 0
