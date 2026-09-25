"""
IMD Binary Grid Reader & Writer
Python native implementation of the imdR binary .grd grid format for India Meteorological Department data.
Supports 0.25° daily rainfall and 1.0° daily maximum/minimum temperature.
"""

import struct
import datetime
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple

IMD_META: Dict[str, Dict[str, Any]] = {
    "rain": {
        "ncols": 135,
        "nrows": 129,
        "xmin": 66.5,
        "ymin": 6.5,
        "res": 0.25,
        "na_val": -999.0,
        "units": "mm/day",
        "description": "IMD 0.25-degree daily gridded rainfall"
    },
    "tmax": {
        "ncols": 31,
        "nrows": 31,
        "xmin": 67.5,
        "ymin": 7.5,
        "res": 1.0,
        "na_val": 99.9,
        "units": "deg_C",
        "description": "IMD 1.0-degree daily gridded maximum temperature"
    },
    "tmin": {
        "ncols": 31,
        "nrows": 31,
        "xmin": 67.5,
        "ymin": 7.5,
        "res": 1.0,
        "na_val": 99.9,
        "units": "deg_C",
        "description": "IMD 1.0-degree daily gridded minimum temperature"
    }
}


def is_leap_year(year: int) -> bool:
    """Check if year is a leap year according to Gregorian calendar rules."""
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def get_days_in_year(year: int) -> int:
    return 366 if is_leap_year(year) else 365


def get_grid_coordinates(variable: str = "rain") -> Tuple[np.ndarray, np.ndarray]:
    """
    Get 1D longitude and latitude coordinate arrays for the IMD grid.
    Returns: (longitudes, latitudes)
    """
    m = IMD_META[variable]
    lons = m["xmin"] + np.arange(m["ncols"]) * m["res"]
    lats = m["ymin"] + np.arange(m["nrows"]) * m["res"]
    return lons, lats


def read_imd_binary(filepath: str, variable: str, year: int) -> np.ndarray:
    """
    Read an IMD binary .grd file into a 3D NumPy array.
    Shape: (ndays, nrows, ncols)
    Missing values (e.g. -999.0 or 99.9) are converted to np.nan.
    """
    m = IMD_META.get(variable)
    if m is None:
        raise ValueError(f"Variable must be one of: {list(IMD_META.keys())}")

    ndays = get_days_in_year(year)
    expected_vals = m["ncols"] * m["nrows"] * ndays
    expected_bytes = expected_vals * 4

    with open(filepath, "rb") as f:
        data = f.read()

    if len(data) != expected_bytes:
        raise ValueError(f"File size mismatch: got {len(data)} bytes, expected {expected_bytes} bytes for {year}")

    # Read binary floats (little-endian 32-bit IEEE float)
    raw = np.frombuffer(data, dtype="<f4")
    # IMD binary stores data as [ndays, nrows, ncols] or [ndays, ncols, nrows]
    # In imdR: arr <- array(raw_vals, dim = c(ncols, nrows, ndays))
    arr = raw.reshape((ndays, m["nrows"], m["ncols"]))

    # Mask missing values
    arr = np.where(np.isclose(arr, m["na_val"], atol=0.05), np.nan, arr)
    return arr


def write_imd_binary(arr: np.ndarray, filepath: str, variable: str, year: int) -> str:
    """
    Write a 3D NumPy array of shape (ndays, nrows, ncols) to an IMD binary .grd file.
    NaNs are replaced with the standard IMD missing value.
    """
    m = IMD_META.get(variable)
    if m is None:
        raise ValueError(f"Variable must be one of: {list(IMD_META.keys())}")

    ndays = get_days_in_year(year)
    if arr.shape[0] != ndays or arr.shape[1] != m["nrows"] or arr.shape[2] != m["ncols"]:
        raise ValueError(f"Array shape {arr.shape} does not match expected ({ndays}, {m['nrows']}, {m['ncols']})")

    clean_arr = np.where(np.isnan(arr), m["na_val"], arr).astype("<f4")
    with open(filepath, "wb") as f:
        f.write(clean_arr.tobytes())
    return filepath


def extract_point_timeseries(
    arr: np.ndarray,
    lat: float,
    lon: float,
    variable: str = "rain",
    year: int = 2026
) -> pd.DataFrame:
    """
    Extract daily time series for a specific latitude and longitude coordinate.
    Uses nearest-neighbor interpolation on the IMD grid.
    """
    m = IMD_META[variable]
    lons, lats = get_grid_coordinates(variable)

    col_idx = int(np.clip(np.round((lon - m["xmin"]) / m["res"]), 0, m["ncols"] - 1))
    row_idx = int(np.clip(np.round((lat - m["ymin"]) / m["res"]), 0, m["nrows"] - 1))

    ndays = get_days_in_year(year)
    dates = [datetime.date(year, 1, 1) + datetime.timedelta(days=i) for i in range(ndays)]
    vals = arr[:, row_idx, col_idx]

    df = pd.DataFrame({
        "date": [d.isoformat() for d in dates],
        "lat": lats[row_idx],
        "lon": lons[col_idx],
        variable: vals
    })
    return df


def extract_bbox_grid(
    arr: np.ndarray,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    variable: str = "rain"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Crop the IMD grid array to a bounding box (e.g. for a district).
    Returns: (cropped_arr, sub_lats, sub_lons)
    """
    m = IMD_META[variable]
    lons, lats = get_grid_coordinates(variable)

    col_mask = (lons >= (min_lon - m["res"] / 2)) & (lons <= (max_lon + m["res"] / 2))
    row_mask = (lats >= (min_lat - m["res"] / 2)) & (lats <= (max_lat + m["res"] / 2))

    sub_arr = arr[:, row_mask, :][:, :, col_mask]
    return sub_arr, lats[row_mask], lons[col_mask]
