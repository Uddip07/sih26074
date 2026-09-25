"""
NASA GPM IMERG Panchayat Spatial Extraction Module
Extracts daily satellite precipitation (precipitationCal) at each Gram Panchayat
centroid across Pune district from downloaded NetCDF4 files.
Outputs: data/interim/panchayat_imerg.csv (gp_code, date, imerg_rainfall_mm)
"""

import os
import sys
import glob
import re
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

DATA_DIR = os.path.join(BASE_DIR, "data", "raw", "satellite", "imerg")
INTERIM_DIR = os.path.join(BASE_DIR, "data", "interim")
OUTPUT_CSV = os.path.join(INTERIM_DIR, "panchayat_imerg.csv")
BOUNDARIES_GEOJSON = os.path.join(BASE_DIR, "data", "raw", "boundaries", "pune_panchayats.geojson")
ERROR_LOG_PATH = os.path.join(INTERIM_DIR, "imerg_extraction_errors.log")


def load_panchayat_centroids() -> pd.DataFrame:
    """
    Load Pune Gram Panchayat polygons and compute metric centroids in WGS84 EPSG:4326.
    Matches the pattern in src/features/build_feature_table.py.
    """
    if not os.path.exists(BOUNDARIES_GEOJSON):
        raise FileNotFoundError(f"Panchayat boundaries not found at {BOUNDARIES_GEOJSON}")

    panchayats_gdf = gpd.read_file(BOUNDARIES_GEOJSON)
    # Project to metric UTM Zone 43N to compute accurate centroids, then reproject to WGS84
    centroids = panchayats_gdf.to_crs("EPSG:32643").geometry.centroid.to_crs("EPSG:4326")
    panchayats_gdf["latitude"] = centroids.y.round(5)
    panchayats_gdf["longitude"] = centroids.x.round(5)
    panchayats_gdf["gp_code"] = panchayats_gdf["gp_code"].astype(str)

    geo_df = panchayats_gdf[["gp_code", "latitude", "longitude"]].drop_duplicates(subset=["gp_code"]).reset_index(drop=True)
    return geo_df


def extract_date_from_filename(filename: str) -> Optional[str]:
    """
    Parse date from standard IMERG filename:
    3B-DAY.MS.MRG.3IMERG.20240101-S000000-E235959.V07B.nc4 -> '2024-01-01'
    """
    basename = os.path.basename(filename)
    match = re.search(r"3IMERG\.(\d{4})(\d{2})(\d{2})-", basename)
    if match:
        year, month, day = match.groups()
        return f"{year}-{month}-{day}"
    return None


def extract_imerg_for_file(
    nc_path: str,
    geo_df: pd.DataFrame
) -> List[Dict[str, Any]]:
    """
    Extract precipitationCal for all panchayat centroids from a single IMERG NetCDF4 file.
    Returns list of dicts: {'gp_code': ..., 'date': ..., 'imerg_rainfall_mm': ...}
    """
    date_str = extract_date_from_filename(nc_path)
    if not date_str:
        return []

    records: List[Dict[str, Any]] = []

    try:
        with xr.open_dataset(nc_path, engine="netcdf4") as ds:
            # Locate precipitation variable
            var_name = None
            for candidate in ["precipitationCal", "precipitation", "precip", "GPM_3IMERGDF_07_precipitation"]:
                if candidate in ds.data_vars:
                    var_name = candidate
                    break

            if not var_name:
                for k in ds.data_vars.keys():
                    if "precip" in str(k).lower():
                        var_name = k
                        break

            if not var_name:
                raise KeyError(f"No precipitation variable found in {nc_path}. Variables: {list(ds.data_vars.keys())}")

            # Locate coordinates
            lon_name = "lon" if "lon" in ds.coords else ("longitude" if "longitude" in ds.coords else None)
            lat_name = "lat" if "lat" in ds.coords else ("latitude" if "latitude" in ds.coords else None)

            if not lon_name or not lat_name:
                raise KeyError(f"Coordinates (lon/lat) not found in {nc_path}. Coords: {list(ds.coords.keys())}")

            lats = geo_df["latitude"].values
            lons = geo_df["longitude"].values
            gp_codes = geo_df["gp_code"].values

            # Try fast vectorized nearest extraction
            try:
                lats_da = xr.DataArray(lats, dims="points")
                lons_da = xr.DataArray(lons, dims="points")
                extracted = ds[var_name].sel({lon_name: lons_da, lat_name: lats_da}, method="nearest")
                if "time" in extracted.dims:
                    extracted = extracted.squeeze("time")
                values = extracted.values

                for gp_code, val in zip(gp_codes, values):
                    rain_val = float(val) if not np.isnan(val) and val >= 0 else 0.0
                    records.append({
                        "gp_code": str(gp_code),
                        "date": date_str,
                        "imerg_rainfall_mm": round(max(0.0, rain_val), 2)
                    })
            except Exception as e_vec:
                # Fallback to per-point extraction if vectorized selection encounters shape mismatch
                for idx, row in geo_df.iterrows():
                    val = ds[var_name].sel(
                        {lon_name: row["longitude"], lat_name: row["latitude"]},
                        method="nearest"
                    ).values
                    rain_val = float(val.item()) if hasattr(val, "item") else float(val)
                    rain_val = rain_val if not np.isnan(rain_val) and rain_val >= 0 else 0.0
                    records.append({
                        "gp_code": str(row["gp_code"]),
                        "date": date_str,
                        "imerg_rainfall_mm": round(max(0.0, rain_val), 2)
                    })

    except Exception as e:
        msg = f"[ERROR] Failed to extract from {os.path.basename(nc_path)}: {e}\n"
        print(msg.strip())
        os.makedirs(INTERIM_DIR, exist_ok=True)
        with open(ERROR_LOG_PATH, "a", encoding="utf-8") as f_err:
            f_err.write(msg)
        return []

    return records


def extract_all_imerg_panchayats(
    imerg_dir: str = DATA_DIR,
    output_csv: str = OUTPUT_CSV,
    force_recompute: bool = False
) -> pd.DataFrame:
    """
    Extract IMERG precipitation across all available .nc4 files for all panchayats.
    Saves to data/interim/panchayat_imerg.csv with columns:
    gp_code, date, imerg_rainfall_mm
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    geo_df = load_panchayat_centroids()
    print(f"Loaded {len(geo_df)} panchayat centroids for IMERG extraction.")

    nc_files = sorted(glob.glob(os.path.join(imerg_dir, "*.nc4")))
    if not nc_files:
        print(f"[WARN] No .nc4 files found in {imerg_dir}.")
        if os.path.exists(output_csv) and not force_recompute:
            return pd.read_csv(output_csv)
        empty_df = pd.DataFrame(columns=["gp_code", "date", "imerg_rainfall_mm"])
        empty_df.to_csv(output_csv, index=False)
        return empty_df

    print(f"Found {len(nc_files)} IMERG NetCDF4 files to process...")

    # Load existing records if available to avoid re-extracting
    existing_dates = set()
    if os.path.exists(output_csv) and not force_recompute:
        try:
            prev_df = pd.read_csv(output_csv)
            if "date" in prev_df.columns:
                existing_dates = set(prev_df["date"].astype(str).unique())
                print(f"Found {len(existing_dates)} existing dates already extracted in {output_csv}.")
        except Exception:
            existing_dates = set()

    all_records: List[Dict[str, Any]] = []
    processed_count = 0

    for idx, nc_path in enumerate(nc_files, 1):
        f_date = extract_date_from_filename(nc_path)
        if f_date and f_date in existing_dates and not force_recompute:
            continue

        file_records = extract_imerg_for_file(nc_path, geo_df)
        if file_records:
            all_records.extend(file_records)
            processed_count += 1
            if processed_count % 10 == 0 or len(nc_files) <= 10:
                print(f"  [{idx}/{len(nc_files)}] Extracted {len(file_records)} panchayat values for date {f_date}")

    if all_records:
        new_df = pd.DataFrame(all_records)
        if os.path.exists(output_csv) and not force_recompute and len(existing_dates) > 0:
            combined_df = pd.concat([pd.read_csv(output_csv), new_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=["gp_code", "date"]).reset_index(drop=True)
        else:
            combined_df = new_df

        combined_df["gp_code"] = combined_df["gp_code"].astype(str)
        combined_df.sort_values(by=["date", "gp_code"], inplace=True)
        combined_df.to_csv(output_csv, index=False)
        print(f"Saved {len(combined_df)} panchayat-date IMERG observations to {output_csv}")
        return combined_df
    elif os.path.exists(output_csv):
        return pd.read_csv(output_csv)
    else:
        empty_df = pd.DataFrame(columns=["gp_code", "date", "imerg_rainfall_mm"])
        empty_df.to_csv(output_csv, index=False)
        return empty_df


if __name__ == "__main__":
    df = extract_all_imerg_panchayats()
    print(df.head())
