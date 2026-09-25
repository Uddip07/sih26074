"""
Web Asset Preparation Utility
Creates optimized lightweight GeoJSON for the interactive web map,
pre-populating downscaled forecast attributes, residuals, and advisory status.
"""

import os
import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import json
import numpy as np
import pandas as pd
import geopandas as gpd
from src.advisory.rule_engine import evaluate_panchayat_advisories, CropStage, AdvisorySeverity
from src.models.residual_xgboost import DownscalingResidualModel

BOUNDARIES_DIR = os.path.join(BASE_DIR, "data", "raw", "boundaries")
INTERIM_DIR = os.path.join(BASE_DIR, "data", "interim")
WEB_DIR = os.path.join(BASE_DIR, "src", "dashboard", "static", "data")


def prepare_web_geojson():
    os.makedirs(WEB_DIR, exist_ok=True)
    
    print("Loading boundaries and feature tables...")
    panchayats_gdf = gpd.read_file(os.path.join(BOUNDARIES_DIR, "pune_panchayats.geojson"))
    blocks_gdf = gpd.read_file(os.path.join(BOUNDARIES_DIR, "pune_blocks.geojson"))
    features_df = pd.read_parquet(os.path.join(INTERIM_DIR, "panchayat_features.parquet"))
    
    # Load trained model
    model = DownscalingResidualModel(model_type="xgboost")
    model.load(os.path.join(BASE_DIR, "outputs", "models", "residual_xgboost.json"))

    # Sample a typical active monsoon forecast date
    monsoon_dates = features_df[features_df["is_monsoon"]]["date"].drop_duplicates().sort_values()
    sample_date = monsoon_dates.iloc[-10]  # Representative monsoon day
    print(f"Selected demo forecast date: {sample_date.strftime('%Y-%m-%d')}")

    # Generate 5-day horizon dates
    all_dates = features_df["date"].drop_duplicates().sort_values().tolist()
    curr_idx = all_dates.index(sample_date)
    horizon_dates = all_dates[curr_idx:curr_idx + 5]
    if len(horizon_dates) < 5:
        horizon_dates = all_dates[-5:]

    print(f"5-Day horizon dates: {[d.strftime('%Y-%m-%d') for d in horizon_dates]}")

    # Prepare features for the 5 dates
    horizon_data = {}
    for day_offset, d in enumerate(horizon_dates, 1):
        sub = features_df[features_df["date"] == d].copy()
        preds = model.predict_downscaled(sub)
        sub["downscaled_rain"] = preds
        sub["residual"] = np.round(preds - sub["block_forecast_rainfall"], 2)
        sub["clean_id"] = sub["panchayat_id"].astype(str).str.split(".").str[0]
        horizon_data[f"day_{day_offset}"] = {
            "date": d.strftime("%Y-%m-%d"),
            "data": sub.set_index("clean_id")
        }

    # Reference day 1 for base polygon attributes
    day1_df = horizon_data["day_1"]["data"]

    # Simplify geometries slightly (0.0015 deg ~ 150m) for fast 60fps rendering in Leaflet
    panchayats_simplified = panchayats_gdf.copy()
    panchayats_simplified["geometry"] = panchayats_gdf.geometry.simplify(0.0015, preserve_topology=True)
    panchayats_simplified["clean_id"] = panchayats_simplified["gp_code"].astype(str).str.split(".").str[0]

    # Attach properties for day 1
    features_list = []
    for _, row in panchayats_simplified.iterrows():
        pid = row["clean_id"]
        if pid not in day1_df.index:
            continue
        p_row = day1_df.loc[pid]
        if isinstance(p_row, pd.DataFrame):
            p_row = p_row.iloc[0]

        # Evaluate advisory for day 1
        bulletin = evaluate_panchayat_advisories(
            panchayat_id=pid,
            panchayat_name=str(row["gp_name"]),
            block_name=str(p_row["block_name"]),
            forecast_date=horizon_data["day_1"]["date"],
            rainfall_mm=float(p_row["downscaled_rain"]),
            rainfall_3day_mm=float(p_row["downscaled_rain"]) * 2.5,
            tmax_c=float(p_row["ground_truth_tmax"]),
            tmin_c=float(p_row["ground_truth_tmin"]),
            rh_pct=float(p_row["block_forecast_rh"]),
            wind_kmh=float(p_row["block_forecast_wind_kmh"]),
            crop_stage=CropStage.VEGETATIVE
        )

        # 3-step Pantone choropleth ramp matching Section 10.4:
        # Agriculture Green (#6CA02D, low risk/rain) -> Advisory Amber (#E8720C, moderate) -> Severe Red (#9B2423, high)
        rain_val = float(p_row["downscaled_rain"])
        if bulletin.overall_status == AdvisorySeverity.SEVERE or rain_val >= 45.0:
            color = "#9B2423"  # Severe Red
            risk_label = "Severe Alert"
        elif bulletin.overall_status == AdvisorySeverity.ADVISORY or rain_val >= 15.0:
            color = "#E8720C"  # Advisory Amber
            risk_label = "Advisory Warning"
        else:
            color = "#6CA02D"  # Agriculture Green
            risk_label = "Normal / Favorable"

        # Multi-day forecast payload
        daily_forecasts = []
        for i in range(1, 6):
            d_key = f"day_{i}"
            if d_key in horizon_data and pid in horizon_data[d_key]["data"].index:
                d_row = horizon_data[d_key]["data"].loc[pid]
                if isinstance(d_row, pd.DataFrame):
                    d_row = d_row.iloc[0]
                daily_forecasts.append({
                    "day": i,
                    "date": horizon_data[d_key]["date"],
                    "downscaled_rain": float(d_row["downscaled_rain"]),
                    "block_forecast_rain": float(d_row["block_forecast_rainfall"]),
                    "residual": float(d_row["residual"]),
                    "tmax": float(d_row["ground_truth_tmax"]),
                    "tmin": float(d_row["ground_truth_tmin"])
                })

        row_props = {
            "gp_code": pid,
            "gp_name": str(row["gp_name"]),
            "block_name": str(p_row["block_name"]),
            "block_lgd": int(p_row["block_lgd"]),
            "elevation_mean": float(p_row["elevation_mean"]),
            "elevation_std": float(p_row["elevation_std"]),
            "slope_mean": float(p_row["slope_mean"]),
            "dist_to_coast_km": float(p_row["dist_to_coast_km"]),
            "dist_to_water_km": float(p_row["dist_to_water_km"]),
            "cropland_pct": float(p_row["landuse_cropland_pct"]),
            "forest_pct": float(p_row["landuse_forest_pct"]),
            "water_pct": float(p_row["landuse_water_pct"]),
            "builtup_pct": float(p_row["landuse_builtup_pct"]),
            "historical_bias": float(p_row["historical_bias"]),
            "downscaled_rain": rain_val,
            "block_forecast_rain": float(p_row["block_forecast_rainfall"]),
            "residual_rain": float(p_row["residual"]),
            "tmax": float(p_row["ground_truth_tmax"]),
            "tmin": float(p_row["ground_truth_tmin"]),
            "rh": float(p_row["block_forecast_rh"]),
            "wind": float(p_row["block_forecast_wind_kmh"]),
            "advisory_status": bulletin.overall_status.value,
            "risk_label": risk_label,
            "fill_color": color,
            "advisories_count": len(bulletin.advisories),
            "bulletin_text": bulletin.gkms_bulletin_text,
            "daily_forecasts": daily_forecasts
        }

        geom_json = row.geometry.__geo_interface__
        features_list.append({
            "type": "Feature",
            "properties": row_props,
            "geometry": geom_json
        })

    web_geojson = {
        "type": "FeatureCollection",
        "features": features_list
    }

    out_file = os.path.join(WEB_DIR, "pune_panchayats_web.geojson")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(web_geojson, f)

    sz = os.path.getsize(out_file)
    print(f"Exported optimized web GeoJSON: {len(features_list)} panchayats, {sz/(1024*1024):.2f} MB saved to {out_file}")

    # Also export simplified block boundaries for map overlay
    blocks_simplified = blocks_gdf.copy()
    blocks_simplified["geometry"] = blocks_gdf.geometry.simplify(0.0015, preserve_topology=True)
    blocks_out = os.path.join(WEB_DIR, "pune_blocks_web.geojson")
    blocks_simplified.to_file(blocks_out, driver="GeoJSON")
    print(f"Exported optimized block boundaries: {len(blocks_simplified)} blocks to {blocks_out}")


if __name__ == "__main__":
    prepare_web_geojson()
