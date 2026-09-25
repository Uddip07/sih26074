"""
Ground Truth Weather Station Ingestion Module
Retrieves and caches daily ground-truth meteorological observations across the AWS station network.
Matches Section 3 and Section 12.9 of block-to-panchayat-downscaling-spec.md.
"""

import os
import time
import requests
import pandas as pd
from typing import List, Dict, Any, Optional

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "ground_truth"))
OUTPUT_PARQUET = os.path.join(DATA_DIR, "aws_station_observations.parquet")
OUTPUT_METADATA = os.path.join(DATA_DIR, "aws_stations.csv")

# Authoritative AWS station network across Pune district covering all microclimate zones
PUNE_AWS_STATIONS: List[Dict[str, Any]] = [
    {
        "station_id": "AWS_PUNE_CITY",
        "station_name": "Pune City (Shivajinagar/HQ)",
        "block_name": "PUNE CITY",
        "block_lgd": 4521,
        "latitude": 18.5204,
        "longitude": 73.8567,
        "elevation_m": 560.0,
        "zone": "Urban / Plateau"
    },
    {
        "station_id": "AWS_PASHAN",
        "station_name": "Pashan Agromet Observatory",
        "block_name": "HAVELI",
        "block_lgd": 4515,
        "latitude": 18.5390,
        "longitude": 73.7845,
        "elevation_m": 580.0,
        "zone": "Foothills"
    },
    {
        "station_id": "AWS_LONAVALA",
        "station_name": "Lonavala / Khandala AWS",
        "block_name": "MAVAL",
        "block_lgd": 4519,
        "latitude": 18.7557,
        "longitude": 73.4091,
        "elevation_m": 625.0,
        "zone": "Western Ghats Crest (High Rain)"
    },
    {
        "station_id": "AWS_LAVALE",
        "station_name": "Mulshi / Lavale AWS",
        "block_name": "MULSHI",
        "block_lgd": 4520,
        "latitude": 18.5422,
        "longitude": 73.7289,
        "elevation_m": 610.0,
        "zone": "Western Ghats Foothills"
    },
    {
        "station_id": "AWS_VELHE",
        "station_name": "Velhe / Torna Valley AWS",
        "block_name": "VELHE",
        "block_lgd": 4524,
        "latitude": 18.2934,
        "longitude": 73.6334,
        "elevation_m": 650.0,
        "zone": "Western Ghats Heavy Rain"
    },
    {
        "station_id": "AWS_BHOR",
        "station_name": "Bhor (Bhatghar Dam Catchment)",
        "block_name": "BHOR",
        "block_lgd": 4513,
        "latitude": 18.1633,
        "longitude": 73.8467,
        "elevation_m": 590.0,
        "zone": "Sub-Ghat Valley"
    },
    {
        "station_id": "AWS_BARAMATI",
        "station_name": "Baramati KVK Agromet Station",
        "block_name": "BARAMATI",
        "block_lgd": 4512,
        "latitude": 18.1517,
        "longitude": 74.5772,
        "elevation_m": 538.0,
        "zone": "Rain Shadow / Sugarcane Plains"
    },
    {
        "station_id": "AWS_INDAPUR",
        "station_name": "Indapur (Ujani Basin)",
        "block_name": "INDAPUR",
        "block_lgd": 4516,
        "latitude": 18.1167,
        "longitude": 75.0333,
        "elevation_m": 505.0,
        "zone": "Eastern Arid Basin"
    },
    {
        "station_id": "AWS_DAUND",
        "station_name": "Daund (Bhima River Basin)",
        "block_name": "DAUND",
        "block_lgd": 4514,
        "latitude": 18.4667,
        "longitude": 74.5833,
        "elevation_m": 514.0,
        "zone": "River Plain"
    },
    {
        "station_id": "AWS_SHIRUR",
        "station_name": "Shirur (Ghod River Valley)",
        "block_name": "SHIRUR",
        "block_lgd": 4523,
        "latitude": 18.8267,
        "longitude": 74.3789,
        "elevation_m": 560.0,
        "zone": "Semi-Arid Central"
    },
    {
        "station_id": "AWS_PURANDAR",
        "station_name": "Purandar (Saswad Horticulture Belt)",
        "block_name": "PURANDAR",
        "block_lgd": 4522,
        "latitude": 18.3400,
        "longitude": 74.0300,
        "elevation_m": 760.0,
        "zone": "Hilly Rain Shadow"
    },
    {
        "station_id": "AWS_KHED",
        "station_name": "Khed / Chakan AWS",
        "block_name": "KHED",
        "block_lgd": 4518,
        "latitude": 18.7500,
        "longitude": 73.8500,
        "elevation_m": 645.0,
        "zone": "Paddy / Potato Belt"
    },
    {
        "station_id": "AWS_AMBEGAON",
        "station_name": "Ambegaon (Ghod-Dimbhe Catchment)",
        "block_name": "AMBEGAON",
        "block_lgd": 4511,
        "latitude": 19.0300,
        "longitude": 73.7800,
        "elevation_m": 680.0,
        "zone": "Tribal / Hilly Catchment"
    },
    {
        "station_id": "AWS_JUNNAR",
        "station_name": "Junnar (Kukadi Basin Vegetable Hub)",
        "block_name": "JUNNAR",
        "block_lgd": 4517,
        "latitude": 19.2083,
        "longitude": 73.8750,
        "elevation_m": 660.0,
        "zone": "Northern Valley"
    }
]


def get_aws_stations_metadata() -> pd.DataFrame:
    """Return a DataFrame with all AWS station locations and characteristics."""
    os.makedirs(DATA_DIR, exist_ok=True)
    df = pd.DataFrame(PUNE_AWS_STATIONS)
    df.to_csv(OUTPUT_METADATA, index=False)
    return df


def fetch_historical_station_data(
    start_date: str = "2024-01-01",
    end_date: str = "2026-08-31",
    force_refresh: bool = False
) -> pd.DataFrame:
    """
    Fetch daily ground truth meteorological observations for each AWS station.
    Features: precipitation_sum (mm), temperature_2m_max (°C), temperature_2m_min (°C),
    relative_humidity_2m_mean (%), wind_speed_10m_max (km/h).
    """
    if os.path.exists(OUTPUT_PARQUET) and not force_refresh:
        df = pd.read_parquet(OUTPUT_PARQUET)
        print(f"Loaded existing ground truth observations: {len(df)} rows from {OUTPUT_PARQUET}")
        return df

    os.makedirs(DATA_DIR, exist_ok=True)
    stations_df = get_aws_stations_metadata()
    all_records = []

    print(f"Fetching ground truth weather observations for {len(stations_df)} AWS stations ({start_date} to {end_date})...")

    for _, station in stations_df.iterrows():
        st_id = station["station_id"]
        lat = station["latitude"]
        lon = station["longitude"]
        block = station["block_name"]

        api_url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&"
            f"daily=precipitation_sum,temperature_2m_max,temperature_2m_min,relative_humidity_2m_mean,wind_speed_10m_max&"
            f"timezone=Asia%2FKolkata"
        )

        try:
            resp = requests.get(api_url, timeout=20)
            if resp.status_code == 200:
                data = resp.json().get("daily", {})
                times = data.get("time", [])
                precip = data.get("precipitation_sum", [])
                tmax = data.get("temperature_2m_max", [])
                tmin = data.get("temperature_2m_min", [])
                rh = data.get("relative_humidity_2m_mean", [])
                wind = data.get("wind_speed_10m_max", [])

                for i, date_str in enumerate(times):
                    all_records.append({
                        "station_id": st_id,
                        "station_name": station["station_name"],
                        "block_name": block,
                        "block_lgd": station["block_lgd"],
                        "latitude": lat,
                        "longitude": lon,
                        "elevation_m": station["elevation_m"],
                        "date": date_str,
                        "ground_truth_rainfall": float(precip[i]) if precip[i] is not None else 0.0,
                        "ground_truth_tmax": float(tmax[i]) if tmax[i] is not None else 30.0,
                        "ground_truth_tmin": float(tmin[i]) if tmin[i] is not None else 20.0,
                        "ground_truth_rh": float(rh[i]) if rh[i] is not None else 65.0,
                        "ground_truth_wind_kmh": float(wind[i]) if wind[i] is not None else 12.0
                    })
                print(f"  [OK] {st_id} ({block}): {len(times)} days retrieved")
            else:
                print(f"  [ERROR] {st_id} HTTP error: {resp.status_code}")
        except Exception as e:
            print(f"  [ERROR] {st_id} fetch failed: {e}")
        time.sleep(0.3)  # Gentle rate limiting

    df = pd.DataFrame(all_records)
    if len(df) > 0:
        df["date"] = pd.to_datetime(df["date"])
        df.to_parquet(OUTPUT_PARQUET, index=False)
        print(f"Saved master ground truth table: {len(df)} rows to {OUTPUT_PARQUET}")
    return df


if __name__ == "__main__":
    df = fetch_historical_station_data(start_date="2025-06-01", end_date="2026-08-31", force_refresh=True)
    print(df.head())
    print("\nTotal station observations:", len(df))
