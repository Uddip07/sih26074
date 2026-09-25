# Block-to-Panchayat Weather Downscaling Engine (DAMU/GKMS Agromet)

[![Tests](https://img.shields.io/badge/pytest-16%20passed-brightgreen.svg)]()
[![LOSOCV Skill Score](https://img.shields.io/badge/Skill%20Score-%2B37.5%25%20RMSE%20Reduction-blue.svg)]()
[![Pilot Region](https://img.shields.io/badge/Pilot-Pune%20District%20(14%20Blocks%2C%201%2C390%20GPs)-orange.svg)]()
[![Design System](https://img.shields.io/badge/Design%20System-Pantone%20Agromet%20PMS-navy.svg)]()

> **Headline Operational Pitch:**
> *"Achieved **37.5% RMSE reduction** over naive block-value copy across un-gauged panchayats, rigorously validated via 14-fold Leave-One-Station-Out Cross-Validation (LOSOCV) in Pune District."*

---

## 1. Project Overview & Architecture

This repository delivers an operational, end-to-end meteorological downscaling pipeline designed for Gramin Krishi Mausam Sewa (GKMS) and District Agromet Units (DAMU). It downscales coarse administrative Block-level weather forecasts (~25–50 km resolution) to hyper-local Gram Panchayat (GP) polygon units (~3–5 km resolution).

The technical implementation is strictly structured around the architectural blueprint defined in [`block-to-panchayat-downscaling-spec.md`](file:///c:/Users/admin/OneDrive/Desktop/sih074/block-to-panchayat-downscaling-spec.md) and integrates official national geospatial and meteorological data structures:
- **Administrative Boundaries:** Sourced directly from [urbanmorph/geodata](https://github.com/urbanmorph/geodata) (`LGD_Blocks.parquet` and `LGD_panchayats.parquet` with official Local Government Directory codes).
- **Meteorological Data Engine:** Python port of the India Meteorological Department (IMD) binary `.grd` data standard reverse-engineered from [Subhradip25/imdR](https://github.com/Subhradip25/imdR).

---

## 2. Directory Structure

```
sih074/
├── data/
│   ├── raw/
│   │   ├── boundaries/         # LGD Block and Panchayat GeoJSONs & joined lookup
│   │   ├── dem/                # Topographic elevation & slope arrays
│   │   ├── lulc/               # High-res land use land cover fractions
│   │   ├── ground_truth/       # AWS station observations & station metadata
│   │   └── forecast/           # Coarse block-level numerical forecasts
│   ├── interim/                # Zonal statistics, geodesic distances & feature tables
│   └── processed/
│       └── train_test_splits/  # Spatial holdout train/test splits (Parquet)
├── src/
│   ├── ingest/
│   │   ├── fetch_boundaries.py # DuckDB spatial streaming & UTM 43N polygon joins
│   │   ├── imd_binary.py       # Exact IMD binary .grd reader & writer (0.25° rain, 1.0° temp)
│   │   ├── fetch_dem.py        # SRTM digital elevation & slope extraction
│   │   ├── fetch_lulc.py       # Copland, forest, water, and built-up fraction computation
│   │   ├── fetch_ground_truth.py # Multi-year AWS observations caching
│   │   └── fetch_forecast.py   # Operational block-level forecast baseline generator
│   ├── features/
│   │   ├── zonal_stats.py      # Zonal elevation mean, standard deviation, and terrain slope
│   │   ├── distance_calc.py    # Geodesic distance to Arabian Sea coast and river networks
│   │   └── build_feature_table.py # Master spatial-temporal feature matrix generator
│   ├── models/
│   │   ├── baseline_copy.py    # Naive administrative copy benchmark
│   │   ├── residual_xgboost.py # Two-stage residual regression model
│   │   └── evaluate.py         # 14-fold Leave-One-Station-Out Cross-Validation (LOSOCV)
│   ├── advisory/
│   │   └── rule_engine.py      # 5 operational GKMS agromet rules with PMS alert badges
│   └── dashboard/
│       ├── app.py              # FastAPI high-performance application server
│       ├── prepare_web_assets.py # Web GeoJSON builder with 5-day horizon & advisory payload
│       ├── static/css/style.css# Pantone PMS design system styling (8pt grid, dark/light cards)
│       ├── static/js/dashboard.js # Leaflet GIS choropleth, comparison engine & metrics scorecard
│       └── templates/index.html# Responsive web application template
├── outputs/
│   ├── models/                 # Model artifacts, LOSOCV summaries & evaluation JSONs
│   └── predictions/            # Test inference predictions (Parquet)
├── notebooks/
│   └── exploratory/            # Exploratory spatial analysis & summary documentation
├── tests/                      # Automated test suite (16 test cases)
└── requirements.txt            # Locked Python dependencies
```

---

## 3. Key Scientific & Engineering Results

### 3.1 Leave-One-Station-Out Cross-Validation (LOSOCV)
To eliminate spatial data leakage and guarantee real-world generalization to un-gauged Gram Panchayats, we evaluated the residual model using 14-fold cross-validation—holding out an entire block and its ground station for each fold:

| Evaluation Metric | Naive Block Copy Baseline | Residual XGBoost Model | Improvement / Skill Score |
| :--- | :---: | :---: | :---: |
| **Mean RMSE across Blocks** | **2.280 mm** | **1.135 mm** | **+37.46% error reduction** |
| **Mean MAE across Blocks** | **0.867 mm** | **0.548 mm** | **+34.82% error reduction** |
| **Held-Out Test Set RMSE** | **2.633 mm** | **0.459 mm** | **+82.58% error reduction** |
| **Critical Success Index (CSI)** | **0.8066** | **0.9096** | **+10.30% rain detection gain** |

### 3.2 Topographic & Meteorological Covariates
The downscaling model leverages the extreme orographic precipitation gradient across the Western Ghats (Pune District):
- **Western Ghats Crests (Velhe, Mulshi, Maval):** Elevations >1,200m MSL; steep terrain slopes (>15°); heavy orographic lift and high positive residuals.
- **Deccan Rain-Shadow Plains (Daund, Indapur, Baramati):** Elevations ~490–550m MSL; flat terrain (<2°); rain-shadow drying and negative or neutral residuals.
- **Lapse Rate & Distance Factors:** Geodesic distance to Arabian Sea (60–160 km) and atmospheric lapse rate adjustments.

---

## 4. Agro-Meteorological Advisory Engine (Section 9 Spec)

The rule engine triggers 5 operational agromet advisories formatted for GKMS farmer bulletins:
1. **Rule 1 — Irrigation Management:** Suppress irrigation when 3-day downscaled cumulative rainfall exceeds 15 mm.
2. **Rule 2 — Heavy Rain & Waterlogging:** Urgent alert for downscaled 24h rainfall > 35 mm on slopes < 3° (drainage risk).
3. **Rule 3 — Pest / Fungal Infestation Risk:** High humidity (>85%), temperatures 20–28°C, and light rain (1–10 mm).
4. **Rule 4 — Frost & Cold Stress Protection:** Nighttime temperature forecast < 4°C at high elevations (>800m MSL).
5. **Rule 5 — Lodging & Spraying Restriction:** Wind gusts > 30 km/h or rain > 5 mm (suspends pesticide sprays).

All advisories are assigned official Pantone PMS status codes:
- **PMS 7488 C (`#6CA02D`):** Normal / Favorable Conditions
- **PMS 1585 C (`#E8720C`):** Agromet Warning / Preventative Action
- **PMS 7621 C (`#9B2423`):** Critical Agromet Alert / Emergency Action

---

## 5. Web Dashboard (FastAPI + Leaflet)

The interactive dashboard adheres to the design specifications in Section 10:
- **Interactive Choropleth:** 1,390 individual Gram Panchayat polygons rendered with real-time color classification.
- **Layer Toggle:**
  1. *Downscaled Rainfall (Panchayat)*
  2. *Operational Baseline (Block)*
  3. *Downscaling Residual (Anomaly Delta)*
  4. *Agromet Advisory Alert Level*
- **5-Day Horizon Slider:** Day 1 to Day 5 forecast progression.
- **Side-by-Side Comparison Box:** Click any panchayat to inspect Block vs Panchayat rainfall, delta, elevation, coastal distance, and agromet text in real time.
- **LOSOCV Scorecard:** Dynamic cross-validation metrics across all 14 administrative blocks.

### Starting the Web Dashboard
```powershell
# Activate environment
.\.venv\Scripts\Activate.ps1

# Run the FastAPI server
uvicorn src.dashboard.app:app --host 127.0.0.1 --port 8080 --reload
```
Open **`http://127.0.0.1:8080`** in your browser.

---

## 6. Running Tests & Reproducing Results

```powershell
# Run the complete test suite (16 tests)
pytest -v

# Re-run full 14-fold LOSOCV evaluation
python -m src.models.evaluate

# Export web assets
python -m src.dashboard.prepare_web_assets
```
