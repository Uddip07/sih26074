# Block-to-Panchayat Weather Downscaling
## Full Implementation & Design Specification

**Domain:** Agro-meteorological advisory services
**Approach:** Statistical/ML downscaling (residual/bias-correction learning)
**Prediction unit:** Panchayat polygon (not raster pixel)

---

## 1. Problem Deconstruction

| Term in problem statement | What it actually means |
|---|---|
| Block level | Administrative polygon, ~50–200 km², irregular shape, one IMD/GKMS forecast value per block |
| Panchayat level | Sub-unit of a block, 10–30 panchayats per block, irregular polygon |
| "High-res from low-res" | Not image super-resolution — this is **polygon-to-polygon disaggregation** |
| Agro-meteorological advisory | Output must map to farmer decisions: irrigation, sowing, pest/disease risk, harvest timing |

**Core question the model must answer:** given one forecast value for an entire block, what differentiates each panchayat inside it (terrain, land use, microclimate history) enough to justify a different number?

---

## 2. Scope Lock (Phase 0 — decide before writing any code)

| Decision | Value |
|---|---|
| Primary variable | Rainfall (mm/day) |
| Secondary variables | Temperature max/min (°C), relative humidity (%) |
| Pilot region | 1 district (choose one with existing AWS/KVK station density) |
| Forecast horizon | 5-day, matching GKMS block-level cadence |
| Coordinate reference system | EPSG:4326 (WGS84) for storage, EPSG:32643/32644 (UTM zone for India) for area/distance calculations |

---

## 3. Data Sources

| Data | Source | Native resolution | Format | Role |
|---|---|---|---|---|
| Block-level forecast (historical) | IMD GKMS/DAMU bulletins, or IMD-NCUM/GFS grid resampled to block polygons | Block polygon / 12 km grid | CSV / GRIB2 | Input (X) |
| Panchayat boundaries | LGD (Local Government Directory) or Census 2011 GIS | Vector polygon | Shapefile/GeoJSON | Prediction unit |
| Block boundaries | LGD | Vector polygon | Shapefile/GeoJSON | Join key |
| DEM (elevation, slope, aspect) | SRTM 30 m via Bhuvan/USGS EarthExplorer | 30 m | GeoTIFF | Static covariate |
| Land use/land cover | Bhuvan LULC or ESA WorldCover | 10–30 m | GeoTIFF | Static covariate |
| AWS/rain gauge ground truth | IMD / state agromet network | Point station | CSV | Label (y), validation |
| Satellite proxy rainfall | IMERG (GPM) | 0.1° (~11 km) | NetCDF | Pseudo-label to densify training |
| Satellite proxy temperature | MODIS LST | 1 km | HDF/GeoTIFF | Pseudo-label |
| Coastline/river network | Natural Earth / Bhuvan hydrology layer | Vector | Shapefile | Distance covariate |

---

## 4. Repository Structure

```
downscaling-project/
├── data/
│   ├── raw/
│   │   ├── boundaries/          # block + panchayat shapefiles
│   │   ├── dem/
│   │   ├── lulc/
│   │   ├── forecast/            # historical block-level forecasts
│   │   └── ground_truth/        # AWS + IMERG + MODIS extracts
│   ├── interim/
│   │   └── panchayat_features.parquet
│   └── processed/
│       └── train_test_splits/
├── src/
│   ├── ingest/
│   │   ├── fetch_dem.py
│   │   ├── fetch_lulc.py
│   │   └── fetch_forecast.py
│   ├── features/
│   │   └── build_feature_table.py
│   ├── models/
│   │   ├── baseline_copy.py
│   │   ├── residual_xgboost.py
│   │   └── evaluate.py
│   ├── advisory/
│   │   └── rule_engine.py
│   └── dashboard/
│       └── app.py
├── notebooks/
│   └── exploratory/
├── outputs/
│   ├── models/
│   └── predictions/
└── requirements.txt
```

**File naming convention:** `{region}_{variable}_{YYYYMMDD}_{resolution}.ext`
Example: `district01_rainfall_20260625_panchayat.geojson`

---

## 5. Data Pipeline (precise steps)

1. **Boundary join** — spatial join panchayat centroids to block polygons (`geopandas.sjoin`, predicate=`within`) → produces `panchayat_id → block_id` lookup table. This is the single most important join in the whole pipeline; validate it manually against a printed map before proceeding.
2. **Reprojection** — reproject all raster layers (DEM, LULC) to EPSG:32643/44 for accurate area/slope calculation; reproject back to EPSG:4326 for storage/serving.
3. **Zonal statistics** — for each panchayat polygon, compute `mean`, `std`, `min`, `max` of elevation and slope using `rasterstats.zonal_stats`.
4. **LULC fractions** — compute percentage of each land-use class within each panchayat polygon (cropland, forest, built-up, water) via categorical zonal stats.
5. **Distance features** — compute panchayat centroid distance to nearest coastline/river using `shapely.distance` after projecting to a metric CRS.
6. **Ground truth gridding** — for IMERG/MODIS, extract the pixel value(s) intersecting each panchayat centroid (or area-weighted average if the panchayat spans multiple pixels).
7. **Historical bias feature** — for each panchayat, compute `mean(ground_truth − block_forecast)` over the available historical record (minimum 1 monsoon season, ideally 3+ years).
8. **Assemble master table** — one row per `(panchayat_id, date)`: block forecast value, all static covariates, historical bias, day-of-year, season flag, target ground truth value.

---

## 6. Feature Set (final table schema)

| Feature | Type | Notes |
|---|---|---|
| `block_forecast_value` | float | Same value repeated for all panchayats in a block |
| `elevation_mean`, `elevation_std` | float | meters |
| `slope_mean` | float | degrees |
| `landuse_cropland_pct`, `landuse_forest_pct`, `landuse_water_pct`, `landuse_builtup_pct` | float | 0–100 |
| `dist_to_coast_km`, `dist_to_water_km` | float | km |
| `historical_bias` | float | ground_truth − forecast, averaged over history |
| `latitude`, `longitude` | float | panchayat centroid |
| `day_of_year` | int | 1–365 |
| `is_monsoon` | bool | June–September flag |
| **target** `residual` | float | `ground_truth − block_forecast_value` |

---

## 7. Modeling

### 7.1 Baseline (build first, always)
Naive spatial copy: every panchayat inherits its parent block's forecast value unchanged. Compute RMSE/MAE against held-out ground truth. This number is the one every subsequent model must beat, and it goes in the final report as the headline comparison.

### 7.2 Residual learning model
```
target = ground_truth − block_forecast_value
prediction = block_forecast_value + model.predict(features)
```

**Model:** XGBoost regressor (or LightGBM)

**Starting hyperparameter grid:**
| Param | Range to search |
|---|---|
| `n_estimators` | 200–800 |
| `max_depth` | 3–8 |
| `learning_rate` | 0.01–0.1 |
| `subsample` | 0.7–1.0 |
| `colsample_bytree` | 0.7–1.0 |
| `min_child_weight` | 1–5 |

Use `GroupKFold` grouped by `panchayat_id` (not random shuffling) to avoid leakage between train/test splits of the same location across dates.

### 7.3 Validation strategy
**Leave-one-station-out cross-validation** — for each fold, hold out all data associated with one AWS station's panchayat, train on the rest, predict on the held-out one. This simulates the real deployment case: predicting for a panchayat with no ground station.

---

## 8. Evaluation Metrics

| Metric | Formula | Purpose |
|---|---|---|
| RMSE | √(mean((y_pred − y_true)²)) | Overall error magnitude |
| MAE | mean(\|y_pred − y_true\|) | Robust to outliers |
| Skill Score | 1 − (RMSE_model / RMSE_baseline) | % improvement over naive copy — headline number |
| CSI (rainfall threshold e.g. 2.5mm) | hits / (hits + misses + false_alarms) | Meteorological standard, not just regression error |

---

## 9. Advisory Rule Engine

| Condition | Advisory |
|---|---|
| Predicted rainfall < 2.5 mm in next 3 days + crop stage = sowing/vegetative | Irrigation advisory issued |
| Predicted rainfall > 50 mm in 24 hr | Waterlogging/drainage advisory |
| Temp > 35°C + humidity > 70% | Fungal/pest risk flag |
| Temp < 5°C forecast | Frost protection advisory |
| Wind speed > 40 km/h forecast | Crop lodging risk, delay spraying operations |

Match output phrasing to existing GKMS advisory templates so agromet officers can adopt it without a new format to learn.

---

## 10. Design System (Dashboard/Demo UI)

### 10.1 Color Palette (Pantone-referenced, with sRGB hex for implementation)

| Role | Pantone (PMS) | Hex (approx. sRGB conversion) | Usage |
|---|---|---|---|
| Primary — Sky/Water | PMS 2955 C | `#003D6B` | Headers, primary buttons, map base layer |
| Secondary — Agriculture Green | PMS 7488 C | `#6CA02D` | Cropland indicators, positive advisory states |
| Accent — Advisory Amber | PMS 1585 C | `#E8720C` | Warning-level advisories (moderate risk) |
| Alert — Severe Red | PMS 7621 C | `#9B2423` | Severe advisories (frost, waterlogging, extreme heat) |
| Neutral — Cool Gray | PMS Cool Gray 7 C | `#8B8D8E` | Body text secondary, borders, disabled states |
| Neutral — Near Black | PMS Neutral Black C | `#2B2B2B` | Primary body text |
| Background | PMS White (paper) | `#FAFAF8` | Page background (warm off-white, not pure white) |
| Surface | PMS Cool Gray 1 C | `#F1F1F0` | Card backgrounds |

> Note: sRGB hex values are standard-practice screen conversions of the Pantone spot colors above; exact on-screen rendering varies by monitor calibration, as with any PMS-to-RGB mapping.

### 10.2 Spacing Scale (8-point grid)

| Token | Value | Usage |
|---|---|---|
| `space-1` | 4px | Icon-to-label gaps |
| `space-2` | 8px | Tight component padding |
| `space-3` | 12px | Form field internal padding |
| `space-4` | 16px | Default card padding |
| `space-5` | 24px | Section spacing within a card |
| `space-6` | 32px | Card-to-card gutter |
| `space-7` | 48px | Section-to-section vertical rhythm |
| `space-8` | 64px | Page-level top/bottom margin |
| `space-9` | 96px | Hero/header block spacing |

### 10.3 Typography

| Role | Font | Size | Line-height | Weight |
|---|---|---|---|---|
| Display (page title) | Inter | 40px | 48px | 700 |
| H1 (section) | Inter | 32px | 40px | 700 |
| H2 (subsection) | Inter | 24px | 32px | 600 |
| H3 (card title) | Inter | 18px | 24px | 600 |
| Body | Inter | 16px | 24px | 400 |
| Body small / caption | Inter | 14px | 20px | 400 |
| Data label / numeric | JetBrains Mono | 14px | 20px | 500 |
| Micro label | Inter | 12px | 16px | 500 (uppercase, +0.04em tracking) |

### 10.4 Component Specs

- **Cards:** 12px corner radius, 1px border in Cool Gray 1 (`#F1F1F0`), `space-4` (16px) internal padding, subtle shadow `0 1px 3px rgba(0,0,0,0.08)`.
- **Buttons (primary):** height 40px, horizontal padding `space-4` (16px), corner radius 8px, background PMS 2955 C, white text, hover state darkens 8%.
- **Map markers:** panchayat polygons filled with a 3-step choropleth ramp from Agriculture Green (low risk) → Advisory Amber (moderate) → Severe Red (high), 60% fill opacity, 1px white polygon borders for separation.
- **Grid gutters (dashboard layout):** 24px between map panel and side panel; 16px between stacked advisory cards.
- **Breakpoints:** mobile < 768px (single column), tablet 768–1024px (map + collapsible side panel), desktop > 1024px (map + fixed side panel).

---

## 11. Build Timeline (5-day pacing)

| Day | Tasks |
|---|---|
| 1 | Boundary overlay + block↔panchayat lookup table; static covariate extraction begins |
| 2 | Finish covariate extraction; baseline (copy-block-value) model + evaluation harness |
| 3 | Feature table assembly; XGBoost residual model training, initial hyperparameter search |
| 4 | Leave-one-station-out validation; iterate features based on error analysis |
| 5 | Advisory rule layer; dashboard build using Section 10 design system; final skill-score writeup |

---

## 12. Free Dataset Directory (exact sources)

All entries below are free; "login" means a one-time free account, not a paywall.

### 12.1 Boundaries (panchayat / village / block / district)

| Source | Link | Notes |
|---|---|---|
| Local Government Directory (LGD) | https://lgdirectory.gov.in/ | Official panchayat/block codes and names; no polygon geometry, codes only |
| UrbanMorph India admin boundary catalog | https://awesome.ecosyste.ms/projects/github.com%2Furbanmorph%2Fgeodata | State/district/subdistrict/block/village polygons aligned to LGD codes; download as Shapefile/GeoJSON/KML/Parquet directly |
| Indian village boundaries (GitHub) | https://github.com/naveenpf/indian_village_boundaries | Community-maintained village-level GeoJSON, useful as a cross-check |

**Critical step:** cross-reference the LGD code list against whichever polygon source you use — codes and polygons rarely come from the same file, so the join has to be done manually (state + district + block name matching).

### 12.2 Terrain (DEM)

| Source | Link | Notes |
|---|---|---|
| USGS EarthExplorer | https://earthexplorer.usgs.gov/ | SRTM 1 Arc-Second Global (30 m), GeoTIFF/DTED/BIL; free registration; Data Sets → Digital Elevation → SRTM |

### 12.3 Land Use / Land Cover

| Source | Link | Notes |
|---|---|---|
| Bhuvan Thematic Services (ISRO) | https://bhuvan-app1.nrsc.gov.in/thematic | LULC at 1:50,000 (2005–06 to 2015–16) and 1:250,000 (2004–present); free registration; use "Get Data" → clip-and-ship for your district |

### 12.4 Block-level historical forecast proxy / gridded observations

| Source | Link | Notes |
|---|---|---|
| IMD Pune gridded data | https://imdpune.gov.in | Source of the binary `.grd` rainfall (0.25°, 1901–present) and tmax/tmin (1.0°, 1951–present) grids |
| `imdR` (R package) | https://github.com/Subhradip25/imdR | Automates downloading and reading the IMD Pune grids directly into rasters — much faster than manual `.grd` parsing |

### 12.5 Ground truth (AWS / rain gauge)

| Source | Link | Notes |
|---|---|---|
| IMD AWS live viewer | http://aws.imd.gov.in/ | Hourly station-wise precipitation by state; view-only, no bulk download |
| India Water Portal archive | https://admin.indiawaterportal.org/articles/station-wise-hourly-rainfall-data-imd-now-available | Community-archived scrape of the above for offline use |

### 12.6 Satellite rainfall proxy (IMERG)

| Source | Link | Notes |
|---|---|---|
| NASA GES DISC | https://disc.gsfc.nasa.gov/datasets?keywords=IMERG | Half-hourly, 0.1° (~11 km); free Earthdata login |
| Google Earth Engine catalog | https://developers.google.com/earth-engine/datasets/catalog/NASA_GPM_L3_IMERG_V07 | Query/aggregate directly in GEE without downloading raw files — usually the faster path for a pilot district |

### 12.7 Satellite temperature proxy (MODIS LST)

| Source | Link | Notes |
|---|---|---|
| NASA AppEEARS | https://appeears.earthdatacloud.nasa.gov/ | Point or area extraction of MOD11A1/MYD11A1 (daily, 1 km); free Earthdata login; easiest way to get a clean per-panchayat time series without handling raw HDF tiles |
| LP DAAC | https://lpdaac.usgs.gov/ | Underlying archive AppEEARS draws from, for direct tile downloads if needed |

### 12.8 Coastline / river distance

| Source | Link | Notes |
|---|---|---|
| Natural Earth | https://www.naturalearthdata.com/downloads/ | Public domain, no login; download "Rivers + lake centerlines" and "Coastline" at 1:10m for India-scale precision |

### 12.9 Point historical weather (gap-filling / validation baseline)

| Source | Link | Notes |
|---|---|---|
| Open-Meteo Historical Weather API | https://open-meteo.com/en/docs/historical-weather-api | Free, no API key, ERA5-based reanalysis back to 1940; useful as an independent sanity-check series for any panchayat centroid when AWS/IMERG disagree |

---

## 13. Appendix

**requirements.txt (core):**
```
geopandas
rasterio
rasterstats
xarray
netCDF4
xgboost
lightgbm
scikit-learn
shapely
folium
pandas
numpy
```

**Headline number for the pitch:** *"X% RMSE reduction over naive block-value copy, validated via leave-one-station-out cross-validation."* This single sentence should anchor the presentation — it is the entire value proposition in one line.
