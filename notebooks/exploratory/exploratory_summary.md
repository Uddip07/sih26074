# Exploratory Weather Downscaling Analysis: Pune District

**Pilot Region:** Pune District, Maharashtra (14 Blocks, 1,351 Panchayats)

**Headline Pitch:** *37.5% RMSE reduction over naive block-value copy, validated via leave-one-station-out cross-validation.*

## 1. Baseline vs Residual Model Performance

| Metric | Baseline (Naive Copy) | Residual Model (XGBoost) | Improvement |
|---|---|---|---|
| RMSE | 2.280 mm | 1.135 mm | **37.46%** |
| MAE | 1.119 mm | 0.614 mm | **45.1%** |

## 2. Top Physical Covariates

- **is_monsoon (27.2%)**: Synoptic seasonal flow dynamics
- **historical_bias (20.9%)**: Localized microclimate persistent residual
- **block_forecast_rainfall (18.7%)**: Coarse NWP anchor value
- **landuse_forest_pct (16.5%)**: Orographic surface roughness and moisture transpiration
- **elevation_mean (4.8%)**: Orographic condensation uplift
- **dist_to_coast_km (3.2%)**: Maritime moisture distance decay
