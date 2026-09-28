# System Architecture

## High-level flow

```text
                 ┌──────────────────────── offline (once, then on retraining) ─────────────────────────┐
                 │                                                                                      │
  Open data ──►  ingest/  ──►  features/  ──►  models/train  ──►  downscaler_operational.joblib         │
  (NWP archive,  (download,    (static GP      (XGBoost per        + evaluation.json, report.html      │
   CHIRPS, ERA5,  area-weight   covariates,     variable, tuned,                                        │
   DEM, land      to blocks     dataset)        calibrated)                                             │
   cover, …)      and GPs)                                                                              │
                 └──────────────────────────────────────────────────────────────────────────────────────┘

                 ┌──────────────────────── daily (pipeline/scheduler) ─────────────────────────────────┐
                 │                                                                                      │
  ECMWF live ──► block forecast ──► features ──► downscaler ──► GP forecast ──► advisory engine ──►    │
  run (Open-      today + 7 days    (same code    point, P10–P90,   (1,349 GPs     rules, crop calendar,  │
  Meteo)          per block         as training)  probabilities,    x 8 days)      ET0, disease models    │
                                                  SHAP                                    │              │
                                                                                          ▼              │
                                                          bulletins mr/hi/en (PDF, SMS, IVR), web issue  │
                 └──────────────────────────────────────────────────────────────────────────────────────┘
                                                                                          │
                                                                                          ▼
                              FastAPI (dashboard/app.py)  ◄──►  SQLite (reviews, subscribers, audit)
                                          │
                                          ▼
                     Gram-Vani web app: maps, panchayat panel, scorecard, officer workflow
```

## Components

### Data ingestion (`src/ingest/`)
Downloads every input from its public source and aggregates it to blocks and Gram Panchayats by exact polygon–grid
area weighting (`src/common/geo.py`). Every output is registered with source, licence and SHA-256 in a provenance
manifest (see [data.md](data.md)).

| Module | Data | Role |
|---|---|---|
| `boundaries` | LGD blocks and Gram Panchayats | spatial units (1,349 modelled GPs in Pune) |
| `nwp_forecast` | ECMWF IFS 0.25° (Open-Meteo Previous Runs + live forecast) | model input (block forecast) |
| `chirps` | CHIRPS v2.0 daily 0.05° (+ preliminary) | rainfall truth, antecedent rain |
| `era5land` | ERA5-Land 0.1° / ERA5 0.25° | temperature, humidity, wind truth |
| `imd_gridded` | IMD 0.25° gauge rainfall | independent check (never trained on) |
| `terrain` | Copernicus DEM 30 m | elevation, slope, windward exposure, Ghat-crest distance |
| `landcover` | ESA WorldCover 10 m | land-cover fractions, reservoirs |
| `hydro` | HydroRIVERS, Natural Earth | distance to rivers / coast |
| `soil` | SoilGrids 2.0 | texture, available water capacity |
| `ndvi` | MODIS 13Q1 (Planetary Computer) | vegetation state, as-of the issue date |
| `stations` | user-supplied AWS / gauge CSVs | validation only (optional) |

### Features (`src/features/`)
`static.py` builds one row of covariates per GP; `dataset.py` expands block forecasts to GPs and adds season,
antecedent rain and NDVI. The **same functions** are used for training and live inference. `bias_encoder.py` gives
each GP its historical bias out-of-fold in time (no leakage), interpolated for GPs never seen in training.

### Downscaling model (`src/models/`)
One XGBoost model per variable (rain, Tmax, Tmin, RH, wind) predicts the **residual** "GP value − block forecast".
It outputs a point forecast, P10/P50/P90 (multi-quantile + split-conformal calibration), probabilities of rain
≥ 2.5 / 15.6 / 64.5 mm (isotonic-calibrated) and TreeSHAP explanations grouped by factor. `evaluate.py` scores it on
never-seen GPs over a held-out test period against six baselines; `report.py` renders the validation report.

### Advisory engine (`src/advisory/`)
Turns each GP's forecast into IMD colour-coded warnings and crop- and stage-aware advice. All thresholds are YAML
(`config/advisory_rules.yaml`, `config/disease_models.yaml`, `config/crop_calendar_<district>.yaml`); bulletins are
rendered in Marathi, Hindi and English as PDF, text, SMS and IVR script.

### Operations (`src/pipeline/`)
- `run.py`: one-command, resumable build of every stage (data → features → training → evaluation → today's forecast).
- `scheduler.py`: runs inside the web server; each morning (after `operational.refresh_hour_local`) it refreshes
  preliminary CHIRPS and stale NDVI, issues the live forecast and publishes it.
- `predict.py`: inference for a live, archived or officer-uploaded (CSV) block forecast; records input completeness.
- `publish.py`: writes compact per-issue JSON and simplified map geometry for the web app.
- `nowcast.py`, `disseminate.py`: short-range rain and a simulated SMS/WhatsApp/IVR outbox.

### Backend API and web app (`src/dashboard/`, entry point `src/main.py`)
FastAPI serves the forecast, advisory, bulletin, evaluation and officer endpoints, plus the web app. The page gets
all of its configuration (map layers, colours, languages, horizons) from `/api/meta`, so no settings live in
JavaScript. SQLite stores officer reviews, subscribers (phone numbers hashed) and an audit log.

## Configuration

| File | Holds |
|---|---|
| `config/settings.yaml` | active district, network/retry policy |
| `config/district_<name>.yaml` | district identity, bbox, periods, data sources, operational horizon and timings |
| `config/model.yaml` | variables, features, XGBoost parameters and tuning, training options |
| `config/advisory_rules.yaml` | warning thresholds, severity colours, ET0 and bulletin settings |
| `config/disease_models.yaml` | crop disease / pest weather-risk criteria |
| `config/crop_calendar_<name>.yaml` | crops per block, growth stages, Kc values |
| `config/dashboard.yaml` | app name, basemap, map layers and colour ramps, geometry simplification |
