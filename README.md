# Block → Gram Panchayat Weather Downscaling for Agromet Advisories

**SIH 2026 problem statement:** *Downscaling of weather forecast from Block level to Panchayat level: inferring
high-resolution plots/data/information from low-resolution plot/data/information/variables for
agro-meteorological advisory services.*

**Pilot:** Pune district, Maharashtra: 14 blocks, **1,349 modelled Gram Panchayats** (LGD), from the Western Ghats
crest (>1,400 m, 3,000+ mm/yr) to the Deccan rain-shadow (~500 m, <600 mm/yr).

The system takes the **block-level 5-day forecast** (one value per block, as in GKMS bulletins) and produces, for
every Gram Panchayat:

* rainfall, Tmax, Tmin, relative humidity and wind for days 1-5, with **P10/P90 ranges** and **rain probabilities**
  (≥ 2.5 / 15.6 / 64.5 mm);
* an **explanation** of why the panchayat differs from its block (terrain, climatology, land cover, ... via TreeSHAP);
* **crop- and stage-aware agromet advisories** (IMD colour-coded) and a **GKMS-style bulletin in Marathi, Hindi and
  English** (PDF, SMS ≤ 160 characters, WhatsApp, IVR script).

---

## Results

All numbers below are generated from `outputs/pune/reports/evaluation.json` by the pipeline; none are typed by
hand. The full report with figures is `outputs/pune/reports/report.html` (also served at `/api/report`).

<!-- RESULTS:START -->
_Run `python -m src.pipeline.run` to generate the results table._
<!-- RESULTS:END -->

**How to read them.** Two different questions are answered:

1. **Disaggregation (perfect prognosis):** given the *observed* block value, how well do we recover each panchayat's
   value? This is the problem statement in its purest form, with NWP forecast error removed.
2. **Operational forecast mode:** given the real ECMWF block forecast (lead 1-5 days), how close are the panchayat
   forecasts to what was observed? This includes the NWP model's own timing and intensity error, which no spatial
   method can remove, so the gains are smaller and the confidence intervals wider.

Every score is on **Gram Panchayats never seen in training (20 % spatial hold-out)** over a **test period never used
for any fitting decision (2026)**, compared with the naive block copy (spec §7.1) and five other baselines, with 95 %
bootstrap confidence intervals (moving blocks of days). A leave-one-block-out CV and an independent check against the
IMD gauge-based grid are included.

---

## Data (all real, all open; nothing simulated)

| Role | Source | Resolution |
|---|---|---|
| Block forecast (model input) | ECMWF IFS open data via Open-Meteo *Previous Runs* archive: true lead-1…5 forecasts, area-averaged to blocks | 0.25° |
| Rainfall truth | CHIRPS v2.0 daily (satellite + gauges), area-weighted to GPs | 0.05° |
| Tmax / Tmin / RH truth | ERA5-Land reanalysis | 0.1° |
| Wind, ET0, radiation truth | ERA5 reanalysis (coarser; documented limitation) | 0.25° |
| Independent rain check | IMD 0.25° gauge-gridded rainfall (never used in training) | 0.25° |
| Boundaries | LGD blocks and Gram Panchayats (urbanmorph/geodata), cleaned and QA-mapped | polygons |
| Terrain | Copernicus DEM GLO-30: elevation stats, slope, aspect, TPI, **windward exposure**, **Ghat-crest distance**, **upwind barrier** | 30 m |
| Land cover | ESA WorldCover 2021, all 4 tiles covering the district, exact class fractions | 10 m |
| Hydrology | HydroRIVERS, Natural Earth coastline, reservoirs ≥ 1 km² from WorldCover | vector |
| Soil | SoilGrids 2.0 texture/SOC + Saxton-Rawls available water capacity | 250 m |
| Rain climatology | CHPclim v2 monthly normals (GP ÷ block ratio: leakage-free spatial prior) | 0.05° |
| Vegetation | MODIS 13Q1 NDVI (Planetary Computer), as-of the issue date | 250 m |

Provenance, licences and SHA-256 of every table: [`data/README.md`](data/README.md) (generated).

## Method

```
block forecast (ECMWF 0.25° → block mean, lead d)                  static GP covariates (terrain, land cover,
        │                                                          hydro, soil, NDVI, CHPclim ratio)
        ▼                                                                    │
  residual r = y_GP − fc_block   ◄──── XGBoost (lead-aware, all 5 variables' forecasts as inputs) ◄──┘
        │                               + leakage-free historical bias (out-of-fold, IDW for unseen GPs)
        ▼
  point forecast, P10/P50/P90 (multi-quantile + split-conformal), P(rain ≥ t) (isotonic-calibrated)
        ▼
  optional mass conservation (area-weighted GP mean = block value)  →  TreeSHAP explanation
        ▼
  rule engine (YAML thresholds, crop calendar, ET0, disease models)  →  bulletins mr/hi/en, SMS, PDF
```

* **Residual learning** (spec §7.2). Rain is modelled in linear or log1p space, chosen on calibration data.
* **Hyper-parameters** tuned with Optuna over the spec §7.2 grid, with GroupKFold by Gram Panchayat.
* **No leakage:** the historical-bias feature is out-of-fold in time and spatially interpolated for unseen GPs;
  antecedent rain uses a 3-day observation latency; NDVI is as-of the issue date; raw lat/lon are excluded.
  Tests in `tests/test_leakage_and_splits.py` enforce this.
* **Baselines:** naive block copy, bias-corrected block (no downscaling), climatology ratio / lapse rate, IDW of
  block forecasts, linear MOS, and the NWP 0.25° grid itself (upper reference).

## Advisories

`config/advisory_rules.yaml` holds the thresholds; DAMU officers edit this file, not code. `config/crop_calendar_pune.yaml`
lists 11 crops by block and stage, with FAO-56 Kc values. The rules are:

* heavy rain (IMD categories and P(≥ 64.5 mm));
* waterlogging (only on flat, clayey land);
* dry spell with an ET0 × Kc irrigation amount;
* kharif sowing window (≥ 65 mm cumulative);
* spray window and fertiliser timing;
* heat, cold wave and frost (raised one level in valleys);
* thunderstorm and lightning;
* strong wind;
* heat-and-humidity pest risk;
* harvest window;
* crop disease models: grape downy and powdery mildew, onion thrips and purple blotch, pomegranate bacterial blight,
  rice blast, potato late blight, soybean and wheat rust.

## Running it

```bash
python -m venv .venv && .venv/Scripts/activate        # Windows (source .venv/bin/activate on Linux)
pip install -r requirements.txt && pip install --no-deps -e .
python -m src.pipeline.run                             # full pipeline (resumable; skips finished stages)
uvicorn src.dashboard.app:app --port 8080              # dashboard at http://127.0.0.1:8080, API docs at /docs
pytest                                                 # tests
```

* **Today's operational forecast:** `python -m src.pipeline.run --only forecast verify --source live`
* **Downscale an official block forecast (CSV):** use the dashboard's *Officer* tab or `POST /api/downscale`.
  The template is at `/api/template/block_forecast.csv`.
* **Another district:** copy `config/district_pune.yaml`, change the LGD code and bbox, and add a crop calendar.
  Then run `python -m src.pipeline.run --config config/district_<name>.yaml`. No code changes are needed.
* **Docker:** `docker build -t agromet . && docker run -p 8080:8080 -v $PWD/data:/app/data -v $PWD/outputs:/app/outputs agromet`

The free Open-Meteo tier allows 10,000 API calls a day. A first full download for one district needs about 15,000
calls, so it takes two days. The fetchers cache per grid node and resume automatically.

## Dashboard

* **Maps:** side-by-side *block forecast (input)* vs *Gram Panchayat forecast (output)* maps, synchronised, with an
  issue-date picker and a day 1-5 slider.
* **Map layers:** rain, rain probabilities, uncertainty, panchayat − block difference, Tmax, Tmin, RH, wind and
  warning level.
* **Finding a panchayat:** search by name or LGD code, or use your GPS location.
* **Panchayat panel:** 5-day block-vs-panchayat chart with ranges, "why this panchayat differs" (SHAP), advisories,
  PDF/text/IVR bulletin in मराठी / हिन्दी / English, SMS preview, and demo registration.
* **Scorecard and Verification tabs:** all numbers are live from the evaluation files.
* **Officer tab:** approve, reject or edit bulletins, publish an issue, simulated dissemination (no message is ever
  sent), live run, CSV upload, 6-hour NWP nowcast, and the audit log.
* **Other:** installable offline PWA (last issue cached), dark mode, keyboard navigation, and warnings that are never
  shown by colour alone.

## Repository layout

```
config/            district, model, advisory rules, crop calendar (YAML)
src/common/        config, paths, geo area-weighting, HTTP (rate-limit aware), provenance manifest
src/ingest/        boundaries, nwp_forecast, chirps, era5land, imd_gridded, imerg (optional), terrain,
                   landcover, hydro, soil, ndvi
src/features/      static covariates, dataset builder (shared by training and inference), bias encoder
src/models/        downscaler, baselines, tuning, train, evaluate, perfect_prog, metrics, postprocess, report
src/advisory/      rules, crop calendar, ET0, disease models, bulletin (PDF/SMS), i18n (mr/hi/en)
src/pipeline/      run (orchestrator), predict (inference), publish, verify, nowcast, disseminate, docs
src/dashboard/     FastAPI app, SQLite workflow store, static web app (Leaflet), PWA
tests/             unit tests (fixtures) + integration tests (skip when outputs are absent)
```

## Limitations (stated plainly)

* **Rainfall truth:** CHIRPS 0.05° is the finest free daily rainfall product, but it is not a rain gauge. Its daily
  timing is weaker than its multi-day totals, and IMD gauges agree better at 2-7-day aggregation (see the truth QA in
  the report).
* **Temperature and wind truth:** these are reanalysis products. Wind truth (ERA5, 0.25°) is coarser than a GP, so
  wind "downscaling" is mostly bias correction.
* **Station validation:** the Maharashtra Mahavedh / IMD AWS station records are not openly downloadable. Drop station
  CSVs into `data/pune/raw/stations/` to add an independent station validation. The pipeline treats them as
  validation-only.
* **Block forecast source:** the model is trained on ECMWF IFS forecasts. Officers can upload official IMD block
  values instead; their error characteristics differ, so re-training on an archive of those bulletins would be
  preferable once one exists.
* **Nowcast layer:** this is short-range NWP, not radar. A radar or INSAT-3D adapter hook is provided.

## Licence

Code: see [LICENSE](LICENSE). Data products keep their own licences (listed in `data/README.md`). Fonts: Noto Sans
(SIL OFL 1.1).
