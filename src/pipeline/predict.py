"""
Operational inference: block forecast in -> panchayat forecast + advisories + bulletins out (feature F5).

Sources of the block forecast (``source``):
* ``live``: today's ECMWF IFS 0.25° run via Open-Meteo, aggregated to blocks exactly as in training;
* ``archive``: a past issue date from ``block_forecasts.parquet`` (re-running a past date);
* ``csv``: a block-forecast table supplied by the DAMU (e.g. the official IMD GKMS block
  values). Columns: ``block_lgd`` or ``block_name``, ``lead_day`` (1-5) and ``rain``,
  ``tmax``, ``tmin``, ``rh``, ``wind``. Validated strictly by ``validate_block_csv``.

Output folder ``outputs/<d>/forecasts/<issue_date>/``:
    block_forecast.parquet   the input, normalised
    gp_forecast.parquet      one row per (GP, lead): point, P10/P50/P90, exceedance probabilities,
                             mass-conserving variant, SHAP family contributions (rain, tmax)
    advisories.json          per GP: overall severity + structured advisories
    bulletins_<lang>.json    per GP rendered bulletin (mr / hi / en)
    meta.json                model version, source, timings, data freshness
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.advisory import bulletin as bl
from src.advisory import rules
from src.advisory.crop_calendar import crops_for
from src.common.config import Config, load_config
from src.common.logging_utils import get_logger
from src.features.dataset import (
    FC_VARS,
    TIME_VARYING_INPUTS,
    add_antecedent,
    add_ndvi,
    add_season,
    add_static,
    expand_to_gps,
    forecast_context,
)
from src.models.downscaler import Downscaler, feature_family
from src.models.postprocess import block_area_weights, conserve

log = get_logger("pipeline.predict")
REQUIRED_CSV = {"lead_day", "rain", "tmax", "tmin", "rh", "wind"}


class BlockForecastError(ValueError):
    pass


def validate_block_csv(df: pd.DataFrame, cfg: Config, issue: date) -> pd.DataFrame:
    from src.ingest.boundaries import load_blocks

    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    missing = REQUIRED_CSV - set(df.columns)
    if missing:
        raise BlockForecastError(f"missing columns: {sorted(missing)}")
    blocks = load_blocks(cfg)[["block_lgd", "block_name"]]
    if "block_lgd" not in df.columns:
        if "block_name" not in df.columns:
            raise BlockForecastError("need block_lgd or block_name")
        df["block_name"] = df["block_name"].astype(str).str.strip().str.upper()
        unknown = set(df["block_name"]) - set(blocks["block_name"])
        if unknown:
            raise BlockForecastError(f"unknown block names: {sorted(unknown)}")
        df = df.merge(blocks, on="block_name")
    else:
        df["block_lgd"] = pd.to_numeric(df["block_lgd"], errors="raise").astype(int)
        unknown = set(df["block_lgd"]) - set(blocks["block_lgd"])
        if unknown:
            raise BlockForecastError(f"unknown block LGD codes: {sorted(unknown)}")
        df = df.drop(columns=[c for c in ["block_name"] if c in df]).merge(blocks, on="block_lgd")
    df["lead_day"] = pd.to_numeric(df["lead_day"], errors="raise").astype(int)
    if not df["lead_day"].isin(cfg.lead_days).all():
        raise BlockForecastError(f"lead_day must be one of {cfg.lead_days}")
    for v, lo, hi in [("rain", 0, 500), ("tmax", -5, 50), ("tmin", -10, 40), ("rh", 0, 100), ("wind", 0, 200)]:
        df[v] = pd.to_numeric(df[v], errors="raise").astype(float)
        bad = ~df[v].between(lo, hi)
        if bad.any():
            raise BlockForecastError(f"{v} outside [{lo}, {hi}] in {int(bad.sum())} rows")
    if (df["tmin"] > df["tmax"]).any():
        raise BlockForecastError("tmin greater than tmax")
    if df.duplicated(["block_lgd", "lead_day"]).any():
        raise BlockForecastError("duplicate (block, lead_day) rows")
    df["issue_date"] = pd.Timestamp(issue)
    df["valid_date"] = df["issue_date"] + pd.to_timedelta(df["lead_day"], unit="D")
    return df[["block_lgd", "block_name", "issue_date", "valid_date", "lead_day", *FC_VARS]]


def get_block_forecast(cfg: Config, issue: date, source: str, csv: Path | pd.DataFrame | None = None) -> pd.DataFrame:
    if source == "archive":
        fc = pd.read_parquet(cfg.paths.interim / "block_forecasts.parquet")
        fc = fc[fc["issue_date"] == pd.Timestamp(issue)]
        if fc.empty:
            raise BlockForecastError(f"no archived forecast issued {issue}")
        return fc
    if source == "csv":
        df = csv if isinstance(csv, pd.DataFrame) else pd.read_csv(csv)
        return validate_block_csv(df, cfg, issue)
    if source == "live":
        from src.ingest.nwp_forecast import fetch_live

        fc = fetch_live(cfg, issue)
        frames = [fc]
        # fc_rain_ante3 needs the lead-1 forecasts of the 3 previous issues; backfill any that were
        # never issued live (e.g. server was off) from the ECMWF previous-runs archive
        have = set(pd.to_datetime(_history(cfg, issue)["issue_date"]).dt.date) if not _history(cfg, issue).empty else set()
        for k in range(1, int(cfg["operational"]["history_backfill_days"]) + 1):
            prev = issue - timedelta(days=k)
            if prev not in have:
                try:
                    frames.append(fetch_live(cfg, prev))
                except Exception as exc:  # noqa: BLE001 - feature becomes NaN, model still runs
                    log.warning("could not backfill forecast issued %s: %s", prev, exc)
        hist = cfg.paths.forecasts / "block_forecast_history.parquet"
        old = pd.read_parquet(hist) if hist.exists() else pd.DataFrame()
        allfc = pd.concat([old, *frames]).drop_duplicates(["block_lgd", "issue_date", "lead_day"], keep="last")
        hist.parent.mkdir(parents=True, exist_ok=True)
        allfc.to_parquet(hist, index=False)
        return fc
    raise ValueError(source)


def _history(cfg: Config, issue: date) -> pd.DataFrame:
    """Lead-1 forecasts of the 3 previous issues (for fc_rain_ante3)."""
    frames = []
    for p in (cfg.paths.interim / "block_forecasts.parquet", cfg.paths.forecasts / "block_forecast_history.parquet"):
        if p.exists():
            f = pd.read_parquet(p)
            frames.append(f[(f["lead_day"] == 1) & (f["issue_date"] >= pd.Timestamp(issue - timedelta(days=3)))
                            & (f["issue_date"] < pd.Timestamp(issue))])
    return pd.concat(frames).drop_duplicates(["block_lgd", "issue_date", "lead_day"]) if frames else pd.DataFrame()


def _obs_rain(cfg: Config) -> pd.DataFrame:
    parts = []
    for fn in ("gp_rain_obs.parquet", "gp_rain_obs_prelim.parquet"):
        p = cfg.paths.interim / fn
        if p.exists():
            d = pd.read_parquet(p)
            d["gp_code"] = d["gp_code"].astype(str)
            parts.append(d.rename(columns={"date": "valid_date"})[["gp_code", "valid_date", "rain_obs"]])
    return pd.concat(parts).drop_duplicates(["gp_code", "valid_date"], keep="last")


def build_features(cfg: Config, fc_block: pd.DataFrame, issue: date) -> pd.DataFrame:
    static = pd.read_parquet(cfg.paths.interim / "gp_static.parquet")
    static["gp_code"] = static["gp_code"].astype(str)
    fc = pd.concat([_history(cfg, issue), fc_block]).rename(columns={v: f"fc_{v}" for v in FC_VARS})
    fc = forecast_context(fc, cfg.lead_days)
    fc = fc[fc["issue_date"] == pd.Timestamp(issue)]
    df = add_season(expand_to_gps(fc, static[["gp_code", "block_lgd"]]))
    df = add_antecedent(df, _obs_rain(cfg), carry_forward_days=int(cfg["operational"]["antecedent_carry_forward_days"]))
    df = add_ndvi(df, cfg)
    return add_static(df, static)


def run(cfg: Config | None = None, issue: date | None = None, source: str = "archive",
        csv: Path | pd.DataFrame | None = None, languages: list[str] | None = None,
        model_path: Path | None = None, write: bool = True) -> dict:
    cfg = cfg or load_config()
    t0 = time.time()
    issue = issue or cfg.today()
    languages = languages or list(cfg["operational"]["languages"])
    model_path = model_path or cfg.paths.models / "downscaler_operational.joblib"
    ds = Downscaler.load(model_path)
    fcb = get_block_forecast(cfg, issue, source, csv)
    X = build_features(cfg, fcb, issue)
    # input completeness: share of GP-days with each time-varying input actually available.
    # Missing inputs are passed to the model as missing (never filled with guesses) and reported.
    completeness = {f: round(float(X[f].notna().mean()), 3) for f in TIME_VARYING_INPUTS}
    norm_path = cfg.paths.reports / "input_coverage_by_month.json"
    if not norm_path.exists():
        raise FileNotFoundError(f"{norm_path} missing - rebuild the dataset stage")
    norm = json.loads(norm_path.read_text(encoding="utf-8"))
    month = str(issue.month)
    tol = float(cfg["operational"]["input_gap_tolerance"])
    input_gaps = {f: {"available": share, "seasonal_norm": norm[f][month]}
                  for f, share in completeness.items() if share < norm[f][month] - tol}
    for f, g in input_gaps.items():
        log.warning("input %s available for %.0f%% of GP-days (seasonal norm %.0f%%)", f,
                    100 * g["available"], 100 * g["seasonal_norm"])
    pred = ds.predict(X)
    out = pd.concat([X[["gp_code", "block_lgd", "issue_date", "valid_date", "lead_day", "fc_rain", "fc_tmax",
                        "fc_tmin", "fc_rh", "fc_wind", "ante_obs7_lag3"]], pred], axis=1)
    static = pd.read_parquet(cfg.paths.interim / "gp_static.parquet")
    static["gp_code"] = static["gp_code"].astype(str)
    w = block_area_weights(static)
    for var in ds.models:
        out[f"{var}_pred_mass"] = conserve(out, var, f"{var}_pred", f"fc_{var}", w)
    fam = feature_family(cfg.model)
    for var in [v for v in ("rain", "tmax") if v in ds.models]:
        c = ds.models[var].contributions(X, fam).add_prefix(f"shap_{var}_")
        out = pd.concat([out, c], axis=1)
    if cfg.model["postprocess"]["mass_conservation"] == "block_forecast":
        for var in ds.models:
            out[f"{var}_pred"] = out[f"{var}_pred_mass"]

    # ---- advisories + bulletins per GP ---------------------------------------------------------------
    info = static.set_index("gp_code")  # carries gp_name, block_name and all static covariates
    adv_out, bulletins = {}, {lang: {} for lang in languages}
    first_day = issue + timedelta(days=1)
    # advisories and bulletins use the validated 5-day window (leads 1-5); the extra horizon days
    # (today, days 6-7) are shown on the dashboard as forecast values only
    adv_frame = out[out["lead_day"].isin(cfg.lead_days)]
    for gp_code, g in adv_frame.groupby("gp_code"):
        r = info.loc[gp_code]
        gp = {"gp_code": gp_code, "gp_name": str(r["gp_name"]), "block_name": str(r["block_name"]),
              "latitude": float(r["latitude"]), "elev_mean": float(r["elev_mean"]), "slope_mean": float(r["slope_mean"]),
              "soil_clay_pct": float(r.get("soil_clay_pct", np.nan)), "tpi_2km": float(r["tpi_2km"])}
        crops = crops_for(gp["block_name"], first_day, cfg)
        ante = g["ante_obs7_lag3"].iloc[0]
        advs = rules.evaluate(g, gp, issue, cfg, crops, None if pd.isna(ante) else float(ante))
        adv_out[gp_code] = {"overall": rules.overall(advs, cfg), "advisories": [a.to_dict() for a in advs],
                            "crops": [c.crop_key for c in crops]}
        for lang in languages:
            bulletins[lang][gp_code] = bl.build(gp, g, advs, crops, issue, lang, cfg)

    meta = {"issue_date": issue.isoformat(), "source": source, "model": model_path.name,
            "model_meta": ds.meta, "n_gps": int(out["gp_code"].nunique()),
            "leads": sorted(int(x) for x in out["lead_day"].unique()),
            "input_completeness": completeness, "input_gaps": input_gaps,
            "validated_leads": list(cfg.lead_days),
            "obs_rain_last_date": str(_obs_rain(cfg)["valid_date"].max().date()),
            "seconds": round(time.time() - t0, 1),
            "severity_counts": pd.Series([v["overall"] for v in adv_out.values()]).value_counts().to_dict()}
    if write:
        d = cfg.paths.forecasts / issue.isoformat()
        d.mkdir(parents=True, exist_ok=True)
        fcb.to_parquet(d / "block_forecast.parquet", index=False)
        out.to_parquet(d / "gp_forecast.parquet", index=False)
        (d / "advisories.json").write_text(json.dumps(adv_out, ensure_ascii=False, default=float), encoding="utf-8")
        for lang, bb in bulletins.items():
            (d / f"bulletins_{lang}.json").write_text(json.dumps(bb, ensure_ascii=False, default=float), encoding="utf-8")
        (d / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        from src.pipeline.publish import publish_issue

        publish_issue(cfg, issue)
    log.info("issue %s (%s): %d GPs in %.1fs; severity %s", issue, source, meta["n_gps"], meta["seconds"],
             meta["severity_counts"])
    return {"meta": meta, "gp_forecast": out, "advisories": adv_out, "bulletins": bulletins}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--issue", type=str, default=None, help="YYYY-MM-DD (default today)")
    ap.add_argument("--source", choices=["live", "archive", "csv"], default="live")
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--model", type=Path)
    a = ap.parse_args()
    run(issue=date.fromisoformat(a.issue) if a.issue else None, source=a.source, csv=a.csv, model_path=a.model)
