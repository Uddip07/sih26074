"""
Training and held-out prediction (audit 1.6, 4.x; features F3, F4, F11-F17).

Split design (no leakage in space *or* time)
--------------------------------------------
* **Temporal:** test = valid dates on/after ``validation.test_start``, never used for any fitting
  decision. Before that, every 5th week is a *calibration* week (early stopping, conformal
  widening, isotonic calibration) with 2-day buffers removed from training on both sides;
  the rest is training. Blocked weeks sample every season (a single end-of-period slice was
  mostly dry season and stopped the boosting after ~40 trees).
* **Spatial:** ``spatial_holdout_fraction`` of GPs (stratified by block) are removed
  from train and calibration. Their test rows are the strictest evaluation:
  **unseen place and unseen time**, exactly the operational case of an ungauged panchayat.

Two models are produced
-----------------------
* ``downscaler_eval.joblib``: fitted on the training split only; it produced every
  reported number.
* ``downscaler_operational.joblib``: refitted with the same hyper-parameters on train
  + test periods for **all** GPs, with the calibration period again held out for early
  stopping / conformal / isotonic. Used by the live forecast pipeline.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from src.common.config import Config, load_config
from src.common.logging_utils import get_logger
from src.features.dataset import load_frame
from src.models.baselines import Baselines
from src.models.downscaler import Downscaler, VariableDownscaler, feature_family, resolve_features
from src.models.postprocess import block_area_weights, conservation_report, conserve
from src.models.tuning import tune

log = get_logger("models.train")
KEYS = ["gp_code", "block_lgd", "issue_date", "valid_date", "lead_day"]


def holdout_gps(static: pd.DataFrame, frac: float, seed: int) -> set[str]:
    rng = np.random.default_rng(seed)
    out: set[str] = set()
    for _, g in static.groupby("block_lgd"):
        codes = g["gp_code"].to_numpy()
        k = int(round(len(codes) * frac))
        if len(codes) >= 5 and k > 0:
            out |= set(rng.choice(codes, size=k, replace=False))
    return out


def calibration_days(dates: pd.Series, test0: pd.Timestamp, every: int = 5, gap: int = 2) -> tuple[set, set]:
    """
    Blocked calibration split inside the pre-test period: every ``every``-th ISO week is a
    calibration week; ``gap`` days on each side are dropped from training (weather autocorrelation
    buffer). Unlike a single end-of-period slice (which was mostly dry season), this samples every
    season, so early stopping, conformal widening and isotonic calibration see monsoon rain.
    Returns (calibration_days, buffer_days).
    """
    days = pd.DatetimeIndex(sorted(dates[dates < test0].unique()))
    week = ((days - days[0]).days // 7).to_numpy()
    cal = days[week % every == every // 2]
    buf = set()
    for d in cal:
        for k in range(1, gap + 1):
            buf |= {d - pd.Timedelta(days=k), d + pd.Timedelta(days=k)}
    cal_set = set(cal)
    return cal_set, buf - cal_set


def split_masks(df: pd.DataFrame, cfg: Config, hold: set[str]) -> dict[str, np.ndarray]:
    v = cfg["validation"]
    d = df["valid_date"]
    is_hold = df["gp_code"].isin(hold).to_numpy()
    test0 = pd.Timestamp(v["test_start"])
    cal_days, buf_days = calibration_days(d, test0)
    in_cal = d.isin(cal_days).to_numpy()
    in_buf = d.isin(buf_days).to_numpy()
    pre = (d < test0).to_numpy()
    return {
        "train": pre & ~in_cal & ~in_buf & ~is_hold,
        "calib": pre & in_cal & ~is_hold,
        "test": ~pre,
        "holdout_gp": is_hold,
    }


def _sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    return df if len(df) <= n else df.sample(n, random_state=seed)


def block_points(cfg: Config) -> pd.DataFrame:
    from src.ingest.boundaries import load_blocks

    b = load_blocks(cfg)
    rp = b.to_crs(cfg.metric_crs).geometry.representative_point().to_crs("EPSG:4326")
    return pd.DataFrame({"block_lgd": b["block_lgd"], "latitude": rp.y.to_numpy(), "longitude": rp.x.to_numpy()})


def block_fc_table(cfg: Config, var: str) -> pd.DataFrame:
    fc = pd.read_parquet(cfg.paths.interim / "block_forecasts.parquet")
    return fc.pivot_table(index=["valid_date", "lead_day"], columns="block_lgd", values=var)


def run(cfg: Config | None = None, variables: list[str] | None = None, do_tune: bool | None = None,
        operational: bool = True) -> dict:
    cfg = cfg or load_config()
    paths = cfg.paths.ensure()
    mcfg = cfg.model
    t0 = time.time()
    df = load_frame(cfg)
    static = pd.read_parquet(paths.interim / "gp_static.parquet")
    static["gp_code"] = static["gp_code"].astype(str)
    seed = int(cfg["validation"]["random_seed"])
    hold = holdout_gps(static, float(cfg["validation"]["spatial_holdout_fraction"]), seed)
    masks = split_masks(df, cfg, hold)
    (paths.models).mkdir(parents=True, exist_ok=True)
    (paths.models / "spatial_holdout_gps.json").write_text(json.dumps(sorted(hold)), encoding="utf-8")
    log.info("rows: train %d | calib %d | test %d (of which unseen-GP %d); holdout GPs %d",
             masks["train"].sum(), masks["calib"].sum(), masks["test"].sum(),
             (masks["test"] & masks["holdout_gp"]).sum(), len(hold))

    weights = block_area_weights(static)
    bpts = block_points(cfg)
    nwp_ref = None
    ref_path = paths.interim / "gp_nwp_grid_reference.parquet"
    if ref_path.exists():
        nwp_ref = pd.read_parquet(ref_path)
        nwp_ref["gp_code"] = nwp_ref["gp_code"].astype(str)

    features = resolve_features(mcfg, list(df.columns))
    fam = feature_family(mcfg)
    log.info("features (%d): %s", len(features), features)
    variables = variables or [v for v, s in mcfg["variables"].items() if s["target"] in df.columns]
    do_tune = mcfg["xgboost"]["tuning"]["enabled"] if do_tune is None else do_tune
    xcfg = mcfg["xgboost"]
    tuning_log: dict = {}
    eval_models: dict[str, VariableDownscaler] = {}
    summary: dict = {"variables": {}, "features": features, "holdout_gps": len(hold)}

    for var in variables:
        spec = mcfg["variables"][var]
        tgt, fcol = spec["target"], spec["forecast"]
        tr = df[masks["train"]]
        ca = df[masks["calib"]]
        te = df[masks["test"]]
        params = dict(xcfg["base"])
        space = spec.get("residual_space", "linear")
        if do_tune:
            t_space = "log1p" if space == "auto" else space
            res = tune(tr, var, tgt, fcol, features, t_space, params, xcfg["search_space"],
                       int(xcfg["tuning"]["n_trials"]), int(xcfg["tuning"]["cv_folds"]),
                       int(xcfg["tuning"]["sample_rows"]), seed, spec.get("lower"), spec.get("upper"))
            params = res["params"]
            tuning_log[var] = res

        model = VariableDownscaler(
            var=var, target=tgt, forecast=fcol, features=features, params=params,
            quantiles=list(xcfg["quantiles"]), thresholds=list(spec.get("exceedance_thresholds", [])),
            residual_space=space, lower=spec.get("lower"), upper=spec.get("upper"),
            conformal=bool(mcfg["training"]["conformal"]),
            early_stopping_rounds=int(mcfg["training"]["early_stopping_rounds"]))
        cap = int(mcfg["training"]["sample_rows_per_variable"])
        model.fit(_sample(tr[tr[tgt].notna()], cap, seed), ca)
        eval_models[var] = model

        # ---- held-out predictions + baselines ------------------------------------------------
        te_ok = te[te[tgt].notna() & te[fcol].notna()]
        pred = model.predict(te_ok)
        bl = Baselines(var, tgt, fcol).fit(_sample(tr, 1_500_000, seed), bpts)
        ref = None
        if nwp_ref is not None:
            m = te_ok[["gp_code", "valid_date", "lead_day"]].merge(
                nwp_ref[["gp_code", "valid_date", "lead_day", f"nwpgrid_{var}"]],
                on=["gp_code", "valid_date", "lead_day"], how="left")
            ref = pd.Series(m[f"nwpgrid_{var}"].to_numpy(), index=te_ok.index)
        base = bl.predict(te_ok, block_fc_table(cfg, var), ref)
        out = te_ok[KEYS + [tgt, fcol, "is_monsoon", "month"]].copy()
        out["unseen_gp"] = out["gp_code"].isin(hold)
        out = pd.concat([out, pred, base.add_prefix("bl_")], axis=1)
        out[f"{var}_pred_mass"] = conserve(out, var, f"{var}_pred", fcol, weights)
        out.to_parquet(paths.models / f"test_predictions_{var}.parquet", index=False)

        # ---- explainability on a test sample ----------------------------------------------------
        smp = _sample(te_ok, 60_000, seed)
        contrib = model.contributions(smp, fam)
        fam_imp = contrib.drop(columns="baseline").abs().mean().sort_values(ascending=False)
        summary["variables"][var] = {
            "residual_space": model.residual_space,
            "space_selection_calib_rmse": model.space_selection_,
            "best_iteration": model.best_iteration_,
            "conformal_margin": model.conformal_margin_,
            "params": {k: v for k, v in params.items() if k in xcfg["search_space"]},
            "feature_importance_gain": model.feature_importance().head(25).round(5).to_dict(),
            "family_mean_abs_shap": fam_imp.round(5).to_dict(),
            "mass_conservation": conservation_report(out, f"{var}_pred", fcol, weights),
            "n_train": int(masks["train"].sum()), "n_test": int(len(out)),
        }
        log.info("[%s] done (%.1f min)", var, (time.time() - t0) / 60)

    meta = {"district": cfg.key, "features": features, "variables": variables,
            "period": [cfg["period"]["start"], cfg["period"]["end"]], "validation": cfg["validation"],
            "trained_utc": pd.Timestamp.utcnow().isoformat(), "kind": "evaluation"}
    Downscaler(eval_models, meta).save(paths.models / "downscaler_eval.joblib")
    (paths.models / "tuning.json").write_text(json.dumps(tuning_log, indent=2, default=float), encoding="utf-8")
    (paths.models / "training_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")

    if operational:
        op_models = {}
        cal_days, buf_days = calibration_days(df["valid_date"], pd.Timestamp(cfg["validation"]["test_start"]))
        in_cal = df["valid_date"].isin(cal_days).to_numpy()
        all_rows = ~in_cal & ~df["valid_date"].isin(buf_days).to_numpy()
        ca_all = df[in_cal]
        for var, em in eval_models.items():
            spec = mcfg["variables"][var]
            m = VariableDownscaler(var=var, target=em.target, forecast=em.forecast, features=features,
                                   params=em.params, quantiles=em.quantiles, thresholds=em.thresholds,
                                   residual_space=em.residual_space, lower=spec.get("lower"),
                                   upper=spec.get("upper"), conformal=em.conformal,
                                   early_stopping_rounds=em.early_stopping_rounds)
            d = df[all_rows]
            m.fit(_sample(d[d[em.target].notna()], int(mcfg["training"]["sample_rows_per_variable"]), seed), ca_all)
            op_models[var] = m
        meta_op = dict(meta, kind="operational", trained_on="all GPs, all dates except calibration slice")
        Downscaler(op_models, meta_op).save(paths.models / "downscaler_operational.joblib")
    log.info("training complete in %.1f min", (time.time() - t0) / 60)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vars", nargs="*")
    ap.add_argument("--no-tune", action="store_true")
    ap.add_argument("--no-operational", action="store_true")
    a = ap.parse_args()
    run(variables=a.vars, do_tune=False if a.no_tune else None, operational=not a.no_operational)
