"""
Evaluation: the single source of truth for every reported number (audit 1.7-1.9, feature F11).

Reads ``outputs/<d>/models/test_predictions_<var>.parquet`` (written by ``train.py``) and
produces ``outputs/<d>/reports/evaluation.json`` plus CSV tables. The README, API and
dashboard read their numbers from this file; nothing is hard-coded anywhere else.

Contents per variable
---------------------
* methods: the downscaler, its mass-conserving variant and all baselines
* subsets: all test rows | **unseen GPs** (spatial hold-out, headline) | seen GPs
* continuous scores overall, by lead day, by season, by block, by observed IMD rain class
* **pooled** and **sample-weighted** summaries; moving-block **bootstrap 95 % CI** of the
  headline skill score
* rain: POD/FAR/CSI/ETS/frequency bias/HSS at 2.5/15.6/64.5 mm (NaN when there are no events),
  Brier score / BSS and reliability of the exceedance probabilities
* all variables: P10-P90 interval coverage and pinball loss

Also: **leave-one-block-out CV** (``lobo``), the **independent IMD gauge-grid check**, and
the **truth QA** (CHIRPS / ERA5-Land vs IMD gauges).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.common.config import Config, load_config
from src.common.logging_utils import get_logger
from src.models import metrics as M

log = get_logger("models.evaluate")

SEASONS = {"winter (DJF)": [12, 1, 2], "pre-monsoon (MAM)": [3, 4, 5], "monsoon (JJAS)": [6, 7, 8, 9],
           "post-monsoon (ON)": [10, 11]}
PRETTY = {"pred": "Downscaler (XGBoost residual)", "pred_mass": "Downscaler, mass-conserving",
          "bl_block_copy": "Naive block copy", "bl_block_bias_corrected": "Block bias-corrected (no downscaling)",
          "bl_climatology_ratio": "Climatology ratio / lapse rate", "bl_idw_blocks": "IDW of block forecasts",
          "bl_linear_mos": "Linear MOS (ridge)", "bl_nwp_grid_reference": "NWP 0.25° grid (upper reference)"}


def _methods(df: pd.DataFrame, var: str) -> dict[str, str]:
    cols = {"pred": f"{var}_pred", "pred_mass": f"{var}_pred_mass"}
    for c in df.columns:
        if c.startswith("bl_"):
            cols[c] = c
    return {k: v for k, v in cols.items() if v in df.columns}


def _cont_table(df: pd.DataFrame, y: str, methods: dict[str, str]) -> dict:
    out = {k: M.continuous(df[y], df[c]) for k, c in methods.items()}
    ref = out["bl_block_copy"]["rmse"]
    ref2 = out.get("bl_block_bias_corrected", {}).get("rmse", np.nan)
    for k in out:
        out[k]["skill_vs_block_copy"] = M.skill(out[k]["rmse"], ref)
        out[k]["skill_vs_bias_corrected_block"] = M.skill(out[k]["rmse"], ref2)
    return out


def _grouped(df: pd.DataFrame, y: str, methods: dict[str, str], by: str | pd.Series) -> dict:
    res = {}
    for key, g in df.groupby(by):
        if len(g) < 50:
            continue
        res[str(key)] = {k: M.continuous(g[y], g[c]) for k, c in methods.items() if k in
                         ("pred", "pred_mass", "bl_block_copy", "bl_block_bias_corrected", "bl_linear_mos")}
        base = res[str(key)]["bl_block_copy"]["rmse"]
        for k in res[str(key)]:
            res[str(key)][k]["skill_vs_block_copy"] = M.skill(res[str(key)][k]["rmse"], base)
    return res


def evaluate_variable(cfg: Config, var: str) -> dict:
    path = cfg.paths.models / f"test_predictions_{var}.parquet"
    df = pd.read_parquet(path)
    spec = cfg.model["variables"][var]
    y = spec["target"]
    methods = _methods(df, var)
    season = df["month"].map({m: s for s, ms in SEASONS.items() for m in ms})
    subsets = {"all": df, "unseen_gp": df[df["unseen_gp"]], "seen_gp": df[~df["unseen_gp"]]}
    res: dict = {"target": y, "n": int(len(df)), "period": [str(df["valid_date"].min().date()),
                                                            str(df["valid_date"].max().date())],
                 "methods": {k: PRETTY.get(k, k) for k in methods}}
    for name, d in subsets.items():
        if len(d) == 0:
            continue
        res[name] = {"overall": _cont_table(d, y, methods),
                     "by_lead": _grouped(d, y, methods, "lead_day"),
                     "by_season": _grouped(d, y, methods, season.loc[d.index])}
    u = subsets["unseen_gp"]
    res["headline"] = {
        "subset": "unseen GPs (spatial hold-out) x test period (temporal hold-out)",
        "vs_block_copy": M.bootstrap_skill(u, y, f"{var}_pred", "bl_block_copy",
                                           n_boot=int(cfg["validation"]["bootstrap_samples"])),
        "vs_bias_corrected_block": M.bootstrap_skill(u, y, f"{var}_pred", "bl_block_bias_corrected",
                                                     n_boot=int(cfg["validation"]["bootstrap_samples"])),
    }
    # fold-style summaries: pooled vs sample-weighted vs unweighted mean over blocks
    per_block = _grouped(u, y, methods, "block_lgd")
    res["unseen_gp_by_block"] = per_block
    sk = pd.DataFrame({b: {"skill": v["pred"]["skill_vs_block_copy"], "n": v["pred"]["n"]}
                       for b, v in per_block.items()}).T
    if len(sk):
        res["headline"]["block_mean_skill_unweighted"] = float(sk["skill"].mean())
        res["headline"]["block_mean_skill_sample_weighted"] = float(np.average(sk["skill"], weights=sk["n"]))

    # probabilistic / interval scores
    q_lo, q_hi = f"{var}_q10", f"{var}_q90"
    if q_lo in df:
        res["intervals"] = {n: {"coverage_p10_p90": M.coverage(d[y], d[q_lo], d[q_hi]),
                                "nominal": 0.8,
                                "pinball_q10": M.pinball(d[y], d[q_lo], 0.1),
                                "pinball_q50": M.pinball(d[y], d[f"{var}_q50"], 0.5),
                                "pinball_q90": M.pinball(d[y], d[q_hi], 0.9),
                                "mean_width": float((d[q_hi] - d[q_lo]).mean())}
                            for n, d in subsets.items() if len(d)}
    if var == "rain":
        res["categorical"] = {}
        for n, d in subsets.items():
            if not len(d):
                continue
            res["categorical"][n] = {str(t): {k: M.categorical(d[y], d[c], t) for k, c in methods.items()}
                                     for t in M.IMD_THRESHOLDS}
        res["probabilistic"] = {}
        for t in M.IMD_THRESHOLDS:
            col = f"rain_p_ge_{str(t).replace('.', 'p')}"
            if col in df:
                ev = (u[y] >= t).astype(float)
                res["probabilistic"][str(t)] = {**M.brier(ev, u[col]),
                                                "reliability": M.reliability(ev, u[col]).to_dict("records")}
        cls = pd.Series(M.rain_class(u[y].to_numpy()), index=u.index)
        res["by_observed_rain_class"] = _grouped(u, y, methods, cls)
        res["by_month_unseen"] = _grouped(u, y, methods, "month")
    return res


# ---------------------------------------------------------------------------------------------
def truth_qa(cfg: Config) -> dict:
    """Agreement of the GP-scale truth sources with IMD gauge-based gridded rainfall (block level)."""
    it = cfg.paths.interim
    out: dict = {}
    imd_p = it / "block_rain_imd.parquet"
    if not imd_p.exists():
        return {"note": "IMD gridded data unavailable"}
    imd = pd.read_parquet(imd_p)
    cands = {"CHIRPS v2.0 (label)": ("block_rain_obs.parquet", "rain_obs"),
             "ERA5 precipitation (0.25°)": ("block_met_obs.parquet", "era5_rain"),
             "GPM IMERG Final V07 (0.1°)": ("block_rain_imerg.parquet", "rain_imerg")}
    fc = it / "block_forecasts.parquet"
    for name, (fn, col) in cands.items():
        if not (it / fn).exists():
            continue
        s = pd.read_parquet(it / fn)[["block_lgd", "date", col]]
        res = {}
        for lag in (0, 1):
            x = s.copy()
            x["date"] = x["date"] + pd.Timedelta(days=lag)
            m = imd.merge(x, on=["block_lgd", "date"]).dropna()
            c = M.continuous(m["rain_imd"], m[col])
            c.update({"csi_2.5": M.categorical(m["rain_imd"], m[col], 2.5)["csi"],
                      "pentad_r": float(m.groupby(["block_lgd", m["date"].dt.to_period("W")])[["rain_imd", col]]
                                        .sum().corr().iloc[0, 1])})
            res[f"lag_{lag}d"] = c
        out[name] = res
    if fc.exists():
        f = pd.read_parquet(fc)
        f1 = f[f["lead_day"] == 1][["block_lgd", "valid_date", "rain"]].rename(columns={"valid_date": "date"})
        f1["date"] = f1["date"] + pd.Timedelta(days=1)
        m = imd.merge(f1, on=["block_lgd", "date"]).dropna()
        out["ECMWF lead-1 block forecast"] = {"lag_1d": M.continuous(m["rain_imd"], m["rain"])}
    out["note"] = ("IMD date D = 24 h ending 08:30 IST on D, so it mostly covers IST calendar day D-1: "
                   "lag_1d compares source day D with IMD date D+1.")
    return out


def imd_independent_check(cfg: Config) -> dict:
    """Aggregate GP rain predictions to IMD 0.25° cells and score against gauge-only IMD rainfall."""
    it = cfg.paths.interim
    cells_p = it / "imd_cells_rain.parquet"
    pred_p = cfg.paths.models / "test_predictions_rain.parquet"
    if not cells_p.exists() or not pred_p.exists():
        return {"note": "missing inputs"}
    from src.common.geo import Grid, area_weights
    from src.ingest.boundaries import load_panchayats

    imd = pd.read_parquet(cells_p)
    pr = pd.read_parquet(pred_p)
    imd = imd[imd["date"].isin(pr["valid_date"] + pd.Timedelta(days=1))]
    if imd.empty:
        return {"note": "IMD gridded rainfall not yet published for the test period",
                "imd_last_date": str(pd.read_parquet(cells_p)["date"].max().date())}
    lats, lons = np.sort(imd["lat"].unique()), np.sort(imd["lon"].unique())
    gps = load_panchayats(cfg, modelled_only=True)
    w = area_weights(gps, "gp_code", Grid(lats, lons, 0.25), cfg.metric_crs)
    # weight of each GP piece within its cell (cell covered area)
    w = w.merge(gps[["gp_code", "area_km2"]], on="gp_code")
    w["a"] = w["weight"] * w["area_km2"]
    w["wc"] = w["a"] / w.groupby(["lat", "lon"])["a"].transform("sum")
    cell_km2 = (0.25 * 111.2) ** 2 * np.cos(np.radians(w["lat"].mean()))
    cover = w.groupby(["lat", "lon"])["a"].sum() / cell_km2
    good = cover[cover > 0.6].index
    w = w.set_index(["lat", "lon"]).loc[lambda x: x.index.isin(good)].reset_index()
    cols = ["rain_pred", "bl_block_copy", "bl_block_bias_corrected", "rain_obs"]
    m = pr[["gp_code", "valid_date", "lead_day"] + cols].merge(w[["gp_code", "lat", "lon", "wc"]], on="gp_code")
    for c in cols:
        m[c] = m[c] * m["wc"]
    agg = m.groupby(["lat", "lon", "valid_date", "lead_day"])[cols].sum().reset_index()
    agg["date"] = agg["valid_date"] + pd.Timedelta(days=1)
    j = agg.merge(imd, on=["lat", "lon", "date"]).dropna()
    res = {c: M.continuous(j["rain_imd"], j[c]) for c in cols}
    for c in cols[:-1]:
        res[c]["skill_vs_block_copy"] = M.skill(res[c]["rmse"], res["bl_block_copy"]["rmse"])
    res["n_cell_days"] = int(len(j))
    res["cells"] = int(len(good))
    res["note"] = "Area-weighted to IMD 0.25° cells with >60 % modelled-GP coverage; IMD date shifted +1 day."
    return res


def station_check(cfg: Config) -> dict:
    """Score GP forecasts at the GPs containing real stations (if the DAMU supplied station CSVs)."""
    from src.ingest import stations

    st = stations.load(cfg)
    if st is None:
        return {"note": "no station CSVs in data/<district>/raw/stations/ - see docs/stations_template.csv"}
    pr = pd.read_parquet(cfg.paths.models / "test_predictions_rain.parquet",
                         columns=["gp_code", "valid_date", "lead_day", "rain_pred", "bl_block_copy"])
    m = st[["station_id", "gp_code", "valid_date", "rain_mm"]].merge(pr, on=["gp_code", "valid_date"]).dropna()
    if m.empty:
        return {"note": "station records do not overlap the test period", "stations": int(st["station_id"].nunique())}
    res = {"stations": int(m["station_id"].nunique()), "station_days": int(len(m)),
           "downscaler": M.continuous(m["rain_mm"], m["rain_pred"]),
           "block_copy": M.continuous(m["rain_mm"], m["bl_block_copy"])}
    res["skill_vs_block_copy"] = M.skill(res["downscaler"]["rmse"], res["block_copy"]["rmse"])
    res["csi_2p5"] = {"downscaler": M.categorical(m["rain_mm"], m["rain_pred"], 2.5)["csi"],
                      "block_copy": M.categorical(m["rain_mm"], m["bl_block_copy"], 2.5)["csi"]}
    return res


def lobo(cfg: Config, variables: list[str] | None = None, sample_rows: int = 700_000) -> dict:
    """
    Leave-one-BLOCK-out CV (honest name for the old "LOSOCV"): for each block, train on all
    other blocks' GPs (training period), predict every GP of the held-out block over the test
    period. The held-out block's hist_bias comes from spatial interpolation only.
    """
    from src.features.dataset import load_frame
    from src.models.downscaler import Downscaler, VariableDownscaler

    ds = Downscaler.load(cfg.paths.models / "downscaler_eval.joblib")
    df = load_frame(cfg)
    v = cfg["validation"]
    test0 = pd.Timestamp(v["test_start"])
    from src.models.train import calibration_days

    cal_days, buf_days = calibration_days(df["valid_date"], test0)
    is_cal = df["valid_date"].isin(cal_days)
    is_buf = df["valid_date"].isin(buf_days)
    variables = variables or list(ds.models)
    out: dict = {}
    for var in variables:
        em = ds.models[var]
        rows = []
        for blk, _ in df.groupby("block_lgd"):
            tr = df[(df["block_lgd"] != blk) & (df["valid_date"] < test0) & ~is_cal & ~is_buf & df[em.target].notna()]
            ca = df[(df["block_lgd"] != blk) & is_cal]
            te = df[(df["block_lgd"] == blk) & (df["valid_date"] >= test0) & df[em.target].notna()
                    & df[em.forecast].notna()]
            if te["gp_code"].nunique() < 5:
                continue
            m = VariableDownscaler(var=var, target=em.target, forecast=em.forecast, features=em.features,
                                   params=em.params, quantiles=em.quantiles, thresholds=[],
                                   residual_space=em.residual_space, lower=em.lower, upper=em.upper,
                                   conformal=False)
            m.fit(tr.sample(min(len(tr), sample_rows), random_state=0), ca)
            p = m.predict(te, with_uncertainty=False)[f"{var}_pred"]
            mod = M.continuous(te[em.target], p)
            base = M.continuous(te[em.target], te[em.forecast])
            rows.append({"block_lgd": int(blk), "n_gps": int(te["gp_code"].nunique()), "n": mod["n"],
                         "model_rmse": mod["rmse"], "block_copy_rmse": base["rmse"],
                         "model_mae": mod["mae"], "block_copy_mae": base["mae"],
                         "skill": M.skill(mod["rmse"], base["rmse"]),
                         "sse_model": mod["rmse"] ** 2 * mod["n"], "sse_base": base["rmse"] ** 2 * base["n"]})
            log.info("LOBO %s block %s: skill %.3f", var, blk, rows[-1]["skill"])
        t = pd.DataFrame(rows)
        out[var] = {
            "folds": t.drop(columns=["sse_model", "sse_base"]).to_dict("records"),
            "pooled_skill": M.skill(np.sqrt(t["sse_model"].sum() / t["n"].sum()),
                                    np.sqrt(t["sse_base"].sum() / t["n"].sum())),
            "mean_skill_unweighted": float(t["skill"].mean()),
            "mean_skill_sample_weighted": float(np.average(t["skill"], weights=t["n"])),
            "folds_with_positive_skill": int((t["skill"] > 0).sum()), "n_folds": int(len(t)),
        }
    return out


def run(cfg: Config | None = None, with_lobo: bool = True) -> dict:
    cfg = cfg or load_config()
    rep = cfg.paths.reports
    rep.mkdir(parents=True, exist_ok=True)
    ev: dict = {"district": cfg.district_name, "generated_utc": pd.Timestamp.utcnow().isoformat(),
                "validation_design": {
                    "temporal": f"test = valid dates >= {cfg['validation']['test_start']}; before that every 5th "
                                "week is calibration (2-day buffers), the rest training",
                    "spatial": f"{int(cfg['validation']['spatial_holdout_fraction'] * 100)} % of GPs per block never "
                               "used in training or calibration",
                    "headline_subset": "unseen GPs x test period"},
                "variables": {}}
    for var in cfg.model["variables"]:
        if (cfg.paths.models / f"test_predictions_{var}.parquet").exists():
            log.info("evaluating %s", var)
            ev["variables"][var] = evaluate_variable(cfg, var)
    ev["truth_qa"] = truth_qa(cfg)
    ev["imd_independent_check"] = imd_independent_check(cfg)
    ev["station_check"] = station_check(cfg)
    if with_lobo:
        ev["leave_one_block_out"] = lobo(cfg)
    elif (rep / "evaluation.json").exists():
        old = json.loads((rep / "evaluation.json").read_text(encoding="utf-8"))
        if "leave_one_block_out" in old:
            ev["leave_one_block_out"] = old["leave_one_block_out"]
    dl = rep / "dl_ablation.json"
    if dl.exists():
        ev["dl_ablation"] = json.loads(dl.read_text(encoding="utf-8"))
    pp = rep / "perfect_prognosis.json"
    if pp.exists():
        ev["perfect_prognosis"] = json.loads(pp.read_text(encoding="utf-8"))
    ts = cfg.paths.models / "training_summary.json"
    if ts.exists():
        ev["training_summary"] = json.loads(ts.read_text(encoding="utf-8"))
    ev["headline"] = headline(ev)
    (rep / "evaluation.json").write_text(json.dumps(sanitize(ev), indent=2, default=_json_default, allow_nan=False),
                                         encoding="utf-8")
    log.info("headline: %s", ev["headline"]["text"])
    return ev


def headline(ev: dict) -> dict:
    r = ev["variables"].get("rain", {})
    h = r.get("headline", {})
    vb = h.get("vs_block_copy", {})
    vbc = h.get("vs_bias_corrected_block", {})
    if not vb:
        return {"text": "evaluation incomplete"}
    txt = (f"Rainfall: {vb['skill'] * 100:.1f}% RMSE reduction vs naive block copy "
           f"(95% CI {vb['ci_low'] * 100:.1f} to {vb['ci_high'] * 100:.1f}%) and {vbc.get('skill', np.nan) * 100:.1f}% vs a "
           f"bias-corrected block forecast, on panchayats never seen in training over "
           f"{r['period'][0]} to {r['period'][1]} (lead days 1-5).")
    per_var = {v: {"skill_vs_block_copy": d["headline"]["vs_block_copy"]["skill"],
                   "ci": [d["headline"]["vs_block_copy"]["ci_low"], d["headline"]["vs_block_copy"]["ci_high"]],
                   "rmse_model": d["unseen_gp"]["overall"]["pred"]["rmse"],
                   "rmse_block_copy": d["unseen_gp"]["overall"]["bl_block_copy"]["rmse"],
                   "skill_vs_bias_corrected_block": d["headline"]["vs_bias_corrected_block"]["skill"]}
               for v, d in ev["variables"].items()}
    pp = ev.get("perfect_prognosis", {}).get("variables", {})
    pp_sum = {v: {"skill_vs_block_copy": d["overall"]["downscaler"]["skill_vs_block_copy"],
                  "mass_conserving_skill": d["overall"]["downscaler_mass_conserving"]["skill_vs_block_copy"],
                  "ci": [d["bootstrap_vs_block_copy"]["ci_low"], d["bootstrap_vs_block_copy"]["ci_high"]],
                  "within_block_anomaly_r": d["within_block_anomaly_correlation"]} for v, d in pp.items()}
    if "rain" in pp_sum:
        r0 = pp_sum["rain"]
        txt = (f"Disaggregation (perfect-prognosis): given only the block value, the model reproduces "
               f"panchayat rainfall with {r0['skill_vs_block_copy'] * 100:.1f}% lower RMSE than copying the block value "
               f"(95% CI {r0['ci'][0] * 100:.1f} to {r0['ci'][1] * 100:.1f}%) on never-seen panchayats. "
               + "Forecast mode (ECMWF lead 1-5 input): " + txt)
    return {"text": txt, "per_variable": per_var, "perfect_prognosis": pp_sum}


def sanitize(o):
    """Recursively convert NaN/inf to None so the file is strict JSON (browsers reject NaN)."""
    if isinstance(o, dict):
        return {str(k): sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [sanitize(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    return o


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    return str(o)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-lobo", action="store_true")
    run(with_lobo=not ap.parse_args().no_lobo)
