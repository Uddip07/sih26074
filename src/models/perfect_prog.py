"""
Perfect-prognosis (PP) disaggregation experiment: the purest test of the problem statement.

The PS asks for *inferring high-resolution information from low-resolution information*.
In forecast mode, most of the daily error is NWP timing/intensity error that no spatial
method can remove, and it masks the downscaling ability. PP mode removes it:

    input  = the OBSERVED block area-mean (CHIRPS rain, ERA5-Land T/RH, ERA5 wind) for day D
    target = the observed value of each Gram Panchayat on day D

The same model code, features and spatio-temporal split are used (train / blocked calibration
weeks / test from 2026, 20 % of GPs never seen). Baselines are the **naive block copy of the
observed block value** and the climatology-ratio disaggregation. The mass-conserving variant
(area-weighted GP mean == block value exactly) is pure redistribution, with no bias correction
at all.

Outputs ``pp_test_predictions_<var>.parquet`` and ``outputs/<d>/reports/perfect_prognosis.json``.
"""

from __future__ import annotations

import json

import pandas as pd

from src.common.config import Config, load_config
from src.common.logging_utils import get_logger
from src.features.dataset import (
    _truth,
    add_antecedent,
    add_ndvi,
    add_season,
    add_static,
    expand_to_gps,
    forecast_context,
)
from src.models import metrics as M
from src.models.downscaler import VariableDownscaler, resolve_features
from src.models.postprocess import block_area_weights, conserve
from src.models.train import holdout_gps, split_masks

log = get_logger("models.pp")
BLOCK_OBS = {"rain": ("block_rain_obs.parquet", "rain_obs"), "tmax": ("block_met_obs.parquet", "tmax_obs"),
             "tmin": ("block_met_obs.parquet", "tmin_obs"), "rh": ("block_met_obs.parquet", "rh_obs"),
             "wind": ("block_met_obs.parquet", "wind_obs")}


def build_frame(cfg: Config) -> pd.DataFrame:
    it = cfg.paths.interim
    static = pd.read_parquet(it / "gp_static.parquet")
    static["gp_code"] = static["gp_code"].astype(str)
    blk = None
    for var, (fn, col) in BLOCK_OBS.items():
        t = pd.read_parquet(it / fn)[["block_lgd", "date", col]].rename(columns={col: f"fc_{var}"})
        blk = t if blk is None else blk.merge(t, on=["block_lgd", "date"], how="outer")
    blk = blk.rename(columns={"date": "valid_date"})
    blk["issue_date"] = blk["valid_date"]
    blk["lead_day"] = 0
    blk = blk[blk["block_lgd"].isin(static["block_lgd"].unique())]
    fc = forecast_context(blk)  # fc_rain_ante3 is NaN in PP mode (no lead-1 forecasts)
    df = add_season(expand_to_gps(fc, static[["gp_code", "block_lgd"]]))
    truth = _truth(cfg)
    df = add_antecedent(df, truth[["gp_code", "valid_date", "rain_obs"]])
    df = df.merge(truth, on=["gp_code", "valid_date"], how="left")
    df = add_ndvi(df, cfg)
    return add_static(df, static)


def run(cfg: Config | None = None, variables: list[str] | None = None, sample_rows: int = 1_200_000) -> dict:
    cfg = cfg or load_config()
    df = build_frame(cfg)
    static = pd.read_parquet(cfg.paths.interim / "gp_static.parquet")
    static["gp_code"] = static["gp_code"].astype(str)
    seed = int(cfg["validation"]["random_seed"])
    hold = holdout_gps(static, float(cfg["validation"]["spatial_holdout_fraction"]), seed)
    masks = split_masks(df, cfg, hold)
    w = block_area_weights(static)
    feats = [f for f in resolve_features(cfg.model, list(df.columns)) if f not in ("fc_rain_ante3", "lead_day")]
    tuned = {}
    tj = cfg.paths.models / "tuning.json"
    if tj.exists():
        tuned = {k: v["params"] for k, v in json.loads(tj.read_text()).items()}
    res: dict = {"description": __doc__.strip().split("\n\n")[0], "variables": {}}
    for var in variables or list(BLOCK_OBS):
        spec = cfg.model["variables"][var]
        tgt, fcol = spec["target"], f"fc_{var}"
        params = tuned.get(var, dict(cfg.model["xgboost"]["base"]))
        m = VariableDownscaler(var=var, target=tgt, forecast=fcol, features=feats, params=params,
                               quantiles=list(cfg.model["xgboost"]["quantiles"]), thresholds=[],
                               residual_space=spec.get("residual_space", "linear"), lower=spec.get("lower"),
                               upper=spec.get("upper"))
        tr = df[masks["train"] & df[tgt].notna().to_numpy() & df[fcol].notna().to_numpy()]
        m.fit(tr.sample(min(len(tr), sample_rows), random_state=seed), df[masks["calib"]])
        te = df[masks["test"] & masks["holdout_gp"] & df[tgt].notna().to_numpy() & df[fcol].notna().to_numpy()]
        p = m.predict(te)
        out = te[["gp_code", "block_lgd", "issue_date", "valid_date", "lead_day", tgt, fcol, "month"]].copy()
        out[f"{var}_pred"] = p[f"{var}_pred"].to_numpy()
        for q in ("q10", "q90"):
            if f"{var}_{q}" in p:
                out[f"{var}_{q}"] = p[f"{var}_{q}"].to_numpy()
        out[f"{var}_pred_mass"] = conserve(out, var, f"{var}_pred", fcol, w)
        if var == "rain":
            out["clim_ratio"] = te[fcol].to_numpy() * te["clim_ratio_month"].to_numpy()
        elif var in ("tmax", "tmin"):
            out["clim_ratio"] = te[fcol].to_numpy() - 0.0065 * te["elev_diff"].to_numpy()
        else:
            out["clim_ratio"] = te[fcol].to_numpy()
        out.to_parquet(cfg.paths.models / f"pp_test_predictions_{var}.parquet", index=False)
        y = out[tgt]
        methods = {"downscaler": f"{var}_pred", "downscaler_mass_conserving": f"{var}_pred_mass",
                   "climatology_ratio": "clim_ratio", "block_copy": fcol}
        cont = {k: M.continuous(y, out[c]) for k, c in methods.items()}
        for k in cont:
            cont[k]["skill_vs_block_copy"] = M.skill(cont[k]["rmse"], cont["block_copy"]["rmse"])
        wet = out[out[fcol] >= 2.5] if var == "rain" else out
        cont_wet = {k: M.continuous(wet[tgt], wet[c]) for k, c in methods.items()}
        for k in cont_wet:
            cont_wet[k]["skill_vs_block_copy"] = M.skill(cont_wet[k]["rmse"], cont_wet["block_copy"]["rmse"])
        # within-block spatial structure: correlation of GP anomalies (GP - block) on each day
        a_obs = y - out[fcol]
        a_pred = out[f"{var}_pred_mass"] - out[fcol]
        spatial_r = float(pd.DataFrame({"o": a_obs, "p": a_pred}).corr().iloc[0, 1])
        res["variables"][var] = {
            "subset": "unseen GPs x test period", "n": int(len(out)), "overall": cont,
            "wet_block_days" if var == "rain" else "all_days": cont_wet,
            "within_block_anomaly_correlation": spatial_r,
            "bootstrap_vs_block_copy": M.bootstrap_skill(out, tgt, f"{var}_pred", fcol,
                                                         n_boot=int(cfg["validation"]["bootstrap_samples"])),
            "bootstrap_mass_conserving_vs_block_copy": M.bootstrap_skill(
                out, tgt, f"{var}_pred_mass", fcol, n_boot=int(cfg["validation"]["bootstrap_samples"])),
        }
        log.info("[PP %s] skill vs block copy %.1f%% (mass-conserving %.1f%%), within-block anomaly r=%.2f", var,
                 100 * cont["downscaler"]["skill_vs_block_copy"],
                 100 * cont["downscaler_mass_conserving"]["skill_vs_block_copy"], spatial_r)
    from src.models.evaluate import sanitize

    (cfg.paths.reports / "perfect_prognosis.json").write_text(json.dumps(sanitize(res), indent=2), encoding="utf-8")
    return res


if __name__ == "__main__":
    run()
