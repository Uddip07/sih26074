"""
Rolling operational verification (feature F29).

Once observations arrive (CHIRPS prelim about 2 days after the fact, ERA5-Land about 5 days),
every issued forecast in ``outputs/<d>/forecasts/*`` is scored against them:

* per issue x lead: RMSE / MAE / bias of the downscaled forecast vs the naive block copy,
  CSI at 2.5 mm
* per GP: a "forecast vs actual" series for the dashboard's verification panel
* a rolling 30-issue summary, written to ``outputs/<d>/reports/verification.json``
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.common.config import Config, load_config
from src.common.logging_utils import get_logger
from src.models import metrics as M

log = get_logger("pipeline.verify")


def _observations(cfg: Config) -> pd.DataFrame:
    it = cfg.paths.interim
    parts = []
    for fn in ("gp_rain_obs.parquet", "gp_rain_obs_prelim.parquet"):
        if (it / fn).exists():
            parts.append(pd.read_parquet(it / fn))
    rain = pd.concat(parts).drop_duplicates(["gp_code", "date"], keep="last") if parts else pd.DataFrame()
    obs = rain
    if (it / "gp_met_obs.parquet").exists():
        met = pd.read_parquet(it / "gp_met_obs.parquet")[["gp_code", "date", "tmax_obs", "tmin_obs", "rh_obs"]]
        obs = rain.merge(met, on=["gp_code", "date"], how="outer") if len(rain) else met
    obs["gp_code"] = obs["gp_code"].astype(str)
    return obs.rename(columns={"date": "valid_date"})


def run(cfg: Config | None = None, refresh_prelim: bool = False) -> dict:
    cfg = cfg or load_config()
    if refresh_prelim:
        from src.ingest import chirps

        try:
            chirps.update_prelim(cfg)
        except Exception as exc:  # noqa: BLE001
            log.warning("prelim refresh failed: %s", exc)
    obs = _observations(cfg)
    rows, per_gp = [], []
    for d in sorted(cfg.paths.forecasts.glob("20*")):
        f = d / "gp_forecast.parquet"
        if not f.exists():
            continue
        # track record = operational (live) issues only; archive replays are scored in evaluation.json
        try:
            if json.loads((d / "meta.json").read_text(encoding="utf-8")).get("source") != "live":
                continue
        except (OSError, ValueError):
            continue
        fc = pd.read_parquet(f)
        m = fc.merge(obs, on=["gp_code", "valid_date"], how="inner")
        if m.empty:
            continue
        for lead, g in m.groupby("lead_day"):
            g = g.dropna(subset=["rain_obs"])
            if g.empty:
                continue
            mod = M.continuous(g["rain_obs"], g["rain_pred"])
            base = M.continuous(g["rain_obs"], g["fc_rain"])
            rec = {"issue_date": d.name, "lead_day": int(lead), "valid_date": str(g["valid_date"].iloc[0].date()),
                   "n": mod["n"], "rain_rmse": mod["rmse"], "rain_rmse_block": base["rmse"],
                   "rain_mae": mod["mae"], "rain_bias": mod["bias"],
                   "rain_skill": M.skill(mod["rmse"], base["rmse"]),
                   "csi_2p5": M.categorical(g["rain_obs"], g["rain_pred"], 2.5)["csi"],
                   "csi_2p5_block": M.categorical(g["rain_obs"], g["fc_rain"], 2.5)["csi"]}
            if "tmax_obs" in g and g["tmax_obs"].notna().any():
                rec["tmax_rmse"] = M.continuous(g["tmax_obs"], g["tmax_pred"])["rmse"]
                rec["tmax_rmse_block"] = M.continuous(g["tmax_obs"], g["fc_tmax"])["rmse"]
            rows.append(rec)
        keep = m[m["lead_day"] == 1][["gp_code", "valid_date", "rain_pred", "fc_rain", "rain_obs"]]
        per_gp.append(keep)
    t = pd.DataFrame(rows)
    live = [d.name for d in cfg.paths.forecasts.glob("20*") if (d / "meta.json").exists()
            and json.loads((d / "meta.json").read_text(encoding="utf-8")).get("source") == "live"]
    res: dict = {"n_issues_verified": int(t["issue_date"].nunique()) if len(t) else 0,
                 "live_issues": sorted(live), "obs_last_date": str(obs["valid_date"].max().date()) if len(obs) else None}
    if len(t):
        recent = t[t["issue_date"].isin(sorted(t["issue_date"].unique())[-30:])]
        agg = recent.assign(se_m=recent["rain_rmse"] ** 2 * recent["n"], se_b=recent["rain_rmse_block"] ** 2 * recent["n"])
        by_lead = agg.groupby("lead_day").apply(
            lambda g: pd.Series({"rmse": np.sqrt(g["se_m"].sum() / g["n"].sum()),
                                 "rmse_block": np.sqrt(g["se_b"].sum() / g["n"].sum())}), include_groups=False)
        by_lead["skill"] = 1 - by_lead["rmse"] / by_lead["rmse_block"]
        res["rolling_30_issues_by_lead"] = by_lead.round(4).reset_index().to_dict("records")
        res["per_issue"] = t.round(4).to_dict("records")
    out = cfg.paths.reports / "verification.json"
    from src.models.evaluate import sanitize

    out.write_text(json.dumps(sanitize(res), indent=2, default=str), encoding="utf-8")
    if per_gp:
        s = pd.concat(per_gp).sort_values("valid_date")
        s.to_parquet(cfg.paths.web / "verification_gp_lead1.parquet", index=False)
    log.info("verified %s issues", res["n_issues_verified"])
    return res


if __name__ == "__main__":
    run(refresh_prelim=True)
