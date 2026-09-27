"""
Master modelling table (vectorised; replaces the old triple-``iterrows`` builder).

One row per ``(gp_code, issue_date, lead_day)``, where ``valid_date = issue_date + lead_day``.

=====================  ==============================================================
Column group           Contents
=====================  ==============================================================
keys                   gp_code, block_lgd, issue_date, valid_date, lead_day
block forecast (X)     fc_rain, fc_tmax, fc_tmin, fc_rh, fc_wind (block area-mean NWP)
forecast context       fc_rain_log1p, fc_rain_issue_sum (total forecast rain over the 5 days
                       of the same issue), fc_rain_ante3 (block lead-1 forecasts valid on
                       issue-2..issue: known at issue time)
antecedent obs         ante_obs7_lag3: GP observed rain summed over [issue-9, issue-3]. The
                       3-day lag reflects real CHIRPS-prelim latency, so the feature is
                       available operationally on the issue date.
season                 doy_sin, doy_cos, month, is_monsoon (of the valid date)
climatology            clim_month_gp, clim_ratio_month (GP / block CHPclim normal for the
                       valid month): leakage-free spatial prior
targets (y)            rain_obs (CHIRPS), tmax_obs, tmin_obs, rh_obs, wind_obs (ERA5-Land),
                       plus rhmax/rhmin/et0/srad for advisories and verification
=====================  ==============================================================

Static GP covariates are **not** duplicated into this table. ``load_frame``
joins them in memory, which keeps the parquet small.

**Leakage note:** nothing here uses the target of the same row. The historical
bias feature is computed later, inside each training fold
(``src/features/bias_encoder.py``).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.common import manifest
from src.common.config import Config, load_config
from src.common.logging_utils import get_logger

log = get_logger("features.dataset")

FC_VARS = ["rain", "tmax", "tmin", "rh", "wind"]


def _truth(cfg: Config) -> pd.DataFrame:
    it = cfg.paths.interim
    rain = pd.read_parquet(it / "gp_rain_obs.parquet")
    met_path = it / "gp_met_obs.parquet"
    rain["gp_code"] = rain["gp_code"].astype(str)
    if met_path.exists():
        met = pd.read_parquet(met_path)
        met["gp_code"] = met["gp_code"].astype(str)
        truth = rain.merge(met, on=["gp_code", "date"], how="outer")
    else:
        log.warning("gp_met_obs.parquet missing - temperature/RH/wind targets unavailable")
        truth = rain
    return truth.rename(columns={"date": "valid_date"})


# ---------------------------------------------------------------------------------------------
# Shared feature builders: used identically by training (build_dataset) and inference
# (src/pipeline/predict.py), so there is no train/serve skew.
# ---------------------------------------------------------------------------------------------
def forecast_context(fc: pd.DataFrame) -> pd.DataFrame:
    """Block-level context features. ``fc`` must hold fc_* columns and, for fc_rain_ante3, the
    lead-1 forecasts of the preceding issues (rows of past issue dates)."""
    fc = fc.copy()
    fc["fc_rain_log1p"] = np.log1p(fc["fc_rain"].clip(lower=0))
    # 5-day (leads 1-5) issue total, as in training, even when the operational horizon is longer
    r15 = fc["fc_rain"].where(fc["lead_day"].between(1, 5), 0.0)
    fc["fc_rain_issue_sum"] = r15.groupby([fc["block_lgd"], fc["issue_date"]]).transform("sum")
    lead1 = fc[fc["lead_day"] == 1][["block_lgd", "valid_date", "fc_rain"]].rename(columns={"fc_rain": "r"})
    if lead1.empty:
        fc["fc_rain_ante3"] = np.nan
        return fc
    lead1 = lead1.drop_duplicates(["block_lgd", "valid_date"]).set_index(["block_lgd", "valid_date"])["r"]
    lead1 = lead1.unstack(0).sort_index()
    lead1 = lead1.reindex(pd.date_range(lead1.index.min(), lead1.index.max(), freq="D"))
    ante3 = lead1.rolling(3, min_periods=3).sum()  # lead-1 forecasts valid on issue-2..issue
    ante3 = ante3.stack(future_stack=True).rename("fc_rain_ante3").reset_index()
    ante3.columns = ["issue_date", "block_lgd", "fc_rain_ante3"]
    return fc.merge(ante3, on=["block_lgd", "issue_date"], how="left")


def add_season(df: pd.DataFrame) -> pd.DataFrame:
    doy = df["valid_date"].dt.dayofyear.to_numpy()
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25).astype("float32")
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25).astype("float32")
    df["month"] = df["valid_date"].dt.month.astype("int8")
    df["is_monsoon"] = df["month"].isin([6, 7, 8, 9]).astype("int8")
    return df


def add_antecedent(df: pd.DataFrame, rain_obs: pd.DataFrame, carry_forward_days: int = 0) -> pd.DataFrame:
    """ante_obs7_lag3: GP observed rain summed over [issue-9, issue-3] (CHIRPS-prelim latency).

    ``carry_forward_days`` (inference only): when preliminary CHIRPS lags more than the nominal 3 days,
    use the most recent complete 7-day window, up to this many days old, instead of leaving it empty."""
    wide = rain_obs.pivot_table(index="valid_date", columns="gp_code", values="rain_obs").sort_index()
    wide = wide.reindex(pd.date_range(wide.index.min(), max(wide.index.max(), df["issue_date"].max()), freq="D"))
    ante = wide.rolling(7, min_periods=5).sum()
    if carry_forward_days:
        ante = ante.ffill(limit=carry_forward_days)
    ante = ante.shift(3)
    ante = ante.stack(future_stack=True).rename("ante_obs7_lag3").reset_index()
    ante.columns = ["issue_date", "gp_code", "ante_obs7_lag3"]
    return df.merge(ante, on=["gp_code", "issue_date"], how="left")


def add_ndvi(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    ndvi_path = cfg.paths.interim / "gp_ndvi.parquet"
    if ndvi_path.exists():
        from src.ingest.ndvi import ndvi_for_issue_dates

        ndvi = pd.read_parquet(ndvi_path)
        ndvi["gp_code"] = ndvi["gp_code"].astype(str)
        df["ndvi"] = ndvi_for_issue_dates(ndvi, df)
    else:
        log.warning("gp_ndvi.parquet missing - NDVI feature excluded")
    return df


def expand_to_gps(fc: pd.DataFrame, gp_keys: pd.DataFrame) -> pd.DataFrame:
    return gp_keys.merge(fc, on="block_lgd", how="inner")


def build_dataset(cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    it = cfg.paths.interim
    static = pd.read_parquet(it / "gp_static.parquet")[["gp_code", "block_lgd"]]
    static["gp_code"] = static["gp_code"].astype(str)

    fc = pd.read_parquet(it / "block_forecasts.parquet")
    fc = fc.rename(columns={v: f"fc_{v}" for v in FC_VARS})
    fc = fc[fc["block_lgd"].isin(static["block_lgd"].unique())]
    fc = forecast_context(fc)
    df = add_season(expand_to_gps(fc, static))
    truth = _truth(cfg)
    df = add_antecedent(df, truth[["gp_code", "valid_date", "rain_obs"]])
    df = df.merge(truth, on=["gp_code", "valid_date"], how="left")
    df = add_ndvi(df, cfg)

    for c in df.columns:
        if df[c].dtype == "float64":
            df[c] = df[c].astype("float32")
    df = df.sort_values(["valid_date", "lead_day", "gp_code"]).reset_index(drop=True)
    out = cfg.paths.processed / "dataset.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    tgt = [c for c in ["rain_obs", "tmax_obs", "tmin_obs", "rh_obs", "wind_obs"] if c in df]
    summary = {
        "rows": int(len(df)), "gps": int(df["gp_code"].nunique()), "valid_dates": int(df["valid_date"].nunique()),
        "leads": sorted(int(x) for x in df["lead_day"].unique()),
        "target_coverage_pct": {c: round(100 * float(df[c].notna().mean()), 2) for c in tgt},
        "forecast_coverage_pct": {f"fc_{v}": round(100 * float(df[f'fc_{v}'].notna().mean()), 2) for v in FC_VARS},
        "period": [str(df["valid_date"].min().date()), str(df["valid_date"].max().date())],
    }
    (cfg.paths.reports / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest.register(cfg.paths.data / "manifest.json", "dataset", out, "derived (see component datasets)",
                      "inherits component licences", "src.features.dataset", extra=summary)
    log.info("dataset: %s", summary)
    return df


def load_frame(cfg: Config | None = None, columns: list[str] | None = None) -> pd.DataFrame:
    """Dataset joined with static GP covariates and month-dependent climatology features."""
    cfg = cfg or load_config()
    df = pd.read_parquet(cfg.paths.processed / "dataset.parquet", columns=columns)
    static = pd.read_parquet(cfg.paths.interim / "gp_static.parquet")
    static["gp_code"] = static["gp_code"].astype(str)
    df = add_static(df, static)
    return df


def add_static(df: pd.DataFrame, static: pd.DataFrame) -> pd.DataFrame:
    """Join static covariates and compute clim_month_gp / clim_ratio_month for each row's valid month."""
    keep = [c for c in static.columns if c not in ("gp_name", "block_name", "block_lgd")]
    s = static[keep].copy()
    for c in s.columns:
        if s[c].dtype == "float64":
            s[c] = s[c].astype("float32")
    out = df.merge(s, on="gp_code", how="left")
    months = out["valid_date"].dt.month.to_numpy()
    gp_m = out[[f"clim_m{m:02d}" for m in range(1, 13)]].to_numpy()
    bl_m = out[[f"blk_clim_m{m:02d}" for m in range(1, 13)]].to_numpy()
    idx = np.arange(len(out))
    out["clim_month_gp"] = gp_m[idx, months - 1].astype("float32")
    # +1 mm/month regularises dry-season ratios (0/0)
    out["clim_ratio_month"] = ((gp_m[idx, months - 1] + 1.0) / (bl_m[idx, months - 1] + 1.0)).astype("float32")
    drop = [f"clim_m{m:02d}" for m in range(1, 13)] + [f"blk_clim_m{m:02d}" for m in range(1, 13)]
    return out.drop(columns=[c for c in drop if c in out.columns])


if __name__ == "__main__":
    build_dataset()
