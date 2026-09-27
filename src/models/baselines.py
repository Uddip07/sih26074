"""
Reference methods every downscaling model must beat (audit 4.2, feature F15).

==========================  ===============================================================
name                        method
==========================  ===============================================================
block_copy                  naive spatial copy: every GP gets its block forecast (spec 7.1)
block_bias_corrected        block forecast + mean block error by (block, season, lead) learned
                            on the training period: *bias correction without downscaling*.
                            Comparing against it isolates the skill that comes from spatial detail.
climatology_ratio           rain: block_fc x (GP normal / block normal) from CHPclim (classic
                            "delta/ratio" disaggregation). Temperatures: block_fc + standard
                            lapse rate (-6.5 °C/km) x (GP - block elevation). RH/wind: copy.
idw_blocks                  inverse-distance weighting of all block forecasts (by block
                            representative points) to the GP location, power 2
linear_mos                  ridge regression (model-output-statistics style) of the target on
                            forecast, elevation difference, climatology ratio, season, lead
nwp_grid_reference          area-weighted value of the 0.25° NWP grid over the GP itself.
                            Upper reference only: GKMS users receive block values, not the grid.
==========================  ===============================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LAPSE_C_PER_M = -0.0065
MOS_FEATURES = ["fc", "elev_diff", "clim_ratio_month", "doy_sin", "doy_cos", "lead_day", "is_monsoon"]


class Baselines:
    def __init__(self, var: str, target: str, forecast: str):
        self.var, self.target, self.forecast = var, target, forecast
        self.block_bias_: pd.Series | None = None
        self.mos_ = None
        self.idw_: dict | None = None

    def fit(self, train: pd.DataFrame, block_points: pd.DataFrame) -> Baselines:
        d = train[[self.target, self.forecast, "block_lgd", "is_monsoon", "lead_day"]].dropna()
        err = d[self.target] - d[self.forecast]
        self.block_bias_ = err.groupby([d["block_lgd"], d["is_monsoon"], d["lead_day"]]).mean()
        X = self._mos_X(train)
        ok = X.notna().all(axis=1) & train[self.target].notna()
        self.mos_ = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X[ok], train.loc[ok, self.target])
        self.idw_ = {"blocks": block_points.reset_index(drop=True)}
        return self

    def _mos_X(self, df: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame({"fc": df[self.forecast]})
        for c in MOS_FEATURES[1:]:
            X[c] = df[c]
        return X

    def predict(self, df: pd.DataFrame, block_fc_lookup: pd.DataFrame | None = None,
                nwp_grid: pd.Series | None = None) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        fc = df[self.forecast].to_numpy(dtype="float64")
        out["block_copy"] = fc

        key = pd.MultiIndex.from_arrays([df["block_lgd"], df["is_monsoon"], df["lead_day"]])
        out["block_bias_corrected"] = fc + self.block_bias_.reindex(key).fillna(0.0).to_numpy()

        if self.var == "rain":
            out["climatology_ratio"] = fc * df["clim_ratio_month"].to_numpy()
        elif self.var in ("tmax", "tmin"):
            out["climatology_ratio"] = fc + LAPSE_C_PER_M * df["elev_diff"].to_numpy()
        else:
            out["climatology_ratio"] = fc

        X = self._mos_X(df)
        ok = X.notna().all(axis=1).to_numpy()
        mos = np.full(len(df), np.nan)
        mos[ok] = self.mos_.predict(X[ok])
        out["linear_mos"] = mos

        if block_fc_lookup is not None:
            out["idw_blocks"] = self._idw(df, block_fc_lookup)
        if nwp_grid is not None:
            out["nwp_grid_reference"] = nwp_grid.reindex(df.index).to_numpy()

        if self.var in ("rain", "wind"):
            out = out.clip(lower=0)
        if self.var == "rh":
            out = out.clip(0, 100)
        return out

    def _idw(self, df: pd.DataFrame, block_fc: pd.DataFrame, power: float = 2.0) -> np.ndarray:
        """block_fc: rows (valid_date, lead_day), columns block_lgd -> forecast value."""
        b = self.idw_["blocks"]
        lat0 = np.radians(b["latitude"].mean())

        def xy(lat, lon):
            return np.c_[np.radians(lon) * 6371 * np.cos(lat0), np.radians(lat) * 6371]

        gp = df[["gp_code", "latitude", "longitude"]].drop_duplicates("gp_code")
        dist = np.linalg.norm(xy(gp["latitude"].to_numpy(), gp["longitude"].to_numpy())[:, None, :]
                              - xy(b["latitude"].to_numpy(), b["longitude"].to_numpy())[None, :, :], axis=2)
        w = 1.0 / np.maximum(dist, 1.0) ** power
        w = w / w.sum(1, keepdims=True)
        wdf = pd.DataFrame(w, index=gp["gp_code"].to_numpy(), columns=b["block_lgd"].to_numpy())
        cols = [c for c in wdf.columns if c in block_fc.columns]
        wdf = wdf[cols].div(wdf[cols].sum(1), axis=0)
        fcm = block_fc[cols]
        key = pd.MultiIndex.from_arrays([df["valid_date"], df["lead_day"]])
        F = fcm.reindex(key).to_numpy()
        W = wdf.reindex(df["gp_code"].to_numpy()).to_numpy()
        return np.nansum(F * W, axis=1) / np.nansum(np.where(np.isfinite(F), W, 0), axis=1)
