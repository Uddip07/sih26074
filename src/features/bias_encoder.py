"""
Leakage-free historical-bias feature (audit 1.5, feature F18).

Spec section 5 step 7 defines ``historical_bias = mean(ground_truth − block_forecast)`` per
panchayat. The old implementation computed it over the **whole** dataset, including
test dates and test panchayats, so each test row's feature contained its own label.

This encoder fixes that:

1. **fit** only on training rows. The bias is kept per ``(gp, season)``, where season =
   monsoon (JJAS) or rest, because the Ghats' orographic bias flips sign seasonally.
   It is shrunk towards the block mean with empirical-Bayes weight ``n/(n+k)``.
2. **fit_transform_oof** gives training rows *out-of-fold* values: data are split into
   contiguous time blocks, and each block's feature comes from the other blocks. This
   mirrors operations, where the bias comes from *past* seasons, never the current day.
3. **transform** for GPs never seen in training (the spatial hold-out, and every GP of
   the held-out block in leave-one-block-out CV) uses **inverse-distance weighting**
   of the k nearest seen GPs. This is how an un-gauged panchayat would be served in
   deployment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

SEASON_COL = "is_monsoon"


class HistoricalBiasEncoder:
    def __init__(self, shrink_k: float = 20.0, idw_k: int = 8, idw_power: float = 2.0, n_time_folds: int = 5):
        self.shrink_k = shrink_k
        self.idw_k = idw_k
        self.idw_power = idw_power
        self.n_time_folds = n_time_folds
        self.table_: pd.DataFrame | None = None
        self.coords_: pd.DataFrame | None = None

    # ------------------------------------------------------------------------------------------
    def _table(self, gp: np.ndarray, block: np.ndarray, season: np.ndarray, resid: np.ndarray) -> pd.DataFrame:
        d = pd.DataFrame({"gp_code": gp, "block_lgd": block, "season": season, "r": resid})
        d = d[np.isfinite(d["r"])]
        g = d.groupby(["gp_code", "block_lgd", "season"])["r"].agg(["sum", "count"]).reset_index()
        b = d.groupby(["block_lgd", "season"])["r"].mean().rename("block_mean").reset_index()
        g = g.merge(b, on=["block_lgd", "season"], how="left")
        w = g["count"] / (g["count"] + self.shrink_k)
        g["bias"] = w * (g["sum"] / g["count"]) + (1 - w) * g["block_mean"]
        return g[["gp_code", "season", "bias"]]

    def fit(self, df: pd.DataFrame, resid: np.ndarray) -> HistoricalBiasEncoder:
        self.table_ = self._table(df["gp_code"].to_numpy(), df["block_lgd"].to_numpy(),
                                  df[SEASON_COL].to_numpy(), resid)
        self.coords_ = (df[["gp_code", "latitude", "longitude"]].drop_duplicates("gp_code").set_index("gp_code"))
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        assert self.table_ is not None and self.coords_ is not None, "fit first"
        key = pd.MultiIndex.from_arrays([df["gp_code"].to_numpy(), df[SEASON_COL].to_numpy()])
        tab = self.table_.set_index(["gp_code", "season"])["bias"]
        out = tab.reindex(key).to_numpy().astype("float64")
        miss = ~np.isfinite(out)
        if miss.any():
            out[miss] = self._idw(df.loc[miss, ["gp_code", "latitude", "longitude", SEASON_COL]])
        return out.astype("float32")

    def _idw(self, rows: pd.DataFrame) -> np.ndarray:
        """Spatial interpolation of seen-GP biases to unseen GPs (per season)."""
        res = np.full(len(rows), np.nan)
        tab = self.table_.merge(self.coords_, left_on="gp_code", right_index=True)
        for season in np.unique(rows[SEASON_COL]):
            src = tab[tab["season"] == season]
            if src.empty:
                continue
            xy_src = _to_xy(src["latitude"].to_numpy(), src["longitude"].to_numpy())
            tree = cKDTree(xy_src)
            m = (rows[SEASON_COL] == season).to_numpy()
            q = rows.loc[m].drop_duplicates("gp_code")
            k = min(self.idw_k, len(src))
            dist, idx = tree.query(_to_xy(q["latitude"].to_numpy(), q["longitude"].to_numpy()), k=k)
            dist = np.atleast_2d(dist)
            idx = np.atleast_2d(idx)
            w = 1.0 / np.maximum(dist, 0.5) ** self.idw_power
            vals = (src["bias"].to_numpy()[idx] * w).sum(1) / w.sum(1)
            lookup = dict(zip(q["gp_code"], vals))
            res[m] = rows.loc[m, "gp_code"].map(lookup).to_numpy()
        return res

    # ------------------------------------------------------------------------------------------
    def fit_transform_oof(self, df: pd.DataFrame, resid: np.ndarray, date_col: str = "valid_date") -> np.ndarray:
        """Out-of-fold (contiguous time blocks) values for training rows; then fit on all rows."""
        dates = df[date_col].to_numpy()
        uniq = np.unique(dates)
        edges = np.array_split(uniq, self.n_time_folds)
        out = np.full(len(df), np.nan, dtype="float64")
        for block_dates in edges:
            in_fold = np.isin(dates, block_dates)
            enc = HistoricalBiasEncoder(self.shrink_k, self.idw_k, self.idw_power)
            enc.fit(df.loc[~in_fold], resid[~in_fold])
            out[in_fold] = enc.transform(df.loc[in_fold])
        self.fit(df, resid)
        return out.astype("float32")


def _to_xy(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Local equirectangular km coordinates (accurate to <0.5 % over one district)."""
    lat0 = np.radians(np.nanmean(lat)) if len(lat) else 0.0
    return np.c_[np.radians(lon) * 6371.0 * np.cos(lat0), np.radians(lat) * 6371.0]
