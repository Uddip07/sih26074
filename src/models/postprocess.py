"""
Post-processing: mass-conservation diagnostics and the optional rescaling (feature F14).

"Downscaling" in the strict sense redistributes a block value among its panchayats
without changing the block total. Our model does two things at once:
(1) **bias-corrects** the NWP block forecast, and (2) **disaggregates** it spatially.

``conservation_report`` measures how far the area-weighted GP mean departs from the
block forecast (the bias-correction component). ``conserve`` optionally rescales the GP
values so that the area-weighted mean equals the block forecast exactly
(multiplicative for rain, additive for the other variables). This gives "pure
disaggregation" output, which is evaluated as a separate variant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MULTIPLICATIVE = {"rain"}


def block_area_weights(static: pd.DataFrame) -> pd.Series:
    """GP weight within its block = GP area / total GP area of the block (non-GP areas excluded)."""
    a = static.set_index("gp_code")["area_km2"]
    blk = static.set_index("gp_code")["block_lgd"]
    return (a / a.groupby(blk).transform("sum")).rename("w_area")


def _group_keys(df: pd.DataFrame) -> list[str]:
    return ["block_lgd", "issue_date", "lead_day"]


def conservation_report(df: pd.DataFrame, pred_col: str, fc_col: str, weights: pd.Series) -> dict:
    d = df[_group_keys(df) + ["gp_code", pred_col, fc_col]].dropna().copy()
    d["w"] = d["gp_code"].map(weights)
    g = d.assign(wp=d[pred_col] * d["w"]).groupby(_group_keys(df)).agg(wp=("wp", "sum"), w=("w", "sum"),
                                                                        fc=(fc_col, "first"))
    blk_mean = g["wp"] / g["w"]
    diff = blk_mean - g["fc"]
    rel = diff / g["fc"].where(g["fc"].abs() > 0.5)
    return {"mean_abs_block_departure": float(diff.abs().mean()), "mean_block_departure": float(diff.mean()),
            "median_abs_relative_departure": float(rel.abs().median()) if rel.notna().any() else np.nan,
            "n_block_forecasts": int(len(g))}


def conserve(df: pd.DataFrame, var: str, pred_col: str, fc_col: str, weights: pd.Series) -> pd.Series:
    d = df[_group_keys(df) + ["gp_code", pred_col, fc_col]].copy()
    d["w"] = d["gp_code"].map(weights).fillna(0)
    grp = d.groupby(_group_keys(df))
    wmean = (d[pred_col] * d["w"]).groupby([d[k] for k in _group_keys(df)]).transform("sum") / \
        grp["w"].transform("sum")
    if var in MULTIPLICATIVE:
        factor = np.where(wmean > 1e-6, d[fc_col] / wmean, np.nan)
        out = np.where(np.isfinite(factor), d[pred_col] * factor, d[fc_col])
        return pd.Series(np.clip(out, 0, None), index=df.index)
    return pd.Series(d[pred_col] + (d[fc_col] - wmean), index=df.index)
