"""
Verification metrics (audit 1.8).

Continuous : RMSE, MAE, mean bias, Pearson r, skill score vs a reference.
Categorical: at IMD thresholds (2.5 / 15.6 / 64.5 mm): hits, misses, false alarms,
             POD, FAR, CSI, ETS (Gilbert skill score), frequency bias, HSS.
             **All undefined scores return NaN** when there are no events. The old
             ``compute_csi`` returned 1.0, which inflated dry-season and heavy-rain CSI.
Probabilistic: Brier score, Brier skill score vs sample climatology, reliability
             table, quantile (pinball) loss, central-interval coverage.
Uncertainty: moving-block bootstrap over *days* (spatially correlated GPs on the
             same day are resampled together) for confidence intervals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

IMD_THRESHOLDS = (2.5, 15.6, 64.5)
IMD_RAIN_CLASSES = [(-0.01, 0.1, "no rain"), (0.1, 2.5, "very light"), (2.5, 15.6, "light"),
                    (15.6, 64.5, "moderate"), (64.5, 115.6, "heavy"), (115.6, 204.5, "very heavy"),
                    (204.5, np.inf, "extremely heavy")]


def _clean(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype="float64")
    p = np.asarray(p, dtype="float64")
    ok = np.isfinite(y) & np.isfinite(p)
    return y[ok], p[ok]


def continuous(y, p) -> dict[str, float]:
    y, p = _clean(y, p)
    if y.size == 0:
        return {"n": 0, "rmse": np.nan, "mae": np.nan, "bias": np.nan, "r": np.nan}
    e = p - y
    r = float(np.corrcoef(y, p)[0, 1]) if y.std() > 0 and p.std() > 0 else np.nan
    return {"n": int(y.size), "rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e))),
            "bias": float(np.mean(e)), "r": r}


def skill(model_rmse: float, ref_rmse: float) -> float:
    return float(1.0 - model_rmse / ref_rmse) if ref_rmse and np.isfinite(ref_rmse) and ref_rmse > 0 else np.nan


def categorical(y, p, threshold: float) -> dict[str, float]:
    y, p = _clean(y, p)
    obs, fc = y >= threshold, p >= threshold
    a = int(np.sum(obs & fc))          # hits
    c = int(np.sum(obs & ~fc))         # misses
    b = int(np.sum(~obs & fc))         # false alarms
    d = int(np.sum(~obs & ~fc))        # correct negatives
    n = a + b + c + d

    def div(x, z):
        return float(x / z) if z > 0 else np.nan

    a_r = (a + b) * (a + c) / n if n else np.nan
    hss_den = (a + c) * (c + d) + (a + b) * (b + d)
    return {
        "threshold": threshold, "hits": a, "misses": c, "false_alarms": b, "correct_neg": d,
        "event_freq": div(a + c, n),
        "pod": div(a, a + c), "far": div(b, a + b), "csi": div(a, a + b + c),
        "ets": div(a - a_r, a + b + c - a_r) if n and np.isfinite(a_r) else np.nan,
        "freq_bias": div(a + b, a + c),
        "hss": div(2 * (a * d - b * c), hss_den),
    }


def brier(y_event: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    y_event = np.asarray(y_event, dtype="float64")
    prob = np.asarray(prob, dtype="float64")
    ok = np.isfinite(y_event) & np.isfinite(prob)
    y_event, prob = y_event[ok], prob[ok]
    if y_event.size == 0:
        return {"brier": np.nan, "bss": np.nan, "base_rate": np.nan}
    bs = float(np.mean((prob - y_event) ** 2))
    base = float(y_event.mean())
    ref = base * (1 - base)
    return {"brier": bs, "bss": float(1 - bs / ref) if ref > 0 else np.nan, "base_rate": base}


def reliability(y_event: np.ndarray, prob: np.ndarray, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y_event, float), "p": np.asarray(prob, float)}).dropna()
    df["bin"] = np.clip((df["p"] * bins).astype(int), 0, bins - 1)
    g = df.groupby("bin").agg(p_mean=("p", "mean"), obs_freq=("y", "mean"), n=("y", "size")).reset_index()
    return g


def pinball(y, q, alpha: float) -> float:
    y, q = _clean(y, q)
    d = y - q
    return float(np.mean(np.maximum(alpha * d, (alpha - 1) * d))) if y.size else np.nan


def coverage(y, lo, hi) -> float:
    y = np.asarray(y, float)
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    ok = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    return float(np.mean((y[ok] >= lo[ok]) & (y[ok] <= hi[ok]))) if ok.any() else np.nan


def rain_class(values: np.ndarray) -> np.ndarray:
    out = np.empty(len(values), dtype=object)
    v = np.asarray(values, float)
    for lo, hi, name in IMD_RAIN_CLASSES:
        out[(v > lo) & (v <= hi) if lo >= 0 else (v <= hi)] = name
    out[~np.isfinite(v)] = None
    return out


def bootstrap_skill(df: pd.DataFrame, y: str, p: str, ref: str, day_col: str = "valid_date",
                    n_boot: int = 300, seed: int = 42, block_days: int = 7) -> dict[str, float]:
    """
    Moving-block bootstrap over days of the RMSE skill score of ``p`` vs ``ref``.
    Returns point estimate plus 2.5/97.5 percentiles.
    """
    d = df[[day_col, y, p, ref]].dropna()
    g = d.assign(e_p=(d[p] - d[y]) ** 2, e_r=(d[ref] - d[y]) ** 2).groupby(day_col)[["e_p", "e_r"]].agg(["sum", "count"])
    sp, cp = g[("e_p", "sum")].to_numpy(), g[("e_p", "count")].to_numpy()
    sr = g[("e_r", "sum")].to_numpy()
    n_days = len(sp)
    point = skill(np.sqrt(sp.sum() / cp.sum()), np.sqrt(sr.sum() / cp.sum()))
    if n_days < 2 * block_days:
        return {"skill": point, "ci_low": np.nan, "ci_high": np.nan, "n_days": n_days}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n_days / block_days))
    vals = []
    for _ in range(n_boot):
        starts = rng.integers(0, n_days - block_days + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(block_days)[None, :]).ravel()[:n_days]
        vals.append(skill(np.sqrt(sp[idx].sum() / cp[idx].sum()), np.sqrt(sr[idx].sum() / cp[idx].sum())))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return {"skill": point, "ci_low": float(lo), "ci_high": float(hi), "n_days": n_days}
