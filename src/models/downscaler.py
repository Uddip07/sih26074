"""
Multi-variable, lead-aware residual downscaling model (spec 7.2; features F3, F4, F12, F13, F14, F16).

For each variable v in {rain, tmax, tmin, rh, wind}:

    residual  r = T(y_gp) - T(fc_block)          T = identity, or log1p for rain (chosen by validation)
    point     y_hat = T^-1( T(fc_block) + f(X) )   f = XGBoost regressor
    quantiles y_q   = T^-1( T(fc_block) + f_q(X) ) q in {0.1, 0.5, 0.9}: one multi-quantile XGBoost,
                    then split-conformal widening on a later calibration period (CQR, Romano et al. 2019)
    rain only P(y >= t) for t in {2.5, 15.6, 64.5} mm: XGBoost classifiers + isotonic calibration

X contains the block forecast of *all* variables, the lead day, season, antecedent
conditions, terrain / land-cover / hydro / soil / vegetation / climatology covariates, and
the leakage-free ``hist_bias`` (see ``src/features/bias_encoder.py``).

**Zero-inflation (F16).** With T = log1p the residual is multiplicative, so predictions are
non-negative by construction and dry block forecasts stay near zero. The exceedance
classifiers model rain *occurrence* separately (the "hurdle" part), and advisories use
those probabilities.

**Explainability (F13).** ``contributions`` returns exact TreeSHAP values (XGBoost
``pred_contribs``) in residual space, aggregated into feature families.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

from src.common.logging_utils import get_logger
from src.features.bias_encoder import HistoricalBiasEncoder

log = get_logger("models.downscaler")


def _fwd(x: np.ndarray, space: str) -> np.ndarray:
    return np.log1p(np.clip(x, 0, None)) if space == "log1p" else x


def _inv(x: np.ndarray, space: str) -> np.ndarray:
    return np.expm1(x) if space == "log1p" else x


def resolve_features(model_cfg: dict, columns: list[str]) -> list[str]:
    fams = model_cfg["features"]
    excl = set(model_cfg["exclude_features"])
    feats: list[str] = []
    for fam in fams.values():
        for f in fam:
            if (f in columns or f == "hist_bias") and f not in excl and f not in feats:
                feats.append(f)
    return feats


def feature_family(model_cfg: dict) -> dict[str, str]:
    return {f: fam for fam, fs in model_cfg["features"].items() for f in fs}


@dataclass
class VariableDownscaler:
    var: str
    target: str
    forecast: str
    features: list[str]
    params: dict
    quantiles: list[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    thresholds: list[float] = field(default_factory=list)
    residual_space: str = "linear"
    lower: float | None = None
    upper: float | None = None
    conformal: bool = True
    early_stopping_rounds: int = 50

    encoder: HistoricalBiasEncoder | None = None
    point_: xgb.XGBRegressor | None = None
    quant_: xgb.XGBRegressor | None = None
    conformal_margin_: float = 0.0
    classifiers_: dict = field(default_factory=dict)
    calibrators_: dict = field(default_factory=dict)
    space_selection_: dict = field(default_factory=dict)
    best_iteration_: int | None = None

    # ------------------------------------------------------------------------------------------
    def _usable(self, df: pd.DataFrame) -> np.ndarray:
        return (df[self.target].notna() & df[self.forecast].notna()).to_numpy()

    def _residual(self, df: pd.DataFrame, space: str) -> np.ndarray:
        return _fwd(df[self.target].to_numpy("float64"), space) - _fwd(df[self.forecast].to_numpy("float64"), space)

    def _X(self, df: pd.DataFrame, hist_bias: np.ndarray) -> pd.DataFrame:
        cols = [f for f in self.features if f != "hist_bias"]
        X = df[cols].astype("float32").copy()
        X["hist_bias"] = hist_bias.astype("float32")
        return X[self.features]

    def _regressor(self, **over) -> xgb.XGBRegressor:
        p = dict(self.params)
        p.update(over)
        return xgb.XGBRegressor(**p)

    # ------------------------------------------------------------------------------------------
    def fit(self, train: pd.DataFrame, calib: pd.DataFrame | None = None) -> VariableDownscaler:
        train = train[self._usable(train)]
        calib = calib[self._usable(calib)] if calib is not None and len(calib) else None
        log.info("[%s] fit on %d rows (%d GPs), calib %s rows", self.var, len(train), train["gp_code"].nunique(),
                 0 if calib is None else len(calib))

        if self.residual_space == "auto":
            self.residual_space = self._select_space(train, calib)

        r_tr = self._residual(train, self.residual_space)
        self.encoder = HistoricalBiasEncoder()
        hb_tr = self.encoder.fit_transform_oof(train, r_tr)
        X_tr = self._X(train, hb_tr)

        eval_set, X_ca, r_ca = None, None, None
        if calib is not None:
            r_ca = self._residual(calib, self.residual_space)
            X_ca = self._X(calib, self.encoder.transform(calib))
            eval_set = [(X_ca, r_ca)]

        self.point_ = self._regressor(early_stopping_rounds=self.early_stopping_rounds if eval_set else None)
        self.point_.fit(X_tr, r_tr, eval_set=eval_set, verbose=False)
        self.best_iteration_ = getattr(self.point_, "best_iteration", None)

        # multi-quantile model (one booster, several outputs)
        from src.common.config import load_config

        qt = load_config().model["xgboost"]["quantile_trees"]
        self.quant_ = self._regressor(objective="reg:quantileerror", quantile_alpha=np.array(self.quantiles),
                                      early_stopping_rounds=None,
                                      n_estimators=max(int(qt["min"]), int(self.params["n_estimators"] * qt["fraction"])))
        self.quant_.fit(X_tr, r_tr, verbose=False)
        if self.conformal and calib is not None:
            q = self.quant_.predict(X_ca)
            lo, hi = q[:, 0], q[:, -1]
            scores = np.maximum(lo - r_ca, r_ca - hi)
            alpha = self.quantiles[0] + (1 - self.quantiles[-1])
            n = len(scores)
            level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
            self.conformal_margin_ = float(max(0.0, np.quantile(scores, level)))

        # exceedance probabilities (rain)
        for t in self.thresholds:
            y_tr = (train[self.target].to_numpy() >= t).astype(int)
            if y_tr.sum() < 50:
                log.warning("[%s] threshold %.1f has only %d events - classifier skipped", self.var, t, y_tr.sum())
                continue
            clf = xgb.XGBClassifier(**{k: v for k, v in self.params.items() if k != "objective"},
                                    objective="binary:logistic", eval_metric="logloss")
            clf.fit(X_tr, y_tr, verbose=False)
            self.classifiers_[t] = clf
            if calib is not None:
                y_ca = (calib[self.target].to_numpy() >= t).astype(int)
                if y_ca.sum() >= 10:
                    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
                    iso.fit(clf.predict_proba(X_ca)[:, 1], y_ca)
                    self.calibrators_[t] = iso
        return self

    def _select_space(self, train: pd.DataFrame, calib: pd.DataFrame | None) -> str:
        """Choose log1p vs linear residual space by calibration-period RMSE (fast models)."""
        if calib is None:
            return "log1p"
        sub = train.sample(min(len(train), 600_000), random_state=0)
        res = {}
        for space in ("linear", "log1p"):
            r = self._residual(sub, space)
            enc = HistoricalBiasEncoder()
            hb = enc.fit_transform_oof(sub, r)
            m = self._regressor(n_estimators=250, early_stopping_rounds=None)
            m.fit(self._X(sub, hb), r, verbose=False)
            pr = _inv(_fwd(calib[self.forecast].to_numpy("float64"), space) +
                      m.predict(self._X(calib, enc.transform(calib))), space)
            pr = np.clip(pr, self.lower, self.upper) if self.lower is not None or self.upper is not None else pr
            res[space] = float(np.sqrt(np.mean((pr - calib[self.target].to_numpy()) ** 2)))
        self.space_selection_ = res
        best = min(res, key=res.get)
        log.info("[%s] residual space selection (calib RMSE): %s -> %s", self.var, res, best)
        return best

    # ------------------------------------------------------------------------------------------
    def _clip(self, x: np.ndarray) -> np.ndarray:
        if self.lower is not None or self.upper is not None:
            return np.clip(x, self.lower, self.upper)
        return x

    def predict(self, df: pd.DataFrame, with_uncertainty: bool = True) -> pd.DataFrame:
        hb = self.encoder.transform(df)
        X = self._X(df, hb)
        base = _fwd(df[self.forecast].to_numpy("float64"), self.residual_space)
        it = (0, self.best_iteration_ + 1) if self.best_iteration_ is not None else None
        r = self.point_.predict(X, iteration_range=it) if it else self.point_.predict(X)
        out = pd.DataFrame(index=df.index)
        out[f"{self.var}_pred"] = self._clip(_inv(base + r, self.residual_space)).astype("float32")
        out[f"{self.var}_hist_bias"] = hb
        if with_uncertainty and self.quant_ is not None:
            q = self.quant_.predict(X)
            q = np.sort(q, axis=1)  # guard against quantile crossing
            q[:, 0] -= self.conformal_margin_
            q[:, -1] += self.conformal_margin_
            for k, a in enumerate(self.quantiles):
                out[f"{self.var}_q{int(round(a * 100)):02d}"] = self._clip(
                    _inv(base + q[:, k], self.residual_space)).astype("float32")
        for t, clf in self.classifiers_.items():
            p = clf.predict_proba(X)[:, 1]
            if t in self.calibrators_:
                p = self.calibrators_[t].predict(p)
            out[f"{self.var}_p_ge_{str(t).replace('.', 'p')}"] = p.astype("float32")
        return out

    def contributions(self, df: pd.DataFrame, families: dict[str, str]) -> pd.DataFrame:
        """TreeSHAP contributions of the point model (residual space), summed by feature family."""
        X = self._X(df, self.encoder.transform(df))
        booster = self.point_.get_booster()
        it = (0, self.best_iteration_ + 1) if self.best_iteration_ is not None else (0, 0)
        contrib = booster.predict(xgb.DMatrix(X), pred_contribs=True, iteration_range=it)
        cols = list(X.columns) + ["bias_term"]
        c = pd.DataFrame(contrib, columns=cols, index=df.index)
        fam = {f: families.get(f, "other") for f in X.columns}
        fam["hist_bias"] = "history"
        grouped = c[list(X.columns)].T.groupby(pd.Series(fam)).sum().T
        grouped["baseline"] = c["bias_term"]
        return grouped

    def feature_importance(self) -> pd.Series:
        imp = self.point_.get_booster().get_score(importance_type="total_gain")
        s = pd.Series(imp, dtype=float).reindex(self.features).fillna(0)
        return (s / s.sum()).sort_values(ascending=False)


class Downscaler:
    """Bundle of per-variable models + metadata. Saved as one joblib file."""

    def __init__(self, models: dict[str, VariableDownscaler], meta: dict):
        self.models = models
        self.meta = meta

    def predict(self, df: pd.DataFrame, with_uncertainty: bool = True) -> pd.DataFrame:
        parts = [m.predict(df, with_uncertainty) for m in self.models.values()]
        out = pd.concat(parts, axis=1)
        # physical consistency: Tmin <= Tmax
        if "tmax_pred" in out and "tmin_pred" in out:
            bad = out["tmin_pred"] > out["tmax_pred"]
            mid = (out.loc[bad, "tmin_pred"] + out.loc[bad, "tmax_pred"]) / 2
            out.loc[bad, "tmin_pred"] = mid - 0.05
            out.loc[bad, "tmax_pred"] = mid + 0.05
        return out

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path: Path) -> Downscaler:
        return joblib.load(path)
