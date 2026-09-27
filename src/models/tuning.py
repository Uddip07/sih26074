"""
Hyper-parameter search (audit 4.1, feature F17).

Optuna TPE search over the spec section 7.2 grid:
n_estimators 200-800, max_depth 3-8, learning_rate 0.01-0.1, subsample 0.7-1.0,
colsample_bytree 0.7-1.0, min_child_weight 1-5.

Cross-validation uses **GroupKFold by gram panchayat** (spec 7.2), so validation GPs are
never seen in training. ``hist_bias`` is recomputed inside every fold with the
leakage-free encoder. Only training-period rows are used; the calibration and test
periods stay untouched.
"""

from __future__ import annotations

import json

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold

from src.common.logging_utils import get_logger
from src.features.bias_encoder import HistoricalBiasEncoder
from src.models.downscaler import _fwd, _inv

log = get_logger("models.tuning")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune(train: pd.DataFrame, var: str, target: str, forecast: str, features: list[str], space: str,
         base_params: dict, search: dict, n_trials: int, folds: int, sample_rows: int, seed: int = 42,
         lower=None, upper=None) -> dict:
    d = train[train[target].notna() & train[forecast].notna()]
    if len(d) > sample_rows:
        keep_gps = d["gp_code"].drop_duplicates().sample(frac=1.0, random_state=seed)
        d = d[d["gp_code"].isin(keep_gps)].sample(sample_rows, random_state=seed)
    y = d[target].to_numpy("float64")
    base = _fwd(d[forecast].to_numpy("float64"), space)
    r = _fwd(y, space) - base
    cols = [f for f in features if f != "hist_bias"]
    X0 = d[cols].astype("float32")

    gkf = GroupKFold(n_splits=folds)
    splits = list(gkf.split(X0, groups=d["gp_code"]))
    fold_bias = []
    for tr, va in splits:
        enc = HistoricalBiasEncoder()
        hb_tr = enc.fit_transform_oof(d.iloc[tr], r[tr])
        hb_va = enc.transform(d.iloc[va])
        fold_bias.append((hb_tr, hb_va))

    def objective(trial: optuna.Trial) -> float:
        p = dict(base_params)
        p.update({
            "n_estimators": trial.suggest_int("n_estimators", *search["n_estimators"], step=50),
            "max_depth": trial.suggest_int("max_depth", *search["max_depth"]),
            "learning_rate": trial.suggest_float("learning_rate", *search["learning_rate"], log=True),
            "subsample": trial.suggest_float("subsample", *search["subsample"]),
            "colsample_bytree": trial.suggest_float("colsample_bytree", *search["colsample_bytree"]),
            "min_child_weight": trial.suggest_int("min_child_weight", *search["min_child_weight"]),
        })
        errs = []
        for (tr, va), (hb_tr, hb_va) in zip(splits, fold_bias):
            Xtr = X0.iloc[tr].assign(hist_bias=hb_tr)[features]
            Xva = X0.iloc[va].assign(hist_bias=hb_va)[features]
            m = xgb.XGBRegressor(**p).fit(Xtr, r[tr], verbose=False)
            pred = _inv(base[va] + m.predict(Xva), space)
            if lower is not None or upper is not None:
                pred = np.clip(pred, lower, upper)
            errs.append(np.sqrt(np.mean((pred - y[va]) ** 2)))
        return float(np.mean(errs))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.enqueue_trial({k: base_params[k] for k in search if k in base_params})
    study.optimize(objective, n_trials=n_trials)
    best = dict(base_params)
    best.update(study.best_params)
    log.info("[%s] tuning: default RMSE %.4f -> best %.4f  %s", var, study.trials[0].value, study.best_value,
             json.dumps(study.best_params))
    return {"params": best, "best_rmse": study.best_value, "default_rmse": study.trials[0].value,
            "n_trials": len(study.trials), "rows": int(len(d)),
            "trials": [{"n": t.number, "rmse": t.value, **t.params} for t in study.trials]}
