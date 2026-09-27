"""
Weather-driven crop disease / pest risk models (feature F31).

All models work on the downscaled daily forecast of a Gram Panchayat (the validated advisory
window) and return a risk level (0 none, 1 low, 2 moderate, 3 high) with the triggering day(s).
The thresholds live in ``config/disease_models.yaml`` (editable by the plant-protection officer);
this module only interprets them. Criteria are simplified, daily-resolution forms of published
rules, suitable for advisory screening:

* grape_downy_mildew: *Plasmopara viticola* primary infection "3-10" rule
  (Baldacci 1947 / Magarey et al. 1991): mean T >= 10 °C, rain >= 10 mm over 24-48 h, and
  leaf wetness (proxied by RH >= 85 %). Pune's October-pruned vineyards are at highest risk
  in Oct-Nov rains.
* grape_powdery_mildew: Gubler-Thomas risk index, daily form: 3+ consecutive days with
  21 <= T <= 30 °C for most of the day (proxy: Tmin >= 15 and Tmax <= 32) and no heavy rain.
* onion_thrips: *Thrips tabaci* outbreaks in hot, dry weather (Tmax >= 30 °C, RH <= 60 %,
  no rain for 3+ days).
* onion_purple_blotch: *Alternaria porri*: RH >= 80 %, 21-30 °C, rain days.
* pomegranate_bacterial_blight: *Xanthomonas axonopodis* pv. *punicae* (oily spot):
  rain + warm (25-35 °C) + humid (RH >= 80 %).
* rice_blast: *Magnaporthe oryzae*: night T 20-26 °C, RH >= 90 %, drizzle/cloudy (rain 1-15 mm).
* potato_late_blight: *Phytophthora infestans*: Hutton-criteria style: Tmin >= 10 °C and
  RH >= 90 % on 2 consecutive days.
* soybean_rust: *Phakopsora pachyrhizi*: 18-28 °C with frequent rain/leaf wetness (RH >= 85 %).
* wheat_rust: leaf/stem rust, 15-25 °C with dew (RH >= 80 %) in Jan-Feb.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.common.config import Config, load_config

OPS = {"ge": np.greater_equal, "gt": np.greater, "le": np.less_equal, "lt": np.less}


@dataclass
class Risk:
    model: str
    level: int
    days: list[int]
    reason: dict


def _longest_run(mask: np.ndarray) -> tuple[int, list[int]]:
    """Length of the longest run of True and the day indices where a run of that length ends."""
    run, best = 0, 0
    for m in mask:
        run = run + 1 if m else 0
        best = max(best, run)
    return best, [int(i) for i in np.where(mask)[0]]


def _favourable(when: dict, series: dict[str, np.ndarray]) -> np.ndarray:
    ok = np.ones(len(series["rain"]), bool)
    for var, cond in when.items():
        if var not in series:
            raise KeyError(f"unknown disease-model variable '{var}' (use one of {sorted(series)})")
        for op, thr in cond.items():
            ok &= OPS[op](series[var], float(thr))
    return ok


def evaluate(model: str, rain, tmax, tmin, rh, cfg: Config | None = None) -> Risk:
    cfg = cfg or load_config()
    if model not in cfg.diseases:
        raise KeyError(f"disease model '{model}' is not defined in config/disease_models.yaml")
    rain, tmax, tmin, rh = (np.asarray(x, float) for x in (rain, tmax, tmin, rh))
    tmean = (tmax + tmin) / 2
    series = {"rain": rain, "rain_2day": rain + np.r_[0.0, rain[:-1]], "tmax": tmax, "tmin": tmin,
              "tmean": tmean, "rh": rh}
    level, days = 0, []
    for lv in cfg.diseases[model]["levels"]:
        fav = _favourable(lv["when"], series)
        run, idx = _longest_run(fav)
        fired = (run >= lv["consecutive"]) if "consecutive" in lv else (fav.sum() >= lv["min_days"])
        if fired:
            level, days = int(lv["level"]), idx
            break
    return Risk(model, level, days, {
        "rain_max": float(np.max(rain)), "tmean_range": [float(tmean.min()), float(tmean.max())],
        "rh_max": float(np.max(rh))})
