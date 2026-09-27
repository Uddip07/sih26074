"""
Weather-driven crop disease / pest risk models (feature F31).

All models work on the 5-day downscaled daily forecast of a Gram Panchayat, and return a
risk level (0 none, 1 low, 2 moderate, 3 high) with the triggering day(s). Criteria are
simplified, daily-resolution forms of published rules, suitable for advisory screening:

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


@dataclass
class Risk:
    model: str
    level: int
    days: list[int]
    reason: dict


def _consecutive(mask: np.ndarray, n: int) -> list[int]:
    run, out = 0, []
    for i, m in enumerate(mask):
        run = run + 1 if m else 0
        if run >= n:
            out.append(i)
    return out


def evaluate(model: str, rain, tmax, tmin, rh) -> Risk:
    rain, tmax, tmin, rh = (np.asarray(x, float) for x in (rain, tmax, tmin, rh))
    tmean = (tmax + tmin) / 2
    idx = np.arange(len(rain))
    if model == "grape_downy_mildew":
        rain2 = rain + np.r_[0, rain[:-1]]
        hit = (tmean >= 10) & (rain2 >= 10) & (rh >= 85)
        mod = (tmean >= 10) & (rain >= 2.5) & (rh >= 80)
        level = 3 if hit.any() else (2 if mod.any() else 0)
        days = idx[hit | mod].tolist()
    elif model == "grape_powdery_mildew":
        fav = (tmin >= 15) & (tmax <= 32) & (rain < 10)
        c3 = _consecutive(fav, 3)
        level = 3 if c3 else (1 if fav.sum() >= 2 else 0)
        days = c3 or idx[fav].tolist()
    elif model == "onion_thrips":
        fav = (tmax >= 30) & (rh <= 60) & (rain < 1)
        c3 = _consecutive(fav, 3)
        level = 3 if c3 else (2 if fav.sum() >= 2 else 0)
        days = idx[fav].tolist()
    elif model == "onion_purple_blotch":
        fav = (rh >= 80) & (tmean >= 21) & (tmean <= 30) & (rain >= 1)
        level = 3 if fav.sum() >= 2 else (2 if fav.any() else 0)
        days = idx[fav].tolist()
    elif model == "pomegranate_bacterial_blight":
        fav = (rain >= 2.5) & (tmean >= 25) & (tmean <= 35) & (rh >= 80)
        level = 3 if fav.sum() >= 2 else (2 if fav.any() else 0)
        days = idx[fav].tolist()
    elif model == "rice_blast":
        fav = (tmin >= 20) & (tmin <= 26) & (rh >= 90) & (rain >= 1) & (rain <= 15)
        level = 3 if fav.sum() >= 2 else (2 if fav.any() else 0)
        days = idx[fav].tolist()
    elif model == "potato_late_blight":
        fav = (tmin >= 10) & (rh >= 90)
        c2 = _consecutive(fav, 2)
        level = 3 if c2 else (1 if fav.any() else 0)
        days = c2 or idx[fav].tolist()
    elif model == "soybean_rust":
        fav = (tmean >= 18) & (tmean <= 28) & (rh >= 85) & (rain >= 1)
        level = 3 if fav.sum() >= 3 else (2 if fav.sum() >= 1 else 0)
        days = idx[fav].tolist()
    elif model == "wheat_rust":
        fav = (tmean >= 15) & (tmean <= 25) & (rh >= 80)
        level = 2 if fav.sum() >= 2 else 0
        days = idx[fav].tolist()
    else:
        raise KeyError(model)
    return Risk(model, int(level), [int(d) for d in days], {
        "rain_max": float(np.max(rain)), "tmean_range": [float(tmean.min()), float(tmean.max())],
        "rh_max": float(np.max(rh))})
