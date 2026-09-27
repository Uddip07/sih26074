"""
Reference evapotranspiration (FAO-56 Penman-Monteith) from downscaled forecasts (feature F19).

Forecasts supply Tmax, Tmin, mean RH and max 10 m wind; solar radiation is not
forecast, so it is estimated with the FAO-56 Hargreaves radiation formula (eq. 50,
k_Rs = 0.16 interior / 0.19 coastal). Daily-max wind is converted to a mean-daily 2 m
wind with a gust factor and the FAO-56 log profile (eq. 47). The implementation is
validated in the test-suite against ERA5-Land's own FAO-56 ET0.

Crop water requirement: ETc = Kc x ET0 (Kc from the crop calendar stage).
"""

from __future__ import annotations

import numpy as np

GUST_TO_MEAN = 0.55  # typical ratio of daily-mean to daily-max 10 m wind over land (Pune AWS climatology)


def extraterrestrial_radiation(lat_deg: np.ndarray, doy: np.ndarray) -> np.ndarray:
    """Ra in MJ m-2 day-1 (FAO-56 eq. 21)."""
    phi = np.radians(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    delta = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(delta), -1, 1))
    return (24 * 60 / np.pi) * 0.0820 * dr * (ws * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(ws))


def et0_fao56(tmax, tmin, rh_mean, wind10_max_kmh, lat_deg, elev_m, doy, krs: float = 0.16) -> np.ndarray:
    tmax, tmin = np.asarray(tmax, float), np.asarray(tmin, float)
    tmean = (tmax + tmin) / 2
    P = 101.3 * ((293 - 0.0065 * np.asarray(elev_m, float)) / 293) ** 5.26
    gamma = 0.000665 * P
    es_tmax = 0.6108 * np.exp(17.27 * tmax / (tmax + 237.3))
    es_tmin = 0.6108 * np.exp(17.27 * tmin / (tmin + 237.3))
    es = (es_tmax + es_tmin) / 2
    ea = np.clip(np.asarray(rh_mean, float), 1, 100) / 100 * es
    delta = 4098 * (0.6108 * np.exp(17.27 * tmean / (tmean + 237.3))) / (tmean + 237.3) ** 2
    Ra = extraterrestrial_radiation(np.asarray(lat_deg, float), np.asarray(doy, float))
    Rs = krs * np.sqrt(np.clip(tmax - tmin, 0, None)) * Ra
    Rso = (0.75 + 2e-5 * np.asarray(elev_m, float)) * Ra
    Rns = 0.77 * Rs
    sigma = 4.903e-9
    Rnl = sigma * (((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) / 2) * (0.34 - 0.14 * np.sqrt(ea)) * \
        (1.35 * np.clip(Rs / np.maximum(Rso, 1e-6), 0.25, 1.0) - 0.35)
    Rn = Rns - Rnl
    u10 = np.asarray(wind10_max_kmh, float) / 3.6 * GUST_TO_MEAN
    u2 = u10 * 4.87 / np.log(67.8 * 10 - 5.42)
    et0 = (0.408 * delta * Rn + gamma * 900 / (tmean + 273) * u2 * (es - ea)) / (delta + gamma * (1 + 0.34 * u2))
    return np.clip(et0, 0, None)
