"""
Agromet advisory rule engine (spec section 9; features F7, F8, F19, F20, F21, F22, F31).

Input: the downscaled 5-day forecast of one Gram Panchayat (lead days 1-5) with point
values, uncertainty quantiles and rain exceedance probabilities, plus the GP's static
attributes (slope, clay, valley position, elevation) and the crops/stages in season.

Output: a list of language-neutral ``Advisory`` objects (message keys + parameters,
IMD colour severity). ``src/advisory/bulletin.py`` renders them in Marathi, Hindi or English.

Improvements over the old engine:
* thresholds live in ``config/advisory_rules.yaml`` (officer-editable), not in code;
* **probability-aware** (P(rain >= 64.5 mm), P(rain >= 2.5 mm) from the model);
* **terrain/soil-aware**: waterlogging needs flat, clayey land; frost risk is raised in valleys;
* **crop- and stage-aware** via the crop calendar (irrigation only for sensitive stages,
  harvest windows only at harvest stage, disease models only for crops in season);
* 3-day rain is the real sum of D+1..D+3 (the old code used ``rain x 2.5``);
* new rules: sowing window, spray window, fertiliser timing, heat, cold/frost,
  thunderstorm/lightning, harvest window, crop disease models, ET0-based irrigation amounts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.advisory import disease_models
from src.advisory.crop_calendar import CropStage, crops_for
from src.advisory.et0 import et0_fao56
from src.common.config import Config, load_config

LEVELS = ["green", "yellow", "orange", "red"]


@dataclass
class Advisory:
    rule: str
    severity: str
    category: str                     # general | crop | disease
    title_key: str
    text_key: str
    action_key: str | None
    params: dict = field(default_factory=dict)
    days: list[int] = field(default_factory=list)
    crop: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _raise(level: str, steps: int = 1) -> str:
    return LEVELS[min(len(LEVELS) - 1, LEVELS.index(level) + steps)]


def _col(fc: pd.DataFrame, name: str, default=np.nan) -> np.ndarray:
    return fc[name].to_numpy(dtype=float) if name in fc else np.full(len(fc), default)


def evaluate(fc: pd.DataFrame, gp: dict, issue: date, cfg: Config | None = None,
             crops: list[CropStage] | None = None, ante_obs7: float | None = None) -> list[Advisory]:
    """
    ``fc``: 5 rows (lead 1..5) with rain_pred, tmax_pred, tmin_pred, rh_pred, wind_pred and, if
    available, rain_q10/q90 and rain_p_ge_2p5 / _15p6 / _64p5.
    ``gp``: dict with block_name, latitude, elev_mean, slope_mean, soil_clay_pct, tpi_2km.
    """
    cfg = cfg or load_config()
    R = cfg.advisory["rules"]
    fc = fc.sort_values("lead_day").reset_index(drop=True)
    days = [issue + timedelta(days=int(k)) for k in fc["lead_day"]]
    rain = np.clip(_col(fc, "rain_pred"), 0, None)
    tmax, tmin = _col(fc, "tmax_pred"), _col(fc, "tmin_pred")
    rh, wind = _col(fc, "rh_pred"), _col(fc, "wind_pred")
    p25, p156, p645 = _col(fc, "rain_p_ge_2p5"), _col(fc, "rain_p_ge_15p6"), _col(fc, "rain_p_ge_64p5")
    rain3 = float(np.nansum(rain[:3]))
    rain2 = float(np.nansum(rain[:2]))
    crops = crops if crops is not None else crops_for(gp["block_name"], days[0], cfg)
    out: list[Advisory] = []

    def dparam(i: int) -> dict:
        return {"day_index": int(i), "date": days[int(i)].isoformat()}

    # 1 heavy rain (IMD categories + probability) ------------------------------------------------
    hr = R["heavy_rain"]
    worst, wi = "green", None
    for i in range(len(rain)):
        lvl = "green"
        if rain[i] >= hr["red_mm"] or (np.isfinite(p645[i]) and p645[i] >= hr["red_prob_64p5"]):
            lvl = "red"
        elif rain[i] >= hr["orange_mm"] or (np.isfinite(p645[i]) and p645[i] >= hr["orange_prob_64p5"]):
            lvl = "orange"
        elif rain[i] >= hr["yellow_mm"]:
            lvl = "yellow"
        if LEVELS.index(lvl) > LEVELS.index(worst):
            worst, wi = lvl, i
    if wi is not None:
        out.append(Advisory("heavy_rain", worst, "general", "heavy_rain_title", "heavy_rain_text", "heavy_rain_action",
                            {"rain": round(float(rain[wi]), 1),
                             "prob": int(round(100 * p645[wi])) if np.isfinite(p645[wi]) else "-", **dparam(wi)},
                            [int(wi)]))

    # 2 waterlogging (terrain + soil aware) -----------------------------------------------------------
    wl = R["waterlogging"]
    if rain3 >= wl["trigger_3day_mm"] and gp.get("slope_mean", 99) <= wl["flat_slope_deg"] \
            and gp.get("soil_clay_pct", 0) >= wl["clay_pct"]:
        out.append(Advisory("waterlogging", "orange" if rain3 >= 2 * wl["trigger_3day_mm"] else "yellow", "general",
                            "waterlogging_title", "waterlogging_text", "waterlogging_action",
                            {"rain3": round(rain3, 1), "slope": round(gp["slope_mean"], 1),
                             "clay": round(gp["soil_clay_pct"])}, [0, 1, 2]))

    # 3 crop water need (ET0 x Kc) and dry-spell irrigation ------------------------------------------------
    doy = np.array([d.timetuple().tm_yday for d in days])
    et0 = et0_fao56(tmax, tmin, rh, wind, gp.get("latitude", 18.5), gp.get("elev_mean", 600), doy)
    ds = R["dry_spell_irrigation"]
    p_dry_ok = (not np.isfinite(p25[:3]).any()) or np.nanmax(p25[:3]) <= ds["max_prob_2p5"]
    for c in crops:
        if rain3 < ds["rain_3day_mm"] and p_dry_ok and c.stage_type in ds["sensitive_stages"]:
            etc = float(np.nansum(et0[:3]) * c.kc)
            irr = max(0.0, etc - 0.8 * rain3)
            out.append(Advisory("dry_spell", "yellow", "crop", "dry_spell_title", "dry_spell_text", "dry_spell_action",
                                {"rain3": round(rain3, 1), "etc": round(etc, 1), "irr": int(round(irr)),
                                 "et0_3day": round(float(np.nansum(et0[:3])), 1)}, [0, 1, 2], c.crop_key))

    # 4 sowing window (kharif onset) ------------------------------------------------------------------------
    sw = R["sowing_window"]
    if days[0].month in sw["months"] and any(c.stage_type == "sowing" for c in crops):
        cum = float((ante_obs7 or 0.0) + rain3)
        key = "sowing_ok" if cum >= sw["cumulative_mm"] else "sowing_wait"
        out.append(Advisory("sowing_window", "green" if key == "sowing_ok" else "yellow", "general",
                            f"{key}_title", f"{key}_text", f"{key}_action", {"cum": int(round(cum))}, [0, 1, 2]))

    # 5 spray window ----------------------------------------------------------------------------------------
    sp = R["spray_window"]
    ok = (rain <= sp["max_rain_mm"]) & (wind <= sp["max_wind_kmh"]) & \
         (~np.isfinite(p25) | (p25 <= sp["max_prob_2p5"]))
    if ok.any():
        idx = [int(i) for i in np.where(ok)[0]]
        out.append(Advisory("spray_window", "green", "general", "spray_ok_title", "spray_ok_text", "spray_ok_action",
                            {"day_indices": idx, "dates": [days[i].isoformat() for i in idx]}, idx))
    else:
        out.append(Advisory("spray_window", "yellow", "general", "spray_no_title", "spray_no_text", "spray_no_action",
                            {}, list(range(len(rain)))))

    # 6 fertiliser timing ----------------------------------------------------------------------------------------
    if rain2 >= R["fertiliser"]["avoid_if_rain_48h_mm"]:
        out.append(Advisory("fertiliser", "yellow", "general", "fert_avoid_title", "fert_avoid_text", "fert_avoid_action",
                            {"rain2": round(rain2, 1)}, [0, 1]))

    # 7 heat --------------------------------------------------------------------------------------------------------
    H = R["heat"]
    i = int(np.nanargmax(tmax)) if np.isfinite(tmax).any() else None
    if i is not None and tmax[i] >= H["yellow_tmax"]:
        lvl = "red" if tmax[i] >= H["red_tmax"] else "orange" if tmax[i] >= H["orange_tmax"] else "yellow"
        out.append(Advisory("heat", lvl, "general", "heat_title", "heat_text", "heat_action",
                            {"tmax": round(float(tmax[i]), 1), **dparam(i)}, [i]))

    # 8 cold wave / frost (valley cold-air pooling raises the level) ----------------------------------------------
    C = R["cold"]
    i = int(np.nanargmin(tmin)) if np.isfinite(tmin).any() else None
    if i is not None and tmin[i] <= C["yellow_tmin"]:
        lvl = "red" if tmin[i] <= C["red_tmin"] else "orange" if tmin[i] <= C["orange_tmin"] else "yellow"
        if gp.get("tpi_2km", 0) <= C["valley_tpi_2km"]:
            lvl = _raise(lvl)
        key = "frost" if tmin[i] <= C["red_tmin"] + 2 else "cold"
        out.append(Advisory(key, lvl, "general", f"{key}_title", f"{key}_text", f"{key}_action",
                            {"tmin": round(float(tmin[i]), 1), **dparam(i)}, [i]))

    # 9 thunderstorm / lightning (pre- and post-monsoon convection) ------------------------------------------------
    T = R["thunderstorm"]
    if days[0].month in T["months"] and np.isfinite(p156).any():
        cand = np.where((p156 >= T["min_prob_15p6"]) & (tmax >= T["min_tmax"]) & (rh >= T["min_rh"]))[0]
        if len(cand):
            i = int(cand[np.argmax(p156[cand])])
            out.append(Advisory("thunderstorm", "orange" if p156[i] >= 0.6 else "yellow", "general", "thunder_title",
                                "thunder_text", "thunder_action", {"prob": int(round(100 * p156[i])), **dparam(i)},
                                [int(c) for c in cand]))

    # 10 wind / lodging (spec rule 5) -------------------------------------------------------------------------------
    W = R["wind"]
    i = int(np.nanargmax(wind)) if np.isfinite(wind).any() else None
    if i is not None and wind[i] >= W["yellow_kmh"]:
        lvl = "red" if wind[i] >= W["red_kmh"] else "orange" if wind[i] >= W["orange_kmh"] else "yellow"
        out.append(Advisory("wind", lvl, "general", "wind_title", "wind_text", "wind_action",
                            {"wind": int(round(float(wind[i]))), **dparam(i)}, [i]))

    # 11 generic pest/fungal risk (spec rule 3) ---------------------------------------------------------------------
    PF = R["pest_fungal_heat_humid"]
    m = (tmax > PF["tmax"]) & (rh > PF["rh"])
    if m.any():
        i = int(np.where(m)[0][0])
        out.append(Advisory("pest_fungal", "yellow", "general", "pest_fungal_title", "pest_fungal_text",
                            "pest_fungal_action", {"tmax": round(float(tmax[i]), 1), "rh": int(round(rh[i]))},
                            [int(x) for x in np.where(m)[0]]))

    # 12 harvest window ---------------------------------------------------------------------------------------------
    HV = R["harvest"]
    for c in crops:
        if c.stage_type != "harvest":
            continue
        dry = rain < HV["rain_mm"]
        run = 0
        best = 0
        for x in dry:
            run = run + 1 if x else 0
            best = max(best, run)
        if best >= HV["dry_days_needed"]:
            out.append(Advisory("harvest", "green", "crop", "harvest_ok_title", "harvest_ok_text", "harvest_ok_action",
                                {"n": int(best)}, [int(x) for x in np.where(dry)[0]], c.crop_key))
        elif (~dry).any():
            out.append(Advisory("harvest", "yellow", "crop", "harvest_no_title", "harvest_no_text", "harvest_no_action",
                                {}, [int(x) for x in np.where(~dry)[0]], c.crop_key))

    # 13 crop disease models ------------------------------------------------------------------------------------------
    for c in crops:
        for dm in c.diseases:
            r = disease_models.evaluate(dm, rain, tmax, tmin, rh)
            if r.level >= 2:
                out.append(Advisory(f"disease:{dm}", "orange" if r.level == 3 else "yellow", "disease",
                                    "disease_title", "disease_title", None,
                                    {"disease": dm, "level": r.level, **r.reason}, r.days, c.crop_key))
    return out


def overall(advisories: list[Advisory]) -> str:
    """IMD-style weather warning level: driven by weather hazards only. Crop-disease risk and farm
    operation windows (spray / harvest / sowing) are advice, not weather warnings, so they never raise
    the panchayat's colour on the map; they are shown with their own risk level in the advisory."""
    lv = [a.severity for a in advisories
          if a.category != "disease" and a.rule not in ("spray_window", "harvest", "sowing_window")]
    return max(lv, key=LEVELS.index) if lv else "green"
