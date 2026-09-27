import re
from datetime import date

import numpy as np
import pandas as pd
import pytest
import yaml

from src.advisory import bulletin, disease_models, rules
from src.advisory.crop_calendar import _in_window, crops_for
from src.advisory.et0 import et0_fao56, extraterrestrial_radiation

GP = {"gp_code": "1", "gp_name": "TESTGAON", "block_name": "BARAMATI", "latitude": 18.2, "elev_mean": 540,
      "slope_mean": 1.5, "soil_clay_pct": 45.0, "tpi_2km": 0.0}
ISSUE = date(2026, 7, 10)


def fc(rain=(0, 0, 0, 0, 0), tmax=(30,) * 5, tmin=(22,) * 5, rh=(70,) * 5, wind=(10,) * 5, **extra):
    d = pd.DataFrame({"lead_day": [1, 2, 3, 4, 5], "rain_pred": rain, "tmax_pred": tmax, "tmin_pred": tmin,
                      "rh_pred": rh, "wind_pred": wind})
    for k, v in extra.items():
        d[k] = v
    return d


def rules_of(advs):
    return {a.rule: a for a in advs}


@pytest.mark.parametrize("mm,level", [(35.4, None), (35.5, "yellow"), (64.4, "yellow"), (64.5, "orange"),
                                      (115.5, "orange"), (115.6, "red")])
def test_heavy_rain_imd_boundaries(cfg, mm, level):
    a = rules_of(rules.evaluate(fc(rain=(mm, 0, 0, 0, 0)), GP, ISSUE, cfg, crops=[]))
    assert (a["heavy_rain"].severity if "heavy_rain" in a else None) == level


def test_heavy_rain_probability_triggers(cfg):
    a = rules_of(rules.evaluate(fc(rain=(20, 0, 0, 0, 0), rain_p_ge_64p5=[0.65, 0, 0, 0, 0]), GP, ISSUE, cfg, crops=[]))
    assert a["heavy_rain"].severity == "red"


def test_waterlogging_needs_flat_clay(cfg):
    wet = fc(rain=(30, 25, 10, 0, 0))
    assert "waterlogging" in rules_of(rules.evaluate(wet, GP, ISSUE, cfg, crops=[]))
    steep = dict(GP, slope_mean=12.0)
    assert "waterlogging" not in rules_of(rules.evaluate(wet, steep, ISSUE, cfg, crops=[]))


def test_three_day_rain_is_real_sum(cfg):
    a = rules_of(rules.evaluate(fc(rain=(1.0, 0.5, 0.5, 40, 40)), GP, ISSUE, cfg,
                                crops=crops_for("BARAMATI", date(2026, 7, 20), cfg)))
    assert a["dry_spell"].params["rain3"] == pytest.approx(2.0)


def test_valley_raises_frost_level(cfg):
    cold = fc(tmin=(6, 8, 9, 9, 9), tmax=(25,) * 5)
    base = rules_of(rules.evaluate(cold, GP, date(2026, 1, 5), cfg, crops=[]))
    valley = rules_of(rules.evaluate(cold, dict(GP, tpi_2km=-30), date(2026, 1, 5), cfg, crops=[]))
    lv = rules.LEVELS
    key = "cold" if "cold" in base else "frost"
    assert lv.index(valley[key].severity) == lv.index(base[key].severity) + 1


def test_spray_window_days(cfg):
    a = rules_of(rules.evaluate(fc(rain=(10, 0, 0, 5, 0), wind=(10, 25, 8, 8, 8)), GP, ISSUE, cfg, crops=[]))
    assert a["spray_window"].params["day_indices"] == [2, 4]


def test_crop_window_wraps_year():
    assert _in_window(date(2026, 1, 10), "12-01", "02-15")
    assert _in_window(date(2025, 12, 10), "12-01", "02-15")
    assert not _in_window(date(2026, 3, 1), "12-01", "02-15")


def test_crop_calendar_block_specific(cfg):
    keys = {c.crop_key for c in crops_for("MAVAL", date(2026, 7, 20), cfg)}
    assert "rice" in keys and "sugarcane" not in keys


def test_extraterrestrial_radiation_fao56_example8():
    # FAO-56 Example 8: 20°S, 3 September -> Ra = 32.2 MJ m-2 day-1
    assert extraterrestrial_radiation(np.array([-20.0]), np.array([246]))[0] == pytest.approx(32.2, abs=0.2)


def test_et0_plausible():
    e = et0_fao56([34, 28], [20, 23], [40, 90], [15, 10], 18.5, 560, [120, 200])
    assert 5 < e[0] < 9 and 1.5 < e[1] < 4.5 and e[0] > e[1]


def test_disease_models():
    r = disease_models.evaluate("grape_downy_mildew", [12, 3, 0, 0, 0], [28] * 5, [20] * 5, [90] * 5)
    assert r.level == 3
    r = disease_models.evaluate("onion_thrips", [0] * 5, [34] * 5, [20] * 5, [45] * 5)
    assert r.level == 3
    r = disease_models.evaluate("grape_downy_mildew", [0] * 5, [34] * 5, [20] * 5, [40] * 5)
    assert r.level == 0


def _placeholders(s: str) -> set:
    return set(re.findall(r"{(\w+)}", str(s)))


def test_i18n_catalogues_complete():
    cats = {lang: yaml.safe_load(open(f"src/advisory/i18n/{lang}.yaml", encoding="utf-8")) for lang in ("en", "mr", "hi")}
    for lang in ("mr", "hi"):
        missing = set(cats["en"]) - set(cats[lang])
        assert not missing, f"{lang} missing {missing}"
        for k, v in cats["en"].items():
            if isinstance(v, str):
                assert _placeholders(v) == _placeholders(cats[lang][k]), (lang, k)
        assert set(cats["en"]["diseases"]) == set(cats[lang]["diseases"])


@pytest.mark.parametrize("lang", ["mr", "hi", "en"])
def test_bulletin_sms_and_pdf(cfg, lang, tmp_path):
    f = fc(rain=(80, 30, 5, 0, 0), wind=(45, 20, 10, 8, 8), rain_q10=[30, 5, 0, 0, 0], rain_q90=[140, 60, 12, 2, 1],
           rain_p_ge_2p5=[1, 0.9, 0.5, 0.1, 0.05], fc_rain=[50, 20, 4, 0, 0])
    crops = crops_for("BARAMATI", date(2026, 9, 21), cfg)
    advs = rules.evaluate(f, GP, date(2026, 9, 20), cfg, crops=crops)
    b = bulletin.build(GP, f, advs, crops, date(2026, 9, 20), lang, cfg)
    assert len(b["sms"]) <= 160 and b["overall_severity"] in rules.LEVELS
    assert "{" not in bulletin.text(b)
    out = bulletin.pdf(b, tmp_path / f"b_{lang}.pdf")
    assert out.read_bytes()[:4] == b"%PDF" and out.stat().st_size > 5000
