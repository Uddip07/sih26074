"""
GKMS-style Gram Panchayat bulletin in Marathi / Hindi / English (feature F9).

Structure follows the IMD-GKMS district/block agromet bulletin, so officers need no new
format: header -> 5-day weather table (with uncertainty) -> weather summary -> general
advisory -> crop-wise advisory (crop + current stage) -> pest & disease alerts -> footer.

Outputs: a structured dict (API/JSON), plain text, a prioritised SMS (<=160 characters),
and a PDF (fpdf2 with HarfBuzz text shaping, so Devanagari conjuncts render correctly).
Fonts are Noto Sans / Noto Sans Devanagari (SIL Open Font Licence, bundled in ``fonts/``).
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.advisory.crop_calendar import CropStage
from src.advisory.rules import LEVELS, Advisory, overall
from src.common.config import Config, load_config

I18N = Path(__file__).parent / "i18n"
FONTS = Path(__file__).parent / "fonts"
LANGS = ("mr", "hi", "en")


@lru_cache(maxsize=8)
def catalog(lang: str) -> dict:
    with open(I18N / f"{lang}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


class _Strict(dict):
    def __missing__(self, key):
        raise KeyError(f"i18n placeholder '{{{key}}}' has no value")


def t(lang: str, key: str, **kw) -> str:
    """Translated string; a key missing from the language catalogue is an error, never English text."""
    cat = catalog(lang)
    if key not in cat:
        raise KeyError(f"i18n key '{key}' missing from {lang}.yaml")
    return str(cat[key]).format_map(_Strict(kw))


def day_label(d: date, lang: str) -> str:
    return f"{catalog(lang)['day_names'][d.weekday()]} {d.day:02d}/{d.month:02d}"


def _day(issue: date, leads: list[int], i: int, lang: str) -> str:
    """Label of advisory day ``i`` (index into the advisory window's ``leads``)."""
    return day_label(issue + timedelta(days=int(leads[int(i)])), lang)


def _render_adv(a: Advisory, lang: str, issue: date, crops: dict[str, CropStage], leads: list[int]) -> dict:
    cat = catalog(lang)
    p = dict(a.params)
    if "day_index" in p:
        p["day"] = _day(issue, leads, p["day_index"], lang)
    if "day_indices" in p:
        p["days"] = ", ".join(_day(issue, leads, i, lang) for i in p["day_indices"])
    crop_name = crops[a.crop].crop_name[lang] if a.crop in crops else ""
    p["crop"] = crop_name
    if a.category == "disease":
        dz = cat["diseases"][p["disease"]]
        title = t(lang, "disease_title", crop=crop_name, disease=dz["name"],
                  level=cat["risk_levels"][int(p["level"])])
        return {"rule": a.rule, "severity": a.severity, "crop": a.crop, "title": title, "text": "",
                "action": dz["action"], "days": a.days}
    return {"rule": a.rule, "severity": a.severity, "crop": a.crop,
            "title": t(lang, a.title_key, **p), "text": t(lang, a.text_key, **p),
            "action": t(lang, a.action_key, **p) if a.action_key else "", "days": a.days}


def weather_summary(fc: pd.DataFrame, lang: str, cfg: Config) -> str:
    th = cfg.advisory["summary"]
    rain = fc["rain_pred"].to_numpy(float)
    total = round(float(np.nansum(rain)), 1)
    n = len(rain)
    n_rain = int((rain >= th["rain_day_mm"]).sum())
    if np.nanmax(rain) >= th["heavy_mm"]:
        s = t(lang, "summary_heavy", total=total, n=n)
    elif np.nanmax(rain) >= th["moderate_mm"]:
        s = t(lang, "summary_moderate", total=total, n_days=n_rain, n=n)
    elif n_rain:
        s = t(lang, "summary_light", total=total, n_days=n_rain, n=n)
    else:
        s = t(lang, "summary_dry", total=total, n=n)
    s += " " + t(lang, "summary_temp", tmax_lo=round(fc["tmax_pred"].min()), tmax_hi=round(fc["tmax_pred"].max()),
                 tmin_lo=round(fc["tmin_pred"].min()), tmin_hi=round(fc["tmin_pred"].max()))
    return s


def build(gp: dict, fc: pd.DataFrame, advisories: list[Advisory], crops: list[CropStage], issue: date,
          lang: str = "mr", cfg: Config | None = None, review: dict | None = None) -> dict:
    cfg = cfg or load_config()
    fc = fc.sort_values("lead_day").reset_index(drop=True)
    leads = [int(x) for x in fc["lead_day"]]
    crop_map = {c.crop_key: c for c in crops}
    rendered = [_render_adv(a, lang, issue, crop_map, leads) for a in advisories]
    sev = overall(advisories, cfg)
    colors = cfg.advisory["severity_colors"]
    table = []
    for _, r in fc.iterrows():
        d = issue + timedelta(days=int(r["lead_day"]))
        table.append({
            "date": d.isoformat(), "label": day_label(d, lang),
            "rain": round(float(r["rain_pred"]), 1),
            "rain_range": [round(float(r.get("rain_q10", np.nan)), 1), round(float(r.get("rain_q90", np.nan)), 1)],
            "rain_prob": int(round(100 * float(r.get("rain_p_ge_2p5", np.nan)))) if np.isfinite(
                r.get("rain_p_ge_2p5", np.nan)) else None,
            "tmax": round(float(r["tmax_pred"]), 1), "tmin": round(float(r["tmin_pred"]), 1),
            "rh": int(round(float(r["rh_pred"]))), "wind": int(round(float(r["wind_pred"]))),
            "block_rain": round(float(r.get("fc_rain", np.nan)), 1),
        })
    crop_lines = [t(lang, "crop_stage_line", crop=c.crop_name[lang], stage=c.stage_name[lang]) for c in crops]
    general = [r for r in rendered if r["crop"] is None and r["rule"] != "normal"]
    if not [a for a in advisories if a.severity != "green"]:
        general.insert(0, {"rule": "normal", "severity": "green", "crop": None, "title": t(lang, "normal_title"),
                           "text": t(lang, "normal_text"), "action": "", "days": []})
    return {
        "lang": lang, "gp_code": gp["gp_code"], "gp_name": gp["gp_name"], "block_name": gp["block_name"],
        "district": cfg.district_name if lang == "en" else cfg["district"][f"name_{lang}"],
        "issue_date": issue.isoformat(),
        "valid": [(issue + timedelta(days=leads[0])).isoformat(), (issue + timedelta(days=leads[-1])).isoformat()],
        "overall_severity": sev, "overall_color": colors[sev]["hex"],
        "overall_label": colors[sev][f"label_{lang}"],
        "table": table, "summary": weather_summary(fc, lang, cfg),
        "general": general,
        "crops": crop_lines,
        "crop_advisories": [r for r in rendered if r["crop"] is not None and not r["rule"].startswith("disease:")],
        "disease_alerts": [r for r in rendered if r["rule"].startswith("disease:")],
        "sms": sms(gp, fc, advisories, issue, lang, crop_map, cfg.advisory["sms"]["max_chars_unicode"]),
        "review": review or {"status": "auto", "note": t(lang, "not_reviewed")},
        "labels": {k: t(lang, k, n=len(fc)) for k in ("bulletin_title", "subtitle", "district", "block", "gp", "issued", "valid",
                                            "weather_forecast", "col_date", "col_rain", "col_rain_prob", "col_tmax",
                                            "col_tmin", "col_rh", "col_wind", "col_range", "overall_status",
                                            "weather_summary", "general_advisory", "crop_advisory",
                                            "disease_advisory", "footer", "reviewed_by")},
    }


def sms(gp: dict, fc: pd.DataFrame, advisories: list[Advisory], issue: date, lang: str,
        crops: dict[str, CropStage], limit: int) -> str:
    """Prioritised SMS: highest-severity items first, trimmed to ``limit`` characters."""
    head = t(lang, "sms_prefix", gp=gp["gp_name"].title() if lang == "en" else gp["gp_name"])
    leads = [int(x) for x in fc.sort_values("lead_day")["lead_day"]]
    items: list[tuple[int, str]] = []
    for a in advisories:
        p = dict(a.params)
        if "day_index" in p:
            p["day"] = _day(issue, leads, p["day_index"], lang)
        if "day_indices" in p:
            p["days"] = ",".join(_day(issue, leads, i, lang).split(" ")[0] for i in p["day_indices"][:3])
        key = {"heavy_rain": "sms_heavy_rain", "dry_spell": "sms_dry_spell", "heat": "sms_heat", "cold": "sms_cold",
               "frost": "sms_cold", "wind": "sms_wind", "thunderstorm": "sms_thunder"}.get(a.rule)
        if a.rule == "spray_window" and a.severity == "green":
            key = "sms_spray_ok"
        if a.rule.startswith("disease:") and a.crop in crops:
            key = "sms_disease"
            p["crop"] = crops[a.crop].crop_name[lang]
            p["disease"] = catalog(lang)["diseases"][p["disease"]]["name"].split(" (")[0]
        if key:
            items.append((LEVELS.index(a.severity), t(lang, key, **p)))
    items.sort(key=lambda x: -x[0])
    if not [i for i in items if i[0] > 0]:
        items.insert(0, (0, t(lang, "sms_normal", total=round(float(fc["rain_pred"].sum())), n=len(fc))))
    msg = head
    for _, s in items:
        if len(msg) + len(s) + 1 > limit:
            continue
        msg += s + " "
    return msg.strip()


def text(b: dict) -> str:
    L = b["labels"]
    lines = [f"{L['bulletin_title']} - {b['gp_name']}", f"{L['block']}: {b['block_name']} | {L['district']}: {b['district']}",
             f"{L['issued']}: {b['issue_date']} | {L['valid']}: {b['valid'][0]} - {b['valid'][1]}",
             f"{L['overall_status']}: {b['overall_label']}", "", L["weather_forecast"]]
    lines.append(f"{L['col_date']:<12}{L['col_rain']:>12}{L['col_tmax']:>16}{L['col_tmin']:>16}{L['col_rh']:>12}{L['col_wind']:>14}")
    for r in b["table"]:
        lines.append(f"{r['label']:<12}{r['rain']:>12}{r['tmax']:>16}{r['tmin']:>16}{r['rh']:>12}{r['wind']:>14}")
    lines += ["", f"{L['weather_summary']}: {b['summary']}", "", L["general_advisory"] + ":"]
    for a in b["general"]:
        lines.append(f" - {a['title']}: {a['text']} {a['action']}".rstrip())
    if b["crops"]:
        lines += ["", L["crop_advisory"] + ": " + "; ".join(b["crops"])]
        for a in b["crop_advisories"]:
            lines.append(f" - {a['title']}: {a['text']} {a['action']}".rstrip())
    if b["disease_alerts"]:
        lines += ["", L["disease_advisory"] + ":"]
        for a in b["disease_alerts"]:
            lines.append(f" - {a['title']}: {a['action']}")
    lines += ["", f"SMS: {b['sms']}", "", L["footer"]]
    return "\n".join(lines)


def pdf(b: dict, out: Path) -> Path:
    from fpdf import FPDF

    doc = FPDF(orientation="P", unit="mm", format="A4")
    doc.set_auto_page_break(True, margin=14)
    doc.add_font("deva", "", str(FONTS / "NotoSansDevanagari-Regular.ttf"))
    doc.add_font("deva", "B", str(FONTS / "NotoSansDevanagari-Bold.ttf"))
    doc.add_font("latin", "", str(FONTS / "NotoSans-Regular.ttf"))
    doc.add_font("latin", "B", str(FONTS / "NotoSans-Bold.ttf"))
    main = "latin" if b["lang"] == "en" else "deva"
    doc.set_fallback_fonts(["latin" if main == "deva" else "deva"])
    doc.set_text_shaping(True)
    doc.add_page()
    L = b["labels"]
    W = doc.w - doc.l_margin - doc.r_margin

    def hexrgb(h: str):
        h = h.lstrip("#")
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

    doc.set_fill_color(0, 60, 113)
    doc.set_text_color(255, 255, 255)
    doc.set_font(main, "B", 15)
    doc.cell(W, 10, f"{L['bulletin_title']}", new_x="LMARGIN", new_y="NEXT", fill=True, align="C")
    doc.set_font(main, "", 8.5)
    doc.cell(W, 6, L["subtitle"], new_x="LMARGIN", new_y="NEXT", fill=True, align="C")
    doc.set_text_color(0, 0, 0)
    doc.ln(2)
    doc.set_font(main, "B", 11)
    doc.cell(W, 7, f"{L['gp']}: {b['gp_name']}   |   {L['block']}: {b['block_name']}   |   {L['district']}: {b['district']}",
             new_x="LMARGIN", new_y="NEXT")
    doc.set_font(main, "", 9.5)
    doc.cell(W, 6, f"{L['issued']}: {b['issue_date']}    {L['valid']}: {b['valid'][0]} - {b['valid'][1]}",
             new_x="LMARGIN", new_y="NEXT")
    doc.set_fill_color(*hexrgb(b["overall_color"]))
    doc.set_text_color(255, 255, 255) if b["overall_severity"] in ("orange", "red", "green") else doc.set_text_color(0, 0, 0)
    doc.set_font(main, "B", 10.5)
    doc.cell(W, 8, f"{L['overall_status']}: {b['overall_label']}", new_x="LMARGIN", new_y="NEXT", fill=True, align="C")
    doc.set_text_color(0, 0, 0)
    doc.ln(2)

    doc.set_font(main, "B", 11)
    doc.cell(W, 7, L["weather_forecast"], new_x="LMARGIN", new_y="NEXT")
    heads = [L["col_date"], L["col_rain"], L["col_range"], L["col_rain_prob"], L["col_tmax"], L["col_tmin"],
             L["col_rh"], L["col_wind"]]
    widths = [22, 20, 26, 22, 24, 24, 20, W - 158]
    doc.set_font(main, "B", 7.5)
    doc.set_fill_color(230, 236, 243)
    for h, w in zip(heads, widths):
        doc.cell(w, 9, h, border=1, align="C", fill=True)
    doc.ln()
    doc.set_font(main, "", 9)
    for r in b["table"]:
        rr = r["rain_range"]
        vals = [r["label"], f"{r['rain']}", f"{rr[0]}-{rr[1]}" if rr[0] == rr[0] else "-",
                f"{r['rain_prob']}%" if r["rain_prob"] is not None else "-", f"{r['tmax']}", f"{r['tmin']}",
                f"{r['rh']}", f"{r['wind']}"]
        for v, w in zip(vals, widths):
            doc.cell(w, 7, v, border=1, align="C")
        doc.ln()
    doc.ln(2)

    def section(title: str, items: list[dict], show_text: bool = True):
        if not items:
            return
        if title:
            doc.set_font(main, "B", 11)
            doc.cell(W, 7, title, new_x="LMARGIN", new_y="NEXT")
        for a in items:
            col = hexrgb({"green": "#2E7D32", "yellow": "#C9A000", "orange": "#E8720C", "red": "#9B2423"}[a["severity"]])
            doc.set_fill_color(*col)
            y = doc.get_y()
            doc.rect(doc.l_margin, y + 1, 2.2, 5, style="F")
            doc.set_x(doc.l_margin + 4)
            doc.set_font(main, "B", 9.5)
            doc.multi_cell(W - 4, 5.5, a["title"], new_x="LMARGIN", new_y="NEXT")
            doc.set_x(doc.l_margin + 4)
            doc.set_font(main, "", 9)
            body = (a["text"] + " " if show_text and a["text"] else "") + a["action"]
            doc.multi_cell(W - 4, 5, body.strip(), new_x="LMARGIN", new_y="NEXT")
            doc.ln(1)

    doc.set_font(main, "B", 11)
    doc.cell(W, 7, L["weather_summary"], new_x="LMARGIN", new_y="NEXT")
    doc.set_font(main, "", 9.5)
    doc.multi_cell(W, 5.5, b["summary"], new_x="LMARGIN", new_y="NEXT")
    doc.ln(1)
    section(L["general_advisory"], b["general"])
    if b["crops"]:
        doc.set_font(main, "B", 11)
        doc.cell(W, 7, L["crop_advisory"], new_x="LMARGIN", new_y="NEXT")
        doc.set_font(main, "", 9)
        doc.multi_cell(W, 5, "; ".join(b["crops"]), new_x="LMARGIN", new_y="NEXT")
        doc.ln(1)
        section("", b["crop_advisories"])
    else:
        section(L["crop_advisory"], b["crop_advisories"])
    section(L["disease_advisory"], b["disease_alerts"], show_text=False)
    doc.ln(2)
    doc.set_font(main, "", 8)
    doc.set_text_color(90, 90, 90)
    rv = b.get("review", {})
    doc.multi_cell(W, 4.5, (f"{L['reviewed_by']}: {rv.get('officer')} ({rv.get('reviewed_utc', '')})"
                            if rv.get("status") == "approved" else rv.get("note", "")), new_x="LMARGIN", new_y="NEXT")
    doc.multi_cell(W, 4.5, L["footer"], new_x="LMARGIN", new_y="NEXT")
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.output(str(out))
    return out
