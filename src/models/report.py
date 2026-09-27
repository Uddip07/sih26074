"""
Validation report: figures + self-contained HTML (feature F11, audit 8.6).

Every figure and number is generated from ``evaluation.json`` and the held-out prediction files,
so the report cannot drift from the evaluation. Output: ``outputs/<d>/reports/report.html``
plus ``figures/*.png``.

Colour roles (validated reference palette, three categorical slots max per chart):
downscaler = blue, naive block copy = orange, bias-corrected block = aqua. Observations are
drawn in neutral ink.
"""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import Config, load_config
from src.common.logging_utils import get_logger

log = get_logger("models.report")
C_MODEL, C_COPY, C_BC = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
VAR_LABEL = {"rain": "Rainfall (mm/day)", "tmax": "Max temperature (°C)", "tmin": "Min temperature (°C)",
             "rh": "Relative humidity (%)", "wind": "Max wind (km/h)"}


def _style(ax, title: str, ylabel: str = "", xlabel: str = ""):
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def _fig(figsize=(6.4, 3.6)):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize, dpi=140)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    return fig, ax


def _save(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path.name


def fig_rmse_by_lead(ev: dict, var: str, out: Path) -> str | None:
    d = ev["variables"][var].get("unseen_gp", {}).get("by_lead")
    if not d:
        return None
    leads = sorted(d, key=int)
    fig, ax = _fig()
    for key, col, lab in [("pred", C_MODEL, "Downscaler"), ("bl_block_copy", C_COPY, "Naive block copy"),
                          ("bl_block_bias_corrected", C_BC, "Block bias-corrected")]:
        y = [d[k][key]["rmse"] if key in d[k] else np.nan for k in leads]
        ax.plot([int(k) for k in leads], y, color=col, lw=2, marker="o", ms=5, label=lab)
        ax.annotate(lab, (int(leads[-1]), y[-1]), xytext=(6, 0), textcoords="offset points", fontsize=8,
                    color=INK, va="center")
    ax.set_xticks([int(k) for k in leads])
    ax.set_xlim(0.7, int(leads[-1]) + 1.6)
    _style(ax, f"{VAR_LABEL[var]}: RMSE by lead day (unseen GPs, test period)", "RMSE", "lead day")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    return _save(fig, out / f"rmse_by_lead_{var}.png")


def fig_scatter(cfg: Config, var: str, out: Path) -> str | None:
    p = cfg.paths.models / f"test_predictions_{var}.parquet"
    if not p.exists():
        return None
    spec = cfg.model["variables"][var]
    df = pd.read_parquet(p, columns=[spec["target"], f"{var}_pred", spec["forecast"], "unseen_gp"])
    df = df[df["unseen_gp"]].dropna()
    if len(df) > 200_000:
        df = df.sample(200_000, random_state=0)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), dpi=140, sharex=True, sharey=True)
    fig.patch.set_facecolor("#fcfcfb")
    y = df[spec["target"]]
    lim = np.nanpercentile(np.r_[y, df[f"{var}_pred"]], 99.7)
    lo = 0 if var in ("rain", "wind") else np.nanpercentile(y, 0.3)
    for ax, col, title in [(axes[0], spec["forecast"], "Naive block copy"), (axes[1], f"{var}_pred", "Downscaler")]:
        ax.set_facecolor("#fcfcfb")
        scale = "log" if var == "rain" else None
        ax.hexbin(y, df[col], gridsize=60, extent=(lo, lim, lo, lim), cmap="Blues", mincnt=1, bins=scale,
                  linewidths=0)
        ax.plot([lo, lim], [lo, lim], color=MUTED, lw=1, ls="--")
        _style(ax, title, "predicted", "observed")
    fig.suptitle(f"{VAR_LABEL[var]}: observed vs predicted, unseen GPs (colour = point density)",
                 x=0.02, ha="left", fontsize=10, color=INK)
    return _save(fig, out / f"scatter_{var}.png")


def fig_block_skill(ev: dict, cfg: Config, var: str, out: Path) -> str | None:
    d = ev["variables"][var].get("unseen_gp_by_block")
    if not d:
        return None
    names = pd.read_parquet(cfg.paths.interim / "blocks.parquet", columns=["block_lgd", "block_name"])
    nm = dict(zip(names["block_lgd"].astype(str), names["block_name"]))
    rows = sorted(((nm.get(k, k), v["pred"]["skill_vs_block_copy"]) for k, v in d.items()), key=lambda x: x[1])
    fig, ax = _fig((6.4, 0.32 * len(rows) + 1.2))
    vals = [r[1] * 100 for r in rows]
    ax.barh([r[0] for r in rows], vals, color=[C_MODEL if v >= 0 else C_COPY for v in vals], height=0.6)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:+.1f}%", (v, i), xytext=(4 if v >= 0 else -4, 0), textcoords="offset points",
                    ha="left" if v >= 0 else "right", va="center", fontsize=8, color=INK)
    ax.axvline(0, color=MUTED, lw=1)
    _style(ax, f"{VAR_LABEL[var]}: RMSE skill vs block copy by block (unseen GPs)", "", "skill score (%)")
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    return _save(fig, out / f"block_skill_{var}.png")


def fig_reliability(ev: dict, out: Path) -> str | None:
    pr = ev["variables"].get("rain", {}).get("probabilistic")
    if not pr:
        return None
    fig, ax = _fig((4.8, 4.2))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    for (t, d), col in zip(pr.items(), [C_MODEL, C_COPY, C_BC]):
        r = pd.DataFrame(d["reliability"])
        if r.empty:
            continue
        ax.plot(r["p_mean"], r["obs_freq"], color=col, lw=2, marker="o", ms=5,
                label=f"P(rain ≥ {t} mm)  BSS {d['bss']:.2f}" if d.get("bss") is not None else f"≥ {t} mm")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _style(ax, "Rain probability reliability (unseen GPs)", "observed frequency", "forecast probability")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    return _save(fig, out / "reliability_rain.png")


def fig_shap(ev: dict, var: str, out: Path) -> str | None:
    d = ev.get("training_summary", {}).get("variables", {}).get(var, {}).get("family_mean_abs_shap")
    if not d:
        return None
    s = pd.Series(d).sort_values()
    fig, ax = _fig((6.0, 0.35 * len(s) + 1.2))
    ax.barh(s.index, s.values, color=C_MODEL, height=0.6)
    _style(ax, f"{VAR_LABEL[var]}: what drives the GP correction (mean |SHAP| by family)", "",
           "mean |contribution| (model residual units)")
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    return _save(fig, out / f"shap_{var}.png")


def fig_lobo(ev: dict, cfg: Config, out: Path) -> str | None:
    lb = ev.get("leave_one_block_out", {}).get("rain")
    if not lb:
        return None
    names = pd.read_parquet(cfg.paths.interim / "blocks.parquet", columns=["block_lgd", "block_name"])
    nm = dict(zip(names["block_lgd"], names["block_name"]))
    f = pd.DataFrame(lb["folds"]).assign(name=lambda x: x["block_lgd"].map(nm)).sort_values("skill")
    fig, ax = _fig((6.4, 0.32 * len(f) + 1.2))
    v = f["skill"] * 100
    ax.barh(f["name"], v, color=[C_MODEL if x >= 0 else C_COPY for x in v], height=0.6)
    for i, x in enumerate(v):
        ax.annotate(f"{x:+.1f}%", (x, i), xytext=(4 if x >= 0 else -4, 0), textcoords="offset points",
                    ha="left" if x >= 0 else "right", va="center", fontsize=8, color=INK)
    ax.axvline(0, color=MUTED, lw=1)
    _style(ax, f"Rainfall, leave-one-block-out CV: pooled skill {lb['pooled_skill'] * 100:+.1f}%", "",
           "RMSE skill vs block copy (%)")
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    return _save(fig, out / "lobo_rain.png")


def fig_case_map(cfg: Config, out: Path) -> str | None:
    """Heaviest-rain test day: block forecast vs downscaled vs observed, same colour scale."""
    p = cfg.paths.models / "test_predictions_rain.parquet"
    if not p.exists():
        return None
    import geopandas as gpd
    import matplotlib.pyplot as plt

    df = pd.read_parquet(p, columns=["gp_code", "valid_date", "lead_day", "rain_obs", "rain_pred", "fc_rain"])
    d1 = df[df["lead_day"] == 1]
    day = d1.groupby("valid_date")["rain_obs"].mean().idxmax()
    s = d1[d1["valid_date"] == day]
    gps = gpd.read_parquet(cfg.paths.interim / "panchayats.parquet")[["gp_code", "geometry"]].merge(s, on="gp_code")
    blocks = gpd.read_parquet(cfg.paths.interim / "blocks.parquet")
    vmax = np.nanpercentile(np.r_[gps["rain_obs"], gps["rain_pred"], gps["fc_rain"]], 98)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), dpi=130)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, col, title in [(axes[0], "fc_rain", "Block forecast (input, ECMWF 0.25° -> block)"),
                           (axes[1], "rain_pred", "Downscaled panchayat forecast"),
                           (axes[2], "rain_obs", "Observed (CHIRPS 0.05°)")]:
        gps.plot(ax=ax, column=col, cmap="Blues", vmin=0, vmax=vmax, linewidth=0)
        blocks.boundary.plot(ax=ax, color=MUTED, linewidth=0.6)
        ax.set_title(title, fontsize=9.5, loc="left", color=INK)
        ax.set_axis_off()
    sm = plt.cm.ScalarMappable(cmap="Blues", norm=plt.Normalize(0, vmax))
    cb = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.01)
    cb.set_label("mm/day", color=MUTED, fontsize=8)
    fig.suptitle(f"Case study, lead day 1, valid {pd.Timestamp(day).date()} (wettest test day)", x=0.02,
                 ha="left", fontsize=11, color=INK)
    fig.savefig(out / "case_map.png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return "case_map.png"


def _img(path: Path) -> str:
    return f'<img alt="{html.escape(path.stem)}" src="data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}">'


def _table(rows: list[list], head: list[str]) -> str:
    h = "".join(f"<th>{html.escape(str(x))}</th>" for x in head)
    b = "".join("<tr>" + "".join(f"<td>{x}</td>" for x in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>"


def _f(x, pct=False, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x * 100:+.1f}%" if pct else f"{x:.{nd}f}"


def build(cfg: Config | None = None) -> Path:
    cfg = cfg or load_config()
    rep = cfg.paths.reports
    figs = rep / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    ev = json.loads((rep / "evaluation.json").read_text(encoding="utf-8"))
    parts = [f"<h1>{html.escape(cfg.district_name)}: block-to-panchayat downscaling validation</h1>",
             f"<p class='lead'>{html.escape(ev['headline']['text'])}</p>",
             "<h2>Validation design</h2><ul>" + "".join(f"<li><b>{k}</b>: {html.escape(str(v))}</li>"
                                                         for k, v in ev["validation_design"].items()) + "</ul>"]
    # headline table
    rows = []
    for v, d in ev["headline"].get("per_variable", {}).items():
        rows.append([VAR_LABEL[v], _f(d["rmse_block_copy"]), _f(d["rmse_model"]), _f(d["skill_vs_block_copy"], True),
                     f"{_f(d['ci'][0], True)} to {_f(d['ci'][1], True)}", _f(d["skill_vs_bias_corrected_block"], True)])
    parts.append("<h2>Headline (unseen panchayats × test period, all leads)</h2>" + _table(
        rows, ["Variable", "RMSE block copy", "RMSE downscaler", "Skill vs copy", "95% CI", "Skill vs bias-corrected block"]))
    pp = ev.get("perfect_prognosis", {}).get("variables", {})
    if pp:
        rows = []
        for v, d in pp.items():
            o = d["overall"]
            b = d["bootstrap_vs_block_copy"]
            rows.append([VAR_LABEL[v], _f(o["block_copy"]["rmse"]), _f(o["climatology_ratio"]["rmse"]),
                         _f(o["downscaler"]["rmse"]), _f(o["downscaler"]["skill_vs_block_copy"], True),
                         f"{_f(b['ci_low'], True)} to {_f(b['ci_high'], True)}",
                         _f(o["downscaler_mass_conserving"]["skill_vs_block_copy"], True),
                         _f(d["within_block_anomaly_correlation"], nd=2)])
        parts.append("<h2>Disaggregation skill (perfect prognosis: observed block value in)</h2>"
                     "<p>Isolates the problem statement - inferring panchayat values from the block value - from NWP "
                     "forecast error. Unseen GPs, test period.</p>" + _table(
                         rows, ["Variable", "RMSE block copy", "RMSE clim. ratio", "RMSE downscaler", "Skill", "95% CI",
                                "Skill, mass-conserving", "Within-block anomaly r"]))
    dl = ev.get("dl_ablation")
    if dl:
        rows = [[k, _f(dl[k]["rmse"]), _f(dl[k]["mae"]), _f(dl[k]["r"]), _f(dl[k].get("skill_vs_block_copy"), True)]
                for k in ("block_copy", "xgboost", "unet")]
        parts.append("<h2>Deep-learning ablation (U-Net on the 0.05° grid, perfect prognosis)</h2>"
                     f"<p>{html.escape(dl['description'])}; {dl['epochs_trained']} epochs.</p>" +
                     _table(rows, ["Method", "RMSE", "MAE", "r", "Skill vs copy"]))
    # all methods for rain
    for var, d in ev["variables"].items():
        ov = d.get("unseen_gp", {}).get("overall", {})
        rows = [[html.escape(d["methods"].get(k, k)), _f(m["rmse"]), _f(m["mae"]), _f(m["bias"]), _f(m["r"]),
                 _f(m.get("skill_vs_block_copy"), True)] for k, m in ov.items()]
        parts.append(f"<h2>{VAR_LABEL[var]}</h2>" + _table(rows, ["Method", "RMSE", "MAE", "Bias", "r", "Skill vs copy"]))
        for fn in (fig_rmse_by_lead(ev, var, figs), fig_scatter(cfg, var, figs), fig_block_skill(ev, cfg, var, figs),
                   fig_shap(ev, var, figs)):
            if fn:
                parts.append(_img(figs / fn))
        iv = d.get("intervals", {}).get("unseen_gp")
        if iv:
            parts.append(f"<p>P10–P90 interval coverage: <b>{iv['coverage_p10_p90'] * 100:.1f}%</b> "
                         f"(nominal 80%), mean width {iv['mean_width']:.2f}.</p>")
        if var == "rain":
            cat = d.get("categorical", {}).get("unseen_gp", {})
            rows = []
            for t, mm in cat.items():
                for k in ("pred", "bl_block_copy", "bl_block_bias_corrected"):
                    if k in mm:
                        c = mm[k]
                        rows.append([f"≥ {t} mm", html.escape(d["methods"][k]), _f(c["pod"], nd=2), _f(c["far"], nd=2),
                                     _f(c["csi"], nd=2), _f(c["ets"], nd=2), _f(c["freq_bias"], nd=2)])
            parts.append("<h3>Rain / no-rain and heavy-rain detection</h3>" +
                         _table(rows, ["Threshold", "Method", "POD", "FAR", "CSI", "ETS", "Freq. bias"]))
            for fn in (fig_reliability(ev, figs), fig_case_map(cfg, figs)):
                if fn:
                    parts.append(_img(figs / fn))
    lb = ev.get("leave_one_block_out", {})
    if lb:
        rows = [[VAR_LABEL[v], _f(r["pooled_skill"], True), _f(r["mean_skill_sample_weighted"], True),
                 _f(r["mean_skill_unweighted"], True), f"{r['folds_with_positive_skill']}/{r['n_folds']}"]
                for v, r in lb.items()]
        parts.append("<h2>Leave-one-block-out cross-validation</h2><p>Each block is held out entirely "
                     "(its historical bias comes only from spatial interpolation of other blocks).</p>" +
                     _table(rows, ["Variable", "Pooled", "Sample-weighted", "Unweighted mean", "Blocks improved"]))
        fn = fig_lobo(ev, cfg, figs)
        if fn:
            parts.append(_img(figs / fn))
    ic = ev.get("imd_independent_check", {})
    parts.append("<h2>Independent check vs IMD gauge-based grid</h2><pre>" +
                 html.escape(json.dumps(ic, indent=1)[:3000]) + "</pre>")
    parts.append("<h2>Ground-truth QA (block level, vs IMD gauges)</h2><pre>" +
                 html.escape(json.dumps(ev.get("truth_qa", {}), indent=1)[:4000]) + "</pre>")
    css = """body{font-family:system-ui,Segoe UI,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;
    color:#0b0b0b;background:#fcfcfb}h1{font-size:22px}h2{font-size:17px;margin-top:32px;border-bottom:1px solid #e4e3df}
    .lead{font-size:15px;background:#eef4fc;padding:12px;border-left:4px solid #2a78d6}
    table{border-collapse:collapse;font-size:13px;margin:8px 0}td,th{border:1px solid #e4e3df;padding:4px 8px;
    text-align:right}th{background:#f3f2ef}td:first-child,th:first-child{text-align:left}img{max-width:100%;
    display:block;margin:12px 0}pre{background:#f3f2ef;padding:8px;font-size:11px;overflow:auto}"""
    out = rep / "report.html"
    out.write_text(f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>Downscaling validation"
                   f"</title><style>{css}</style></head><body>{''.join(parts)}</body></html>", encoding="utf-8")
    log.info("report written: %s", out)
    return out


if __name__ == "__main__":
    build()
