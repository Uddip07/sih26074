"""
FastAPI server: forecast API, bulletins, officer workflow, dissemination simulator, dashboard.

Every number shown in the UI comes from pipeline outputs (``evaluation.json``, per-issue
forecasts). There are no hard-coded metrics (audit 1.9).

Security (audit 6.7):
* CORS is **off** by default; allowed origins come from ``AGROMET_CORS_ORIGINS`` (comma list),
  and credentials are never allowed with a wildcard.
* Write endpoints (run pipeline, review, publish, disseminate) require the header
  ``X-Admin-Token`` equal to ``AGROMET_ADMIN_TOKEN``. If that variable is unset (local dev),
  they are accepted only from 127.0.0.1 / ::1.
"""

from __future__ import annotations

import io
import json
import os
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.common.config import load_config
from src.common.logging_utils import get_logger
from src.dashboard.db import Store

log = get_logger("dashboard")
cfg = load_config()
HERE = Path(__file__).parent
STATIC = HERE / "static"
TEMPLATES = HERE / "templates"
store = Store(cfg.paths.outputs / "app.db")

app = FastAPI(title="Block-to-Panchayat Agromet Downscaling API", version="2.0.0",
              description="Panchayat-level forecasts (today + 7 days) downscaled from block forecasts, with GKMS-style "
                          "advisories in Marathi, Hindi and English.")
app.add_middleware(GZipMiddleware, minimum_size=1000)
_origins = [o.strip() for o in os.environ.get("AGROMET_CORS_ORIGINS", "").split(",") if o.strip()]
if _origins:
    app.add_middleware(CORSMiddleware, allow_origins=_origins, allow_credentials=False,
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type", "X-Admin-Token"])
app.mount("/static", StaticFiles(directory=STATIC), name="static")


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
def admin(request: Request, x_admin_token: str | None = Header(default=None)) -> str:
    token = os.environ.get("AGROMET_ADMIN_TOKEN")
    if token:
        if x_admin_token != token:
            raise HTTPException(401, "invalid or missing X-Admin-Token")
        return "officer"
    host = request.client.host if request.client else ""
    if host not in ("127.0.0.1", "::1", "localhost", "testclient"):
        raise HTTPException(403, "write endpoints are localhost-only unless AGROMET_ADMIN_TOKEN is set")
    return "officer(local)"


def _json_file(p: Path, what: str):
    if not p.exists():
        raise HTTPException(404, f"{what} not generated yet - run the pipeline (python -m src.pipeline.run)")
    return json.loads(p.read_text(encoding="utf-8"))


def _issues() -> list[str]:
    idx = cfg.paths.web / "issues" / "index.json"
    return json.loads(idx.read_text(encoding="utf-8"))["issues"] if idx.exists() else []


def _issue(issue_date: str | None) -> str:
    issues = _issues()
    if not issues:
        raise HTTPException(404, "no forecast issues available yet")
    if issue_date is None:
        return issues[0]
    if issue_date not in issues:
        raise HTTPException(404, f"issue {issue_date} not available; have {issues[:10]}")
    return issue_date


@lru_cache(maxsize=16)
def _gp_forecast(issue: str) -> pd.DataFrame:
    return pd.read_parquet(cfg.paths.forecasts / issue / "gp_forecast.parquet")


@lru_cache(maxsize=16)
def _bulletins(issue: str, lang: str) -> dict:
    return _json_file(cfg.paths.forecasts / issue / f"bulletins_{lang}.json", "bulletins")


@lru_cache(maxsize=16)
def _advisories(issue: str) -> dict:
    return _json_file(cfg.paths.forecasts / issue / "advisories.json", "advisories")


@lru_cache(maxsize=1)
def _gps() -> pd.DataFrame:
    import geopandas as gpd

    g = gpd.read_parquet(cfg.paths.interim / "panchayats.parquet")
    return g[g["modelled"]].reset_index(drop=True)


def _clear_caches() -> None:
    _gp_forecast.cache_clear()
    _bulletins.cache_clear()
    _advisories.cache_clear()


@app.on_event("startup")
def _start_live_refresh() -> None:
    """Issue today's live forecast (today + 7 days) once a day in the background.
    Disable with AGROMET_AUTO_REFRESH=0 (tests, read-only deployments)."""
    import sys

    if os.environ.get("AGROMET_AUTO_REFRESH", "1") == "0" or "pytest" in sys.modules:
        return
    from src.pipeline.scheduler import start_background

    start_background(cfg, on_issue=_clear_caches)


def _lang(lang: str) -> str:
    if lang not in cfg["operational"]["languages"]:
        raise HTTPException(400, f"lang must be one of {cfg['operational']['languages']}")
    return lang


def _clean(o):
    from src.models.evaluate import sanitize

    return sanitize(o)


# ---------------------------------------------------------------------------------------------
# pages
# ---------------------------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index():
    return HTMLResponse((TEMPLATES / "index.html").read_text(encoding="utf-8"))


@app.get("/sw.js", include_in_schema=False)
def service_worker():
    return FileResponse(STATIC / "js" / "sw.js", media_type="application/javascript")


@app.get("/manifest.webmanifest", include_in_schema=False)
def manifest():
    return FileResponse(STATIC / "manifest.webmanifest", media_type="application/manifest+json")


# ---------------------------------------------------------------------------------------------
# read API
# ---------------------------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok", "district": cfg.district_name, "issues": len(_issues()),
            "model_ready": (cfg.paths.models / "downscaler_operational.joblib").exists(),
            "auth": "token" if os.environ.get("AGROMET_ADMIN_TOKEN") else "localhost-only (dev)"}


@app.get("/api/meta")
def meta():
    """Everything the page needs to render: district, horizons, languages, map layers, colours."""
    ev_path = cfg.paths.reports / "evaluation.json"
    ev = json.loads(ev_path.read_text(encoding="utf-8")) if ev_path.exists() else {}
    bq = cfg.paths.reports / "boundary_qa.json"
    from src.advisory.crop_calendar import all_crops

    d = cfg["district"]
    return _clean({"district": cfg.district_name,
                   "district_names": {"en": cfg.district_name, **{k[5:]: v for k, v in d.items() if k.startswith("name_")}},
                   "state": d["state"], "languages": cfg["operational"]["languages"],
                   "default_language": cfg["operational"]["default_language"],
                   "lead_days": cfg.lead_days, "live_lead_days": cfg.live_lead_days,
                   "boundaries": json.loads(bq.read_text()) if bq.exists() else None,
                   "headline": ev.get("headline"), "forecast_model": cfg["forecast"]["model"],
                   "forecast_label": cfg["forecast"]["label"],
                   "truth": cfg["ground_truth"], "crops": all_crops(cfg),
                   "severity_colors": cfg.advisory["severity_colors"],
                   "dashboard": cfg.dashboard, "issues": _issues()})


@app.get("/api/evaluation")
def evaluation():
    return _json_file(cfg.paths.reports / "evaluation.json", "evaluation")


@app.get("/api/evaluation/summary")
def evaluation_summary():
    """What the Scorecard shows: forecast-mode skill on unseen GPs over the test period."""
    ev = _json_file(cfg.paths.reports / "evaluation.json", "evaluation")
    gt = cfg["ground_truth"]
    out = {"headline": ev["headline"], "validation_design": ev["validation_design"],
           "test_period": [cfg["validation"]["test_start"], cfg["period"]["end"]],
           "tested_leads": cfg.lead_days, "forecast_model": cfg["forecast"]["model"],
           "truth_labels": [gt["rainfall"]["label"], gt["temperature_humidity_wind"]["label"]],
           "variables": {}}
    for v, dv in ev["variables"].items():
        out["variables"][v] = {"by_lead": dv["unseen_gp"]["by_lead"]}
    return out


@app.get("/api/report", response_class=HTMLResponse)
def report():
    p = cfg.paths.reports / "report.html"
    if not p.exists():
        raise HTTPException(404, "report not generated yet - run: python -m src.pipeline.run --only report")
    return HTMLResponse(p.read_text(encoding="utf-8"))


@app.get("/api/verification")
def verification():
    return _json_file(cfg.paths.reports / "verification.json", "verification")


@app.get("/api/issues")
def issues():
    return {"issues": _issues()}


@app.get("/api/issues/{issue_date}")
def issue_payload(issue_date: str):
    p = cfg.paths.web / "issues" / f"{_issue(issue_date)}.json"
    return Response(p.read_bytes(), media_type="application/json")


@app.get("/api/geo/{layer}")
def geo(layer: str):
    if layer not in ("panchayats", "blocks", "non_gp_areas", "urban_areas", "ghat_crest"):
        raise HTTPException(404, "unknown layer")
    p = cfg.paths.web / "geo" / f"{layer}.geojson"
    if not p.exists():
        raise HTTPException(404, "geometry not published yet")
    return FileResponse(p, media_type="application/geo+json", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/forecast")
def forecast(gp_code: str, issue_date: str | None = None):
    issue = _issue(issue_date)
    df = _gp_forecast(issue)
    g = df[df["gp_code"] == gp_code].sort_values("lead_day")
    if g.empty:
        raise HTTPException(404, f"GP {gp_code} not in issue {issue}")
    rows = json.loads(g.drop(columns=["issue_date"]).to_json(orient="records", date_format="iso"))
    rev = store.get_review(issue, gp_code)
    return {"issue_date": issue, "gp_code": gp_code, "days": rows, "advisories": _advisories(issue).get(gp_code),
            "review": rev}


@app.get("/api/forecast/block/{block_lgd}")
def forecast_block(block_lgd: int, issue_date: str | None = None):
    issue = _issue(issue_date)
    df = _gp_forecast(issue)
    g = df[df["block_lgd"] == block_lgd]
    if g.empty:
        raise HTTPException(404, "block not found")
    agg = g.groupby("lead_day").agg(block_rain=("fc_rain", "first"), gp_rain_min=("rain_pred", "min"),
                                    gp_rain_mean=("rain_pred", "mean"), gp_rain_max=("rain_pred", "max"),
                                    block_tmax=("fc_tmax", "first"), gp_tmax_min=("tmax_pred", "min"),
                                    gp_tmax_max=("tmax_pred", "max")).reset_index()
    return {"issue_date": issue, "block_lgd": block_lgd, "n_gps": int(g["gp_code"].nunique()),
            "summary": _clean(agg.round(2).to_dict("records"))}


@app.get("/api/advisory/{gp_code}")
def advisory(gp_code: str, issue_date: str | None = None, lang: str = "mr"):
    issue = _issue(issue_date)
    b = _bulletins(issue, _lang(lang)).get(gp_code)
    if b is None:
        raise HTTPException(404, "GP not found")
    rev = store.get_review(issue, gp_code)
    b = dict(b)
    if rev.get("status") in ("approved", "rejected"):
        b["review"] = {"status": rev["status"], "officer": rev["officer"], "reviewed_utc": rev["updated_utc"],
                       "note": rev.get("note")}
        for k, v in (rev.get("edits") or {}).items():  # officer overrides of rendered text
            if k in ("summary", "sms"):
                b[k] = v
    return b


@app.get("/api/bulletin/{gp_code}.{fmt}")
def bulletin_file(gp_code: str, fmt: str, issue_date: str | None = None, lang: str = "mr"):
    b = advisory(gp_code, issue_date, lang)
    from src.advisory import bulletin as bl

    if fmt == "txt":
        return PlainTextResponse(bl.text(b))
    if fmt == "json":
        return JSONResponse(b)
    if fmt == "pdf":
        out = cfg.paths.bulletins / b["issue_date"] / f"{gp_code}_{lang}.pdf"
        bl.pdf(b, out)
        return FileResponse(out, media_type="application/pdf",
                            filename=f"agromet_{b['gp_name']}_{b['issue_date']}_{lang}.pdf")
    raise HTTPException(400, "fmt must be pdf|txt|json")


@app.get("/api/export/{issue_date}.csv")
def export_csv(issue_date: str, block_lgd: int | None = None):
    issue = _issue(issue_date)
    df = _gp_forecast(issue)
    if block_lgd:
        df = df[df["block_lgd"] == block_lgd]
    cols = [c for c in df.columns if not c.startswith("shap_")]
    names = _gps()[["gp_code", "gp_name", "block_name"]]
    df = names.merge(df[cols], on="gp_code")
    buf = io.StringIO()
    df.round(3).to_csv(buf, index=False)
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": f"attachment; filename=gp_forecast_{issue}.csv"})


@app.get("/api/locate")
def locate(lat: float = Query(..., ge=-90, le=90), lon: float = Query(..., ge=-180, le=180)):
    from shapely.geometry import Point

    g = _gps()
    hit = g[g.contains(Point(lon, lat))]
    if hit.empty:
        d = g.to_crs(cfg.metric_crs).distance(
            __import__("geopandas").GeoSeries([Point(lon, lat)], crs=4326).to_crs(cfg.metric_crs).iloc[0])
        k = int(d.idxmin())
        if d.iloc[k] > cfg.dashboard["locate_max_distance_m"]:
            raise HTTPException(404, "location is outside the modelled panchayats")
        hit = g.iloc[[k]]
    r = hit.iloc[0]
    return {"gp_code": r["gp_code"], "gp_name": r["gp_name"], "block_name": r["block_name"]}


@app.get("/api/search")
def search(q: str = Query(..., min_length=2), limit: int = 15):
    g = _gps()
    m = g[g["gp_name"].str.contains(q.strip().upper(), regex=False) | g["gp_code"].str.startswith(q.strip())]
    return [{"gp_code": r.gp_code, "gp_name": r.gp_name, "block_name": r.block_name,
             "lat": round(r.latitude, 4), "lon": round(r.longitude, 4)} for r in m.head(limit).itertuples()]


@app.get("/api/nowcast")
def nowcast():
    from src.pipeline.nowcast import nowcast as nc

    return nc(cfg)


@app.get("/api/ivr/{gp_code}", response_class=PlainTextResponse)
def ivr(gp_code: str, issue_date: str | None = None, lang: str = "mr"):
    b = advisory(gp_code, issue_date, lang)
    from src.pipeline.disseminate import ivr_script

    return ivr_script(b)


@app.get("/api/outbox")
def outbox(limit: int = 200):
    return store.outbox(limit)


@app.get("/api/audit")
def audit(limit: int = 200):
    return store.audit(limit)


@app.get("/api/review/{issue_date}")
def review_list(issue_date: str):
    issue = _issue(issue_date)
    return {"issue": store.issue_status(issue), "reviews": store.reviews(issue)}


# ---------------------------------------------------------------------------------------------
# write API (officer)
# ---------------------------------------------------------------------------------------------
class ReviewIn(BaseModel):
    status: str = Field(pattern="^(draft|approved|rejected)$")
    officer: str = Field(min_length=2, max_length=80)
    note: str = Field(default="", max_length=1000)
    edits: dict[str, str] = Field(default_factory=dict)


@app.post("/api/review/{issue_date}/{gp_code}")
def review(issue_date: str, gp_code: str, body: ReviewIn, actor: str = Depends(admin)):
    issue = _issue(issue_date)
    if gp_code not in _advisories(issue):
        raise HTTPException(404, "GP not in issue")
    return store.review(issue, gp_code, body.status, body.officer, body.note, body.edits)


class PublishIn(BaseModel):
    officer: str = Field(min_length=2, max_length=80)


@app.post("/api/review/{issue_date}/publish")
def publish(issue_date: str, body: PublishIn, actor: str = Depends(admin)):
    return store.publish(_issue(issue_date), body.officer)


class SubscribeIn(BaseModel):
    name: str = Field(default="", max_length=80)
    phone: str = Field(min_length=10, max_length=20)
    gp_code: str
    lang: str = "mr"
    channel: str = Field(default="sms", pattern="^(sms|whatsapp|ivr)$")


@app.post("/api/subscribe")
def subscribe(body: SubscribeIn):
    if body.gp_code not in set(_gps()["gp_code"]):
        raise HTTPException(404, "unknown GP")
    _lang(body.lang)
    try:
        return store.subscribe(body.name, body.phone, body.gp_code, body.lang, body.channel)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/disseminate/{issue_date}")
def disseminate(issue_date: str, actor: str = Depends(admin)):
    from src.pipeline.disseminate import simulate

    issue = _issue(issue_date)
    if store.issue_status(issue).get("status") != "published":
        raise HTTPException(409, "publish the issue (officer approval) before dissemination")
    return simulate(cfg, store, issue, lambda gp, lang: advisory(gp, issue, lang))


@app.post("/api/run")
def run_pipeline(source: str = Query("live", pattern="^(live|archive)$"), issue_date: str | None = None,
                 actor: str = Depends(admin)):
    from src.pipeline.predict import BlockForecastError, run

    d = date.fromisoformat(issue_date) if issue_date else cfg.today()
    try:
        res = run(cfg, d, source)
    except BlockForecastError as exc:
        raise HTTPException(400, str(exc)) from exc
    _clear_caches()
    store.log(actor, "run", {"issue_date": d.isoformat(), "source": source})
    return res["meta"]


@app.post("/api/downscale")
async def downscale_csv(file: UploadFile = File(...), issue_date: str = Form(...), actor: str = Depends(admin)):
    """Upload an official block forecast (CSV) and get panchayat forecasts + advisories for it."""
    from src.pipeline.predict import BlockForecastError, run

    raw = await file.read()
    if len(raw) > 2_000_000:
        raise HTTPException(413, "file too large")
    try:
        df = pd.read_csv(io.BytesIO(raw))
        res = run(cfg, date.fromisoformat(issue_date), "csv", df)
    except (BlockForecastError, ValueError, pd.errors.ParserError) as exc:
        raise HTTPException(400, f"invalid block forecast: {exc}") from exc
    _clear_caches()
    store.log(actor, "downscale_csv", {"issue_date": issue_date, "rows": int(len(df))})
    return res["meta"]


@app.get("/api/template/block_forecast.csv", response_class=PlainTextResponse)
def csv_template():
    from src.ingest.boundaries import load_blocks

    b = load_blocks(cfg, modelled_only=True)
    lines = ["block_lgd,block_name,lead_day,rain,tmax,tmin,rh,wind"]
    for r in b.itertuples():
        for k in cfg.lead_days:  # values left empty: the officer enters the official block forecast
            lines.append(f"{r.block_lgd},{r.block_name},{k},,,,,")
    return "\n".join(lines)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.dashboard.app:app", host=cfg.dashboard["server"]["host"], port=int(cfg.dashboard["server"]["port"]))
    _ = (np, timedelta)
