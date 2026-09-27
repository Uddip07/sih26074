import json
from datetime import date

import pandas as pd
import pytest

from src.common.config import load_config
from src.pipeline.predict import BlockForecastError, validate_block_csv

CFG = load_config()
HAVE_BOUNDARIES = (CFG.paths.interim / "blocks.parquet").exists()
HAVE_EVAL = (CFG.paths.reports / "evaluation.json").exists()
HAVE_ISSUES = (CFG.paths.web / "issues" / "index.json").exists()
needs_boundaries = pytest.mark.skipif(not HAVE_BOUNDARIES, reason="run the boundaries stage first")


def _csv(**over):
    rows = []
    for name in ("BARAMATI", "MAVAL"):
        for k in range(1, 6):
            rows.append({"block_name": name, "lead_day": k, "rain": 5.0, "tmax": 31, "tmin": 21, "rh": 70, "wind": 12})
    df = pd.DataFrame(rows)
    for k, v in over.items():
        df.loc[0, k] = v
    return df


@needs_boundaries
def test_csv_validation_accepts_good():
    out = validate_block_csv(_csv(), CFG, date(2026, 9, 1))
    assert len(out) == 10 and (out["valid_date"] - out["issue_date"]).dt.days.tolist() == out["lead_day"].tolist()


@needs_boundaries
@pytest.mark.parametrize("over", [{"rain": -1}, {"rh": 130}, {"lead_day": 7}, {"block_name": "NOWHERE"},
                                  {"tmin": 40}])
def test_csv_validation_rejects_bad(over):
    with pytest.raises(BlockForecastError):
        validate_block_csv(_csv(**over), CFG, date(2026, 9, 1))


def test_config_complete():
    for key in ("district", "crs", "bbox", "period", "validation", "forecast", "ground_truth"):
        assert key in CFG.raw
    assert set(CFG.model["variables"]) == {"rain", "tmax", "tmin", "rh", "wind"}
    assert "latitude" in CFG.model["exclude_features"]


@needs_boundaries
def test_boundary_qa_consistent():
    qa = json.loads((CFG.paths.reports / "boundary_qa.json").read_text())
    assert qa["gram_panchayats"] == sum(qa["gps_per_block"].values())
    assert abs(qa["gp_plus_nongp_area_km2"] - qa["block_area_km2"]) / qa["block_area_km2"] < 0.01


@pytest.fixture(scope="module")
def client():
    if not HAVE_BOUNDARIES:
        pytest.skip("pipeline outputs absent")
    from fastapi.testclient import TestClient

    from src.dashboard.app import app

    return TestClient(app)


def test_api_health_and_meta(client):
    assert client.get("/api/health").json()["status"] == "ok"
    m = client.get("/api/meta").json()
    assert m["district"] == "Pune" and "severity_colors" in m


def test_api_search_and_locate(client):
    r = client.get("/api/search", params={"q": "PIMP"}).json()
    assert r and all("PIMP" in x["gp_name"] for x in r)
    loc = client.get("/api/locate", params={"lat": r[0]["lat"], "lon": r[0]["lon"]}).json()
    assert "gp_code" in loc


def test_api_write_endpoints_protected(client, monkeypatch):
    monkeypatch.setenv("AGROMET_ADMIN_TOKEN", "secret")
    r = client.post("/api/run?source=archive")
    assert r.status_code == 401


@pytest.mark.skipif(not HAVE_EVAL, reason="evaluation not generated")
def test_evaluation_json_is_strict_and_consistent():
    txt = (CFG.paths.reports / "evaluation.json").read_text()
    assert "NaN" not in txt and "Infinity" not in txt
    ev = json.loads(txt)
    rain = ev["variables"]["rain"]
    h = ev["headline"]["per_variable"]["rain"]
    assert h["rmse_model"] == pytest.approx(rain["unseen_gp"]["overall"]["pred"]["rmse"])
    assert ev["headline"]["text"] and "%" in ev["headline"]["text"]


@pytest.mark.skipif(not HAVE_ISSUES, reason="no forecast issues")
def test_api_issue_forecast_bulletin(client):
    iss = client.get("/api/issues").json()["issues"][0]
    payload = client.get(f"/api/issues/{iss}").json()
    code = next(iter(payload["gp"]))
    f = client.get("/api/forecast", params={"gp_code": code, "issue_date": iss}).json()
    assert len(f["days"]) == len(payload["leads"]) and set(payload["leads"]) >= {1, 2, 3, 4, 5}
    for lang in ("mr", "hi", "en"):
        b = client.get(f"/api/advisory/{code}", params={"issue_date": iss, "lang": lang}).json()
        assert len(b["sms"]) <= 160
    pdf = client.get(f"/api/bulletin/{code}.pdf", params={"issue_date": iss, "lang": "mr"})
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"


def test_no_hardcoded_headline_numbers():
    for p in ("src/dashboard/templates/index.html", "src/dashboard/static/js/dashboard.js", "src/dashboard/app.py"):
        s = open(p, encoding="utf-8").read()
        assert "37.5" not in s and "82.58" not in s and "82.6" not in s
