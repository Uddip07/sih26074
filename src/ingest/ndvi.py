"""
Vegetation state: MODIS 16-day NDVI (MOD13Q1 Terra + MYD13Q1 Aqua, 250 m, collection 6.1).

Accessed as Cloud-Optimised GeoTIFFs via the Microsoft Planetary Computer STAC
API (anonymous SAS tokens, no account needed).

* Pixels are masked with the MODIS ``pixel_reliability`` layer (0 good, 1 marginal kept;
  2 snow, 3 cloudy dropped).
* GP value = mean of valid pixels inside the GP polygon.
* Operational latency: a composite is only usable ``LATENCY_DAYS`` after its
  period ends (processing + publication delay). The dataset feature ``ndvi`` for an
  issue date is the latest composite *available* on that date, so there is no
  look-ahead.

Outputs: ``data/<d>/interim/gp_ndvi.parquet`` (time series) and ``gp_ndvi_static.parquet``
(GP mean NDVI over the archive, a static "greenness" covariate).
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.features import rasterize
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

from src.common import manifest
from src.common.config import Config, load_config
from src.common.logging_utils import get_logger
from src.ingest.boundaries import load_panchayats

log = get_logger("ingest.ndvi")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token/{account}/{container}"
COLLECTION = "modis-13Q1-061"
LATENCY_DAYS = 8
SOURCE = "MODIS MOD13Q1/MYD13Q1 v6.1 (Didan 2021) via Microsoft Planetary Computer"
LICENCE = "NASA EOSDIS data - no restrictions on reuse (cite LP DAAC)"


def _search(cfg: Config) -> list[dict]:
    """STAC search in 3-month windows (avoids relying on server-side pagination caps)."""
    start = pd.Timestamp(cfg["period"]["start"]) - pd.Timedelta(days=40)
    end = pd.Timestamp.today().normalize()
    items: dict[str, dict] = {}
    for w0 in pd.date_range(start, end, freq="3MS").union([start]):
        w1 = min(end, w0 + pd.DateOffset(months=3) - pd.Timedelta(days=1))
        body = {"collections": [COLLECTION], "bbox": list(cfg.bbox), "limit": 200,
                "datetime": f"{w0:%Y-%m-%d}/{w1:%Y-%m-%d}"}
        r = requests.post(STAC, json=body, timeout=120)
        r.raise_for_status()
        for f in r.json()["features"]:
            items[f["id"]] = f
    return list(items.values())


def build(cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    paths = cfg.paths.ensure()
    out = paths.interim / "gp_ndvi.parquet"
    gps = load_panchayats(cfg, modelled_only=True)
    tokens: dict[tuple[str, str], str] = {}

    def signed(href: str) -> str:
        # SAS tokens are issued per storage account/container (taken from the asset URL)
        account = href.split("//")[1].split(".")[0]
        container = href.split(".net/")[1].split("/")[0]
        if (account, container) not in tokens:
            tokens[(account, container)] = requests.get(SAS.format(account=account, container=container),
                                                        timeout=60).json()["token"]
        return href + "?" + tokens[(account, container)]

    items = _search(cfg)
    log.info("MODIS NDVI: %d tile-composites found", len(items))

    label_cache: dict[str, tuple] = {}

    def process(item: dict) -> list[dict] | None:
        nd = signed(item["assets"]["250m_16_days_NDVI"]["href"])
        rel = signed(item["assets"]["250m_16_days_pixel_reliability"]["href"])
        for attempt in range(4):
            try:
                with rasterio.open(nd) as s1, rasterio.open(rel) as s2:
                    b = transform_bounds("EPSG:4326", s1.crs, *cfg.bbox, densify_pts=41)
                    win = from_bounds(*b, transform=s1.transform).intersection(
                        rasterio.windows.Window(0, 0, s1.width, s1.height)).round_offsets().round_lengths()
                    a = s1.read(1, window=win).astype("float32")
                    q = s2.read(1, window=win)
                    t = s1.window_transform(win)
                    crs = s1.crs
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == 3:
                    log.warning("NDVI %s failed: %s", item["id"], exc)
                    return None
        tile = item["id"].split(".")[2]
        key = f"{tile}_{win.col_off}_{win.row_off}_{a.shape}"
        if key not in label_cache:
            gm = gps.to_crs(crs)
            label_cache[key] = rasterize(((g, k + 1) for k, g in enumerate(gm.geometry)), out_shape=a.shape,
                                         transform=t, fill=0, all_touched=True, dtype="int32")
        lab = label_cache[key]
        ok = (lab > 0) & (a > -3000) & np.isin(q, [0, 1])
        n = len(gps)
        s = np.bincount(lab[ok], weights=a[ok] * 1e-4, minlength=n + 1)[1:]
        c = np.bincount(lab[ok], minlength=n + 1)[1:]
        p = item["properties"]
        return [{"gp_code": gp, "start": p["start_datetime"][:10], "end": p["end_datetime"][:10],
                 "platform": item["id"][:3], "sum": float(ss), "n": int(cc)}
                for gp, ss, cc in zip(gps["gp_code"], s, c) if cc > 0]

    rows = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        for k, res in enumerate(ex.map(process, items), 1):
            if res:
                rows += res
            if k % 40 == 0:
                log.info("  composites processed %d/%d", k, len(items))
    if not rows:
        raise RuntimeError("no NDVI composites could be read")
    df = pd.DataFrame(rows)
    # A GP can straddle the h24/h25 tile edge: combine pixel sums across tiles
    df = df.groupby(["gp_code", "start", "end", "platform"], as_index=False)[["sum", "n"]].sum()
    df["ndvi"] = (df["sum"] / df["n"]).astype("float32")
    df["start"], df["end"] = pd.to_datetime(df["start"]), pd.to_datetime(df["end"])
    df["available_date"] = df["end"] + pd.Timedelta(days=LATENCY_DAYS)
    df = df[["gp_code", "platform", "start", "end", "available_date", "ndvi", "n"]].sort_values(["gp_code", "end"])
    df.to_parquet(out, index=False)
    static = df.groupby("gp_code")["ndvi"].agg(ndvi_mean="mean", ndvi_std="std").reset_index()
    static.to_parquet(paths.interim / "gp_ndvi_static.parquet", index=False)
    manifest.register(paths.data / "manifest.json", "gp_ndvi", out, SOURCE, LICENCE, "src.ingest.ndvi",
                      extra={"composites": int(df[["platform", "end"]].drop_duplicates().shape[0]),
                             "latency_days": LATENCY_DAYS})
    log.info("NDVI: %d GP-composites, %d GPs, NDVI %.2f..%.2f", len(df), df["gp_code"].nunique(),
             df["ndvi"].min(), df["ndvi"].max())
    return df


def ndvi_for_issue_dates(ndvi: pd.DataFrame, keys: pd.DataFrame) -> pd.Series:
    """Latest composite available on each row's issue_date (as-of join, no look-ahead)."""
    left = keys[["gp_code", "issue_date"]].reset_index()
    left["issue_date"] = left["issue_date"].astype("datetime64[ns]")
    left = left.sort_values("issue_date")
    right = ndvi[["gp_code", "available_date", "ndvi"]].copy()
    right["available_date"] = right["available_date"].astype("datetime64[ns]")
    right = right.sort_values("available_date")
    m = pd.merge_asof(left, right, left_on="issue_date", right_on="available_date", by="gp_code",
                      direction="backward", tolerance=pd.Timedelta(days=40))
    return m.set_index("index")["ndvi"].reindex(keys.index).astype("float32")


if __name__ == "__main__":
    build()
