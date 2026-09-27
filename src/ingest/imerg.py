"""
Optional: NASA GPM IMERG Final daily V07 (0.1°) as a second satellite rainfall truth.

Runs only when NASA Earthdata credentials are available (``~/.netrc`` / ``_netrc`` with
``machine urs.earthdata.nasa.gov``, or the ``EARTHDATA_USERNAME`` / ``EARTHDATA_PASSWORD``
environment variables). Otherwise the stage is skipped with a clear message; CHIRPS (0.05°, no
login) is the primary rainfall truth.

Each daily file is subset to the district, **area-weighted** to GPs and blocks (the spec asks for
area weighting when a GP spans several pixels, not centroid sampling), and then deleted.
Outputs: ``gp_rain_imerg.parquet`` and ``block_rain_imerg.parquet``; the truth QA compares both
against the IMD gauges.
"""

from __future__ import annotations

import netrc
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

from src.common import manifest
from src.common.config import Config, load_config
from src.common.geo import Grid, area_weights, stack_to_polygons, weight_matrix
from src.common.logging_utils import get_logger
from src.ingest.boundaries import load_blocks, load_panchayats

log = get_logger("ingest.imerg")
URL = ("https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDF.07/{y}/{m:02d}/"
       "3B-DAY.MS.MRG.3IMERG.{y}{m:02d}{d:02d}-S000000-E235959.V07B.nc4")
HOST = "urs.earthdata.nasa.gov"


def credentials() -> tuple[str, str] | None:
    u, p = os.environ.get("EARTHDATA_USERNAME"), os.environ.get("EARTHDATA_PASSWORD")
    if u and p:
        return u, p
    for name in (".netrc", "_netrc"):
        f = Path.home() / name
        if f.exists():
            try:
                auth = netrc.netrc(f).authenticators(HOST)
                if auth:
                    return auth[0], auth[2]
            except (netrc.NetrcParseError, OSError):
                pass
    return None


class _Session(requests.Session):
    """Keep the Authorization header only for Earthdata redirects (NASA's documented pattern)."""

    def rebuild_auth(self, prepared_request, response):
        headers = prepared_request.headers
        url = prepared_request.url
        if "Authorization" in headers:
            orig = requests.utils.urlparse(response.request.url).hostname
            redir = requests.utils.urlparse(url).hostname
            if orig != redir and redir != HOST and orig != HOST:
                del headers["Authorization"]


def build(cfg: Config | None = None) -> pd.DataFrame | None:
    cfg = cfg or load_config()
    cred = credentials()
    if cred is None:
        log.warning("IMERG skipped: no Earthdata credentials (set EARTHDATA_USERNAME/PASSWORD or ~/.netrc)")
        return None
    s = _Session()
    s.auth = cred
    w, so, e, n = cfg.bbox
    tmp = cfg.paths.raw / "imerg"
    tmp.mkdir(parents=True, exist_ok=True)
    days = pd.date_range(cfg["period"]["start"], cfg["period"]["end"], freq="D")
    fields, lats, lons, got = [], None, None, []
    for d in days:
        f = tmp / f"{d:%Y%m%d}.nc4"
        try:
            if not f.exists():
                r = s.get(URL.format(y=d.year, m=d.month, d=d.day), timeout=180)
                r.raise_for_status()
                f.write_bytes(r.content)
            with xr.open_dataset(f) as ds:
                var = "precipitation" if "precipitation" in ds else "precipitationCal"
                da = ds[var].isel(time=0).sel(lon=slice(w, e), lat=slice(so, n))
                if da.dims[0] == "lon":
                    da = da.transpose("lat", "lon")
                fields.append(da.to_numpy().astype("float32"))
                lats, lons = da["lat"].to_numpy(), da["lon"].to_numpy()
            got.append(d)
        except Exception as exc:  # noqa: BLE001
            log.warning("IMERG %s failed: %s", d.date(), exc)
        finally:
            f.unlink(missing_ok=True)
    if not got:
        return None
    stack = np.stack(fields)
    grid = Grid(np.round(lats, 3), np.round(lons, 3), 0.1)
    out = []
    for polys, id_col in ((load_panchayats(cfg, modelled_only=True), "gp_code"), (load_blocks(cfg), "block_lgd")):
        wts = area_weights(polys, id_col, grid, cfg.metric_crs)
        ids = list(polys[id_col])
        W = weight_matrix(wts, id_col, ids, len(grid.lats), len(grid.lons))
        df = pd.DataFrame(stack_to_polygons(stack, W), index=pd.DatetimeIndex(got, name="date"), columns=ids)
        out.append(df.stack(future_stack=True).rename("rain_imerg").reset_index().rename(columns={"level_1": id_col}))
    out[0].to_parquet(cfg.paths.interim / "gp_rain_imerg.parquet", index=False)
    out[1].to_parquet(cfg.paths.interim / "block_rain_imerg.parquet", index=False)
    manifest.register(cfg.paths.data / "manifest.json", "gp_rain_imerg", cfg.paths.interim / "gp_rain_imerg.parquet",
                      "NASA GPM IMERG Final daily V07B (GES DISC)", "NASA open data", "src.ingest.imerg",
                      extra={"days": len(got)})
    return out[0]


if __name__ == "__main__":
    build()
