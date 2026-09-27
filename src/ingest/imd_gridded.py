"""
IMD 0.25° gauge-based gridded daily rainfall (Pai et al. 2014). Used as an **independent** check.

IMD interpolates about 6,900 rain gauges onto a 0.25° grid. It is the Indian
reference rainfall dataset. It is **never used for training**; it serves two
purposes:

1. **Truth QA:** how well CHIRPS (our GP-scale label) agrees with IMD gauges
   over the district.
2. **Independent verification:** GP downscaled forecasts are aggregated (area-
   weighted) to IMD cells and scored against gauge-based rainfall that played
   no part in the modelling.

Download protocol: HTTP POST to imdpune.gov.in (the same protocol as the imdR
package, reimplemented here). The binary reader is ``src/ingest/imd_binary.py``.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.common import manifest
from src.common.config import Config, load_config
from src.common.geo import Grid, area_weights, stack_to_polygons, weight_matrix
from src.common.logging_utils import get_logger
from src.ingest.boundaries import load_blocks
from src.ingest.imd_binary import IMD_META, get_days_in_year, get_grid_coordinates, read_imd_binary

log = get_logger("ingest.imd")

ENDPOINTS = {
    "rain": ("https://imdpune.gov.in/cmpg/Griddata/rainfall.php",
             "https://imdpune.gov.in/cmpg/Griddata/Rainfall_25_Bin.html", "rain"),
}


def download_year(cfg: Config, year: int, variable: str = "rain") -> Path | None:
    dest = cfg.paths.shared_raw / "imd" / variable / f"{year}.grd"
    m = IMD_META[variable]
    expected = m["ncols"] * m["nrows"] * get_days_in_year(year) * 4
    if dest.exists() and dest.stat().st_size == expected:
        return dest
    url, referer, field = ENDPOINTS[variable]
    try:
        r = requests.post(url, data={field: str(year)}, timeout=600,
                          headers={"User-Agent": "Mozilla/5.0", "Referer": referer, "Origin": "https://imdpune.gov.in"})
    except requests.RequestException as exc:
        log.warning("IMD %s %s download failed: %s", variable, year, exc)
        return None
    if r.status_code != 200 or len(r.content) != expected:
        log.warning("IMD %s %s not available (HTTP %s, %d B, expected %d)", variable, year, r.status_code,
                    len(r.content), expected)
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(r.content)
    return dest


def build(cfg: Config | None = None) -> pd.DataFrame | None:
    cfg = cfg or load_config()
    paths = cfg.paths.ensure()
    y0 = date.fromisoformat(cfg["period"]["start"]).year
    y1 = date.fromisoformat(cfg["period"]["end"]).year
    lons, lats = get_grid_coordinates("rain")
    w_, s_, e_, n_ = cfg.bbox
    ri = np.where((lats >= s_ - 0.25) & (lats <= n_ + 0.25))[0]
    ci = np.where((lons >= w_ - 0.25) & (lons <= e_ + 0.25))[0]
    grid = Grid(lats[ri], lons[ci], 0.25)
    blocks = load_blocks(cfg)
    wb = area_weights(blocks, "block_lgd", grid, cfg.metric_crs)
    W = weight_matrix(wb, "block_lgd", list(blocks["block_lgd"]), len(grid.lats), len(grid.lons))

    cell_frames, block_frames, years = [], [], []
    for y in range(y0, y1 + 1):
        p = download_year(cfg, y)
        if p is None:
            continue
        arr = read_imd_binary(str(p), "rain", y)[:, ri][:, :, ci]
        days = pd.date_range(f"{y}-01-01", periods=arr.shape[0], freq="D")
        # IMD pads not-yet-analysed days of the current year with the missing value
        valid_day = np.isfinite(arr).any(axis=(1, 2))
        arr, days = arr[valid_day], days[valid_day]
        years.append(y)
        blk = stack_to_polygons(arr, W)
        bdf = pd.DataFrame(blk, index=pd.DatetimeIndex(days, name="date"), columns=list(blocks["block_lgd"]))
        block_frames.append(bdf.stack(future_stack=True).rename("rain_imd").reset_index()
                            .rename(columns={"level_1": "block_lgd"}))
        ii, jj = np.meshgrid(np.arange(len(grid.lats)), np.arange(len(grid.lons)), indexing="ij")
        for t, d in enumerate(days):
            cell_frames.append(pd.DataFrame({"date": d, "lat": grid.lats[ii.ravel()], "lon": grid.lons[jj.ravel()],
                                             "rain_imd": arr[t].ravel()}))
    if not years:
        log.warning("no IMD years available - independent verification will be skipped")
        return None
    period = (pd.Timestamp(cfg["period"]["start"]), pd.Timestamp(cfg["period"]["end"]))
    blk = pd.concat(block_frames)
    blk = blk[(blk["date"] >= period[0]) & (blk["date"] <= period[1])]
    cells = pd.concat(cell_frames).dropna()
    cells = cells[(cells["date"] >= period[0]) & (cells["date"] <= period[1])]
    blk.to_parquet(paths.interim / "block_rain_imd.parquet", index=False)
    cells.to_parquet(paths.interim / "imd_cells_rain.parquet", index=False)
    manifest.register(paths.data / "manifest.json", "imd_rain_025", paths.interim / "imd_cells_rain.parquet",
                      "IMD 0.25° gridded rainfall (Pai et al. 2014), imdpune.gov.in",
                      "IMD data policy - free for research/non-commercial use with acknowledgement",
                      "src.ingest.imd_gridded", extra={"years": years, "cell_days": int(len(cells))})
    log.info("IMD gridded rain: years %s, %d cell-days, %d block-days", years, len(cells), len(blk))
    return blk


if __name__ == "__main__":
    build()
