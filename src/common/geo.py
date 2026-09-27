"""
Geospatial helpers shared by the ingest and feature stages.

The key operation is **polygon-to-grid area weighting**: every gridded product
(NWP forecast 0.25°, ERA5-Land 0.1°, CHIRPS 0.05°, IMD 0.25°) is converted to
polygon values (block or Gram Panchayat) by an area-weighted mean over the grid
cells the polygon overlaps. Areas are computed in the metric CRS (UTM 43N), so
the weights are true surface-area fractions, not degree-squared fractions.
"""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box


@dataclass(frozen=True)
class Grid:
    """A regular lat/lon grid described by its cell *centres*."""

    lats: np.ndarray  # ascending
    lons: np.ndarray  # ascending
    res: float

    @classmethod
    def covering(cls, bbox: tuple[float, float, float, float], res: float, origin: float = 0.0) -> Grid:
        """Grid aligned to ``origin + k*res`` centres that fully covers ``bbox``."""
        west, south, east, north = bbox
        def centres(lo: float, hi: float) -> np.ndarray:
            k0 = np.floor((lo - origin) / res + 0.5) - 1
            k1 = np.ceil((hi - origin) / res - 0.5) + 1
            return np.round(origin + np.arange(k0, k1 + 1) * res, 6)
        return cls(centres(south, north), centres(west, east), res)

    def cells(self) -> gpd.GeoDataFrame:
        h = self.res / 2
        rows = []
        for i, la in enumerate(self.lats):
            for j, lo in enumerate(self.lons):
                rows.append((i, j, float(la), float(lo), box(lo - h, la - h, lo + h, la + h)))
        return gpd.GeoDataFrame(
            pd.DataFrame(rows, columns=["i", "j", "lat", "lon", "geometry"]), geometry="geometry", crs="EPSG:4326"
        )


def area_weights(
    polygons: gpd.GeoDataFrame,
    id_col: str,
    grid: Grid,
    metric_crs: str,
) -> pd.DataFrame:
    """
    Fraction of each polygon's area falling in each grid cell.

    Returns columns ``[id_col, i, j, lat, lon, weight]``; weights of each polygon
    sum to 1 (cells outside the grid are ignored and the rest renormalised).
    """
    # Intersect in geographic coordinates (grid cells are exact lat/lon rectangles there), then
    # measure the pieces' true areas in the metric CRS. Intersecting after projection would bend
    # the cell edges and leave slivers where polygon and cell edges coincide.
    cells = grid.cells()
    polys = polygons[[id_col, "geometry"]].to_crs("EPSG:4326")
    inter = gpd.overlay(polys, cells, how="intersection", keep_geom_type=True)
    inter["a"] = inter.to_crs(metric_crs).geometry.area
    inter = inter[inter["a"] > 0]
    # Edges that coincide with cell edges leave numerical slivers after reprojection: drop pieces
    # below 1e-6 of the polygon's area, then renormalise.
    tot = inter.groupby(id_col)["a"].transform("sum")
    inter = inter[inter["a"] / tot >= 1e-6]
    tot = inter.groupby(id_col)["a"].transform("sum")
    inter["weight"] = inter["a"] / tot
    out = inter[[id_col, "i", "j", "lat", "lon", "weight"]].reset_index(drop=True)
    missing = set(polygons[id_col]) - set(out[id_col])
    if missing:
        raise ValueError(f"{len(missing)} polygons do not overlap the grid, e.g. {sorted(missing)[:5]}")
    return out


def apply_weights(values: np.ndarray, weights: pd.DataFrame, id_col: str) -> pd.Series:
    """
    Area-weighted polygon means of one 2-D field ``values[i, j]``.
    NaN cells are dropped and the remaining weights renormalised.
    """
    v = values[weights["i"].to_numpy(), weights["j"].to_numpy()]
    w = weights["weight"].to_numpy().copy()
    ok = np.isfinite(v)
    w[~ok] = 0.0
    df = pd.DataFrame({id_col: weights[id_col].to_numpy(), "vw": np.where(ok, v, 0.0) * w, "w": w})
    g = df.groupby(id_col, sort=False)[["vw", "w"]].sum()
    return (g["vw"] / g["w"].replace(0, np.nan)).rename("value")


def weight_matrix(weights: pd.DataFrame, id_col: str, ids: list, n_i: int, n_j: int):
    """
    Sparse (n_polygons x n_cells) matrix W so that ``W @ field.ravel()`` gives the
    polygon means for a whole stack of fields at once (fast for thousands of days).
    """
    from scipy.sparse import csr_matrix

    index = {pid: k for k, pid in enumerate(ids)}
    rows = weights[id_col].map(index).to_numpy()
    cols = weights["i"].to_numpy() * n_j + weights["j"].to_numpy()
    return csr_matrix((weights["weight"].to_numpy(), (rows, cols)), shape=(len(ids), n_i * n_j))


def stack_to_polygons(stack: np.ndarray, W) -> np.ndarray:
    """
    ``stack`` has shape (T, n_i, n_j) -> returns (T, n_polygons), NaN-aware
    (NaN cells get zero weight and the remaining weights are renormalised).
    """
    T = stack.shape[0]
    flat = stack.reshape(T, -1)
    ok = np.isfinite(flat)
    num = (W @ np.where(ok, flat, 0.0).T).T
    den = (W @ ok.T.astype(float)).T
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 1e-9, num / den, np.nan)
