"""
Deep-learning ablation (feature F34): a U-Net super-resolving the *block-mean* rainfall field to
the 0.05° CHIRPS grid, in perfect-prognosis mode (observed block value in), compared on equal
terms with the XGBoost downscaler.

Inputs per day (channels on the 0.05° grid covering the district):
  0  block field: every pixel carries its block's observed area-mean rain (log1p)
  1  CHPclim normal for the month (log1p), the classic spatial prior
  2  mean elevation (standardised), from the Copernicus DEM aggregated to 0.05°
  3  windward exposure proxy: west-east elevation gradient (standardised)
  4  block mask (1 inside modelled blocks)
  5-6 day-of-year sin/cos (broadcast)
Target: log1p(CHIRPS) - log1p(block field) (residual, as in spec 7.2).

Fairness: pixels dominated by the spatial hold-out GPs are **masked out of the loss**, the test
period (2026) is never used for training or early stopping (blocked calibration weeks are), and
predictions are aggregated to GPs with the same area weights before scoring against the same
unseen-GP x test-period rows.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import xarray as xr

from src.common.config import Config, load_config
from src.common.geo import area_weights, stack_to_polygons, weight_matrix
from src.common.logging_utils import get_logger
from src.ingest.boundaries import load_blocks, load_panchayats
from src.ingest.chirps import grid_of
from src.models import metrics as M
from src.models.train import calibration_days, holdout_gps

log = get_logger("models.dl")


def _inputs(cfg: Config):
    raw = cfg.paths.raw / "chirps"
    cube = xr.open_dataarray(next(raw.glob("chirps_final_*.nc")))
    clim = xr.open_dataarray(raw / "chpclim_monthly.nc").interp(lat=cube["lat"], lon=cube["lon"], method="nearest")
    grid = grid_of(cube)
    nI, nJ = len(grid.lats), len(grid.lons)
    blocks = load_blocks(cfg, modelled_only=True)
    wb = area_weights(blocks, "block_lgd", grid, cfg.metric_crs)
    # pixel -> dominant block (by area share of the pixel)
    wb["pix"] = wb["i"] * nJ + wb["j"]
    blocks_m = blocks.set_index("block_lgd")
    wb["area"] = wb["weight"] * wb["block_lgd"].map(blocks_m["area_km2"])
    dom = wb.sort_values("area", ascending=False).drop_duplicates("pix")
    pix_block = np.full(nI * nJ, -1)
    pix_block[dom["pix"].to_numpy()] = dom["block_lgd"].to_numpy()
    bobs = pd.read_parquet(cfg.paths.interim / "block_rain_obs.parquet")
    bw = bobs.pivot(index="date", columns="block_lgd", values="rain_obs").reindex(pd.DatetimeIndex(cube["time"].values))
    days = pd.DatetimeIndex(cube["time"].values)
    T = len(days)
    bfield = np.zeros((T, nI * nJ), dtype="float32")
    inside = pix_block >= 0
    col = {b: k for k, b in enumerate(bw.columns)}
    idx = np.array([col.get(b, -1) for b in pix_block[inside]])
    vals = bw.to_numpy()
    bfield[:, inside] = np.where(idx >= 0, vals[:, np.clip(idx, 0, None)], 0)
    bfield = np.nan_to_num(bfield.reshape(T, nI, nJ))
    # static: elevation at 0.05° from the GP-weighted terrain table is not gridded; use the DEM directly
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    elev = np.zeros((nI, nJ), dtype="float32")
    with rasterio.open(cfg.paths.raw / "dem" / "dem_utm30.tif") as src:
        from rasterio.transform import from_origin

        dst_t = from_origin(grid.lons[0] - grid.res / 2, grid.lats[-1] + grid.res / 2, grid.res, grid.res)
        tmp = np.zeros((nI, nJ), dtype="float32")
        reproject(rasterio.band(src, 1), tmp, dst_transform=dst_t, dst_crs="EPSG:4326", resampling=Resampling.average)
        elev = tmp[::-1]  # north-up -> ascending latitude rows
    grad = np.gradient(elev, axis=1)
    y = cube.to_numpy().astype("float32")
    return dict(days=days, y=y, bfield=bfield, clim=clim.to_numpy().astype("float32"), elev=elev, grad=grad,
                mask=inside.reshape(nI, nJ).astype("float32"), grid=grid, nI=nI, nJ=nJ)


def run(cfg: Config | None = None, epochs: int = 25, seed: int = 42) -> dict:
    import torch
    from torch import nn

    cfg = cfg or load_config()
    torch.manual_seed(seed)
    np.random.seed(seed)
    D = _inputs(cfg)
    days, nI, nJ = D["days"], D["nI"], D["nJ"]
    months = days.month.to_numpy()
    doy = days.dayofyear.to_numpy()
    ez = (D["elev"] - D["elev"].mean()) / (D["elev"].std() + 1e-6)
    gz = (D["grad"] - D["grad"].mean()) / (D["grad"].std() + 1e-6)

    def x_of(t_idx):
        b = np.log1p(D["bfield"][t_idx])
        c = np.log1p(D["clim"][months[t_idx] - 1] / 30.0)
        n = len(t_idx)
        chans = [b, c, np.broadcast_to(ez, (n, nI, nJ)), np.broadcast_to(gz, (n, nI, nJ)),
                 np.broadcast_to(D["mask"], (n, nI, nJ)),
                 np.broadcast_to(np.sin(2 * np.pi * doy[t_idx] / 365.25)[:, None, None], (n, nI, nJ)),
                 np.broadcast_to(np.cos(2 * np.pi * doy[t_idx] / 365.25)[:, None, None], (n, nI, nJ))]
        return np.stack(chans, 1).astype("float32")

    target = np.log1p(np.nan_to_num(D["y"])) - np.log1p(D["bfield"])
    valid = np.isfinite(D["y"]) & (D["mask"][None] > 0)

    # spatial hold-out pixels (dominated by unseen GPs) are excluded from the loss
    static = pd.read_parquet(cfg.paths.interim / "gp_static.parquet")
    static["gp_code"] = static["gp_code"].astype(str)
    hold = holdout_gps(static, float(cfg["validation"]["spatial_holdout_fraction"]), int(cfg["validation"]["random_seed"]))
    gps = load_panchayats(cfg, modelled_only=True)
    wg = area_weights(gps, "gp_code", D["grid"], cfg.metric_crs)
    wg["a"] = wg["weight"] * wg["gp_code"].map(gps.set_index("gp_code")["area_km2"])
    wg["held"] = wg["gp_code"].isin(hold)
    share = wg.groupby(["i", "j"]).apply(lambda g: g.loc[g["held"], "a"].sum() / g["a"].sum(), include_groups=False)
    hold_pix = np.zeros((nI, nJ), bool)
    for (i, j), s in share.items():
        hold_pix[i, j] = s > 0.5
    test0 = pd.Timestamp(cfg["validation"]["test_start"])
    cal, buf = calibration_days(pd.Series(days), test0)
    is_test = days >= test0
    is_cal = days.isin(list(cal))
    is_train = ~is_test & ~is_cal & ~days.isin(list(buf))

    # pad to a multiple of 8 for the U-Net
    pI, pJ = (-nI) % 8, (-nJ) % 8

    def pad(a):
        return np.pad(a, [(0, 0)] * (a.ndim - 2) + [(0, pI), (0, pJ)])

    class UNet(nn.Module):
        def __init__(self, c_in=7, w=32):
            super().__init__()

            def blk(a, b):
                return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.GELU(), nn.Conv2d(b, b, 3, padding=1), nn.GELU())
            self.e1, self.e2, self.e3 = blk(c_in, w), blk(w, 2 * w), blk(2 * w, 4 * w)
            self.pool = nn.MaxPool2d(2)
            self.u2, self.d2 = nn.ConvTranspose2d(4 * w, 2 * w, 2, 2), blk(4 * w, 2 * w)
            self.u1, self.d1 = nn.ConvTranspose2d(2 * w, w, 2, 2), blk(2 * w, w)
            self.out = nn.Conv2d(w, 1, 1)

        def forward(self, x):
            e1 = self.e1(x)
            e2 = self.e2(self.pool(e1))
            e3 = self.e3(self.pool(e2))
            d2 = self.d2(torch.cat([self.u2(e3), e2], 1))
            d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
            return self.out(d1)[:, 0]

    net = UNet()
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_mask = torch.tensor(pad((valid & ~hold_pix[None]).astype("float32")))
    tgt = torch.tensor(pad(np.nan_to_num(target).astype("float32")))
    tr_idx, ca_idx = np.where(is_train)[0], np.where(is_cal)[0]
    best, best_state, patience = np.inf, None, 0
    for ep in range(epochs):
        net.train()
        perm = np.random.permutation(tr_idx)
        for k in range(0, len(perm), 16):
            b = perm[k:k + 16]
            x = torch.tensor(pad(x_of(b)))
            m = loss_mask[b]
            loss = (((net(x) - tgt[b]) ** 2) * m).sum() / m.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = []
            for k in range(0, len(ca_idx), 32):
                b = ca_idx[k:k + 32]
                m = loss_mask[b]
                vl.append(float((((net(torch.tensor(pad(x_of(b)))) - tgt[b]) ** 2) * m).sum() / m.sum().clamp(min=1)))
        v = float(np.mean(vl))
        log.info("U-Net epoch %d: calib loss %.4f", ep + 1, v)
        if v < best - 1e-4:
            best, best_state, patience = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 4:
                break
    net.load_state_dict(best_state)
    net.eval()
    te_idx = np.where(is_test)[0]
    preds = []
    with torch.no_grad():
        for k in range(0, len(te_idx), 32):
            b = te_idx[k:k + 32]
            r = net(torch.tensor(pad(x_of(b)))).numpy()[:, :nI, :nJ]
            preds.append(np.clip(np.expm1(np.log1p(D["bfield"][b]) + r), 0, None))
    pf = np.concatenate(preds)
    ids = list(gps["gp_code"])
    W = weight_matrix(wg, "gp_code", ids, nI, nJ)
    gp_pred = stack_to_polygons(pf, W)
    dl = pd.DataFrame(gp_pred, index=pd.DatetimeIndex(days[te_idx], name="valid_date"), columns=ids)
    dl = dl.stack(future_stack=True).rename("rain_dl").reset_index().rename(columns={"level_1": "gp_code"})
    ref = pd.read_parquet(cfg.paths.models / "pp_test_predictions_rain.parquet")
    m = ref.merge(dl, on=["gp_code", "valid_date"], how="inner")
    res = {"description": "U-Net super-resolution of the block rain field (perfect prognosis), unseen GPs x test period",
           "epochs_trained": ep + 1, "n": int(len(m)),
           "unet": M.continuous(m["rain_obs"], m["rain_dl"]),
           "xgboost": M.continuous(m["rain_obs"], m["rain_pred"]),
           "block_copy": M.continuous(m["rain_obs"], m["fc_rain"])}
    for k in ("unet", "xgboost"):
        res[k]["skill_vs_block_copy"] = M.skill(res[k]["rmse"], res["block_copy"]["rmse"])
    from src.models.evaluate import sanitize

    (cfg.paths.reports / "dl_ablation.json").write_text(json.dumps(sanitize(res), indent=2), encoding="utf-8")
    log.info("DL ablation: U-Net skill %.1f%% vs XGBoost %.1f%% (block copy RMSE %.3f)",
             100 * res["unet"]["skill_vs_block_copy"], 100 * res["xgboost"]["skill_vs_block_copy"], res["block_copy"]["rmse"])
    return res


if __name__ == "__main__":
    run()
