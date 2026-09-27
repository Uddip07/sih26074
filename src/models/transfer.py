"""
Cross-district transfer test (feature F32): apply a model trained in district A to district B.

    python -m src.models.transfer --source config/district_pune.yaml --target config/district_satara.yaml

District B needs only its own data stages (boundaries ... dataset); no model is trained there. The
source district's operational model predicts every GP of district B over B's test period and is
scored against B's observations and B's naive block copy. This is the hardest generalisation test:
new blocks, new terrain, and a new rainfall regime (e.g. Mahabaleshwar ~6,000 mm/yr).
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from src.common.config import Config, load_config
from src.common.logging_utils import get_logger
from src.features.dataset import load_frame
from src.models import metrics as M
from src.models.downscaler import Downscaler

log = get_logger("models.transfer")


def run(source: Config, target: Config) -> dict:
    ds = Downscaler.load(source.paths.models / "downscaler_operational.joblib")
    df = load_frame(target)
    test0 = pd.Timestamp(target["validation"]["test_start"])
    df = df[df["valid_date"] >= test0]
    res: dict = {"source": source.key, "target": target.key, "period_start": str(test0.date()),
                 "n_gps": int(df["gp_code"].nunique()), "variables": {}}
    missing = [f for f in next(iter(ds.models.values())).features if f not in df.columns and f != "hist_bias"]
    if missing:
        raise ValueError(f"target district lacks features {missing}")
    for var, m in ds.models.items():
        d = df[df[m.target].notna() & df[m.forecast].notna()]
        p = m.predict(d, with_uncertainty=False)[f"{var}_pred"]
        mod, base = M.continuous(d[m.target], p), M.continuous(d[m.target], d[m.forecast])
        tmp = d[["valid_date", m.target, m.forecast]].assign(pred=p.to_numpy())
        res["variables"][var] = {"downscaler": mod, "block_copy": base,
                                 "skill_vs_block_copy": M.skill(mod["rmse"], base["rmse"]),
                                 "bootstrap": M.bootstrap_skill(tmp, m.target, "pred", m.forecast,
                                                                n_boot=int(target["validation"]["bootstrap_samples"]))}
        log.info("[transfer %s->%s %s] skill %.1f%%", source.key, target.key, var,
                 100 * res["variables"][var]["skill_vs_block_copy"])
    from src.models.evaluate import sanitize

    out = target.paths.reports / f"transfer_from_{source.key}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sanitize(res), indent=2), encoding="utf-8")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="district config the model was trained on")
    ap.add_argument("--target", required=True, help="district config to apply it to")
    a = ap.parse_args()
    run(load_config(a.source), load_config(a.target))
