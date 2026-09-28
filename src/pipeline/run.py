"""
One-command pipeline runner (audit 7.1, features F24/F25).

    python -m src.pipeline.run                         # everything, skipping stages whose outputs exist
    python -m src.pipeline.run --from static           # rebuild from a stage onwards
    python -m src.pipeline.run --only train evaluate   # selected stages
    python -m src.pipeline.run --config config/district_<name>.yaml   # another district, no code changes
    python -m src.pipeline.run --only forecast                         # today's operational (live) forecast

Stage order (each stage's outputs are checked before running):
boundaries -> terrain -> landcover -> hydro -> soil -> ndvi -> chirps -> era5 -> imd -> nwp ->
static -> dataset -> train -> perfect_prognosis -> dl_ablation -> evaluate -> report -> forecast -> docs
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import dataclass

from src.common.config import Config, load_config
from src.common.logging_utils import get_logger

log = get_logger("pipeline")


@dataclass
class Stage:
    name: str
    outputs: Callable[[Config], list]
    run: Callable[[Config, argparse.Namespace], object]
    optional: bool = False  # failure is logged, pipeline continues: only for validation-only extras


def _it(c: Config, *names):
    return [c.paths.interim / n for n in names]


def _stages() -> list[Stage]:
    from src.ingest import (
        boundaries,
        chirps,
        era5land,
        hydro,
        imd_gridded,
        landcover,
        ndvi,
        nwp_forecast,
        soil,
        terrain,
    )

    def train(c, a):
        from src.models.train import run

        return run(c, do_tune=False if a.no_tune else None)

    def perfect_prognosis(c, a):
        from src.models.perfect_prog import run

        return run(c)

    def evaluate(c, a):
        from src.models.evaluate import run

        return run(c, with_lobo=not a.no_lobo)

    def report(c, a):
        from src.models.report import build

        return build(c)

    def forecast(c, a):
        from src.pipeline.predict import run

        return run(c, c.today(), "live")["meta"]

    def docs(c, a):
        from src.pipeline.docs import build

        return build(c)

    def static(c, a):
        from src.features.static import build_static

        return build_static(c)

    def dataset(c, a):
        from src.features.dataset import build_dataset

        return build_dataset(c)

    return [
        Stage("boundaries", lambda c: _it(c, "panchayats.parquet", "blocks.parquet"), lambda c, a: boundaries.build(c)),
        Stage("terrain", lambda c: _it(c, "gp_terrain.parquet"), lambda c, a: terrain.build(c)),
        Stage("landcover", lambda c: _it(c, "gp_landcover.parquet"), lambda c, a: landcover.build(c)),
        Stage("hydro", lambda c: _it(c, "gp_hydro.parquet"), lambda c, a: hydro.build(c)),
        Stage("soil", lambda c: _it(c, "gp_soil.parquet"), lambda c, a: soil.build(c)),
        Stage("ndvi", lambda c: _it(c, "gp_ndvi.parquet"), lambda c, a: ndvi.build(c)),
        Stage("chirps", lambda c: _it(c, "gp_rain_obs.parquet", "gp_climatology.parquet"), lambda c, a: chirps.build(c)),
        Stage("era5", lambda c: _it(c, "gp_met_obs.parquet"), lambda c, a: era5land.build(c)),
        Stage("imd", lambda c: _it(c, "block_rain_imd.parquet"), lambda c, a: imd_gridded.build(c), optional=True),
        Stage("nwp", lambda c: _it(c, "block_forecasts.parquet"), lambda c, a: nwp_forecast.fetch_archive(c)),
        Stage("static", lambda c: _it(c, "gp_static.parquet"), static),
        Stage("dataset", lambda c: [c.paths.processed / "dataset.parquet"], dataset),
        Stage("train", lambda c: [c.paths.models / "downscaler_eval.joblib",
                                  c.paths.models / "downscaler_operational.joblib"], train),
        Stage("perfect_prognosis", lambda c: [c.paths.reports / "perfect_prognosis.json"], perfect_prognosis),
        Stage("dl_ablation", lambda c: [c.paths.reports / "dl_ablation.json"],
              lambda c, a: __import__("src.models.dl_ablation", fromlist=["run"]).run(c, epochs=30), optional=True),
        Stage("evaluate", lambda c: [c.paths.reports / "evaluation.json"], evaluate),
        Stage("report", lambda c: [c.paths.reports / "report.html"], report),
        Stage("forecast", lambda c: [c.paths.forecasts / c.today().isoformat() / "meta.json"], forecast),
        Stage("docs", lambda c: [c.paths.root / "docs" / "data.md"], docs),
    ]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--from", dest="from_stage")
    ap.add_argument("--force", action="store_true", help="re-run even if outputs exist")
    ap.add_argument("--no-tune", action="store_true")
    ap.add_argument("--no-lobo", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    cfg.paths.ensure()
    stages = _stages()
    names = [s.name for s in stages]
    if a.only:
        bad = set(a.only) - set(names)
        if bad:
            raise SystemExit(f"unknown stages {bad}; valid: {names}")
        todo = [s for s in stages if s.name in a.only]
        force = True
    elif a.from_stage:
        todo = stages[names.index(a.from_stage):]
        force = True
    else:
        todo, force = stages, a.force
    timings = {}
    for s in todo:
        outs = s.outputs(cfg)
        if not force and outs and all(p.exists() for p in outs):
            log.info("[skip] %s (outputs present)", s.name)
            continue
        t0 = time.time()
        log.info("[run ] %s", s.name)
        try:
            s.run(cfg, a)
        except Exception as exc:
            if s.optional:
                log.warning("[warn] optional stage %s failed: %s", s.name, exc)
                continue
            raise
        timings[s.name] = round(time.time() - t0, 1)
        log.info("[done] %s in %.1fs", s.name, timings[s.name])
    (cfg.paths.reports / "pipeline_timings.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
