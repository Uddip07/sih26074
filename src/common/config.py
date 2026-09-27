"""
Configuration loading.

A district is described entirely by ``config/district_<key>.yaml``. Model,
advisory and crop settings live in their own YAML files so that agromet
officers can edit thresholds without touching code.

The active district config is chosen by (highest priority first):
1. an explicit ``path`` argument,
2. the ``AGROMET_CONFIG`` environment variable,
3. ``config/district_pune.yaml``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
DEFAULT_DISTRICT_CONFIG = CONFIG_DIR / "district_pune.yaml"


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass(frozen=True)
class Paths:
    """All filesystem locations for one district. Nothing else builds paths."""

    root: Path
    key: str

    @property
    def data(self) -> Path:
        return self.root / "data" / self.key

    @property
    def raw(self) -> Path:
        return self.data / "raw"

    @property
    def interim(self) -> Path:
        return self.data / "interim"

    @property
    def processed(self) -> Path:
        return self.data / "processed"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs" / self.key

    @property
    def models(self) -> Path:
        return self.outputs / "models"

    @property
    def reports(self) -> Path:
        return self.outputs / "reports"

    @property
    def forecasts(self) -> Path:
        return self.outputs / "forecasts"

    @property
    def bulletins(self) -> Path:
        return self.outputs / "bulletins"

    @property
    def web(self) -> Path:
        return self.outputs / "web"

    @property
    def shared_raw(self) -> Path:
        """District-independent downloads (e.g. Natural Earth, global climatology)."""
        return self.root / "data" / "_shared"

    def ensure(self) -> Paths:
        for p in (self.raw, self.interim, self.processed, self.models, self.reports,
                  self.forecasts, self.bulletins, self.web, self.shared_raw):
            p.mkdir(parents=True, exist_ok=True)
        return self


class Config:
    """Read-only view over the district YAML plus the shared model/advisory YAMLs."""

    def __init__(self, district_path: Path | str | None = None):
        path = Path(district_path or os.environ.get("AGROMET_CONFIG") or DEFAULT_DISTRICT_CONFIG)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        self.path = path
        self.raw: dict[str, Any] = _read_yaml(path)
        self.model: dict[str, Any] = _read_yaml(CONFIG_DIR / "model.yaml")
        self.advisory: dict[str, Any] = _read_yaml(CONFIG_DIR / "advisory_rules.yaml")
        crop_file = CONFIG_DIR / f"crop_calendar_{self.key}.yaml"
        self.crops: dict[str, Any] = _read_yaml(crop_file) if crop_file.exists() else {}
        self.paths = Paths(PROJECT_ROOT, self.key)

    # -- convenience accessors -------------------------------------------------
    def __getitem__(self, item: str) -> Any:
        return self.raw[item]

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    @property
    def key(self) -> str:
        return self.raw["district"]["key"]

    @property
    def district_name(self) -> str:
        return self.raw["district"]["name"]

    @property
    def metric_crs(self) -> str:
        return self.raw["crs"]["metric"]

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        b = self.raw["bbox"]
        return (b["west"], b["south"], b["east"], b["north"])

    @property
    def lead_days(self) -> list[int]:
        return list(self.raw["forecast"]["lead_days"])

    @property
    def live_lead_days(self) -> list[int]:
        """Operational horizon (may extend beyond the validated ``lead_days``)."""
        return list(self.raw.get("operational", {}).get("live_lead_days", self.lead_days))

    def resolve(self, relative: str) -> Path:
        p = Path(relative)
        return p if p.is_absolute() else PROJECT_ROOT / p


@lru_cache(maxsize=8)
def load_config(path: str | None = None) -> Config:
    return Config(path)
