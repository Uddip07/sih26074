"""
Configuration loading.

A district is described entirely by ``config/district_<key>.yaml``. Model,
advisory and crop settings live in their own YAML files so that agromet
officers can edit thresholds without touching code.

The active district config is chosen by (highest priority first):
1. an explicit ``path`` argument,
2. the ``AGROMET_CONFIG`` environment variable,
3. ``active_district`` in ``config/settings.yaml``.

Every YAML is read strictly: a missing file or a missing key is an error, never a silent default.
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
SETTINGS_FILE = CONFIG_DIR / "settings.yaml"


class ConfigError(RuntimeError):
    pass


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"configuration file not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or not data:
        raise ConfigError(f"configuration file is empty or not a mapping: {path}")
    return data


def active_district_path() -> Path:
    """District config chosen by AGROMET_CONFIG, else ``active_district`` in config/settings.yaml."""
    p = os.environ.get("AGROMET_CONFIG") or _read_yaml(SETTINGS_FILE)["active_district"]
    return Path(p)


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
        path = Path(district_path) if district_path else active_district_path()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        self.path = path
        self.raw: dict[str, Any] = _read_yaml(path)
        self.model: dict[str, Any] = _read_yaml(CONFIG_DIR / "model.yaml")
        self.advisory: dict[str, Any] = _read_yaml(CONFIG_DIR / "advisory_rules.yaml")
        self.diseases: dict[str, Any] = _read_yaml(CONFIG_DIR / "disease_models.yaml")
        self.dashboard: dict[str, Any] = _read_yaml(CONFIG_DIR / "dashboard.yaml")
        self.crops: dict[str, Any] = _read_yaml(CONFIG_DIR / f"crop_calendar_{self.key}.yaml")
        self.paths = Paths(PROJECT_ROOT, self.key)

    # -- convenience accessors -------------------------------------------------
    def __getitem__(self, item: str) -> Any:
        return self.raw[item]

    def get(self, dotted: str) -> Any:
        """Dotted lookup (``"forecast.timezone"``); a missing key is an error."""
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                raise KeyError(f"'{dotted}' missing from {self.path.name}")
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
        return list(self.raw["operational"]["live_lead_days"])

    @property
    def timezone(self) -> str:
        return self.raw["forecast"]["timezone"]

    def today(self):
        """Current date in the district's timezone (not the server clock's)."""
        from datetime import datetime
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo(self.timezone)).date()

    def now(self):
        from datetime import datetime
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo(self.timezone))

    def resolve(self, relative: str) -> Path:
        p = Path(relative)
        return p if p.is_absolute() else PROJECT_ROOT / p


@lru_cache(maxsize=8)
def load_config(path: str | None = None) -> Config:
    return Config(path)
