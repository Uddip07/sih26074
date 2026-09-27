"""Crop calendar lookup: which crops are grown in a block and which stage they are in on a date (F8)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from src.common.config import Config, load_config


@dataclass(frozen=True)
class CropStage:
    crop_key: str
    crop_name: dict
    stage_name: dict
    stage_type: str
    kc: float
    diseases: tuple[str, ...]


def _in_window(d: date, start: str, end: str) -> bool:
    md = (d.month, d.day)
    s = tuple(int(x) for x in start.split("-"))
    e = tuple(int(x) for x in end.split("-"))
    return s <= md <= e if s <= e else (md >= s or md <= e)


def crops_for(block_name: str, on: date, cfg: Config | None = None, only: list[str] | None = None) -> list[CropStage]:
    cfg = cfg or load_config()
    out = []
    for key, c in cfg.crops.get("crops", {}).items():
        if only and key not in only:
            continue
        if block_name.upper() not in [b.upper() for b in c["blocks"]] and not only:
            continue
        for st in c["stages"]:
            if _in_window(on, st["start"], st["end"]):
                out.append(CropStage(key, c["name"], st["name"], st["type"], float(st.get("kc", 1.0)),
                                     tuple(c.get("diseases", []))))
                break
    return out


def all_crops(cfg: Config | None = None) -> dict[str, dict]:
    cfg = cfg or load_config()
    return {k: {"name": v["name"], "blocks": v["blocks"]} for k, v in cfg.crops.get("crops", {}).items()}
