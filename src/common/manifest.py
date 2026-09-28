"""
Data provenance manifest.

Every dataset the pipeline writes is registered here with its source, licence,
retrieval time, row count and SHA-256, so anyone can check where a number came
from and whether a file changed. The manifest lives at
``data/<district>/manifest.json`` and is rendered into ``docs/data.md``.
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def register(
    manifest_path: Path,
    dataset: str,
    path: Path,
    source: str,
    licence: str,
    produced_by: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Add/replace one dataset entry in the manifest."""
    entry = {
        "path": str(path.relative_to(manifest_path.parents[2])) if path.is_relative_to(manifest_path.parents[2]) else str(path),
        "source": source,
        "licence": licence,
        "produced_by": produced_by,
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "bytes": path.stat().st_size if path.exists() else None,
        "sha256": sha256_file(path) if path.exists() and path.is_file() else None,
    }
    if extra:
        entry.update(extra)
    with _LOCK:
        data: dict[str, Any] = {}
        if manifest_path.exists():
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        data[dataset] = entry
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return entry


def load(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))
