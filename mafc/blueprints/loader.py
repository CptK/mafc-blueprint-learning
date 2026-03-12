from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from mafc.blueprints.models import Blueprint

SUPPORTED_BLUEPRINT_EXTENSIONS = {".yaml", ".yml", ".json"}


def load_blueprint(path: str | Path) -> Blueprint:
    """Load and validate a single blueprint file from disk."""
    blueprint_path = Path(path)
    if not blueprint_path.is_file():
        raise FileNotFoundError(f"Blueprint file not found: {blueprint_path}")

    payload = _read_blueprint_payload(blueprint_path)
    return Blueprint.model_validate(payload)


def load_blueprints(path: str | Path) -> list[Blueprint]:
    """Load one blueprint file or every supported blueprint file in a directory tree."""
    blueprint_path = Path(path)
    if blueprint_path.is_file():
        return [load_blueprint(blueprint_path)]
    if not blueprint_path.is_dir():
        raise FileNotFoundError(f"Blueprint path not found: {blueprint_path}")

    blueprint_files = sorted(
        file_path
        for file_path in blueprint_path.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_BLUEPRINT_EXTENSIONS
    )
    return [load_blueprint(file_path) for file_path in blueprint_files]


def _read_blueprint_payload(path: Path) -> dict[str, Any]:
    """Parse a raw JSON or YAML blueprint file into a mapping payload."""
    suffix = path.suffix.lower()
    contents = path.read_text(encoding="utf-8")

    if suffix in {".yaml", ".yml"}:
        payload = yaml.safe_load(contents)
    elif suffix == ".json":
        payload = json.loads(contents)
    else:
        raise ValueError(f"Unsupported blueprint file extension '{suffix}' for {path}")

    if not isinstance(payload, dict):
        raise ValueError(f"Blueprint file must contain a mapping at top level: {path}")
    return payload
