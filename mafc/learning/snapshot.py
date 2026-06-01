"""Registry snapshot/restore helpers used by the learning loop.

The on-disk layout matches what ``scripts/run_learning.py`` already writes
for per-epoch snapshots: one ``<name>.yaml`` file per blueprint in the
target directory.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from mafc.blueprints.loader import load_blueprints
from mafc.blueprints.registry import BlueprintRegistry


def snapshot_registry(registry: BlueprintRegistry, directory: Path) -> Path:
    """Write every blueprint in ``registry`` as a YAML file under ``directory``.

    Idempotent: callers may overwrite an existing snapshot dir freely.
    Returns the directory written to.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Wipe any pre-existing blueprint files so a smaller registry doesn't leave stale entries.
    for stale in directory.glob("*.yaml"):
        stale.unlink()
    for bp in registry.get_all():
        bp_dict = bp.model_dump(by_alias=True)
        with open(directory / f"{bp.name}.yaml", "w") as f:
            yaml.dump(bp_dict, f, default_flow_style=False, allow_unicode=True)
    return directory


def restore_registry(directory: Path) -> BlueprintRegistry:
    """Load a registry from a snapshot directory written by ``snapshot_registry``."""
    return BlueprintRegistry(load_blueprints(directory))


def restore_registry_in_place(registry: BlueprintRegistry, directory: Path) -> None:
    """Restore ``registry`` to the state captured in ``directory`` without rebinding.

    The rollback path needs the existing ``BlueprintRegistry`` instance to keep its
    object identity so the selector and pipeline (both hold a reference) continue
    to see the restored state without being rewired.
    """
    registry.replace_all(load_blueprints(directory))
