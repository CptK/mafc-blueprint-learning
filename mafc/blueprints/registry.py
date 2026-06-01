from __future__ import annotations

from pathlib import Path

from mafc.blueprints.loader import load_blueprints
from mafc.blueprints.models import Blueprint


class BlueprintRegistry:
    """In-memory registry for validated blueprint definitions."""

    def __init__(self, blueprints: list[Blueprint] | None = None):
        """Initialize the registry and optionally register a starting blueprint set."""
        self._blueprints_by_name: dict[str, Blueprint] = {}
        for blueprint in blueprints or []:
            self.register(blueprint)

    def register(self, blueprint: Blueprint | list[Blueprint]) -> None:
        """Register one blueprint by name."""
        blueprints = blueprint if isinstance(blueprint, list) else [blueprint]
        for bp in blueprints:
            existing = self._blueprints_by_name.get(bp.name)
            if existing is not None:
                raise ValueError(f"Blueprint '{bp.name}' is already registered")
            self._blueprints_by_name[bp.name] = bp

    def get(self, name: str) -> Blueprint:
        """Return one blueprint by name or raise when it is unknown."""
        try:
            return self._blueprints_by_name[name]
        except KeyError as exc:
            raise KeyError(f"Unknown blueprint '{name}'") from exc

    def replace(self, old_name: str, blueprint: Blueprint) -> None:
        """Replace the blueprint registered under old_name with blueprint, preserving insertion order.

        old_name and blueprint.name may differ when the updater renames the blueprint.
        Raises if old_name is unknown or if blueprint.name is already taken by a different slot.
        """
        if old_name not in self._blueprints_by_name:
            raise KeyError(f"Cannot replace unknown blueprint '{old_name}'; use register() instead.")
        if blueprint.name != old_name and blueprint.name in self._blueprints_by_name:
            raise ValueError(f"Cannot replace '{old_name}' with '{blueprint.name}': name already registered.")
        # Rebuild the dict to swap the key in-place, preserving insertion order.
        self._blueprints_by_name = {
            (blueprint.name if k == old_name else k): (blueprint if k == old_name else v)
            for k, v in self._blueprints_by_name.items()
        }

    def remove(self, name: str) -> None:
        """Remove a blueprint by name. Raises if the name is unknown."""
        if name not in self._blueprints_by_name:
            raise KeyError(f"Cannot remove unknown blueprint '{name}'.")
        del self._blueprints_by_name[name]

    def contains(self, name: str) -> bool:
        """Return True if a blueprint with this name is registered."""
        return name in self._blueprints_by_name

    def get_all(self) -> list[Blueprint]:
        """Return all registered blueprints in insertion order."""
        return list(self._blueprints_by_name.values())

    def replace_all(self, blueprints: list[Blueprint]) -> None:
        """Atomically replace the registry contents with ``blueprints``.

        Used by rollback to restore a snapshotted registry without breaking
        existing references (selectors, pipelines hold a reference to this
        registry object; rebinding them on every rollback would be intrusive).
        """
        new_map: dict[str, Blueprint] = {}
        for bp in blueprints:
            if bp.name in new_map:
                raise ValueError(f"Duplicate blueprint name in replace_all: '{bp.name}'")
            new_map[bp.name] = bp
        self._blueprints_by_name = new_map

    @classmethod
    def from_path(cls, path: str | Path) -> "BlueprintRegistry":
        """Build a registry from one blueprint file or a directory of blueprints."""
        return cls(load_blueprints(path))
