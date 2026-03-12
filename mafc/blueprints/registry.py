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

    def get_all(self) -> list[Blueprint]:
        """Return all registered blueprints in insertion order."""
        return list(self._blueprints_by_name.values())

    @classmethod
    def from_path(cls, path: str | Path) -> "BlueprintRegistry":
        """Build a registry from one blueprint file or a directory of blueprints."""
        return cls(load_blueprints(path))
