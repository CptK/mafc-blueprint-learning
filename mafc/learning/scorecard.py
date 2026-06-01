"""Per-blueprint scorecard built from ``ExecutionResult``s.

Tracks rolling outcome statistics for every blueprint that has been executed
during a learning run.

Stores raw ``(y_true, y_pred)`` pairs per blueprint so per-class metrics
(precision, recall, F1, macro-F1) can be computed on demand without
re-walking results.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mafc.eval.metrics import classification_block
from mafc.learning.execution import ExecutionResult


@dataclass
class _BlueprintEntry:
    n_runs: int = 0
    n_correct: int = 0
    n_errored: int = 0
    sum_cost_usd: float = 0.0
    sum_iterations: int = 0
    sum_duration_ms: int = 0
    y_true: list[str] = field(default_factory=list)
    y_pred: list[str] = field(default_factory=list)


class BlueprintScorecard:
    """Thread-safe accumulator of per-blueprint outcome statistics."""

    def __init__(self) -> None:
        self._by_bp: dict[str, _BlueprintEntry] = {}
        # RLock so to_dict() can call entry() while holding the lock without
        # deadlocking; both readers and writers share the same critical section.
        self._lock = threading.RLock()

    def record(self, result: ExecutionResult) -> None:
        with self._lock:
            entry = self._by_bp.setdefault(result.blueprint_name, _BlueprintEntry())
            entry.n_runs += 1
            entry.sum_cost_usd += result.cost_usd
            entry.sum_iterations += result.n_iterations
            entry.sum_duration_ms += result.duration_ms
            if result.predicted_label is None:
                entry.n_errored += 1
                return
            entry.y_true.append(result.ground_truth)
            entry.y_pred.append(result.predicted_label)
            if result.correct:
                entry.n_correct += 1

    def blueprint_names(self) -> list[str]:
        with self._lock:
            return list(self._by_bp.keys())

    def entry(self, blueprint_name: str) -> _BlueprintEntry | None:
        with self._lock:
            entry = self._by_bp.get(blueprint_name)
            if entry is None:
                return None
            # Return a defensive copy so the caller can read outside the lock.
            return _BlueprintEntry(
                n_runs=entry.n_runs,
                n_correct=entry.n_correct,
                n_errored=entry.n_errored,
                sum_cost_usd=entry.sum_cost_usd,
                sum_iterations=entry.sum_iterations,
                sum_duration_ms=entry.sum_duration_ms,
                y_true=list(entry.y_true),
                y_pred=list(entry.y_pred),
            )

    def to_dict(self, labels: Iterable[str] | None = None) -> dict[str, dict[str, Any]]:
        """Render per-blueprint stats. If ``labels`` is given, also embed classification metrics."""
        out: dict[str, dict[str, Any]] = {}
        with self._lock:
            snapshot = {name: self.entry(name) for name in self._by_bp}
        label_list = list(labels) if labels else None
        for name, entry in snapshot.items():
            if entry is None:
                continue
            scored = len(entry.y_true)
            block: dict[str, Any] = {
                "n_runs": entry.n_runs,
                "n_correct": entry.n_correct,
                "n_errored": entry.n_errored,
                "accuracy": round(entry.n_correct / scored, 4) if scored else None,
                "avg_cost_usd": round(entry.sum_cost_usd / entry.n_runs, 6) if entry.n_runs else None,
                "avg_iterations": (round(entry.sum_iterations / entry.n_runs, 2) if entry.n_runs else None),
                "avg_duration_ms": (round(entry.sum_duration_ms / entry.n_runs) if entry.n_runs else None),
            }
            if label_list and scored:
                block["classification"] = classification_block(entry.y_true, entry.y_pred, label_list)
            out[name] = block
        return out

    def save_json(self, path: Path, labels: Iterable[str] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(labels), f, indent=2)
