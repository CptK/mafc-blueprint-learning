"""Checkpoint and run-state I/O for the Strategy.md baseline.

A run directory holds:
  * ``strategy.md``            — latest document (canonical; resume target).
  * ``strategy_epoch{N}.md``   — per-epoch snapshot for inspection/rollback.
  * ``fold_log.jsonl``         — one line per batch fold (epoch, batch, ok, changelog).
  * ``state.json``             — run metadata + progress, enough to resume.

Resume model: point the driver's ``--resume-from`` at a ``strategy_*.md`` and set
``--start-epoch`` to continue epoch numbering. ``state.json`` records where the
previous run stopped so resuming is a manual, explicit decision rather than magic.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

STRATEGY_FILENAME = "strategy.md"
STATE_FILENAME = "state.json"
FOLD_LOG_FILENAME = "fold_log.jsonl"


@dataclass
class RunState:
    """Serializable metadata describing a strategy-building run's progress."""

    model: str
    batch_size: int
    max_words: int
    seed: int
    n_records: int
    epochs_planned: int
    start_epoch: int
    resumed_from: str | None = None
    epochs_completed: int = 0
    last_epoch_index: int | None = None
    total_folds: int = 0
    failed_folds: int = 0
    total_consolidations: int = 0
    # Next work item to process, updated after every batch so a crashed run can
    # resume exactly where it stopped (the shuffle is deterministic per epoch).
    resume_epoch_index: int = 0
    resume_batch_index: int = 0
    extra: dict = field(default_factory=dict)


def epoch_filename(epoch_index: int) -> str:
    """Snapshot filename for a given 0-based epoch index (1-based in the name)."""
    return f"strategy_epoch{epoch_index + 1}.md"


def write_document(out_dir: Path, text: str, *, epoch_index: int | None = None) -> Path:
    """Write the canonical ``strategy.md`` and, if given, an epoch snapshot.

    Returns the path of the epoch snapshot when ``epoch_index`` is provided,
    otherwise the canonical path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    canonical = out_dir / STRATEGY_FILENAME
    canonical.write_text(text, encoding="utf-8")
    if epoch_index is None:
        return canonical
    snapshot = out_dir / epoch_filename(epoch_index)
    snapshot.write_text(text, encoding="utf-8")
    return snapshot


def append_fold_log(out_dir: Path, entry: dict) -> None:
    """Append one fold record to ``fold_log.jsonl``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / FOLD_LOG_FILENAME).open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def reset_fold_log(out_dir: Path) -> None:
    """Remove any existing ``fold_log.jsonl`` so a fresh run starts clean.

    Call only for fresh (non-resume) runs; resuming should keep appending.
    """
    log_path = out_dir / FOLD_LOG_FILENAME
    if log_path.exists():
        log_path.unlink()


def save_state(out_dir: Path, state: RunState) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / STATE_FILENAME).write_text(
        json.dumps(asdict(state), indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_state(out_dir: Path) -> RunState | None:
    path = out_dir / STATE_FILENAME
    if not path.exists():
        return None
    return RunState(**json.loads(path.read_text(encoding="utf-8")))


def load_document(path: Path) -> str:
    """Read an existing strategy document to resume/continue from."""
    return path.read_text(encoding="utf-8")
