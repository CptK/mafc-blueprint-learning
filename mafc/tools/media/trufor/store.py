"""Persistent store for TruFor scores, so datasets can be scored once up front.

A store is a directory:

    <store>/index.json        sha256 -> record
    <store>/maps/<sha256>.npz optional localization/confidence maps

Keying on the file's sha256 (not its path) means a store stays valid when files
are moved or a dataset is re-downloaded, and the same image shared by several
datasets is only ever scored once.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os

import numpy as np

from mafc.common.logger import logger

INDEX_FILENAME = "index.json"
MAPS_DIRNAME = "maps"


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class ScoreRecord:
    """One scored media file."""

    score: float
    source_name: str | None = None  # original filename, for humans; not used for lookup
    image_size: list[int] | None = None  # [height, width]
    n_frames: int | None = None  # set for videos: how many frames were sampled
    frame_scores: list[float] = field(default_factory=list)  # per-frame scores, videos only
    has_maps: bool = False
    created: str = ""

    def __post_init__(self):
        if not self.created:
            self.created = datetime.now(timezone.utc).isoformat(timespec="seconds")


class ScoreStore:
    """Reads/writes a score store directory. Loads the index lazily."""

    def __init__(self, path: str | Path, writable: bool = True):
        self.path = Path(path)
        self.writable = writable
        self._index: dict[str, ScoreRecord] | None = None
        self._dirty = False

    @property
    def index(self) -> dict[str, ScoreRecord]:
        if self._index is None:
            self._load()
            assert self._index is not None  # _load always assigns
        return self._index

    def _load(self) -> None:
        index_path = self.path / INDEX_FILENAME
        self._index = {}
        if not index_path.is_file():
            return
        try:
            raw = json.loads(index_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"[TruFor] Could not read store {index_path}: {e}. Treating it as empty.")
            return
        for key, value in raw.get("records", {}).items():
            try:
                self._index[key] = ScoreRecord(**value)
            except TypeError:
                logger.warning(f"[TruFor] Skipping malformed record {key} in {index_path}")

    def get(self, sha256: str) -> ScoreRecord | None:
        return self.index.get(sha256)

    def put(self, sha256: str, record: ScoreRecord) -> None:
        if not self.writable:
            raise RuntimeError(f"store {self.path} is read-only")
        self.index[sha256] = record
        self._dirty = True

    def save(self) -> None:
        """Atomically rewrites index.json."""
        if not self.writable or not self._dirty:
            return
        self.path.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": 1,
            "model": "trufor",
            "records": {k: asdict(v) for k, v in self.index.items()},
        }
        tmp = self.path / f"{INDEX_FILENAME}.tmp"
        tmp.write_text(json.dumps(payload, indent=1))
        os.replace(tmp, self.path / INDEX_FILENAME)
        self._dirty = False

    # --- optional localization/confidence maps -------------------------------

    def map_path(self, sha256: str) -> Path:
        return self.path / MAPS_DIRNAME / f"{sha256}.npz"

    def save_maps(self, sha256: str, localization_map: np.ndarray, confidence_map: np.ndarray | None) -> None:
        path = self.map_path(sha256)
        path.parent.mkdir(parents=True, exist_ok=True)
        if confidence_map is None:
            np.savez_compressed(path, map=localization_map)
        else:
            np.savez_compressed(path, map=localization_map, conf=confidence_map)

    def load_maps(self, sha256: str) -> dict[str, np.ndarray] | None:
        path = self.map_path(sha256)
        if not path.is_file():
            return None
        with np.load(path) as data:
            return {key: data[key] for key in data.files}

    def __len__(self) -> int:
        return len(self.index)

    def __repr__(self) -> str:
        return f"ScoreStore({self.path}, {len(self)} records, writable={self.writable})"
