"""Persistent store for Sightengine scores, so datasets can be scored once up front.

A store is a directory:

    <store>/index.json   sha256 -> record

Keying on the file's sha256 (not its path) means a store stays valid when files
are moved or a dataset is re-downloaded, and the same image shared by several
datasets is only ever scored once. Unlike TruFor, Sightengine runs remotely
(one paid API call per file), so precomputing is about not paying twice rather
than not recomputing a local model.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os

from mafc.common.logger import logger

INDEX_FILENAME = "index.json"


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class SightengineRecord:
    """The raw Sightengine scores for one media file, as returned by the API.

    Only the model outputs are stored — thresholds and the human-readable
    verdict are re-derived at read time, so the tool's thresholds can change
    without re-scoring. Video frame scores are stored pre-aggregated (a video's
    frames come from the API, not a local sampler), so changing
    ``video_aggregation`` does require re-precomputing videos.
    """

    ai_generated_score: float | None = None
    deepfake_score: float | None = None
    ai_speech_score: float | None = None  # videos only; None if not checked
    top_generator: str | None = None
    top_generator_score: float | None = None
    source_name: str | None = None  # original filename, for humans; not used for lookup
    n_frames: int | None = None  # videos: number of frames the API scored
    aggregation: str | None = None  # videos: how frame scores were combined
    notes: list[str] = field(default_factory=list)  # dynamic notes (truncation, async, ai_speech failure)
    created: str = ""

    def __post_init__(self):
        if not self.created:
            self.created = datetime.now(timezone.utc).isoformat(timespec="seconds")


class SightengineStore:
    """Reads/writes a score store directory. Loads the index lazily."""

    def __init__(self, path: str | Path, writable: bool = True):
        self.path = Path(path)
        self.writable = writable
        self._index: dict[str, SightengineRecord] | None = None
        self._dirty = False

    @property
    def index(self) -> dict[str, SightengineRecord]:
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
            logger.warning(f"[Sightengine] Could not read store {index_path}: {e}. Treating it as empty.")
            return
        for key, value in raw.get("records", {}).items():
            try:
                self._index[key] = SightengineRecord(**value)
            except TypeError:
                logger.warning(f"[Sightengine] Skipping malformed record {key} in {index_path}")

    def get(self, sha256: str) -> SightengineRecord | None:
        return self.index.get(sha256)

    def put(self, sha256: str, record: SightengineRecord) -> None:
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
            "model": "sightengine",
            "records": {k: asdict(v) for k, v in self.index.items()},
        }
        tmp = self.path / f"{INDEX_FILENAME}.tmp"
        tmp.write_text(json.dumps(payload, indent=1))
        os.replace(tmp, self.path / INDEX_FILENAME)
        self._dirty = False

    def __len__(self) -> int:
        return len(self.index)

    def __repr__(self) -> str:
        return f"SightengineStore({self.path}, {len(self)} records, writable={self.writable})"
