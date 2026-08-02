"""Persistent store for GenD scores, so datasets can be scored once up front.

A store is a directory:

    <store>/index.json   sha256 -> record

Keying on the file's sha256 (not its path) means a store stays valid when files
are moved or a dataset is re-downloaded, and the same image shared by several
datasets is only ever scored once. GenD runs locally, so this is about not
recomputing a slow model rather than not paying twice.

Mirrors mafc.tools.media.sightengine.store so the three detectors' stores can be
read the same way.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
class GenDRecord:
    """GenD's raw output for one media file.

    `p_fake` is None when no face was detected. That is a distinct outcome from
    a low score — the model was never given anything to judge — so consumers
    must branch on it rather than defaulting it to 0.
    """

    p_fake: float | None = None
    n_faces: int = 0
    n_faces_skipped: int = 0  # found, but below min_face_px
    face_scores: list[float] = field(default_factory=list)
    model_name: str | None = None
    source_name: str | None = None  # original filename, for humans; not used for lookup
    n_frames: int | None = None  # videos: frames sampled
    aggregation: str | None = None  # videos: how frame scores were combined
    notes: list[str] = field(default_factory=list)
    created: str = ""

    def __post_init__(self):
        if not self.created:
            self.created = datetime.now(timezone.utc).isoformat(timespec="seconds")


class GenDStore:
    """Reads/writes a score store directory. Loads the index lazily."""

    def __init__(self, path: str | Path, writable: bool = True):
        self.path = Path(path)
        self.writable = writable
        self._index: dict[str, GenDRecord] | None = None
        self._dirty = False

    @property
    def index(self) -> dict[str, GenDRecord]:
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
            logger.warning(f"[GenD] Could not read store {index_path}: {e}. Treating it as empty.")
            return
        for key, value in raw.get("records", {}).items():
            try:
                self._index[key] = GenDRecord(**value)
            except TypeError:
                logger.warning(f"[GenD] Skipping malformed record {key} in {index_path}")

    def get(self, sha256: str) -> GenDRecord | None:
        return self.index.get(sha256)

    def put(self, sha256: str, record: GenDRecord) -> None:
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
            "model": "gend",
            "records": {k: asdict(v) for k, v in self.index.items()},
        }
        tmp = self.path / f"{INDEX_FILENAME}.tmp"
        tmp.write_text(json.dumps(payload, indent=1))
        os.replace(tmp, self.path / INDEX_FILENAME)
        self._dirty = False

    def __len__(self) -> int:
        return len(self.index)

    def __repr__(self) -> str:
        return f"GenDStore({self.path}, {len(self)} records, writable={self.writable})"
