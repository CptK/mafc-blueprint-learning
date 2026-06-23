"""Loading of VeriTaS ``claims.json`` files for the training subsystem.

A ``claims.json`` is ``{"claims": [{id, text, date, language, media, integrity:
{score, decisive_property}, ...}, ...]}``. We only need the fields relevant to
the magnitude regressor; everything else is ignored.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel


class ClaimRecord(BaseModel):
    """Minimal projection of a benchmark claim for training purposes."""

    id: str
    text: str = ""
    date: str | None = None
    language: str | None = None
    n_media: int = 0
    has_media: bool = False
    integrity_score: float
    decisive_property: str | None = None

    @property
    def magnitude(self) -> float:
        return abs(self.integrity_score)


def _to_record(raw: dict) -> ClaimRecord | None:
    integrity = raw.get("integrity") or {}
    score = integrity.get("score")
    if score is None:
        return None
    media = raw.get("media") or []
    return ClaimRecord(
        id=str(raw["id"]),
        text=raw.get("text") or "",
        date=raw.get("date"),
        language=raw.get("language"),
        n_media=len(media),
        has_media=len(media) > 0,
        integrity_score=float(score),
        decisive_property=integrity.get("decisive_property"),
    )


def load_claims(path: Path) -> list[ClaimRecord]:
    """Load and project one ``claims.json``; claims without a score are dropped."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    claims = raw["claims"] if isinstance(raw, dict) else raw
    out: list[ClaimRecord] = []
    for c in claims:
        rec = _to_record(c)
        if rec is not None:
            out.append(rec)
    return out


def load_many(paths: list[Path]) -> dict[str, ClaimRecord]:
    """Load several ``claims.json`` files into a ``{claim_id: ClaimRecord}`` map.

    Later files win on id collisions (last-write-wins), matching how the
    strategy/blueprint loaders concatenate splits.
    """
    by_id: dict[str, ClaimRecord] = {}
    for p in paths:
        for rec in load_claims(p):
            by_id[rec.id] = rec
    return by_id


def resolve_claims_paths(paths: list[Path]) -> list[Path]:
    """Accept either a ``claims.json`` file or a directory containing one."""
    resolved: list[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            cand = p / "claims.json"
            if cand.exists():
                resolved.append(cand)
        elif p.exists():
            resolved.append(p)
    return resolved
