"""Serialization helpers for ArticleAnalysis — shared across scripts and the pipeline."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from mafc.learning.models import ActionEvidenceLink, ArticleAnalysis


def analysis_to_dict(a: ArticleAnalysis) -> dict:
    return dataclasses.asdict(a)


def analysis_from_dict(d: dict) -> ArticleAnalysis:
    links = d.get("action_evidence_links")
    return ArticleAnalysis(
        claim_type=d["claim_type"],
        verdict_summary=d["verdict_summary"],
        key_evidence=d.get("key_evidence") or [],
        evidence_types=d.get("evidence_types") or [],
        action_evidence_links=([ActionEvidenceLink(**lnk) for lnk in links] if links else None),
        investigative_steps=d.get("investigative_steps"),
        search_queries=d.get("search_queries"),
        process_richness=d.get("process_richness", "result_only"),
        notes=d.get("notes"),
    )


def load_analyses(path: Path) -> dict[str, ArticleAnalysis]:
    """Load {claim_id: ArticleAnalysis} from a JSON file. Returns {} if not found."""
    if not path.exists():
        return {}
    with open(path) as f:
        raw: dict[str, dict] = json.load(f)
    return {claim_id: analysis_from_dict(d) for claim_id, d in raw.items()}


def save_analyses(analyses: dict[str, ArticleAnalysis], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({cid: analysis_to_dict(a) for cid, a in analyses.items()}, f, indent=2)
