"""Normalised reader over VeriTaS execution traces.

Two trace shapes exist on disk and both embed a ``judge_run`` and a ``summary``:

- ``*.fact_check_trace.json`` (FactCheckAgent / blueprint runs): has ``iterations``
  (each with ``delegated_tasks``, ``new_errors``, ``evidence_count_before/after``),
  a ``blueprint`` block with ``selection.claim_features``, ``judge_run`` and a rich
  ``summary`` (``runtime_seconds``, ``total_calls``, ``timings``, token counts,
  ``delegated_tasks``, ``required_checks``).
- ``*.strategy_trace.json`` (StrategyAgent runs): has ``rounds`` (each with
  ``tool_calls``, ``done``, ``evidence_count_after``) instead of ``iterations`` and
  no ``blueprint`` block, but the same ``judge_run`` / ``summary`` essentials.

This module flattens whichever fields are present into a ``NormalisedTrace`` so the
feature extractor degrades gracefully across trace types. Nothing here touches the
ground-truth ``integrity.score`` (that join happens in ``features``); ``true_label``
inside the trace is deliberately ignored to avoid leakage.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

# Recognised trace-file suffixes, richest first.
TRACE_SUFFIXES = (".fact_check_trace.json", ".strategy_trace.json", ".judge_trace.json")


class EvidenceView(BaseModel):
    source: str | None = None
    takeaways_text: str | None = None
    is_useful: bool = False


class NormalisedTrace(BaseModel):
    """Trace fields the feature extractor consumes, joined by claim id."""

    claim_id: str
    trace_kind: str  # "fact_check" | "strategy" | "judge"

    # Judge conditioning / hedging
    judge_label: str | None = None  # the judge's predicted 7-class label
    judge_direction: str | None = None  # coarsened direction of judge_label
    judge_justification: str | None = None
    judge_output_tokens: int | None = None
    judge_repair_fired: bool = False
    judge_errors_present: bool = False

    # Evidence
    evidence: list[EvidenceView] = []
    evidence_count: int | None = None

    # Search-struggle / iteration signals
    n_iterations: int = 0
    max_iterations: int | None = None
    hit_max_iterations: bool = False
    evidence_growth: list[int] = []  # evidence count after each iteration/round
    n_delegated_tasks: int = 0
    n_errors: int = 0
    retrieval_failures: int = 0  # errors mentioning failed retrieval

    # Runtime / cost
    runtime_seconds: float | None = None
    total_calls: int | None = None
    total_output_tokens: int | None = None
    total_input_tokens: int | None = None

    # Difficulty priors
    claim_features: dict = {}
    blueprint_name: str | None = None


_RETRIEVAL_FAIL_MARKERS = ("failed to retrieve", "could not retrieve", "retrieval failed")


def _count_retrieval_failures(errors: list[str]) -> int:
    n = 0
    for e in errors:
        low = str(e).lower()
        if any(m in low for m in _RETRIEVAL_FAIL_MARKERS):
            n += 1
    return n


def _parse_evidence(judge_run: dict) -> list[EvidenceView]:
    out: list[EvidenceView] = []
    result = ((judge_run or {}).get("summary") or {}).get("result") or {}
    for ev in result.get("evidences") or []:
        take = ev.get("takeaways")
        take_text = None
        if isinstance(take, dict):
            take_text = take.get("text")
        elif isinstance(take, str):
            take_text = take
        out.append(
            EvidenceView(
                source=ev.get("source"),
                takeaways_text=take_text,
                is_useful=take is not None,
            )
        )
    return out


def _judge_fields(trace: dict, norm: NormalisedTrace) -> None:
    judge = trace.get("judge_run") or {}
    decision = judge.get("decision") or {}
    norm.judge_label = decision.get("label")
    norm.judge_justification = decision.get("justification")
    norm.judge_repair_fired = bool(judge.get("repair_response"))
    jsummary = judge.get("summary") or {}
    norm.judge_output_tokens = jsummary.get("total_output_tokens")
    norm.judge_errors_present = bool(jsummary.get("errors"))
    norm.evidence = _parse_evidence(judge)
    norm.evidence_count = judge.get("evidence_count")


def _factcheck_struggle(trace: dict, norm: NormalisedTrace) -> None:
    iterations = trace.get("iterations") or []
    norm.n_iterations = len(iterations)
    growth: list[int] = []
    n_tasks = 0
    for it in iterations:
        growth.append(it.get("evidence_count_after", 0))
        n_tasks += len(it.get("delegated_tasks") or [])
    norm.evidence_growth = growth
    norm.n_delegated_tasks = n_tasks
    bp = trace.get("blueprint") or {}
    norm.blueprint_name = bp.get("name")
    norm.max_iterations = bp.get("max_iterations")
    if norm.max_iterations is not None:
        norm.hit_max_iterations = norm.n_iterations >= norm.max_iterations
    selection = bp.get("selection") or {}
    norm.claim_features = selection.get("claim_features") or {}


def _strategy_struggle(trace: dict, norm: NormalisedTrace) -> None:
    rounds = trace.get("rounds") or []
    norm.n_iterations = len(rounds)
    growth: list[int] = []
    n_tasks = 0
    for r in rounds:
        growth.append(r.get("evidence_count_after", 0))
        n_tasks += len(r.get("tool_calls") or [])
    norm.evidence_growth = growth
    norm.n_delegated_tasks = n_tasks
    # Strategy runs are single-pass; max rounds isn't recorded on the trace.
    if rounds:
        norm.max_iterations = max(r.get("round", 0) for r in rounds) or None
        norm.hit_max_iterations = any(not r.get("done", False) for r in rounds[-1:])


def _summary_fields(trace: dict, norm: NormalisedTrace) -> None:
    summary = trace.get("summary") or {}
    errors = summary.get("errors") or []
    norm.n_errors = len(errors)
    norm.retrieval_failures = _count_retrieval_failures([str(e) for e in errors])
    norm.runtime_seconds = summary.get("runtime_seconds")
    norm.total_calls = summary.get("total_calls")
    norm.total_output_tokens = summary.get("total_output_tokens")
    norm.total_input_tokens = summary.get("total_input_tokens")


def normalise_trace(trace: dict, claim_id: str, trace_kind: str) -> NormalisedTrace:
    norm = NormalisedTrace(claim_id=claim_id, trace_kind=trace_kind)
    _judge_fields(trace, norm)
    if trace_kind == "fact_check":
        _factcheck_struggle(trace, norm)
    elif trace_kind == "strategy":
        _strategy_struggle(trace, norm)
    _summary_fields(trace, norm)
    if norm.evidence_count is None:
        norm.evidence_count = len(norm.evidence)
    return norm


def _kind_and_id(path: Path) -> tuple[str, str] | None:
    name = path.name
    for suffix in TRACE_SUFFIXES:
        if name.endswith(suffix):
            stem = name[: -len(suffix)]
            cid = stem.split("benchmark_")[-1] if "benchmark_" in stem else stem
            kind = suffix.strip(".").replace("_trace.json", "")
            return kind, cid
    return None


def discover_traces(trace_dir: Path) -> dict[str, Path]:
    """Map ``claim_id -> richest trace path`` in a directory.

    When multiple trace kinds exist for one claim, the richest (fact_check >
    strategy > judge) wins so the feature extractor gets maximal coverage.
    """
    priority = {"fact_check": 0, "strategy": 1, "judge": 2}
    best: dict[str, tuple[int, Path]] = {}
    for path in sorted(Path(trace_dir).glob("*trace*.json")):
        parsed = _kind_and_id(path)
        if parsed is None:
            continue
        kind, cid = parsed
        rank = priority.get(kind, 9)
        if cid not in best or rank < best[cid][0]:
            best[cid] = (rank, path)
    return {cid: p for cid, (_, p) in best.items()}


def load_normalised(path: Path) -> NormalisedTrace | None:
    parsed = _kind_and_id(Path(path))
    if parsed is None:
        return None
    kind, cid = parsed
    try:
        trace = json.loads(Path(path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return normalise_trace(trace, cid, kind)
