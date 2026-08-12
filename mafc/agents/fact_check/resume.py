"""Rebuild a fact-check session's state from a recorded trace.

Lets an already-completed run be continued instead of re-run. The motivating case
is the refine node: it only changes what happens *after* the blueprint graph would
have finalized, so re-running the whole investigation would bury that effect in
resampling noise at temperature 1.0. Resuming holds the entire pre-finalize
trajectory fixed — same evidence, same ledger, same history — leaving the refine
iterations as the only thing that varies.

Recovered from the trace: every sub-agent's reported evidence, the check ledger
with its reasons, the node and action history, and the delegated-task record (so
the planner can see what already ran and avoid repeating it).

Two things do not survive serialization and are reconstructed approximately:

``Evidence.action``  the concrete Action subclass is gone; only its name and repr
                     were recorded. A stand-in carries both. Nothing downstream
                     reads more than that — ``format_evidence_block`` uses source
                     and takeaways, and the referent digest keys off the action
                     *name*, which is preserved.

``iteration``        resumes at the count of recorded iterations, so the remaining
                     budget matches what the original run had left.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ezmm import MultimodalSequence

from mafc.agents.fact_check.models import (
    CheckStatus,
    DelegatedTaskRecord,
    FactCheckSessionState,
)
from mafc.blueprints.models import Blueprint
from mafc.blueprints.topology import analyze_blueprint_topology
from mafc.common.action import Action
from mafc.common.evidence import Evidence
from mafc.common.logger import logger


class RecordedAction(Action):
    """Stand-in for an Action recovered from a trace.

    Carries the original action's name so anything keying off it — notably the
    reverse-image-search detection in ``mafc.common.media_referent`` — behaves as
    it did in the original run, plus the recorded repr for trace readability.
    """

    def __init__(self, name: str, repr_text: str | None = None):
        """Rebuild an action from its recorded name and representation."""
        self._save_parameters(locals())
        self.name = name
        self.repr_text = repr_text

    def __str__(self) -> str:
        return self.repr_text or self.name


def _as_sequence(value: Any) -> MultimodalSequence | None:
    """Rebuild a MultimodalSequence from its serialized form, or None."""
    if value is None:
        return None
    text = value if isinstance(value, str) else (value.get("text") if isinstance(value, dict) else str(value))
    if not text:
        return None
    try:
        return MultimodalSequence(text)
    except Exception:
        # A media reference pointing outside the registry this process opened.
        # Losing the item is better than losing the whole claim, so fall back to
        # the text with references stripped by MultimodalSequence's own parser.
        return None


def evidence_from_dict(raw: dict) -> Evidence | None:
    """Rebuild one Evidence item, or None when nothing usable survives."""
    body = _as_sequence(raw.get("raw"))
    takeaways = _as_sequence(raw.get("takeaways"))
    if body is None and takeaways is None:
        return None
    return Evidence(
        raw=body if body is not None else MultimodalSequence(""),
        action=RecordedAction(
            name=str(raw.get("action") or "recorded_action"),
            repr_text=raw.get("action_repr"),
        ),
        source=str(raw.get("source") or ""),
        preview=raw.get("preview"),
        takeaways=takeaways,
        # Carried over deliberately: dropping it would remove the judge's referent
        # block, which is worth more than the change being measured.
        referent=raw.get("referent"),
    )


def evidences_from_trace(trace: dict) -> list[Evidence]:
    """Collect every sub-agent's reported evidence, de-duplicated by source.

    Sub-agents report their whole accumulated set each time they are consulted, so
    the same source recurs across tasks; the original run deduplicated before the
    judge saw it and this mirrors that.
    """
    out: list[Evidence] = []
    seen: set[str] = set()
    for iteration in trace.get("iterations") or []:
        for task in iteration.get("delegated_tasks") or []:
            child = task.get("child_trace") or {}
            reported = ((child.get("summary") or {}).get("result") or {}).get("evidences") or []
            for item in reported:
                source = str(item.get("source") or "")
                key = f"{source}|{str(item.get('raw'))[:200]}"
                if key in seen:
                    continue
                evidence = evidence_from_dict(item)
                if evidence is not None:
                    seen.add(key)
                    out.append(evidence)
    return out


def _delegated_tasks_from_trace(trace: dict) -> dict[str, DelegatedTaskRecord]:
    records: dict[str, DelegatedTaskRecord] = {}
    for iteration in trace.get("iterations") or []:
        for task in iteration.get("delegated_tasks") or []:
            task_id = str(task.get("task_id") or "")
            if not task_id:
                continue
            records[task_id] = DelegatedTaskRecord(
                task_id=task_id,
                agent_type=str(task.get("agent_type") or ""),
                child_session_id=str(task.get("child_session_id") or ""),
                instruction=str(task.get("instruction") or ""),
                iteration=int(iteration.get("iteration") or 0),
                follow_up_to=task.get("follow_up_to"),
                rationale=task.get("rationale"),
            )
    return records


def state_from_trace(trace: dict, blueprint: Blueprint) -> FactCheckSessionState:
    """Rebuild the orchestration state as it stood when the run finalized.

    ``blueprint`` must be the one the original run selected; the caller resolves
    it by name so the graph, checks and budget match.
    """
    topology = analyze_blueprint_topology(blueprint)
    summary = trace.get("summary") or {}
    iterations = trace.get("iterations") or []

    status: dict[str, CheckStatus] = {}
    for check_id, value in (summary.get("required_checks") or {}).items():
        try:
            status[check_id] = CheckStatus(value)
        except ValueError:
            logger.warning(f"[resume] unknown check status {value!r} for {check_id}; treating as unchecked")
            status[check_id] = CheckStatus.UNCHECKED

    defs = {check.id: check for check in blueprint.required_checks if check.id in status}
    node_history = list(summary.get("node_history") or [])
    last = iterations[-1] if iterations else {}
    current = last.get("node_after") or (
        node_history[-1] if node_history else blueprint.verification_graph.start_node
    )

    return FactCheckSessionState(
        selected_blueprint=blueprint,
        current_node_id=current,
        node_layers=dict(topology.node_layers),
        max_layer=topology.max_layer,
        iteration=len(iterations),
        required_check_status=status,
        required_check_reasons=dict(summary.get("required_check_reasons") or {}),
        required_check_defs=defs,
        action_history=list(summary.get("action_history") or []),
        node_history=node_history,
        delegated_tasks=_delegated_tasks_from_trace(trace),
        evidences=evidences_from_trace(trace),
    )


def load_trace(path: str | Path) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(f"[resume] could not read {path}: {exc}")
        return None
