from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from mafc.blueprints.models import Blueprint, BlueprintNode, BlueprintRequiredCheck
from mafc.common.evidence import Evidence


class CheckStatus(str, Enum):
    """Status of one required blueprint check during orchestration."""

    UNCHECKED = "unchecked"
    SUPPORTED = "supported"
    REFUTED = "refuted"
    UNCLEAR = "unclear"


class PlannerDecisionType(str, Enum):
    """Actions the blueprint-guided planner may request at an action node."""

    DELEGATE = "delegate"
    FINALIZE = "finalize"


@dataclass
class DelegationTask:
    """One mid-level task assigned by the planner to an agent capability."""

    task_id: str
    agent_type: str
    instruction: str
    follow_up_to: str | None = None
    rationale: str | None = None


@dataclass
class PlannerCheckUpdate:
    """One required-check status update emitted by the planner."""

    id: str
    status: CheckStatus
    reason: str


@dataclass
class PlannerDecision:
    """Structured planner output for the execution phase of an action node."""

    decision_type: PlannerDecisionType
    rationale: str
    tasks: list[DelegationTask] = field(default_factory=list)
    final_answer: str | None = None


@dataclass
class RoutingDecision:
    """Structured output for the routing phase: where to go after node execution."""

    next_node_id: str  # a valid node ID in the blueprint graph, or "finalize"
    rationale: str
    check_updates: list[PlannerCheckUpdate] = field(default_factory=list)
    final_answer: str | None = None
    coercion_warnings: list[str] = field(default_factory=list)


@dataclass
class DelegatedTaskRecord:
    """Tracked execution record for one delegated task and its child session."""

    task_id: str
    agent_type: str
    child_session_id: str
    instruction: str
    iteration: int
    follow_up_to: str | None = None
    rationale: str | None = None


@dataclass
class FactCheckSessionState:
    """Mutable orchestration state for one top-level fact-check session."""

    selected_blueprint: Blueprint
    current_node_id: str
    node_layers: dict[str, int]
    max_layer: int
    iteration: int = 0
    required_check_status: dict[str, CheckStatus] = field(default_factory=dict)
    required_check_reasons: dict[str, str] = field(default_factory=dict)
    required_check_defs: dict[str, BlueprintRequiredCheck] = field(default_factory=dict)
    """Definitions of all ACTIVE checks (blueprint-level plus checks of visited
    nodes). Node-attached checks join when execution first reaches their node,
    so a claim only ever carries the checks of the path it actually takes."""
    action_history: list[str] = field(default_factory=list)
    node_history: list[str] = field(default_factory=list)
    delegated_tasks: dict[str, DelegatedTaskRecord] = field(default_factory=dict)
    evidences: list[Evidence] = field(default_factory=list)
    final_answer: str | None = None
    last_synthesis: str | None = None

    def activate_node_checks(self, node: BlueprintNode) -> list[str]:
        """Activate the checks a node references on first visit; returns new ids.

        Definitions are looked up in the blueprint's root required_checks — the
        single place check definitions live. Idempotent: already-active ids are
        left untouched, so converging paths never duplicate or reset checks.
        """
        ids = getattr(node, "activates_checks", None) or []
        if not ids:
            return []
        defs = {check.id: check for check in self.selected_blueprint.required_checks}
        added: list[str] = []
        for check_id in ids:
            if check_id in self.required_check_status or check_id not in defs:
                continue
            self.required_check_status[check_id] = CheckStatus.UNCHECKED
            self.required_check_defs[check_id] = defs[check_id]
            added.append(check_id)
        return added
