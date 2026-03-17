from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from mafc.blueprints.models import Blueprint
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
    action_history: list[str] = field(default_factory=list)
    node_history: list[str] = field(default_factory=list)
    delegated_tasks: dict[str, DelegatedTaskRecord] = field(default_factory=dict)
    evidences: list[Evidence] = field(default_factory=list)
    final_answer: str | None = None
    last_synthesis: str | None = None
