from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from mafc.agents.fact_check.models import (
    CheckStatus,
    DelegationTask,
    PlannerCheckUpdate,
    PlannerDecision,
    PlannerDecisionType,
    RoutingDecision,
)
from mafc.utils.parsing import extract_json_object, strip_json_fences


class PlannerCheckUpdatePayload(BaseModel):
    """Validated JSON payload for one check-status update."""

    model_config = ConfigDict(extra="ignore")

    id: str
    status: CheckStatus
    reason: str
    original_status: str | None = None  # set when status was coerced

    @field_validator("status", mode="before")
    @classmethod
    def coerce_status(cls, v: object) -> object:
        valid = {s.value for s in CheckStatus}
        if isinstance(v, str) and v not in valid:
            return CheckStatus.UNCLEAR
        return v


class DelegationTaskPayload(BaseModel):
    """Validated JSON payload for one delegated task."""

    model_config = ConfigDict(extra="ignore")

    task_id: str
    agent_type: str
    instruction: str
    follow_up_to: str | None = None
    rationale: str | None = None


class PlannerDecisionPayload(BaseModel):
    """Validated JSON payload for the execution phase of an action node."""

    model_config = ConfigDict(extra="ignore")

    decision_type: PlannerDecisionType
    rationale: str
    tasks: list[DelegationTaskPayload] = []
    final_answer: str | None = None


class RoutingDecisionPayload(BaseModel):
    """Validated JSON payload for the routing phase."""

    model_config = ConfigDict(extra="ignore")

    next_node_id: str
    rationale: str
    check_updates: list[PlannerCheckUpdatePayload] = []
    final_answer: str | None = None


def parse_planner_decision(response_text: str) -> PlannerDecision:
    """Parse action-node planner output into a strongly typed decision."""
    payload = json.loads(extract_json_object(strip_json_fences(response_text.strip())))
    parsed = PlannerDecisionPayload.model_validate(payload)
    return PlannerDecision(
        decision_type=parsed.decision_type,
        rationale=parsed.rationale,
        tasks=[
            DelegationTask(
                task_id=item.task_id,
                agent_type=item.agent_type,
                instruction=item.instruction,
                follow_up_to=item.follow_up_to,
                rationale=item.rationale,
            )
            for item in parsed.tasks
        ],
        final_answer=parsed.final_answer,
    )


def try_parse_planner_decision(response_text: str) -> PlannerDecision | None:
    """Return a parsed planner decision or None when the response is invalid."""
    try:
        return parse_planner_decision(response_text)
    except (json.JSONDecodeError, ValidationError, ValueError):
        return None


def parse_routing_decision(response_text: str) -> RoutingDecision:
    """Parse routing-phase output into a strongly typed routing decision."""
    payload = json.loads(extract_json_object(strip_json_fences(response_text.strip())))
    parsed = RoutingDecisionPayload.model_validate(payload)
    raw_updates = payload.get("check_updates") or []
    coercion_warnings: list[str] = []
    for raw, item in zip(raw_updates, parsed.check_updates):
        raw_status = raw.get("status") if isinstance(raw, dict) else None
        if raw_status is not None and raw_status != item.status.value:
            coercion_warnings.append(
                f"check_update '{item.id}': invalid status '{raw_status}' coerced to 'unclear'"
            )
    return RoutingDecision(
        next_node_id=parsed.next_node_id,
        rationale=parsed.rationale,
        check_updates=[
            PlannerCheckUpdate(id=item.id, status=item.status, reason=item.reason)
            for item in parsed.check_updates
        ],
        final_answer=parsed.final_answer,
        coercion_warnings=coercion_warnings,
    )


def try_parse_routing_decision(response_text: str) -> RoutingDecision | None:
    """Return a parsed routing decision or None when the response is invalid."""
    try:
        return parse_routing_decision(response_text)
    except (json.JSONDecodeError, ValidationError, ValueError):
        return None
