from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict, ValidationError

from mafc.agents.fact_check.models import (
    CheckStatus,
    DelegationTask,
    PlannerCheckUpdate,
    PlannerDecision,
    PlannerDecisionType,
)
from mafc.utils.parsing import extract_json_object


class PlannerCheckUpdatePayload(BaseModel):
    """Validated JSON payload for one check-status update."""

    model_config = ConfigDict(extra="forbid")

    id: str
    status: CheckStatus
    reason: str


class DelegationTaskPayload(BaseModel):
    """Validated JSON payload for one delegated task."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    agent_type: str
    instruction: str
    follow_up_to: str | None = None
    rationale: str | None = None


class PlannerDecisionPayload(BaseModel):
    """Validated JSON payload for one planner step decision."""

    model_config = ConfigDict(extra="forbid")

    decision_type: PlannerDecisionType
    rationale: str
    tasks: list[DelegationTaskPayload] = []
    instruction: str | None = None
    target_node_id: str | None = None
    final_answer: str | None = None
    check_updates: list[PlannerCheckUpdatePayload] = []


def parse_planner_decision(response_text: str) -> PlannerDecision:
    """Parse planner output into a strongly typed orchestration decision."""
    text = response_text.strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.startswith("```")]
        text = "\n".join(lines).strip()

    payload = json.loads(extract_json_object(text))
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
        instruction=parsed.instruction,
        target_node_id=parsed.target_node_id,
        final_answer=parsed.final_answer,
        check_updates=[
            PlannerCheckUpdate(id=item.id, status=item.status, reason=item.reason)
            for item in parsed.check_updates
        ],
    )


def try_parse_planner_decision(response_text: str) -> PlannerDecision | None:
    """Return a parsed planner decision or None when the response is invalid."""
    try:
        return parse_planner_decision(response_text)
    except (json.JSONDecodeError, ValidationError, ValueError):
        return None
