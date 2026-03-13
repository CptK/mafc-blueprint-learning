from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ezmm import MultimodalSequence

from mafc.agents.agent import AgentResult
from mafc.agents.common import AgentSession
from mafc.agents.fact_check.models import FactCheckSessionState, PlannerDecision
from mafc.common.claim import Claim
from mafc.common.evidence import Evidence
from mafc.common.modeling.message import Message


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize_filename(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return sanitized or "fact_check_trace"


def _serialize_multimodal(content: MultimodalSequence | None) -> dict[str, Any] | None:
    if content is None:
        return None
    return {
        "text": str(content),
        "images": [image.reference for image in content.images],
        "videos": [video.reference for video in content.videos],
    }


def _serialize_claim(claim: Claim | None) -> dict[str, Any] | None:
    if claim is None:
        return None
    return {
        "id": claim.id,
        "dataset": claim.dataset,
        "text": str(claim),
        "author": claim.author,
        "date": claim.date.isoformat() if claim.date else None,
        "origin": claim.origin,
        "meta_info": claim.meta_info,
        "images": [image.reference for image in claim.images],
        "videos": [video.reference for video in claim.videos],
    }


def _serialize_message(message: Message) -> dict[str, Any]:
    return {
        "role": message.role.value,
        "content": _serialize_multimodal(message.content),
    }


def _serialize_evidence(evidence: Evidence) -> dict[str, Any]:
    return {
        "source": evidence.source,
        "action": evidence.action.name,
        "action_repr": str(evidence.action),
        "raw": _serialize_multimodal(evidence.raw),
        "takeaways": _serialize_multimodal(evidence.takeaways),
    }


def _serialize_result(result: AgentResult) -> dict[str, Any]:
    return {
        "status": result.status.value if result.status is not None else None,
        "result": _serialize_multimodal(result.result),
        "errors": list(result.errors),
        "evidences": [_serialize_evidence(evidence) for evidence in result.evidences],
        "message_count": len(result.messages),
    }


def _serialize_dataclass(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    return value


class FactCheckTraceRecorder:
    """Captures one fact-check run as structured JSON for later graphing."""

    trace_version = 1

    def __init__(self, trace_dir: str | Path | None, session: AgentSession, agent_name: str):
        self.enabled = trace_dir is not None
        self.path: Path | None = None
        self._event_seq = 0
        self._current_iteration_index: int | None = None
        self._last_iteration_node_id: str | None = None
        self._task_node_ids: dict[str, str] = {}
        self.trace: dict[str, Any] = {
            "trace_version": self.trace_version,
            "agent": agent_name,
            "session_id": session.id,
            "parent_session_id": session.parent_session_id,
            "status": None,
            "started_at": _timestamp(),
            "ended_at": None,
            "goal": _serialize_multimodal(session.goal),
            "claim": _serialize_claim(session.claim),
            "blueprint": None,
            "iterations": [],
            "events": [],
            "flow": {
                "nodes": [],
                "edges": [],
            },
            "summary": {
                "result": None,
                "errors": [],
                "evidence_count": 0,
                "message_count": 0,
                "required_checks": {},
                "required_check_reasons": {},
                "node_history": [],
                "action_history": [],
                "delegated_tasks": {},
            },
        }
        if self.enabled:
            trace_dir_path = Path(trace_dir)
            trace_dir_path.mkdir(parents=True, exist_ok=True)
            filename = f"{_sanitize_filename(session.id)}.fact_check_trace.json"
            self.path = trace_dir_path / filename

        self._add_flow_node("run", "run", f"Run {session.id}", None)
        self.record_event("run_started", {"session_id": session.id})

    def set_claim(self, claim: Claim | None) -> None:
        self.trace["claim"] = _serialize_claim(claim)

    def set_blueprint(self, blueprint_name: str, max_iterations: int, start_node_id: str) -> None:
        self.trace["blueprint"] = {
            "name": blueprint_name,
            "max_iterations": max_iterations,
            "start_node_id": start_node_id,
        }
        self.record_event(
            "blueprint_selected",
            {
                "name": blueprint_name,
                "max_iterations": max_iterations,
                "start_node_id": start_node_id,
            },
        )

    def start_iteration(self, iteration: int, node_before: str, evidence_count: int) -> None:
        iteration_node_id = f"iteration:{iteration}"
        iteration_record = {
            "iteration": iteration,
            "started_at": _timestamp(),
            "ended_at": None,
            "node_before": node_before,
            "node_after": node_before,
            "planner_messages": [],
            "planner_response": None,
            "decision": None,
            "check_updates": [],
            "delegated_tasks": [],
            "new_errors": [],
            "evidence_count_before": evidence_count,
            "evidence_count_after": evidence_count,
        }
        self.trace["iterations"].append(iteration_record)
        self._current_iteration_index = len(self.trace["iterations"]) - 1
        self._add_flow_node(iteration_node_id, "iteration", f"Iteration {iteration}", iteration)
        self._add_flow_edge(self._last_iteration_node_id or "run", iteration_node_id, "next")
        self._last_iteration_node_id = iteration_node_id
        self.record_event(
            "iteration_started",
            {
                "iteration": iteration,
                "node_before": node_before,
                "evidence_count_before": evidence_count,
            },
            iteration=iteration,
            flow_node_id=iteration_node_id,
        )

    def record_planner_messages(self, messages: list[Message], iteration: int) -> None:
        record = self._current_iteration()
        record["planner_messages"] = [_serialize_message(message) for message in messages]
        self.record_event(
            "planner_prompt",
            {"messages": record["planner_messages"]},
            iteration=iteration,
        )

    def record_planner_response(self, response_text: str, iteration: int) -> None:
        record = self._current_iteration()
        record["planner_response"] = response_text
        self.record_event(
            "planner_response",
            {"response_text": response_text},
            iteration=iteration,
        )

    def record_decision(self, decision: PlannerDecision, iteration: int) -> None:
        record = self._current_iteration()
        serialized = _serialize_dataclass(decision)
        record["decision"] = serialized
        record["check_updates"] = serialized.get("check_updates", [])
        self.record_event(
            "planner_decision",
            serialized,
            iteration=iteration,
        )

    def record_node_transition(
        self,
        *,
        iteration: int,
        from_node: str,
        to_node: str,
        requested_target: str | None,
    ) -> None:
        record = self._current_iteration()
        record["node_after"] = to_node
        self.record_event(
            "node_transition",
            {
                "from_node": from_node,
                "to_node": to_node,
                "requested_target": requested_target,
            },
            iteration=iteration,
        )

    def record_delegated_task(
        self,
        *,
        iteration: int,
        task_id: str,
        agent_type: str,
        instruction: str,
        follow_up_to: str | None,
        rationale: str | None,
        child_session_id: str,
    ) -> None:
        task_node_id = f"task:{iteration}:{task_id}"
        self._task_node_ids[task_id] = task_node_id
        self._current_iteration()["delegated_tasks"].append(
            {
                "task_id": task_id,
                "agent_type": agent_type,
                "instruction": instruction,
                "follow_up_to": follow_up_to,
                "rationale": rationale,
                "child_session_id": child_session_id,
                "result": None,
            }
        )
        self._add_flow_node(task_node_id, "task", f"{agent_type}:{task_id}", iteration)
        self._add_flow_edge(f"iteration:{iteration}", task_node_id, "delegates")
        if follow_up_to and follow_up_to in self._task_node_ids:
            self._add_flow_edge(self._task_node_ids[follow_up_to], task_node_id, "follow_up")
        self.record_event(
            "delegation_created",
            {
                "task_id": task_id,
                "agent_type": agent_type,
                "instruction": instruction,
                "follow_up_to": follow_up_to,
                "rationale": rationale,
                "child_session_id": child_session_id,
            },
            iteration=iteration,
            flow_node_id=task_node_id,
        )

    def record_delegated_task_result(self, *, iteration: int, task_id: str, result: AgentResult) -> None:
        for task in self._current_iteration()["delegated_tasks"]:
            if task["task_id"] == task_id:
                task["result"] = _serialize_result(result)
                break
        self.record_event(
            "delegation_completed",
            {
                "task_id": task_id,
                "result": _serialize_result(result),
            },
            iteration=iteration,
            flow_node_id=self._task_node_ids.get(task_id),
        )

    def record_synthesis(
        self,
        *,
        iteration: int | None,
        stage: str,
        instruction: str | None,
        answer: str,
        evidence_count: int,
    ) -> None:
        self.record_event(
            "synthesis",
            {
                "stage": stage,
                "instruction": instruction,
                "answer": answer,
                "evidence_count": evidence_count,
            },
            iteration=iteration,
        )

    def record_error(self, *, phase: str, message: str, iteration: int | None = None) -> None:
        if iteration is not None and self._current_iteration_index is not None:
            self._current_iteration()["new_errors"].append(message)
        self.record_event(
            "error",
            {
                "phase": phase,
                "message": message,
            },
            iteration=iteration,
        )

    def finish_iteration(self, *, iteration: int, evidence_count_after: int, new_errors: list[str]) -> None:
        record = self._current_iteration()
        record["ended_at"] = _timestamp()
        record["evidence_count_after"] = evidence_count_after
        for error in new_errors:
            if error not in record["new_errors"]:
                record["new_errors"].append(error)
        self.record_event(
            "iteration_finished",
            {
                "iteration": iteration,
                "node_after": record["node_after"],
                "evidence_count_after": evidence_count_after,
                "new_errors": list(record["new_errors"]),
            },
            iteration=iteration,
        )
        self._current_iteration_index = None

    def finalize(
        self,
        *,
        session: AgentSession,
        state: FactCheckSessionState | None,
        result: AgentResult | None,
        errors: list[str],
    ) -> None:
        self.trace["ended_at"] = _timestamp()
        self.trace["status"] = session.status.value
        if result is not None:
            self.trace["summary"]["result"] = _serialize_multimodal(result.result)
            self.trace["summary"]["message_count"] = len(result.messages)
        self.trace["summary"]["errors"] = list(errors)
        self.trace["summary"]["evidence_count"] = len(session.evidences)
        if state is not None:
            self.trace["summary"]["required_checks"] = {
                key: value.value for key, value in state.required_check_status.items()
            }
            self.trace["summary"]["required_check_reasons"] = dict(state.required_check_reasons)
            self.trace["summary"]["node_history"] = list(state.node_history)
            self.trace["summary"]["action_history"] = list(state.action_history)
            self.trace["summary"]["delegated_tasks"] = {
                task_id: _serialize_dataclass(task) for task_id, task in state.delegated_tasks.items()
            }
        self.record_event(
            "run_finished",
            {
                "status": self.trace["status"],
                "evidence_count": self.trace["summary"]["evidence_count"],
                "error_count": len(errors),
            },
        )
        if self.enabled and self.path is not None:
            self.path.write_text(json.dumps(self.trace, indent=2, ensure_ascii=True), encoding="utf-8")

    def record_event(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        iteration: int | None = None,
        flow_node_id: str | None = None,
    ) -> None:
        self._event_seq += 1
        self.trace["events"].append(
            {
                "seq": self._event_seq,
                "ts": _timestamp(),
                "event_type": event_type,
                "iteration": iteration,
                "flow_node_id": flow_node_id,
                "payload": payload,
            }
        )

    def _current_iteration(self) -> dict[str, Any]:
        assert self._current_iteration_index is not None, "No iteration is currently active."
        return self.trace["iterations"][self._current_iteration_index]

    def _add_flow_node(self, node_id: str, node_type: str, label: str, iteration: int | None) -> None:
        nodes: list[dict[str, Any]] = self.trace["flow"]["nodes"]
        if any(node["id"] == node_id for node in nodes):
            return
        nodes.append(
            {
                "id": node_id,
                "type": node_type,
                "label": label,
                "iteration": iteration,
            }
        )

    def _add_flow_edge(self, source: str, target: str, edge_type: str) -> None:
        edges: list[dict[str, Any]] = self.trace["flow"]["edges"]
        edge = {
            "source": source,
            "target": target,
            "type": edge_type,
        }
        if edge not in edges:
            edges.append(edge)
