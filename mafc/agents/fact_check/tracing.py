from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from mafc.agents.agent import AgentResult
from mafc.agents.common import AgentSession
from mafc.agents.fact_check.models import FactCheckSessionState, PlannerDecision
from mafc.agents.tracing import (
    BaseTraceRecorder,
    sanitize_filename,
    serialize_claim,
    serialize_message,
    serialize_multimodal,
    serialize_result,
    timestamp,
)
from mafc.blueprints.selector import BlueprintSelectionResult
from mafc.common.claim import Claim
from mafc.common.modeling.message import Message
from mafc.common.trace import TraceScope

_RAW_TRUNCATE_CHARS = 20000


def _serialize_dataclass(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    return value


class FactCheckTraceRecorder(BaseTraceRecorder):
    """Captures one fact-check run as structured JSON for later graphing."""

    trace_version = 1

    def __init__(
        self,
        trace_dir: str | Path | None,
        session: AgentSession,
        agent_name: str,
        trace_scope: TraceScope | None = None,
        true_label: str | None = None,
    ):
        super().__init__(trace_dir, session, trace_scope)
        self._current_iteration_index: int | None = None
        self._last_iteration_node_id: str | None = None
        self._task_node_ids: dict[str, str] = {}
        self.trace: dict[str, Any] = {
            "trace_version": self.trace_version,
            "agent": agent_name,
            "session_id": session.id,
            "parent_session_id": session.parent_session_id,
            "status": None,
            "started_at": timestamp(),
            "ended_at": None,
            "goal": serialize_multimodal(session.goal),
            "claim": serialize_claim(session.claim),
            "blueprint": None,
            "iterations": [],
            "judge_run": None,
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
                "true_label": true_label,
                "total_cost_usd": 0.0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "by_model": {},
                "runtime_seconds": None,
            },
        }
        if self.enabled:
            assert self.trace_dir is not None
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{sanitize_filename(session.id, 'fact_check_trace')}.fact_check_trace.json"
            self.path = self.trace_dir / filename

        self._add_flow_node("run", "run", f"Run {session.id}", None)
        self.record_event("run_started", {"session_id": session.id})

    # Override record_event to include ``iteration`` and ``flow_node_id``.
    def record_event(  # type: ignore[override]
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        iteration: int | None = None,
        flow_node_id: str | None = None,
    ) -> None:
        self._event_seq += 1
        event: dict[str, Any] = {
            "seq": self._event_seq,
            "ts": timestamp(),
            "event_type": event_type,
            "iteration": iteration,
            "flow_node_id": flow_node_id,
            "payload": payload,
        }
        self.trace["events"].append(event)
        if self.scope is not None:
            self.scope.append_event(event_type, event)

    def set_claim(self, claim: Claim | None) -> None:
        self.trace["claim"] = serialize_claim(claim)

    def set_blueprint(self, selection_result: BlueprintSelectionResult) -> None:
        bp = selection_result.selected_blueprint
        selection: dict[str, Any] = {
            "mode": selection_result.selection_mode.value,
            "claim_features": selection_result.claim_features.model_dump(),
            "all_blueprints": selection_result.all_blueprints,
            "surviving_blueprints": selection_result.surviving_blueprints,
            "rejected_blueprints": [
                {"blueprint_name": r.blueprint_name, "reason": r.reason}
                for r in selection_result.rejected_blueprints
            ],
            "reason": selection_result.reason,
            "llm_prompt": selection_result.llm_prompt,
            "llm_raw_response": selection_result.llm_raw_response,
        }
        self.trace["blueprint"] = {
            "name": bp.name,
            "max_iterations": bp.policy_constraints.max_iterations,
            "start_node_id": bp.verification_graph.start_node,
            "selection": selection,
        }
        self.record_event(
            "blueprint_selected",
            {
                "name": bp.name,
                "max_iterations": bp.policy_constraints.max_iterations,
                "start_node_id": bp.verification_graph.start_node,
                "selection": selection,
            },
        )

    def start_iteration(self, iteration: int, node_before: str, evidence_count: int) -> None:
        iteration_node_id = f"iteration:{iteration}"
        iteration_record = {
            "iteration": iteration,
            "started_at": timestamp(),
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
        record["planner_messages"] = [serialize_message(message) for message in messages]
        self.record_event(
            "planner_prompt",
            {"messages": record["planner_messages"]},
            iteration=iteration,
        )

    def record_planner_response(self, response_text: str, iteration: int) -> None:
        self._current_iteration()["planner_response"] = response_text
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
        self.record_event("planner_decision", serialized, iteration=iteration)

    def record_node_transition(
        self,
        *,
        iteration: int,
        from_node: str,
        to_node: str,
        requested_target: str | None,
    ) -> None:
        self._current_iteration()["node_after"] = to_node
        self.record_event(
            "node_transition",
            {"from_node": from_node, "to_node": to_node, "requested_target": requested_target},
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
                task["result"] = serialize_result(result, raw_truncate=_RAW_TRUNCATE_CHARS)
                break
        self.record_event(
            "delegation_completed",
            {"task_id": task_id, "result": serialize_result(result, raw_truncate=_RAW_TRUNCATE_CHARS)},
            iteration=iteration,
            flow_node_id=self._task_node_ids.get(task_id),
        )

    def record_delegated_task_trace(
        self,
        *,
        iteration: int,
        task_id: str,
        child_trace: dict[str, Any],
    ) -> None:
        for task in self._current_iteration()["delegated_tasks"]:
            if task["task_id"] == task_id:
                task["child_trace"] = child_trace
                break
        self.record_event(
            "delegation_trace_attached",
            {
                "task_id": task_id,
                "child_agent": child_trace.get("agent"),
                "child_session_id": child_trace.get("session_id"),
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
            {"stage": stage, "instruction": instruction, "answer": answer, "evidence_count": evidence_count},
            iteration=iteration,
        )

    def record_error(self, *, phase: str, message: str, iteration: int | None = None) -> None:
        if iteration is not None and self._current_iteration_index is not None:
            self._current_iteration()["new_errors"].append(message)
        self.record_event("error", {"phase": phase, "message": message}, iteration=iteration)

    def finish_iteration(self, *, iteration: int, evidence_count_after: int, new_errors: list[str]) -> None:
        record = self._current_iteration()
        record["ended_at"] = timestamp()
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

    def record_judge_run(self, judge_trace: dict[str, Any]) -> None:
        self.trace["judge_run"] = judge_trace
        self.record_event(
            "judge_run_attached",
            {
                "label": (judge_trace.get("decision") or {}).get("label"),
                "session_id": judge_trace.get("session_id"),
            },
        )

    def finalize(
        self,
        *,
        session: AgentSession,
        state: FactCheckSessionState | None,
        result: AgentResult | None,
        errors: list[str],
    ) -> None:
        self.trace["ended_at"] = timestamp()
        self.trace["status"] = session.status.value
        if result is not None:
            self.trace["summary"]["result"] = serialize_multimodal(result.result)
            self.trace["summary"]["message_count"] = len(result.messages)
        self.trace["summary"]["errors"] = list(errors)
        self.trace["summary"]["evidence_count"] = len(session.evidences)
        try:
            started = datetime.fromisoformat(self.trace["started_at"])
            ended = datetime.fromisoformat(self.trace["ended_at"])
            self.trace["summary"]["runtime_seconds"] = round((ended - started).total_seconds(), 1)
        except Exception:
            pass
        # Aggregate usage from child traces and judge run before writing stats.
        self._aggregate_child_usage()
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
        self._write_usage_stats()
        # fact_check uses self.enabled rather than self.write_file (root session is still written)
        if self.scope is not None:
            self.scope.set_summary(self.trace)
        if self.enabled and self.path is not None:
            import json

            self.path.write_text(json.dumps(self.trace, indent=2, ensure_ascii=True), encoding="utf-8")

    def _aggregate_child_usage(self) -> None:
        """Sum usage from all child traces and the judge run into this recorder's accumulators."""
        for iteration in self.trace.get("iterations", []):
            for task in iteration.get("delegated_tasks", []):
                child_summary = (task.get("child_trace") or {}).get("summary") or {}
                self._total_cost += child_summary.get("total_cost_usd", 0.0)
                self._total_input_tokens += child_summary.get("total_input_tokens", 0)
                self._total_output_tokens += child_summary.get("total_output_tokens", 0)
                for m_name, m_stats in (child_summary.get("by_model") or {}).items():
                    entry = self._by_model.setdefault(
                        m_name, {"cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0}
                    )
                    entry["cost_usd"] += m_stats.get("cost_usd", 0.0)
                    entry["input_tokens"] += m_stats.get("input_tokens", 0)
                    entry["output_tokens"] += m_stats.get("output_tokens", 0)
        judge_summary = (self.trace.get("judge_run") or {}).get("summary") or {}
        self._total_cost += judge_summary.get("total_cost_usd", 0.0)
        self._total_input_tokens += judge_summary.get("total_input_tokens", 0)
        self._total_output_tokens += judge_summary.get("total_output_tokens", 0)
        for m_name, m_stats in (judge_summary.get("by_model") or {}).items():
            entry = self._by_model.setdefault(
                m_name, {"cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0}
            )
            entry["cost_usd"] += m_stats.get("cost_usd", 0.0)
            entry["input_tokens"] += m_stats.get("input_tokens", 0)
            entry["output_tokens"] += m_stats.get("output_tokens", 0)

    def _current_iteration(self) -> dict[str, Any]:
        assert self._current_iteration_index is not None, "No iteration is currently active."
        return self.trace["iterations"][self._current_iteration_index]

    def _add_flow_node(self, node_id: str, node_type: str, label: str, iteration: int | None) -> None:
        nodes: list[dict[str, Any]] = self.trace["flow"]["nodes"]
        if any(node["id"] == node_id for node in nodes):
            return
        nodes.append({"id": node_id, "type": node_type, "label": label, "iteration": iteration})

    def _add_flow_edge(self, source: str, target: str, edge_type: str) -> None:
        edges: list[dict[str, Any]] = self.trace["flow"]["edges"]
        edge = {"source": source, "target": target, "type": edge_type}
        if edge not in edges:
            edges.append(edge)
