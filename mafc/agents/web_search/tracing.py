from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from mafc.agents.agent import AgentResult
from mafc.agents.common import AgentSession
from mafc.agents.tracing import (
    BaseTraceRecorder,
    sanitize_filename,
    serialize_evidence,
    serialize_message,
    serialize_multimodal,
    serialize_result,
    timestamp,
)
from mafc.common.evidence import Evidence
from mafc.common.modeling.message import Message
from mafc.common.trace import TraceScope
from mafc.tools.web_search.common import Source, WebSource

_RAW_TRUNCATE_CHARS = 20000


def get_web_search_trace_path(trace_dir: str | Path, session_id: str) -> Path:
    return Path(trace_dir) / f"{sanitize_filename(session_id, 'web_search_trace')}.web_search_trace.json"


def _serialize_source(source: Source) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "reference": source.reference,
        "content": serialize_multimodal(source.content),
        "takeaways": serialize_multimodal(source.takeaways),
    }
    if isinstance(source, WebSource):
        payload.update(
            {
                "url": source.url,
                "title": source.title,
                "preview": source.preview,
                "release_date": source.release_date.isoformat() if source.release_date else None,
            }
        )
    return payload


class WebSearchTraceRecorder(BaseTraceRecorder):
    trace_version = 1

    def __init__(
        self,
        trace_dir: str | Path | None,
        session: AgentSession,
        agent_name: str,
        trace_scope: TraceScope | None = None,
    ):
        super().__init__(trace_dir, session, trace_scope)
        self._current_iteration_index: int | None = None
        self.trace = {
            "trace_version": self.trace_version,
            "agent": agent_name,
            "session_id": session.id,
            "parent_session_id": session.parent_session_id,
            "status": None,
            "started_at": timestamp(),
            "ended_at": None,
            "goal": serialize_multimodal(session.goal),
            "iterations": [],
            "events": [],
            "summary": {
                "result": None,
                "errors": [],
                "evidence_count": 0,
                "message_count": 0,
                "seen_queries": [],
                "total_cost_usd": 0.0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "by_model": {},
            },
        }
        self._finalize_init(session)

    # Override to include ``step`` in the event dict.
    def record_event(self, event_type: str, payload: dict[str, Any], *, step: int | None = None) -> None:  # type: ignore[override]
        self._event_seq += 1
        event: dict[str, Any] = {
            "seq": self._event_seq,
            "ts": timestamp(),
            "event_type": event_type,
            "step": step,
            "payload": payload,
        }
        self.trace["events"].append(event)
        if self.scope is not None:
            self.scope.append_event(event_type, event)

    def start_iteration(self, *, step: int, evidence_count: int, seen_queries: set[str]) -> None:
        self.trace["iterations"].append(
            {
                "step": step,
                "started_at": timestamp(),
                "ended_at": None,
                "planner_messages": [],
                "planner_response": None,
                "planner_repair_prompt": None,
                "planner_repair_response": None,
                "resolved_plan": None,
                "search_results": [],
                "selected_sources": [],
                "selection_prompt": None,
                "selection_response": None,
                "retrievals": [],
                "synthesis": None,
                "new_errors": [],
                "evidence_count_before": evidence_count,
                "evidence_count_after": evidence_count,
                "seen_queries_before": sorted(seen_queries),
                "seen_queries_after": sorted(seen_queries),
            }
        )
        self._current_iteration_index = len(self.trace["iterations"]) - 1
        self.record_event(
            "iteration_started",
            {
                "step": step,
                "evidence_count_before": evidence_count,
                "seen_queries_before": sorted(seen_queries),
            },
            step=step,
        )

    def record_planner_messages(self, messages: list[Message], *, step: int) -> None:
        record = self._current_iteration()
        record["planner_messages"] = [serialize_message(message) for message in messages]
        self.record_event("planner_prompt", {"messages": record["planner_messages"]}, step=step)

    def record_planner_response(self, response_text: str, *, step: int) -> None:
        self._current_iteration()["planner_response"] = response_text
        self.record_event("planner_response", {"response_text": response_text}, step=step)

    def record_planner_repair(self, *, prompt: str, response_text: str, step: int) -> None:
        record = self._current_iteration()
        record["planner_repair_prompt"] = prompt
        record["planner_repair_response"] = response_text
        self.record_event(
            "planner_repair",
            {"prompt": prompt, "response_text": response_text},
            step=step,
        )

    def record_resolved_plan(
        self,
        *,
        step: int,
        queries: list[str],
        done: bool,
        should_terminate: bool,
        fallback_used: bool = False,
    ) -> None:
        payload = {
            "queries": list(queries),
            "done": done,
            "should_terminate": should_terminate,
            "fallback_used": fallback_used,
        }
        self._current_iteration()["resolved_plan"] = payload
        self.record_event("resolved_plan", payload, step=step)

    def record_search_result(
        self,
        *,
        step: int,
        query_text: str,
        sources: list[Source] | None,
        errors: list[str],
        marked_seen: bool,
    ) -> None:
        payload = {
            "query_text": query_text,
            "sources": None if sources is None else [_serialize_source(source) for source in sources],
            "errors": list(errors),
            "marked_seen": marked_seen,
        }
        self._current_iteration()["search_results"].append(payload)
        self.record_event("search_result", payload, step=step)

    def record_selected_sources(
        self,
        *,
        step: int,
        selected_sources: list[tuple[str, list[Source] | None]],
        selection_prompt: str | None = None,
        selection_response: str | None = None,
    ) -> None:
        payload = [
            {
                "query_text": query_text,
                "sources": None if sources is None else [_serialize_source(source) for source in sources],
            }
            for query_text, sources in selected_sources
        ]
        record = self._current_iteration()
        record["selected_sources"] = payload
        if selection_prompt is not None:
            record["selection_prompt"] = selection_prompt
        if selection_response is not None:
            record["selection_response"] = selection_response
        self.record_event("selected_sources", {"selected_sources": payload}, step=step)

    def record_retrieval(
        self,
        *,
        step: int,
        query_text: str,
        source: WebSource,
        retrieved_content: str | None,
        evidence: Evidence | None,
        irrelevant: bool = False,
    ) -> None:
        payload = {
            "query_text": query_text,
            "source": _serialize_source(source),
            "retrieved_content": (
                retrieved_content[:_RAW_TRUNCATE_CHARS]
                + f"… [{len(retrieved_content) - _RAW_TRUNCATE_CHARS} chars truncated]"
                if retrieved_content and len(retrieved_content) > _RAW_TRUNCATE_CHARS
                else retrieved_content
            ),
            "evidence": (
                serialize_evidence(evidence, raw_truncate=_RAW_TRUNCATE_CHARS)
                if evidence is not None
                else None
            ),
            "irrelevant": irrelevant,
        }
        self._current_iteration()["retrievals"].append(payload)
        self.record_event("retrieval_result", payload, step=step)

    def record_synthesis(
        self,
        *,
        step: int | None,
        stage: str,
        instruction: str,
        answer: str,
        evidence_count: int,
    ) -> None:
        payload = {
            "stage": stage,
            "instruction": instruction,
            "answer": answer,
            "evidence_count": evidence_count,
        }
        if step is not None and self._current_iteration_index is not None:
            self._current_iteration()["synthesis"] = payload
        self.record_event("synthesis", payload, step=step)

    def record_error(self, *, step: int | None, phase: str, message: str) -> None:
        if step is not None and self._current_iteration_index is not None:
            self._current_iteration()["new_errors"].append(message)
        self.record_event("error", {"phase": phase, "message": message}, step=step)

    def finish_iteration(
        self, *, step: int, evidence_count: int, seen_queries: set[str], new_errors: list[str]
    ) -> None:
        record = self._current_iteration()
        record["ended_at"] = timestamp()
        record["evidence_count_after"] = evidence_count
        record["seen_queries_after"] = sorted(seen_queries)
        for error in new_errors:
            if error not in record["new_errors"]:
                record["new_errors"].append(error)
        self.record_event(
            "iteration_finished",
            {
                "step": step,
                "evidence_count_after": evidence_count,
                "seen_queries_after": sorted(seen_queries),
                "new_errors": list(record["new_errors"]),
            },
            step=step,
        )
        self._current_iteration_index = None

    def finalize(
        self,
        *,
        session: AgentSession,
        result: AgentResult | None,
        errors: list[str],
        seen_queries: set[str],
    ) -> None:
        self.trace["ended_at"] = timestamp()
        self.trace["status"] = session.status.value
        self.trace["summary"]["errors"] = list(errors)
        self.trace["summary"]["evidence_count"] = len(session.evidences)
        self.trace["summary"]["message_count"] = len(result.messages) if result is not None else 0
        self.trace["summary"]["seen_queries"] = sorted(seen_queries)
        if result is not None:
            self.trace["summary"]["result"] = serialize_result(result, raw_truncate=_RAW_TRUNCATE_CHARS)
        self.record_event(
            "run_finished",
            {
                "status": session.status.value,
                "evidence_count": len(session.evidences),
                "error_count": len(errors),
            },
        )
        self._write_usage_stats()
        self._persist()

    def _make_path(self, session_id: str) -> Path:
        assert self.trace_dir is not None
        return get_web_search_trace_path(self.trace_dir, session_id)

    def _current_iteration(self) -> dict[str, Any]:
        assert self._current_iteration_index is not None, "No iteration is currently active."
        return cast(dict[str, Any], self.trace["iterations"][self._current_iteration_index])
