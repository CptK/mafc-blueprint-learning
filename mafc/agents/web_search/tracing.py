from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ezmm import MultimodalSequence

from mafc.agents.agent import AgentResult
from mafc.agents.common import AgentSession
from mafc.common.evidence import Evidence
from mafc.common.modeling.message import Message
from mafc.common.trace import TraceScope
from mafc.tools.web_search.common import Source, WebSource


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize_filename(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return sanitized or "web_search_trace"


def get_web_search_trace_path(trace_dir: str | Path, session_id: str) -> Path:
    return Path(trace_dir) / f"{_sanitize_filename(session_id)}.web_search_trace.json"


_RAW_TRUNCATE_CHARS = 20000


def _serialize_multimodal(
    content: MultimodalSequence | None, truncate: int | None = None
) -> dict[str, Any] | None:
    if content is None:
        return None
    text = str(content)
    if truncate is not None and len(text) > truncate:
        text = text[:truncate] + f"… [{len(str(content)) - truncate} chars truncated]"
    return {
        "text": text,
        "images": [image.reference for image in content.images],
        "videos": [video.reference for video in content.videos],
    }


def _serialize_message(message: Message) -> dict[str, Any]:
    return {
        "role": message.role.value,
        "content": _serialize_multimodal(message.content),
    }


def _serialize_source(source: Source) -> dict[str, Any]:
    payload = {
        "reference": source.reference,
        "content": _serialize_multimodal(source.content),
        "takeaways": _serialize_multimodal(source.takeaways),
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


def _serialize_evidence(evidence: Evidence) -> dict[str, Any]:
    return {
        "source": evidence.source,
        "action": evidence.action.name,
        "action_repr": str(evidence.action),
        "preview": evidence.preview,
        "raw": _serialize_multimodal(evidence.raw, truncate=_RAW_TRUNCATE_CHARS),
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


class WebSearchTraceRecorder:
    trace_version = 1

    def __init__(
        self,
        trace_dir: str | Path | None,
        session: AgentSession,
        agent_name: str,
        trace_scope: TraceScope | None = None,
    ):
        self.enabled = trace_dir is not None
        self.trace_dir = Path(trace_dir) if trace_dir is not None else None
        self.write_file = trace_dir is not None and session.parent_session_id is None
        self.path: Path | None = None
        self.scope = trace_scope
        self._event_seq = 0
        self._current_iteration_index: int | None = None
        self.trace: dict[str, Any] = {
            "trace_version": self.trace_version,
            "agent": agent_name,
            "session_id": session.id,
            "parent_session_id": session.parent_session_id,
            "status": None,
            "started_at": _timestamp(),
            "ended_at": None,
            "goal": _serialize_multimodal(session.goal),
            "iterations": [],
            "events": [],
            "summary": {
                "result": None,
                "errors": [],
                "evidence_count": 0,
                "message_count": 0,
                "seen_queries": [],
                "events_path": None,
            },
        }
        if self.enabled:
            assert self.trace_dir is not None
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            self.path = get_web_search_trace_path(self.trace_dir, session.id)

        self.record_event("run_started", {"session_id": session.id})

    def start_iteration(self, *, step: int, evidence_count: int, seen_queries: set[str]) -> None:
        self.trace["iterations"].append(
            {
                "step": step,
                "started_at": _timestamp(),
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
        record["planner_messages"] = [_serialize_message(message) for message in messages]
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
            "evidence": _serialize_evidence(evidence) if evidence is not None else None,
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
        record["ended_at"] = _timestamp()
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
        self.trace["ended_at"] = _timestamp()
        self.trace["status"] = session.status.value
        self.trace["summary"]["errors"] = list(errors)
        self.trace["summary"]["evidence_count"] = len(session.evidences)
        self.trace["summary"]["message_count"] = len(result.messages) if result is not None else 0
        self.trace["summary"]["seen_queries"] = sorted(seen_queries)
        self.trace["summary"]["events_path"] = (
            str(self.trace_dir / f"{_sanitize_filename(session.id)}.trace.jsonl")
            if self.trace_dir is not None
            else None
        )
        if result is not None:
            self.trace["summary"]["result"] = _serialize_result(result)
        self.record_event(
            "run_finished",
            {
                "status": session.status.value,
                "evidence_count": len(session.evidences),
                "error_count": len(errors),
            },
        )
        if self.scope is not None:
            self.scope.set_summary(self.trace)
        if self.write_file and self.path is not None:
            self.path.write_text(json.dumps(self.trace, indent=2, ensure_ascii=True), encoding="utf-8")

    def record_event(self, event_type: str, payload: dict[str, Any], *, step: int | None = None) -> None:
        self._event_seq += 1
        event = {
            "seq": self._event_seq,
            "ts": _timestamp(),
            "event_type": event_type,
            "step": step,
            "payload": payload,
        }
        self.trace["events"].append(event)
        if self.scope is not None:
            self.scope.append_event(event_type, event)

    def _current_iteration(self) -> dict[str, Any]:
        assert self._current_iteration_index is not None, "No iteration is currently active."
        return self.trace["iterations"][self._current_iteration_index]
