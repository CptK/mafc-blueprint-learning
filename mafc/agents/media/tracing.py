from __future__ import annotations

from pathlib import Path
from typing import Any

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
from mafc.tools.tool_result import ToolResult

_RAW_TRUNCATE_CHARS = 2000


def get_media_trace_path(trace_dir: str | Path, session_id: str) -> Path:
    return Path(trace_dir) / f"{sanitize_filename(session_id, 'media_trace')}.media_trace.json"


def _serialize_tool_result(tool_name: str, tool_result: ToolResult) -> dict[str, Any]:
    from mafc.tools.web_search.google_vision import GoogleRisResults

    raw = tool_result.raw
    sources_serialized = None
    if isinstance(raw, GoogleRisResults) and raw.sources:
        sources_serialized = [
            {
                "reference": s.reference,
                "url": getattr(s, "url", s.reference),
                "title": getattr(s, "title", None),
            }
            for s in raw.sources
        ]
    return {
        "tool": tool_name,
        "sources": sources_serialized,
        "raw_text": str(raw)[:_RAW_TRUNCATE_CHARS] if raw is not None else None,
        "takeaways": serialize_multimodal(tool_result.takeaways),
    }


class MediaTraceRecorder(BaseTraceRecorder):
    trace_version = 1

    def __init__(
        self,
        trace_dir: str | Path | None,
        session: AgentSession,
        agent_name: str,
        trace_scope: TraceScope | None = None,
    ):
        super().__init__(trace_dir, session, trace_scope)
        self.trace = {
            "trace_version": self.trace_version,
            "agent": agent_name,
            "session_id": session.id,
            "parent_session_id": session.parent_session_id,
            "status": None,
            "started_at": timestamp(),
            "ended_at": None,
            "goal": serialize_multimodal(session.goal),
            "media_items": [],
            "planner_messages": [],
            "planner_response": None,
            "planned_tools": [],
            "tool_results": [],
            "evidences": [],
            "synthesis": None,
            "events": [],
            "summary": {
                "result": None,
                "errors": [],
                "evidence_count": 0,
                "message_count": 0,
                "total_cost_usd": 0.0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "by_model": {},
            },
        }
        if self.enabled:
            assert self.trace_dir is not None
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            self.path = get_media_trace_path(self.trace_dir, session.id)

        self.record_event("run_started", {"session_id": session.id})

    def record_media_items(self, media_references: list[str]) -> None:
        self.trace["media_items"] = list(media_references)
        self.record_event("media_items", {"media_items": list(media_references)})

    def record_planner_messages(self, messages: list[Message]) -> None:
        self.trace["planner_messages"] = [serialize_message(m) for m in messages]
        self.record_event("planner_prompt", {"messages": self.trace["planner_messages"]})

    def record_planner_response(self, response_text: str) -> None:
        self.trace["planner_response"] = response_text
        self.record_event("planner_response", {"response_text": response_text})

    def record_planned_tools(self, tools: list[str]) -> None:
        self.trace["planned_tools"] = list(tools)
        self.record_event("planned_tools", {"tools": list(tools)})

    def record_tool_result(self, tool_name: str, tool_result: ToolResult) -> None:
        payload = _serialize_tool_result(tool_name, tool_result)
        self.trace["tool_results"].append(payload)
        self.record_event("tool_result", payload)

    def record_evidences(self, evidences: list[Evidence]) -> None:
        self.trace["evidences"] = [serialize_evidence(e, raw_truncate=_RAW_TRUNCATE_CHARS) for e in evidences]
        self.record_event("evidences_collected", {"evidence_count": len(evidences)})

    def record_synthesis(self, answer: str, evidence_count: int) -> None:
        payload = {"answer": answer, "evidence_count": evidence_count}
        self.trace["synthesis"] = payload
        self.record_event("synthesis", payload)

    def record_error(self, phase: str, message: str) -> None:
        self.record_event("error", {"phase": phase, "message": message})

    def finalize(self, *, session: AgentSession, result: AgentResult | None, errors: list[str]) -> None:
        self.trace["ended_at"] = timestamp()
        self.trace["status"] = session.status.value
        self.trace["summary"]["errors"] = list(errors)
        self.trace["summary"]["evidence_count"] = len(session.evidences)
        self.trace["summary"]["message_count"] = len(result.messages) if result is not None else 0
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
