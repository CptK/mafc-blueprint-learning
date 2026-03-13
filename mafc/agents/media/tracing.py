from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ezmm import MultimodalSequence

from mafc.agents.agent import AgentResult
from mafc.agents.common import AgentSession
from mafc.common.claim import Claim
from mafc.common.evidence import Evidence
from mafc.common.modeling.message import Message
from mafc.common.trace import TraceScope
from mafc.tools.tool_result import ToolResult


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize_filename(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return sanitized or "media_trace"


def get_media_trace_path(trace_dir: str | Path, session_id: str) -> Path:
    return Path(trace_dir) / f"{_sanitize_filename(session_id)}.media_trace.json"


_RAW_TRUNCATE_CHARS = 2000


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


def _serialize_claim(claim: Claim | None) -> dict[str, Any] | None:
    if claim is None:
        return None
    return {
        "id": claim.id,
        "text": str(claim),
        "author": claim.author,
        "date": claim.date.isoformat() if claim.date else None,
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


def _serialize_tool_result(tool_name: str, tool_result: ToolResult) -> dict[str, Any]:
    from mafc.tools.web_search.google_vision import GoogleRisResults

    raw = tool_result.raw
    sources_serialized = None
    if isinstance(raw, GoogleRisResults) and raw.sources:
        sources_serialized = []
        for s in raw.sources:
            sources_serialized.append(
                {
                    "reference": s.reference,
                    "url": getattr(s, "url", s.reference),
                    "title": getattr(s, "title", None),
                }
            )
    return {
        "tool": tool_name,
        "sources": sources_serialized,
        "raw_text": str(raw)[:_RAW_TRUNCATE_CHARS] if raw is not None else None,
        "takeaways": _serialize_multimodal(tool_result.takeaways),
    }


class MediaTraceRecorder:
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
        self.trace: dict[str, Any] = {
            "trace_version": self.trace_version,
            "agent": agent_name,
            "session_id": session.id,
            "parent_session_id": session.parent_session_id,
            "status": None,
            "started_at": _timestamp(),
            "ended_at": None,
            "goal": _serialize_multimodal(session.goal),
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
                "events_path": None,
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
        self.trace["planner_messages"] = [_serialize_message(m) for m in messages]
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
        self.trace["evidences"] = [_serialize_evidence(e) for e in evidences]
        self.record_event("evidences_collected", {"evidence_count": len(evidences)})

    def record_synthesis(self, answer: str, evidence_count: int) -> None:
        payload = {"answer": answer, "evidence_count": evidence_count}
        self.trace["synthesis"] = payload
        self.record_event("synthesis", payload)

    def record_error(self, phase: str, message: str) -> None:
        self.record_event("error", {"phase": phase, "message": message})

    def finalize(self, *, session: AgentSession, result: AgentResult | None, errors: list[str]) -> None:
        self.trace["ended_at"] = _timestamp()
        self.trace["status"] = session.status.value
        self.trace["summary"]["errors"] = list(errors)
        self.trace["summary"]["evidence_count"] = len(session.evidences)
        self.trace["summary"]["message_count"] = len(result.messages) if result is not None else 0
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

    def record_event(self, event_type: str, payload: dict[str, Any]) -> None:
        self._event_seq += 1
        event = {
            "seq": self._event_seq,
            "ts": _timestamp(),
            "event_type": event_type,
            "payload": payload,
        }
        self.trace["events"].append(event)
        if self.scope is not None:
            self.scope.append_event(event_type, event)
