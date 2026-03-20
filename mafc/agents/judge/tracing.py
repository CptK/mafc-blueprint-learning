from __future__ import annotations

from pathlib import Path

from mafc.agents.agent import AgentResult
from mafc.agents.common import AgentSession
from mafc.agents.tracing import (
    BaseTraceRecorder,
    sanitize_filename,
    serialize_claim,
    serialize_message,
    serialize_result,
    timestamp,
)
from mafc.common.modeling.message import Message
from mafc.common.trace import TraceScope

_RAW_TRUNCATE_CHARS = 2000


def get_judge_trace_path(trace_dir: str | Path, session_id: str) -> Path:
    return Path(trace_dir) / f"{sanitize_filename(session_id, 'judge_trace')}.judge_trace.json"


class JudgeTraceRecorder(BaseTraceRecorder):
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
            "claim": serialize_claim(session.claim),
            "evidence_count": len(session.evidences),
            "class_definitions": {},
            "prompt_messages": [],
            "model_response": None,
            "repair_prompt": None,
            "repair_response": None,
            "decision": None,
            "events": [],
            "summary": {
                "result": None,
                "label": None,
                "justification": None,
                "errors": [],
                "evidence_count": len(session.evidences),
                "message_count": 0,
                "total_cost_usd": 0.0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "by_model": {},
            },
        }
        self._finalize_init(session)

    def _make_path(self, session_id: str) -> Path:
        assert self.trace_dir is not None
        return get_judge_trace_path(self.trace_dir, session_id)

    def record_class_definitions(self, class_definitions: dict[str, str]) -> None:
        self.trace["class_definitions"] = {str(k): v for k, v in class_definitions.items()}
        self.record_event("class_definitions", {"class_definitions": self.trace["class_definitions"]})

    def record_prompt_messages(self, messages: list[Message]) -> None:
        self.trace["prompt_messages"] = [serialize_message(m) for m in messages]
        self.record_event("prompt_messages", {"messages": self.trace["prompt_messages"]})

    def record_model_response(self, response_text: str) -> None:
        self.trace["model_response"] = response_text
        self.record_event("model_response", {"response_text": response_text})

    def record_repair(self, *, prompt: str, response_text: str) -> None:
        self.trace["repair_prompt"] = prompt
        self.trace["repair_response"] = response_text
        self.record_event("repair", {"prompt": prompt, "response_text": response_text})

    def record_decision(self, label: str, justification: str) -> None:
        payload = {"label": label, "justification": justification}
        self.trace["decision"] = payload
        self.trace["summary"]["label"] = label
        self.trace["summary"]["justification"] = justification
        self.record_event("decision", payload)

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
