"""Shared base for all agent trace recorders.

Each agent-specific recorder subclasses ``BaseTraceRecorder`` and adds only
the fields and ``record_*`` methods that are unique to that agent.
"""

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

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_filename(value: str, default: str = "trace") -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return sanitized or default


def serialize_multimodal(
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


def serialize_message(message: Message) -> dict[str, Any]:
    return {
        "role": message.role.value,
        "content": serialize_multimodal(message.content),
    }


def serialize_evidence(evidence: Evidence, raw_truncate: int = 20000) -> dict[str, Any]:
    return {
        "source": evidence.source,
        "action": evidence.action.name,
        "action_repr": str(evidence.action),
        "preview": evidence.preview,
        "raw": serialize_multimodal(evidence.raw, truncate=raw_truncate),
        "takeaways": serialize_multimodal(evidence.takeaways),
    }


def serialize_result(result: AgentResult, raw_truncate: int = 20000) -> dict[str, Any]:
    return {
        "status": result.status.value if result.status is not None else None,
        "result": serialize_multimodal(result.result),
        "errors": list(result.errors),
        "evidences": [serialize_evidence(e, raw_truncate=raw_truncate) for e in result.evidences],
        "message_count": len(result.messages),
    }


def serialize_claim(claim: Claim | None) -> dict[str, Any] | None:
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


# ---------------------------------------------------------------------------
# Base recorder
# ---------------------------------------------------------------------------


class BaseTraceRecorder:
    """Common infrastructure shared by all agent trace recorders.

    Subclasses must:
    - Call ``super().__init__(trace_dir, session, trace_scope)``
    - Build ``self.trace`` after calling super
    - Set ``self.path`` for the agent-specific file path
    - Call ``self.record_event("run_started", ...)`` at the end of their ``__init__``
    - Call ``self._write_usage_stats()`` and ``self._persist()`` at the end of ``finalize()``
    """

    trace_version = 1

    def __init__(
        self,
        trace_dir: str | Path | None,
        session: AgentSession,
        trace_scope: TraceScope | None = None,
    ) -> None:
        self.enabled = trace_dir is not None
        self.trace_dir = Path(trace_dir) if trace_dir is not None else None
        self.write_file = trace_dir is not None and session.parent_session_id is None
        self.path: Path | None = None
        self.scope = trace_scope
        self._event_seq = 0
        self._total_cost: float = 0.0
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._by_model: dict[str, dict] = {}
        # Subclass is responsible for assigning self.trace after calling super().__init__
        self.trace: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Usage tracking
    # ------------------------------------------------------------------

    def add_usage(self, response: "Response", model_name: str) -> None:  # type: ignore[name-defined]
        from mafc.common.modeling.model import Response as _Response  # noqa: F401

        self._total_cost += response.total_cost
        in_tok = response.input_token_count or 0
        out_tok = response.output_token_count or 0
        self._total_input_tokens += in_tok
        self._total_output_tokens += out_tok
        entry = self._by_model.setdefault(
            model_name, {"cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0}
        )
        entry["cost_usd"] += response.total_cost
        entry["input_tokens"] += in_tok
        entry["output_tokens"] += out_tok

    # ------------------------------------------------------------------
    # Event logging
    # ------------------------------------------------------------------

    def record_event(self, event_type: str, payload: dict[str, Any], **kwargs: Any) -> None:
        """Append a structured event to ``self.trace["events"]``.

        Subclasses that need extra fields in the event dict (e.g. ``step``,
        ``iteration``) should override this method.
        """
        self._event_seq += 1
        event: dict[str, Any] = {
            "seq": self._event_seq,
            "ts": timestamp(),
            "event_type": event_type,
            **kwargs,
            "payload": payload,
        }
        self.trace["events"].append(event)
        if self.scope is not None:
            self.scope.append_event(event_type, event)

    # ------------------------------------------------------------------
    # Finalize helpers
    # ------------------------------------------------------------------

    def _write_usage_stats(self) -> None:
        """Write accumulated usage counters into ``self.trace["summary"]``."""
        self.trace["summary"]["total_cost_usd"] = round(self._total_cost, 6)
        self.trace["summary"]["total_input_tokens"] = self._total_input_tokens
        self.trace["summary"]["total_output_tokens"] = self._total_output_tokens
        self.trace["summary"]["by_model"] = {
            name: {
                "cost_usd": round(stats["cost_usd"], 6),
                "input_tokens": stats["input_tokens"],
                "output_tokens": stats["output_tokens"],
            }
            for name, stats in self._by_model.items()
        }

    def _persist(self) -> None:
        """Sync trace to the TraceScope and flush to disk if enabled."""
        if self.scope is not None:
            self.scope.set_summary(self.trace)
        if self.write_file and self.path is not None:
            self.path.write_text(json.dumps(self.trace, indent=2, ensure_ascii=True), encoding="utf-8")
