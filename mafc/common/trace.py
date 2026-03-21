from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class _TraceState:
    trace_id: str
    trace_dir: Path | None
    next_seq: int = 1


class TraceScope:
    """Hierarchical in-memory trace scope with append-only event persistence."""

    def __init__(self, state: _TraceState, node: dict[str, Any], path: list[str]):
        self._state = state
        self._node = node
        self._path = path

    @classmethod
    def root(
        cls,
        *,
        scope_type: str,
        trace_id: str,
        trace_dir: str | Path | None = None,
        key: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TraceScope:
        state = _TraceState(trace_id=trace_id, trace_dir=Path(trace_dir) if trace_dir is not None else None)
        node: dict[str, Any] = {
            "scope_type": scope_type,
            "key": key,
            "metadata": metadata or {},
            "events": [],
            "children": [],
            "summary": None,
        }
        return cls(state=state, node=node, path=[scope_type if key is None else f"{scope_type}:{key}"])

    def child_scope(
        self,
        scope_type: str,
        *,
        key: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TraceScope:
        child: dict[str, Any] = {
            "scope_type": scope_type,
            "key": key,
            "metadata": metadata or {},
            "events": [],
            "children": [],
            "summary": None,
        }
        self._node["children"].append(child)
        path_part = scope_type if key is None else f"{scope_type}:{key}"
        return TraceScope(state=self._state, node=child, path=[*self._path, path_part])

    def append_event(self, event_type: str, payload: dict[str, Any]) -> None:
        event = {
            "seq": self._state.next_seq,
            "scope_path": list(self._path),
            "event_type": event_type,
            "payload": payload,
        }
        self._state.next_seq += 1
        self._node["events"].append(event)

    def set_summary(self, summary: dict[str, Any]) -> None:
        self._node["summary"] = summary

    def snapshot(self) -> dict[str, Any]:
        return self._clone_node(self._node)

    def _clone_node(self, node: dict[str, Any]) -> dict[str, Any]:
        return {
            "scope_type": node["scope_type"],
            "key": node["key"],
            "metadata": dict(node["metadata"]),
            "events": [dict(event) for event in node["events"]],
            "children": [self._clone_node(child) for child in node["children"]],
            "summary": node["summary"],
        }
