"""Lightweight trace recorder for the standalone Strategy fact-checker.

Self-contained — no dependency on the blueprint fact-check tracing. It records
the planning rounds, tool delegations, and judge run, and aggregates token/cost
usage from this agent's own LLM calls plus every sub-agent's trace summary.

The emitted ``trace`` dict deliberately mirrors the few keys the benchmark's
result-extraction helpers read (``judge_run.decision.label``,
``summary.total_cost_usd`` / token counts, ``trace_path``) so it plugs into the
existing eval with no changes there.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def _timestamp() -> str:
    return datetime.now().isoformat()


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)[:200] or "strategy_trace"


class StrategyRunTrace:
    """Accumulates one Strategy fact-check run as a JSON-serializable dict."""

    def __init__(
        self,
        trace_dir: str | Path | None,
        session_id: str,
        claim_text: str,
        strategy_word_count: int,
        true_label: str | None = None,
    ) -> None:
        self.trace_dir = Path(trace_dir) if trace_dir else None
        self.path: Path | None = (
            self.trace_dir / f"{_sanitize(session_id)}.strategy_trace.json" if self.trace_dir else None
        )
        self.trace: dict[str, Any] = {
            "agent": "StrategyAgent",
            "session_id": session_id,
            "status": None,
            "started_at": _timestamp(),
            "ended_at": None,
            "claim": claim_text,
            "strategy_word_count": strategy_word_count,
            "rounds": [],
            "judge_run": None,
            "trace_path": str(self.path) if self.path else None,
            "summary": {
                "result": None,
                "errors": [],
                "evidence_count": 0,
                "true_label": true_label,
                "total_cost_usd": 0.0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "by_model": {},
            },
        }

    # ---- usage accounting -------------------------------------------------

    def add_usage(self, response: Any, model_name: str) -> None:
        """Add one of this agent's own LLM calls to the totals."""
        cost = float(getattr(response, "total_cost", 0.0) or 0.0)
        in_tok = int(getattr(response, "input_token_count", 0) or 0)
        out_tok = int(getattr(response, "output_token_count", 0) or 0)
        self._add(model_name, cost, in_tok, out_tok)

    def absorb_child(self, child_trace: dict | None) -> None:
        """Add a sub-agent's (web_search / media / judge) usage from its trace summary."""
        summary = (child_trace or {}).get("summary") or {}
        self._add(
            None,
            float(summary.get("total_cost_usd", 0.0) or 0.0),
            int(summary.get("total_input_tokens", 0) or 0),
            int(summary.get("total_output_tokens", 0) or 0),
            by_model=summary.get("by_model") or {},
        )

    def _add(
        self,
        model_name: str | None,
        cost: float,
        in_tok: int,
        out_tok: int,
        by_model: dict | None = None,
    ) -> None:
        s = self.trace["summary"]
        s["total_cost_usd"] = round(s["total_cost_usd"] + cost, 6)
        s["total_input_tokens"] += in_tok
        s["total_output_tokens"] += out_tok
        if model_name is not None:
            entry = s["by_model"].setdefault(
                model_name, {"cost_usd": 0.0, "calls": 0, "input_tokens": 0, "output_tokens": 0}
            )
            entry["cost_usd"] = round(entry["cost_usd"] + cost, 6)
            entry["calls"] += 1
            entry["input_tokens"] += in_tok
            entry["output_tokens"] += out_tok
        for m_name, m_stats in (by_model or {}).items():
            entry = s["by_model"].setdefault(
                m_name, {"cost_usd": 0.0, "calls": 0, "input_tokens": 0, "output_tokens": 0}
            )
            entry["cost_usd"] = round(entry["cost_usd"] + m_stats.get("cost_usd", 0.0), 6)
            entry["calls"] += m_stats.get("calls", 0)
            entry["input_tokens"] += m_stats.get("input_tokens", 0)
            entry["output_tokens"] += m_stats.get("output_tokens", 0)

    # ---- structured records ----------------------------------------------

    def record_round(
        self,
        *,
        round_index: int,
        prompt: str,
        response_text: str,
        reasoning: str,
        tool_calls: list[dict],
        done: bool,
        evidence_count_after: int,
    ) -> None:
        self.trace["rounds"].append(
            {
                "round": round_index,
                "prompt": prompt,
                "response": response_text,
                "reasoning": reasoning,
                "tool_calls": tool_calls,
                "done": done,
                "evidence_count_after": evidence_count_after,
            }
        )

    def record_judge(self, judge_trace: dict | None) -> None:
        self.trace["judge_run"] = judge_trace
        self.absorb_child(judge_trace)

    def record_error(self, message: str) -> None:
        self.trace["summary"]["errors"].append(message)

    def finalize(self, *, status: str, result_text: str | None, evidence_count: int) -> None:
        self.trace["ended_at"] = _timestamp()
        self.trace["status"] = status
        self.trace["summary"]["result"] = result_text
        self.trace["summary"]["evidence_count"] = evidence_count
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.trace, indent=2, ensure_ascii=True), encoding="utf-8")
