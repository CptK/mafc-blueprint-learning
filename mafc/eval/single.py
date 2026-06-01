"""Single-sample fact-check execution helpers.

Hosts the per-sample logic that used to live inside ``mafc/eval/runner.py`` so
the learning loop can call it without reimplementing agent construction or
result extraction.

Two entry points:
- ``build_fact_check_agent`` — assemble a fully-wired ``FactCheckAgent`` from a
  ``BenchmarkRunConfig``.
- ``run_fact_check`` — execute one sample with an existing agent and return a
  result dict identical in shape to what the benchmark runner writes.

The extract_* helpers pull individual fields out of an ``AgentResult`` for
callers that need partial information (e.g. the learning executor).
"""

from __future__ import annotations

import hashlib
import json
import time
import traceback as _traceback
from pathlib import Path
from typing import Any, Protocol

from mafc.agents.common import AgentSession
from mafc.agents.fact_check.agent import FactCheckAgent
from mafc.agents.judge.agent import JudgeAgent
from mafc.agents.media.agent import MediaAgent
from mafc.agents.web_search.agent import WebSearchAgent
from mafc.blueprints import BlueprintRegistry, BlueprintSelector
from mafc.common.logger import logger
from mafc.common.modeling import make_model
from mafc.common.modeling.prompt import Prompt
from mafc.eval.run_config import BenchmarkRunConfig
from mafc.tools.web_search.google_search import GoogleSearchPlatform


class _HasSampleExtraFields(Protocol):
    def sample_extra_fields(self, sample: Any) -> dict[str, Any]: ...


# ---------------------------------------------------------------------------
# Agent construction
# ---------------------------------------------------------------------------


def build_fact_check_agent(
    config: BenchmarkRunConfig,
    benchmark: Any,
    trace_dir: Path | None,
    cache_dir: Path | None = None,
) -> FactCheckAgent:
    """Build a FactCheckAgent with every sub-agent wired from ``config``.

    ``benchmark`` is used for the judge's class definitions and extra rules.
    """
    fc_cfg = config.agents.fact_check
    ws_cfg = config.agents.web_search
    media_cfg = config.agents.media
    judge_cfg = config.agents.judge
    bp_cfg = config.blueprints

    planner_model = make_model(
        fc_cfg.model, temperature=fc_cfg.temperature, max_response_length=fc_cfg.max_response_length
    )
    worker_model = make_model(
        ws_cfg.model, temperature=ws_cfg.temperature, max_response_length=ws_cfg.max_response_length
    )
    summarization_model = (
        make_model(
            ws_cfg.summarization_model,
            temperature=ws_cfg.summarization_temperature or ws_cfg.temperature,
            top_p=ws_cfg.summarization_top_p or ws_cfg.top_p,
            top_k=ws_cfg.summarization_top_k or ws_cfg.top_k,
            max_response_length=ws_cfg.summarization_max_response_length or ws_cfg.max_response_length,
            thinking=ws_cfg.summarization_thinking,
            presence_penalty=ws_cfg.summarization_presence_penalty,
        )
        if ws_cfg.summarization_model
        else worker_model
    )
    media_model = make_model(
        media_cfg.model, temperature=media_cfg.temperature, max_response_length=media_cfg.max_response_length
    )
    judge_model = make_model(
        judge_cfg.model, temperature=judge_cfg.temperature, max_response_length=judge_cfg.max_response_length
    )
    selector_model = make_model(
        bp_cfg.selector_model, max_response_length=bp_cfg.selector_max_response_length
    )

    registry = BlueprintRegistry.from_path(bp_cfg.config_dir)
    selector = BlueprintSelector(model=selector_model, registry=registry, default_blueprint_name="generic")

    media_agent = MediaAgent(model=media_model, summarization_model=media_model)
    web_search_agent = WebSearchAgent(
        main_model=worker_model,
        summarization_model=summarization_model,
        n_workers=ws_cfg.workers,
        max_iterations=ws_cfg.max_iterations,
        max_queries_per_step=ws_cfg.max_queries_per_step,
        max_results_per_query=ws_cfg.max_results_per_query,
        search_tool=GoogleSearchPlatform(cache_dir=cache_dir),
    )
    judge_agent = JudgeAgent(
        model=judge_model,
        class_definitions=benchmark.class_definitions,
        extra_judge_rules=benchmark.extra_judge_rules,
    )
    return FactCheckAgent(
        model=planner_model,
        blueprint_selector=selector,
        delegation_agents={"media": [media_agent], "web_search": [web_search_agent]},
        judge_agent=judge_agent,
        n_workers=fc_cfg.workers,
        trace_dir=str(trace_dir) if trace_dir else None,
    )


def compute_agent_fingerprint(config: BenchmarkRunConfig) -> str:
    """Produce a stable short hash of the agent-configuration parts that affect outcomes.

    Two configs that produce equivalent agent behaviour should hash to the same
    fingerprint; any change that could alter a fact-check result (model name,
    temperature, max iterations, the benchmark's label space) should change it.
    """
    payload = {
        "fact_check": config.agents.fact_check.model_dump(),
        "web_search": config.agents.web_search.model_dump(),
        "media": config.agents.media.model_dump(),
        "judge": config.agents.judge.model_dump(),
        "blueprints": {
            "selector_model": config.blueprints.selector_model,
            "selector_max_response_length": config.blueprints.selector_max_response_length,
        },
        # Label scheme affects the verdict label space the agent is allowed to
        # emit, so two runs with different schemes must NOT share cache entries.
        "label_scheme": config.benchmark.label_scheme,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Result extraction
# ---------------------------------------------------------------------------


def extract_predicted_label(agent_result) -> str | None:
    judge_run = (agent_result.trace or {}).get("judge_run") or {}
    decision = judge_run.get("decision") or {}
    return decision.get("label") or None


def extract_cost(agent_result) -> dict[str, Any]:
    trace_summary = (agent_result.trace or {}).get("summary") or {}
    return {
        "cost_usd": trace_summary.get("total_cost_usd", 0.0),
        "input_tokens": trace_summary.get("total_input_tokens", 0),
        "output_tokens": trace_summary.get("total_output_tokens", 0),
        "total_tokens": (
            trace_summary.get("total_input_tokens", 0) + trace_summary.get("total_output_tokens", 0)
        ),
    }


def extract_blueprint_info(agent_result) -> dict[str, Any]:
    trace = agent_result.trace or {}
    bp = trace.get("blueprint") or {}
    selection = bp.get("selection") or {}
    return {
        "blueprint_name": bp.get("name") or "unknown",
        "selection_mode": selection.get("mode") or "unknown",
        "n_iterations": len(trace.get("iterations") or []),
    }


def extract_required_check_statuses(agent_result) -> dict[str, str]:
    """Return ``{check_id: status_str}`` from the trace's summary, or {} if absent."""
    summary = (agent_result.trace or {}).get("summary") or {}
    return dict(summary.get("required_checks") or {})


def extract_node_history(agent_result) -> list[str]:
    """Return the ordered list of visited node ids from the trace summary."""
    summary = (agent_result.trace or {}).get("summary") or {}
    return list(summary.get("node_history") or [])


def extract_judge_reason(agent_result) -> str | None:
    """Return the judge's reasoning text, or None if the judge did not run."""
    judge_run = (agent_result.trace or {}).get("judge_run") or {}
    decision = judge_run.get("decision") or {}
    # Different judge implementations may put the rationale under different keys;
    # fall back across the most common ones.
    for key in ("reason", "reasoning", "justification", "rationale"):
        value = decision.get(key)
        if value:
            return str(value)
    return None


def extract_trace_path(agent_result) -> str | None:
    """Return the path the trace was written to, or None if traces are disabled."""
    return (agent_result.trace or {}).get("trace_path")


# ---------------------------------------------------------------------------
# Single-sample run
# ---------------------------------------------------------------------------


def _zero_cost() -> dict[str, Any]:
    return {"cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def run_fact_check(
    sample,
    agent: FactCheckAgent,
    *,
    benchmark: _HasSampleExtraFields | None = None,
) -> dict[str, Any]:
    """Execute one sample through ``agent`` and return a result dict.

    The returned dict mirrors what the benchmark runner writes per sample, with
    additional fields for in-trace state (required checks, node history, judge
    reasoning) that downstream consumers like the learning executor need.

    Args:
        sample: A benchmark sample (or any object exposing ``id``, ``input``,
            and ``label.value``). ``input.date`` is consulted for the search
            cutoff if present.
        agent: A pre-built ``FactCheckAgent``. Callers that need ad-hoc agent
            construction should use ``build_fact_check_agent`` first.
        benchmark: Optional benchmark used solely for ``sample_extra_fields``
            (e.g. raw ground-truth scores for regression metrics). Pass ``None``
            when calling from contexts that don't need benchmark-specific fields.
    """
    start = time.monotonic()
    errors: list[str] = []
    predicted: str | None = None
    cost = _zero_cost()
    blueprint_info = {"blueprint_name": "unknown", "selection_mode": "unknown", "n_iterations": 0}
    required_checks: dict[str, str] = {}
    node_history: list[str] = []
    judge_reason: str | None = None
    trace_path: str | None = None

    try:
        session = AgentSession(
            id=f"benchmark:{sample.id}",
            goal=Prompt(text="Fact-check this claim using the selected blueprint."),
            claim=sample.input,
            cutoff_date=sample.input.date.date() if sample.input.date is not None else None,
        )
        result = agent.run(session, true_label=sample.label.value)
        predicted = extract_predicted_label(result)
        errors = list(result.errors)
        cost = extract_cost(result)
        blueprint_info = extract_blueprint_info(result)
        required_checks = extract_required_check_statuses(result)
        node_history = extract_node_history(result)
        judge_reason = extract_judge_reason(result)
        trace_path = extract_trace_path(result)
    except Exception as e:
        errors.append(f"{type(e).__name__}: {e}")
        logger.error(f"[run_fact_check] Exception for sample {sample.id}:\n{_traceback.format_exc()}")

    ground_truth = sample.label.value
    result_dict: dict[str, Any] = {
        "claim_id": sample.id,
        "ground_truth": ground_truth,
        "predicted": predicted,
        "correct": predicted == ground_truth if predicted is not None else False,
        "errors": errors,
        "duration_ms": round((time.monotonic() - start) * 1000),
        "cost": cost,
        "required_checks": required_checks,
        "node_history": node_history,
        "judge_reason": judge_reason,
        "trace_path": trace_path,
        **blueprint_info,
    }
    if benchmark is not None:
        result_dict.update(benchmark.sample_extra_fields(sample))
    return result_dict
