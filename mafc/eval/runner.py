"""Core benchmark runner: builds agents from config, processes samples, writes results."""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

from mafc.agents.common import AgentSession
from mafc.agents.fact_check.agent import FactCheckAgent
from mafc.agents.judge.agent import JudgeAgent
from mafc.agents.media.agent import MediaAgent
from mafc.agents.web_search.agent import WebSearchAgent
from mafc.blueprints import BlueprintRegistry, BlueprintSelector
from mafc.common.logger import logger
from mafc.common.modeling import make_model
from mafc.common.modeling.prompt import Prompt
from mafc.eval.benchmark import Benchmark
from mafc.eval.run_config import BenchmarkRunConfig
from mafc.eval.veritas.benchmark import VeriTaS


def _build_fact_check_agent(
    config: BenchmarkRunConfig, benchmark: Benchmark, trace_dir: Path | None
) -> FactCheckAgent:
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
    media_model = make_model(
        media_cfg.model, temperature=media_cfg.temperature, max_response_length=media_cfg.max_response_length
    )
    judge_model = make_model(
        judge_cfg.model, temperature=judge_cfg.temperature, max_response_length=judge_cfg.max_response_length
    )
    selector_model = make_model(bp_cfg.selector_model)

    registry = BlueprintRegistry.from_path(bp_cfg.config_dir)
    selector = BlueprintSelector(model=selector_model, registry=registry, default_blueprint_name="generic")

    media_agent = MediaAgent(model=media_model, summarization_model=media_model)
    web_search_agent = WebSearchAgent(
        main_model=worker_model,
        summarization_model=worker_model,
        n_workers=ws_cfg.workers,
        max_iterations=ws_cfg.max_iterations,
        max_results_per_query=ws_cfg.max_results_per_query,
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


def _extract_predicted_label(agent_result) -> str | None:
    judge_run = (agent_result.trace or {}).get("judge_run") or {}
    decision = judge_run.get("decision") or {}
    return decision.get("label") or None


def _extract_cost(agent_result) -> dict[str, Any]:
    trace_summary = (agent_result.trace or {}).get("summary") or {}
    return {
        "cost_usd": trace_summary.get("total_cost_usd", 0.0),
        "input_tokens": trace_summary.get("total_input_tokens", 0),
        "output_tokens": trace_summary.get("total_output_tokens", 0),
        "total_tokens": (
            trace_summary.get("total_input_tokens", 0) + trace_summary.get("total_output_tokens", 0)
        ),
    }


def _run_sample(
    config: BenchmarkRunConfig, benchmark, sample, trace_dir: Path | None, agent: FactCheckAgent | None = None
) -> dict[str, Any]:
    start = time.monotonic()
    errors: list[str] = []
    predicted: str | None = None
    cost: dict[str, Any] = {"cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    try:
        if agent is None:
            agent = _build_fact_check_agent(config, benchmark, trace_dir)
        session = AgentSession(
            id=f"benchmark:{sample.id}",
            goal=Prompt(text="Fact-check this claim using the selected blueprint."),
            claim=sample.input,
        )
        result = agent.run(session, true_label=sample.label.value)
        predicted = _extract_predicted_label(result)
        errors = list(result.errors)
        cost = _extract_cost(result)
    except Exception as e:
        errors.append(f"{type(e).__name__}: {e}")

    ground_truth = sample.label.value
    return {
        "claim_id": sample.id,
        "ground_truth": ground_truth,
        "predicted": predicted,
        "correct": predicted == ground_truth if predicted is not None else False,
        "errors": errors,
        "duration_ms": round((time.monotonic() - start) * 1000),
        "cost": cost,
        **benchmark.sample_extra_fields(sample),
    }


def _compute_summary(results: list[dict[str, Any]], run_duration_s: float, benchmark=None) -> dict[str, Any]:
    total = len(results)
    if total == 0:
        return {
            "total": 0,
            "completed": 0,
            "errored": 0,
            "accuracy": None,
            "run_duration_s": run_duration_s,
            "cost": {"cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        }

    scored = [r for r in results if r["predicted"] is not None]
    correct = sum(1 for r in scored if r["correct"])

    total_cost_usd = sum((r.get("cost") or {}).get("cost_usd", 0.0) for r in results)
    total_input_tokens = sum((r.get("cost") or {}).get("input_tokens", 0) for r in results)
    total_output_tokens = sum((r.get("cost") or {}).get("output_tokens", 0) for r in results)

    summary: dict[str, Any] = {
        "total": total,
        "completed": len(scored),
        "errored": total - len(scored),
        "correct": correct,
        "accuracy": correct / len(scored) if scored else None,
        "avg_duration_ms": round(sum(r["duration_ms"] for r in results) / total),
        "run_duration_s": round(run_duration_s, 1),
        "cost": {
            "cost_usd": round(total_cost_usd, 6),
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
        },
    }

    if benchmark is not None:
        summary["metrics"] = benchmark.compute_metrics(results)

    return summary


def _log_sample_result(result: dict[str, Any]) -> None:
    status = "✓" if result["correct"] else "✗"
    logger.info(
        f"[Runner] {status} {result['claim_id']}: "
        f"predicted={result['predicted']} gt={result['ground_truth']}"
    )


def run_benchmark(config: BenchmarkRunConfig, run_dir: Path, skip_ids: set[str] | None = None) -> None:
    bm_cfg = config.benchmark
    data_path = bm_cfg.data_path or f"data/{bm_cfg.name}_{bm_cfg.split}"
    benchmark = VeriTaS(data_path=data_path, variant=bm_cfg.split, label_scheme=bm_cfg.label_scheme)

    # Select samples
    samples = list(benchmark)
    if bm_cfg.sample_ids is not None:
        id_set = set(bm_cfg.sample_ids)
        samples = [s for s in samples if s.id in id_set]
    elif bm_cfg.first_n is not None:
        samples = samples[: bm_cfg.first_n]

    # Skip already-completed samples (resume)
    if skip_ids:
        samples = [s for s in samples if s.id not in skip_ids]

    logger.info(f"[Runner] {len(samples)} samples to process (skipped {len(skip_ids or {})})")

    trace_dir = run_dir / "traces" if config.run.traces else None
    if trace_dir:
        trace_dir.mkdir(parents=True, exist_ok=True)

    run_start = time.monotonic()

    pbar = tqdm(total=len(samples), desc="Benchmark", unit="claim", dynamic_ncols=True)

    def _handle_result(result: dict[str, Any]) -> None:
        logger.save_benchmark_result(result)
        _log_sample_result(result)
        status = "✓" if result["correct"] else "✗"
        cost_usd = (result.get("cost") or {}).get("cost_usd", 0.0)
        pbar.set_postfix_str(f"{status} {result['claim_id']} | ${cost_usd:.4f}")
        pbar.update(1)

    if config.run.concurrency <= 1:
        agent = _build_fact_check_agent(config, benchmark, trace_dir)
        for sample in samples:
            result = _run_sample(config, benchmark, sample, trace_dir, agent=agent)
            _handle_result(result)
    else:
        timeout = config.run.timeout_per_sample
        # Each worker thread gets its own agent so that each has its own
        # ScrapeMMRetriever and asyncio event loop. A single shared agent
        # would funnel all concurrent samples through one event loop, flooding
        # it and causing SSL teardown races and segfaults on macOS.
        _thread_local = threading.local()

        def _submit(sample):
            if not hasattr(_thread_local, "agent"):
                _thread_local.agent = _build_fact_check_agent(config, benchmark, trace_dir)
            return _run_sample(config, benchmark, sample, trace_dir, agent=_thread_local.agent)

        with ThreadPoolExecutor(max_workers=config.run.concurrency) as executor:
            futures = {executor.submit(_submit, s): s for s in samples}
            for future in as_completed(futures):
                sample = futures[future]
                _zero_cost = {"cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
                try:
                    result = future.result(timeout=timeout)
                except TimeoutError:
                    result = {
                        "claim_id": sample.id,
                        "ground_truth": sample.label.value,
                        "predicted": None,
                        "correct": False,
                        "errors": [f"Timed out after {timeout}s"],
                        "duration_ms": (timeout or 0) * 1000,
                        "cost": _zero_cost,
                    }
                except Exception as e:
                    result = {
                        "claim_id": sample.id,
                        "ground_truth": sample.label.value,
                        "predicted": None,
                        "correct": False,
                        "errors": [f"{type(e).__name__}: {e}"],
                        "duration_ms": 0,
                        "cost": _zero_cost,
                    }
                _handle_result(result)

    pbar.close()

    run_duration_s = time.monotonic() - run_start

    # Compute summary over ALL results in the file (covers resumed runs too)
    all_results: list[dict[str, Any]] = []
    if logger.results_path.exists():
        with open(logger.results_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        all_results.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    summary = _compute_summary(all_results, run_duration_s, benchmark=benchmark)
    logger.write_benchmark_summary(summary)

    # Save human-readable metrics report + confusion matrix PNG plots
    metrics = summary.get("metrics") or {}
    if metrics:
        report = benchmark.format_metrics_report(metrics)
        if report:
            report_path = run_dir / "metrics_report.txt"
            report_path.write_text(report, encoding="utf-8")
            logger.info(f"[Runner] Metrics report saved to {report_path}")
        for plot_path in benchmark.save_metric_plots(metrics, run_dir):
            logger.info(f"[Runner] Confusion matrix saved to {plot_path}")

    accuracy_str = f"{summary['accuracy']:.1%}" if summary["accuracy"] is not None else "n/a"
    cost = summary.get("cost") or {}
    logger.info(
        f"[Runner] Done. Accuracy: {accuracy_str} ({summary['correct']}/{summary['completed']}) | "
        f"Cost: ${cost.get('cost_usd', 0.0):.4f} | "
        f"Tokens: {cost.get('total_tokens', 0):,} "
        f"(in={cost.get('input_tokens', 0):,}, out={cost.get('output_tokens', 0):,})"
    )
