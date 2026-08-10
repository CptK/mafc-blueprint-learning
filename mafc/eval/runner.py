"""Core benchmark runner: builds agents from config, processes samples, writes results."""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

from mafc.agents.fact_check.agent import FactCheckAgent
from mafc.common.logger import logger
from mafc.eval.metrics import format_blueprint_stats_report
from mafc.eval.run_config import BenchmarkRunConfig
from mafc.eval.single import build_fact_check_agent, run_fact_check
from mafc.eval.veritas.benchmark import VeriTaS


def _run_sample(
    config: BenchmarkRunConfig, benchmark, sample, trace_dir: Path | None, agent: FactCheckAgent | None = None
) -> dict[str, Any]:
    if agent is None:
        agent = build_fact_check_agent(config, benchmark, trace_dir)
    return run_fact_check(sample, agent, benchmark=benchmark)


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

    # Per-blueprint stats
    bp_groups: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        bp = r.get("blueprint_name") or "unknown"
        bp_groups.setdefault(bp, []).append(r)

    blueprint_stats: dict[str, Any] = {}
    for bp_name, bp_results in sorted(bp_groups.items()):
        bp_scored = [r for r in bp_results if r.get("predicted") is not None]
        bp_correct = sum(1 for r in bp_scored if r.get("correct"))
        bp_cost = sum((r.get("cost") or {}).get("cost_usd", 0.0) for r in bp_results)
        bp_iters = [r["n_iterations"] for r in bp_results if r.get("n_iterations") is not None]
        entry: dict[str, Any] = {
            "count": len(bp_results),
            "completed": len(bp_scored),
            "errored": len(bp_results) - len(bp_scored),
            "correct": bp_correct,
            "accuracy": round(bp_correct / len(bp_scored), 4) if bp_scored else None,
            "avg_cost_usd": round(bp_cost / len(bp_results), 6) if bp_results else None,
            "avg_duration_ms": (
                round(sum(r["duration_ms"] for r in bp_results) / len(bp_results)) if bp_results else None
            ),
        }
        if bp_iters:
            entry["avg_iterations"] = round(sum(bp_iters) / len(bp_iters), 2)
        if benchmark is not None and bp_scored:
            bp_metrics = benchmark.compute_metrics(bp_results)
            entry["macro_f1"] = (bp_metrics.get("macro") or {}).get("f1")
            entry["weighted_f1"] = (bp_metrics.get("weighted") or {}).get("f1")
        blueprint_stats[bp_name] = entry

    summary["blueprint_stats"] = blueprint_stats

    # Selection mode distribution
    selection_mode_counts: dict[str, int] = {}
    for r in results:
        mode = r.get("selection_mode") or "unknown"
        selection_mode_counts[mode] = selection_mode_counts.get(mode, 0) + 1
    summary["selection_mode_counts"] = selection_mode_counts

    return summary


def _log_sample_result(result: dict[str, Any]) -> None:
    status = "✓" if result["correct"] else "✗"
    logger.info(
        f"[Runner] {status} {result['claim_id']}: predicted={result['predicted']} gt={result['ground_truth']}"
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

    cache_dir = run_dir / "temp"
    if config.run.concurrency <= 1:
        agent = build_fact_check_agent(config, benchmark, trace_dir, cache_dir=cache_dir)
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
                _thread_local.agent = build_fact_check_agent(
                    config, benchmark, trace_dir, cache_dir=cache_dir
                )
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
    report_parts: list[str] = []
    if metrics:
        classification_report = benchmark.format_metrics_report(metrics)
        if classification_report:
            report_parts.append(classification_report)
    blueprint_report = format_blueprint_stats_report(
        summary.get("blueprint_stats") or {},
        summary.get("selection_mode_counts"),
    )
    if blueprint_report:
        report_parts.append(blueprint_report)
    if report_parts:
        report_path = run_dir / "metrics_report.txt"
        report_path.write_text("\n\n".join(report_parts), encoding="utf-8")
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
