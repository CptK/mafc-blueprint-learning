#!/usr/bin/env python3
"""Continue an existing one-round strategy run with a second investigation round.

Reuses the round-1 evidence recorded in <run_dir>/traces instead of re-running
the whole investigation: for each claim the planner sees the round-1 evidence
and action history and decides round-2 tool calls (or done). Only claims that
gain new evidence are re-judged; all others keep their original prediction, so
API cost is limited to one planner call per claim plus the requested follow-up
tool calls and judging of changed claims.

Claims that errored in the source run (no judge prompt in the trace) are re-run
from scratch.

Caveat: round 1 originally ran under a "FINAL round" directive, so this is an
approximation of a native max_rounds=2 run — round-1 behavior is frozen.

Usage:
  python scripts/continue_round2.py --run-dir out/<run> --out-dir out/<run>-round2 \
      [--workers 8] [--first-n N] [--only-ids ids.json|id1,id2]
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ezmm
from ezmm import MultimodalSequence
from ezmm.common.registry import ItemRegistry
from tqdm import tqdm

import mafc  # noqa: F401 — loads config/.env

# ezmm's ItemRegistry shares one SQLite cursor across threads; without a lock
# concurrent media resolution segfaults (same patch as scripts/run_benchmark.py).
_registry_lock = threading.RLock()
for _name in ("get", "get_by_path", "add_item", "get_cached", "update_file_path", "contains"):
    _orig = getattr(ItemRegistry, _name)

    def _make_locked(m):
        def _locked(self, *args, **kwargs):
            with _registry_lock:
                return m(self, *args, **kwargs)

        return _locked

    setattr(ItemRegistry, _name, _make_locked(_orig))

from mafc.agents.common import AgentSession  # noqa: E402
from mafc.agents.web_search.actions import InspectWebSource  # noqa: E402
from mafc.common.evidence import Evidence  # noqa: E402
from mafc.common.logger import logger  # noqa: E402
from mafc.common.modeling.prompt import Prompt  # noqa: E402
from mafc.eval.run_config import BenchmarkRunConfig  # noqa: E402
from mafc.eval.single import build_fact_check_agent, run_fact_check  # noqa: E402
from mafc.eval.veritas.benchmark import VeriTaS  # noqa: E402
from mafc.eval.veritas.metrics import compute_veritas_metrics, format_veritas_metrics_report  # noqa: E402
from mafc.single_file_strategy.tracing import StrategyRunTrace  # noqa: E402

_write_lock = threading.Lock()
_thread_local = threading.local()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, help="Source run directory (one-round run to extend).")
    p.add_argument("--out-dir", required=True, help="Output directory for the continued run.")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--first-n", type=int, default=None)
    p.add_argument("--only-ids", default=None, help="Comma-separated claim_ids or a JSON file with a list.")
    return p.parse_args()


def parse_round1_evidence(trace: dict) -> list[Evidence] | None:
    """Reconstruct the evidence list the round-1 judge saw from the trace's judge prompt."""
    jr = trace.get("judge_run")
    if not jr or not jr.get("prompt_messages"):
        return None
    user_text = None
    for m in jr["prompt_messages"]:
        if m.get("role") == "user":
            c = m.get("content")
            user_text = c.get("text") if isinstance(c, dict) else str(c)
    if not user_text:
        return None
    start = user_text.find("Accepted evidence:")
    if start == -1:
        return None
    end = user_text.find("Return strict JSON", start)
    block = user_text[start + len("Accepted evidence:") : end if end != -1 else None]
    # Blocks are formatted as "- Source: <source>\n  Summary: <multiline summary>"
    evidences: list[Evidence] = []
    for match in re.split(r"\n(?=- Source: )", block.strip()):
        m = re.match(r"- Source: (.*?)\n  Summary: (.*)", match, flags=re.S)
        if not m:
            continue
        source, summary = m.group(1).strip(), m.group(2).strip()
        if not source or not summary:
            continue
        evidences.append(
            Evidence(
                raw=MultimodalSequence(summary),
                action=InspectWebSource(
                    query_text="round-1 investigation", source_url=source, source_title=None
                ),
                source=source,
                takeaways=MultimodalSequence(summary),
            )
        )
    return evidences


def parse_round1_history(trace: dict) -> list[str]:
    history = []
    for rd in trace.get("rounds") or []:
        for tc in rd.get("tool_calls") or []:
            tool = tc.get("tool") or "?"
            instr = (tc.get("instruction") or "")[:80]
            history.append(f"{tool}: {instr} -> completed in round 1")
    return history


def get_agent(config, benchmark, trace_dir):
    """One StrategyAgent per worker thread (agents are not shared across threads)."""
    if not hasattr(_thread_local, "agent"):
        _thread_local.agent = build_fact_check_agent(config, benchmark, trace_dir=trace_dir)
        _thread_local.agent.max_rounds = 2
    return _thread_local.agent


def continue_one(sample, orig_row, trace_path, config, benchmark, trace_dir) -> dict:
    """Run the round-2 increment for one claim; returns a results.jsonl row."""
    agent = get_agent(config, benchmark, trace_dir)
    start = time.monotonic()

    trace_data = json.loads(Path(trace_path).read_text()) if trace_path and Path(trace_path).exists() else {}
    evidences = parse_round1_evidence(trace_data)
    if evidences is None or not evidences:
        # Errored or evidence-less in the source run: re-run from scratch.
        row = run_fact_check(sample, agent, benchmark=benchmark)
        row["round2_mode"] = "full_rerun"
        return row

    claim = sample.input
    n_before = len(evidences)
    history = parse_round1_history(trace_data)
    errors: list[str] = []
    session = AgentSession(
        id=f"benchmark:{sample.id}",
        goal=Prompt(text="Fact-check this claim (round-2 continuation)."),
        claim=claim,
        cutoff_date=claim.date.date() if claim.date is not None else None,
    )
    trace = StrategyRunTrace(
        trace_dir,
        session_id=session.id,
        claim_text=str(claim),
        strategy_word_count=len(agent.strategy_md.split()),
        true_label=sample.label.value,
    )

    try:
        agent._run_round(session, claim, evidences, history, errors, trace, round_idx=2, last_round=True)
        new_evidence = len(evidences) > n_before

        if not new_evidence:
            trace.finalize(status="completed", result_text=None, evidence_count=n_before)
            row = dict(orig_row)
            row["round2_mode"] = "kept"
            row["errors"] = errors
            return row

        session.evidences = list(evidences)
        agent._judge(session, claim, evidences, errors, trace)
        decision = (trace.trace.get("judge_run") or {}).get("decision") or {}
        predicted = decision.get("label")
        trace.finalize(status="completed", result_text=None, evidence_count=len(evidences))

        if predicted is None:
            row = dict(orig_row)
            row["round2_mode"] = "judge_failed_kept"
            row["errors"] = errors
            return row

        summary = trace.trace.get("summary") or {}
        row = dict(orig_row)
        row.update(
            predicted=predicted,
            correct=predicted == sample.label.value,
            errors=errors,
            duration_ms=round((time.monotonic() - start) * 1000),
            cost={
                "cost_usd": summary.get("total_cost_usd", 0.0),
                "input_tokens": summary.get("total_input_tokens", 0),
                "output_tokens": summary.get("total_output_tokens", 0),
                "total_tokens": summary.get("total_input_tokens", 0) + summary.get("total_output_tokens", 0),
            },
            judge_reason=decision.get("justification"),
            trace_path=trace.trace.get("trace_path"),
            round2_mode="rejudged",
            round2_new_evidence=len(evidences) - n_before,
        )
        return row
    except Exception as exc:  # noqa: BLE001 — never lose a claim to one failure
        logger.error(f"[round2] {sample.id} failed:\n{traceback.format_exc()}")
        row = dict(orig_row)
        row["round2_mode"] = "error_kept"
        row.setdefault("errors", []).append(f"round2: {type(exc).__name__}: {exc}")
        return row


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    trace_dir = out_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    # Share the source run's media registry so trace media references resolve.
    ezmm.set_ezmm_path(run_dir / "temp")

    config = BenchmarkRunConfig.from_yaml(run_dir / "config.yaml")
    logger.set_log_level(config.run.log_level.lower())  # type: ignore[arg-type]
    bm_cfg = config.benchmark
    data_path = bm_cfg.data_path or f"data/{bm_cfg.name}_{bm_cfg.split}"
    benchmark = VeriTaS(data_path=data_path, variant=bm_cfg.split, label_scheme=bm_cfg.label_scheme)
    samples_by_id = {s.id: s for s in benchmark}

    orig_rows: dict[str, dict] = {}
    for line in open(run_dir / "results.jsonl"):
        r = json.loads(line)
        orig_rows[r["claim_id"]] = r

    out_results = out_dir / "results.jsonl"
    done_ids: set[str] = set()
    if out_results.exists():
        for line in open(out_results):
            try:
                done_ids.add(json.loads(line)["claim_id"])
            except json.JSONDecodeError:
                pass

    only_ids: set[str] | None = None
    if args.only_ids:
        if Path(args.only_ids).exists():
            only_ids = set(json.load(open(args.only_ids)))
        else:
            only_ids = {x.strip() for x in args.only_ids.split(",")}

    jobs = []
    for cid, row in orig_rows.items():
        if cid in done_ids or cid not in samples_by_id:
            continue
        if only_ids is not None and cid not in only_ids:
            continue
        tp = run_dir / "traces" / f"benchmark_{cid}.strategy_trace.json"
        jobs.append((samples_by_id[cid], row, tp))
    if args.first_n:
        jobs = jobs[: args.first_n]

    print(f"claims={len(orig_rows)} already_done={len(done_ids)} jobs={len(jobs)}")
    if not jobs:
        return

    counts: dict[str, int] = {}
    cost_total = 0.0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(continue_one, s, r, tp, config, benchmark, trace_dir) for s, r, tp in jobs]
        with tqdm(total=len(jobs), desc="round 2", unit="claim") as pbar:
            for fut in as_completed(futures):
                row = fut.result()
                mode = row.get("round2_mode", "?")
                counts[mode] = counts.get(mode, 0) + 1
                if mode in ("rejudged", "full_rerun"):
                    cost_total += (row.get("cost") or {}).get("cost_usd", 0.0)
                with _write_lock:
                    with open(out_results, "a") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                pbar.set_postfix({**counts, "cost": f"${cost_total:.2f}"})
                pbar.update(1)

    # Metrics over the merged results
    all_rows = [json.loads(line) for line in open(out_results)]
    metrics = compute_veritas_metrics(all_rows, label_scheme=bm_cfg.label_scheme)
    report = format_veritas_metrics_report(metrics, label_scheme=bm_cfg.label_scheme)
    (out_dir / "metrics_report.txt").write_text(report)
    (out_dir / "summary.json").write_text(json.dumps({"modes": counts, "metrics": metrics}, indent=1))
    print(report)
    print(f"\nmodes: {counts}")


if __name__ == "__main__":
    main()
