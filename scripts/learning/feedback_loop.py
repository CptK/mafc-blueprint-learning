#!/usr/bin/env python3
"""Benchmark-feedback loop for blueprint sets — mechanized curation on TRAINING data.

Closes the loop that manual curation performed by hand (per-blueprint error
attribution → targeted rewording → subset validation), using only the training
split so the evaluation quarter is never touched. Costs are kept low by an
escalation ladder: routing is traces-free, screening covers only the buckets
that carry real traffic, and validation re-runs only previously-run claims for
a paired comparison.

Rounds (subcommands; state accumulates in --workdir):

  route     Selector-only pass over the training claims (NO traces, ~cents).
            Yields the true per-blueprint traffic distribution — catches
            runtime over-absorption (a 16%-share blueprint drawing 40% of
            traffic) before any expensive run.
  screen    Run full traces on N claims per high-traffic bucket (default 20
            per bucket over buckets covering --traffic-cover of routed
            traffic). Prints the report and flags suspicious buckets.
  confirm   Add more claims (default 30) for flagged (or named) buckets.
  report    Recompute the report over all accumulated results.
  update    For flagged/named buckets, feed failure cases to BlueprintUpdater
            (outcome-aware) and write the revised set to a NEW directory —
            the original blueprint directory is never modified.
  validate  Re-run this loop's previously-run claims for the updated buckets
            against the new blueprint dir and compare PAIRED old vs new.

Typical session:
    python scripts/learning/feedback_loop.py route    --workdir out/feedback/v3 \\
        --config config/experiments/veritas_eom_v3.yaml \\
        --training-dir data/veritas_2025_with_fact_checks
    python scripts/learning/feedback_loop.py screen   --workdir out/feedback/v3
    python scripts/learning/feedback_loop.py confirm  --workdir out/feedback/v3
    python scripts/learning/feedback_loop.py update   --workdir out/feedback/v3
    python scripts/learning/feedback_loop.py validate --workdir out/feedback/v3
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

from mafc.common.logger import logger
from mafc.eval.run_config import BenchmarkRunConfig

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


_STATE_FILENAME = "state.json"
_NEAR_MISS_THRESHOLD = 0.34  # one 7-class bin (1/3) + epsilon

_FEEDBACK_UPDATE_HINT = """\
NOTE: The claims below are REAL benchmark results of this exact blueprint on \
training data, partitioned by outcome. Revise the blueprint to fix the recurring \
failure patterns you observe in the incorrect cases WITHOUT regressing the \
correct ones. Focus on what the verification process got wrong (wrong referent, \
missing corroboration, premature verdicts, unused evidence types) — not on the \
topics of the claims. Keep the blueprint's name EXACTLY unchanged. Do not make \
the description broader; a broader description attracts traffic the blueprint \
cannot serve.\
"""


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


def _load_state(workdir: Path) -> dict:
    path = workdir / _STATE_FILENAME
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"runs": [], "flagged": [], "blueprint_dirs": []}


def _save_state(workdir: Path, state: dict) -> None:
    with open(workdir / _STATE_FILENAME, "w") as f:
        json.dump(state, f, indent=2)


def _require(state: dict, key: str, hint: str):
    value = state.get(key)
    if not value:
        raise SystemExit(f"state is missing '{key}' — {hint}")
    return value


def _label_numeric_map() -> dict[str, float]:
    from mafc.eval.veritas.labels import LABEL_NUMERIC_7

    return {label.value: v for label, v in LABEL_NUMERIC_7.items()}


# ---------------------------------------------------------------------------
# Result aggregation
# ---------------------------------------------------------------------------


def _load_results(state: dict, blueprint_dir: str) -> dict[str, dict]:
    """Latest result per claim across this loop's runs executed with blueprint_dir."""
    results: dict[str, dict] = {}
    num = _label_numeric_map()
    for run in state["runs"]:
        if run["blueprint_dir"] != blueprint_dir:
            continue
        results_path = Path(run["run_dir"]) / "results.jsonl"
        if not results_path.exists():
            continue
        with open(results_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("predicted") in num and r.get("gt_integrity_score") is not None:
                    results[r["claim_id"]] = r
    return results


def _sq_err(r: dict, num: dict[str, float]) -> float:
    return (num[r["predicted"]] - r["gt_integrity_score"]) ** 2


def _cls(x: float) -> str:
    return "i" if x > 1 / 6 else ("c" if x < -1 / 6 else "u")


def _is_flip(r: dict, num: dict[str, float]) -> bool:
    p, g = _cls(num[r["predicted"]]), _cls(r["gt_integrity_score"])
    return "u" not in (p, g) and p != g


def _build_report(
    state: dict,
    flag_margin: float,
    min_n: int,
    min_flips: int,
) -> dict:
    blueprint_dir = _require(state, "blueprint_dir", "run 'route' first")
    results = _load_results(state, blueprint_dir)
    num = _label_numeric_map()
    routing: dict[str, str] = state.get("routing", {})

    if not results:
        return {"pool": {"n": 0}, "buckets": {}, "flagged": []}

    pool_mse = sum(_sq_err(r, num) for r in results.values()) / len(results)
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for r in results.values():
        by_bucket[r.get("blueprint_name") or "unknown"].append(r)

    route_counts: dict[str, int] = defaultdict(int)
    for bp in routing.values():
        route_counts[bp] += 1
    n_routed = sum(route_counts.values()) or 1

    buckets: dict[str, dict] = {}
    flagged: list[str] = []
    for bp, rs in sorted(by_bucket.items(), key=lambda x: -len(x[1])):
        mse = sum(_sq_err(r, num) for r in rs) / len(rs)
        flips = [r for r in rs if _is_flip(r, num)]
        worst = sorted(rs, key=lambda r: -_sq_err(r, num))[:5]
        entry = {
            "n": len(rs),
            "mse": round(mse, 4),
            "excess_vs_pool": round(mse - pool_mse, 4),
            "flips": len(flips),
            "routed_share": round(route_counts.get(bp, 0) / n_routed, 4),
            "worst_claims": [
                {
                    "claim_id": r["claim_id"],
                    "gt": r["gt_integrity_score"],
                    "predicted": r["predicted"],
                    "sq_err": round(_sq_err(r, num), 3),
                }
                for r in worst
            ],
        }
        is_flagged = len(rs) >= min_n and (mse >= pool_mse + flag_margin or len(flips) >= min_flips)
        entry["flagged"] = is_flagged
        if is_flagged:
            flagged.append(bp)
        buckets[bp] = entry

    return {
        "pool": {"n": len(results), "mse": round(pool_mse, 4)},
        "buckets": buckets,
        "flagged": flagged,
    }


def _print_report(report: dict) -> None:
    pool = report["pool"]
    print(f"\n=== Feedback report ===  pool: n={pool['n']} MSE={pool.get('mse')}")
    print(f"{'blueprint':<48} {'n':>4} {'MSE':>7} {'excess':>7} {'flips':>5} {'route%':>7}  flag")
    for bp, e in report["buckets"].items():
        print(
            f"{bp:<48} {e['n']:>4} {e['mse']:>7.4f} {e['excess_vs_pool']:>+7.4f} "
            f"{e['flips']:>5} {e['routed_share']:>6.1%}  {'⚑' if e['flagged'] else ''}"
        )
    if report["flagged"]:
        print(f"\nFlagged: {', '.join(report['flagged'])}")
    else:
        print("\nNo buckets flagged.")


# ---------------------------------------------------------------------------
# Round: route
# ---------------------------------------------------------------------------


def cmd_route(args: argparse.Namespace) -> None:
    workdir: Path = args.workdir
    workdir.mkdir(parents=True, exist_ok=True)
    state = _load_state(workdir)

    config = BenchmarkRunConfig.from_yaml(args.config)
    if config.blueprints is None:
        raise SystemExit("config must have a 'blueprints' section (strategy configs have no selector)")
    blueprint_dir = args.blueprint_dir or config.blueprints.config_dir

    state.update(
        {
            "base_config": str(args.config),
            "training_dir": str(args.training_dir),
            "blueprint_dir": blueprint_dir,
            "original_blueprint_dir": blueprint_dir,
            "label_scheme": config.benchmark.label_scheme,
        }
    )

    # ezmm setup mirrors scripts/run_benchmark.py: benchmark loading registers
    # media items, and the ItemRegistry's shared SQLite cursor is not thread-safe.
    import ezmm
    from ezmm.common.registry import ItemRegistry

    ezmm.set_ezmm_path(workdir / "temp")
    registry_lock = threading.RLock()
    for name in ("get", "get_by_path", "add_item", "get_cached", "update_file_path", "contains"):
        orig = getattr(ItemRegistry, name)

        def make_locked(m):
            def locked(self, *a, **kw):
                with registry_lock:
                    return m(self, *a, **kw)

            return locked

        setattr(ItemRegistry, name, make_locked(orig))

    from mafc.blueprints import BlueprintRegistry, BlueprintSelector
    from mafc.common.modeling import make_model
    from mafc.eval.veritas.benchmark import VeriTaS

    benchmark = VeriTaS(
        data_path=str(args.training_dir),
        variant=args.split,
        label_scheme=config.benchmark.label_scheme,
    )
    samples = list(benchmark)
    if args.limit:
        samples = samples[: args.limit]
    logger.info(f"[route] Routing {len(samples)} training claims with {config.blueprints.selector_model}.")

    bp_registry = BlueprintRegistry.from_path(blueprint_dir)
    thread_local = threading.local()

    def route_one(sample) -> tuple[str, str, str]:
        if not hasattr(thread_local, "selector"):
            thread_local.selector = BlueprintSelector(
                model=make_model(
                    config.blueprints.selector_model,
                    max_response_length=config.blueprints.selector_max_response_length,
                ),
                registry=bp_registry,
                default_blueprint_name="generic",
            )
        selection = thread_local.selector.select(sample.input)
        return sample.id, selection.selected_blueprint.name, selection.selection_mode.value

    routing: dict[str, str] = {}
    modes: dict[str, int] = defaultdict(int)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(route_one, s): s for s in samples}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                claim_id, bp_name, mode = future.result()
            except Exception as e:
                logger.warning(f"[route] {futures[future].id}: {type(e).__name__}: {e}")
                continue
            routing[claim_id] = bp_name
            modes[mode] += 1
            if i % 200 == 0:
                logger.info(f"[route] {i}/{len(samples)} routed.")

    state["routing"] = routing
    state["routing_split"] = args.split
    _save_state(workdir, state)

    counts: dict[str, int] = defaultdict(int)
    for bp in routing.values():
        counts[bp] += 1
    total = sum(counts.values()) or 1
    print(f"\n=== Routed traffic ({total} claims, modes: {dict(modes)}) ===")
    for bp, c in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {bp:<48} {c:>5}  {c / total:>6.1%}")
    print(f"\nRouting saved to {workdir / _STATE_FILENAME}.")


# ---------------------------------------------------------------------------
# Rounds: screen / confirm (trace runs via scripts/run_benchmark.py --resume)
# ---------------------------------------------------------------------------


def _pick_claims(
    state: dict,
    buckets: list[str],
    per_bucket: int,
    seed: int,
) -> dict[str, list[str]]:
    """Sample per-bucket claim ids from the routing, excluding already-run ids."""
    routing: dict[str, str] = _require(state, "routing", "run 'route' first")
    already_run: set[str] = set()
    for run in state["runs"]:
        already_run.update(run["sample_ids"])

    rng = random.Random(seed)
    picks: dict[str, list[str]] = {}
    for bucket in buckets:
        candidates = sorted(cid for cid, bp in routing.items() if bp == bucket and cid not in already_run)
        rng.shuffle(candidates)
        picks[bucket] = candidates[:per_bucket]
    return picks


def _launch_run(
    state: dict, workdir: Path, round_name: str, sample_ids: list[str], concurrency: int | None
) -> Path:
    """Create a run dir with a config and execute it via run_benchmark --resume."""
    run_dir = workdir / "runs" / round_name
    if run_dir.exists():
        raise SystemExit(f"{run_dir} already exists — each round gets a fresh name.")
    run_dir.mkdir(parents=True)

    with open(state["base_config"]) as f:
        cfg = yaml.safe_load(f)
    cfg["benchmark"]["data_path"] = state["training_dir"]
    cfg["benchmark"]["split"] = state.get("routing_split", "2025_train")
    cfg["benchmark"]["sample_ids"] = sample_ids
    cfg["benchmark"].pop("first_n", None)
    cfg["blueprints"]["config_dir"] = state["blueprint_dir"]
    if concurrency:
        cfg.setdefault("run", {})["concurrency"] = concurrency
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    logger.info(f"[{round_name}] Running {len(sample_ids)} claims → {run_dir}")
    proc = subprocess.run(
        [sys.executable, "scripts/run_benchmark.py", "--resume", str(run_dir)],
        cwd=REPO_ROOT,
        env={**__import__("os").environ, "PYTHONPATH": str(REPO_ROOT)},
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"benchmark run failed (exit {proc.returncode}); resume with: "
            f"PYTHONPATH=. python scripts/run_benchmark.py --resume {run_dir}"
        )
    return run_dir


def _record_run(state: dict, workdir: Path, round_name: str, run_dir: Path, sample_ids: list[str]) -> None:
    state["runs"].append(
        {
            "round": round_name,
            "run_dir": str(run_dir),
            "sample_ids": sample_ids,
            "blueprint_dir": state["blueprint_dir"],
        }
    )
    _save_state(workdir, state)


def cmd_screen(args: argparse.Namespace) -> None:
    state = _load_state(args.workdir)
    routing: dict[str, str] = _require(state, "routing", "run 'route' first")

    counts: dict[str, int] = defaultdict(int)
    for bp in routing.values():
        counts[bp] += 1
    total = sum(counts.values())
    covered, cum = [], 0
    for bp, c in sorted(counts.items(), key=lambda x: -x[1]):
        covered.append(bp)
        cum += c
        if cum / total >= args.traffic_cover:
            break
    logger.info(f"[screen] Buckets covering {cum / total:.0%} of traffic: {covered}")

    picks = _pick_claims(state, covered, args.per_bucket, args.seed)
    sample_ids = [cid for ids in picks.values() for cid in ids]
    if not sample_ids:
        raise SystemExit("No unrun claims left to sample.")
    for bp, ids in picks.items():
        logger.info(f"[screen]   {bp}: {len(ids)} claims")

    round_name = f"screen-{len(state['runs']) + 1:02d}"
    run_dir = _launch_run(state, args.workdir, round_name, sample_ids, args.concurrency)
    _record_run(state, args.workdir, round_name, run_dir, sample_ids)
    cmd_report(args)


def cmd_confirm(args: argparse.Namespace) -> None:
    state = _load_state(args.workdir)
    buckets = args.blueprints or state.get("flagged") or []
    if not buckets:
        raise SystemExit("No flagged buckets and none named via --blueprints.")

    picks = _pick_claims(state, buckets, args.per_bucket, args.seed)
    sample_ids = [cid for ids in picks.values() for cid in ids]
    if not sample_ids:
        raise SystemExit("No unrun claims left for these buckets.")
    for bp, ids in picks.items():
        logger.info(f"[confirm]   {bp}: {len(ids)} claims")

    round_name = f"confirm-{len(state['runs']) + 1:02d}"
    run_dir = _launch_run(state, args.workdir, round_name, sample_ids, args.concurrency)
    _record_run(state, args.workdir, round_name, run_dir, sample_ids)
    cmd_report(args)


def cmd_report(args: argparse.Namespace) -> None:
    state = _load_state(args.workdir)
    report = _build_report(state, args.flag_margin, args.min_n, args.min_flips)
    _print_report(report)
    state["flagged"] = report["flagged"]
    state["last_report"] = report
    _save_state(args.workdir, state)


# ---------------------------------------------------------------------------
# Round: update (writes a NEW blueprint dir — originals are never touched)
# ---------------------------------------------------------------------------


def _next_feedback_dir(original_dir: str, workdir: Path) -> Path:
    base = Path(original_dir).name
    k = 1
    while (workdir / "blueprints" / f"{base}-fb{k}").exists():
        k += 1
    return workdir / "blueprints" / f"{base}-fb{k}"


def cmd_update(args: argparse.Namespace) -> None:
    state = _load_state(args.workdir)
    buckets = args.blueprints or state.get("flagged") or []
    if not buckets:
        raise SystemExit("No flagged buckets and none named via --blueprints.")
    blueprint_dir = _require(state, "blueprint_dir", "run 'route' first")

    from mafc.blueprints.loader import load_blueprint
    from mafc.common.claim import Claim
    from mafc.common.modeling import make_model
    from mafc.learning.analysis_io import load_analyses
    from mafc.learning.blueprint_contrast import enforce_iteration_floor
    from mafc.learning.blueprint_updater import BlueprintUpdater
    from mafc.learning.execution import ExecutionResult
    from mafc.learning.models import ClaimLearningRecord

    results = _load_results(state, blueprint_dir)
    num = _label_numeric_map()
    analyses = load_analyses(Path(state["training_dir"]) / "article_analyses.json")
    with open(Path(state["training_dir"]) / "claims.json") as f:
        claims_by_id = {str(c["id"]): c["text"] for c in json.load(f)["claims"]}

    routing: dict[str, str] = state.get("routing", {})
    route_counts: dict[str, int] = defaultdict(int)
    for bp in routing.values():
        route_counts[bp] += 1
    n_routed = sum(route_counts.values()) or 1

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else _next_feedback_dir(state["original_blueprint_dir"], args.workdir)
    )
    if out_dir.exists():
        raise SystemExit(f"{out_dir} already exists — refusing to overwrite.")
    out_dir.mkdir(parents=True)
    for src in Path(blueprint_dir).glob("*.yaml"):
        shutil.copy(src, out_dir / src.name)

    model = make_model(args.model, max_response_length=args.max_tokens)
    updater = BlueprintUpdater(
        model=model,
        use_execution_outcomes=True,
        outcome_error_threshold=_NEAR_MISS_THRESHOLD,
    )

    updated: list[str] = []
    for bucket in buckets:
        bucket_results = [r for r in results.values() if r.get("blueprint_name") == bucket]
        if not bucket_results:
            logger.warning(f"[update] No results for '{bucket}' — skipping.")
            continue
        bucket_results.sort(key=lambda r: -_sq_err(r, num))
        selected = bucket_results[: args.max_records]

        records = []
        for r in selected:
            cid = r["claim_id"]
            records.append(
                ClaimLearningRecord(
                    claim=Claim(claims_by_id.get(cid, ""), id=cid),
                    article_analysis=analyses.get(cid),
                    assigned_blueprint=bucket,
                    execution_result=ExecutionResult(
                        claim_id=cid,
                        blueprint_name=bucket,
                        ground_truth=r["ground_truth"],
                        predicted_label=r["predicted"],
                        correct=bool(r.get("correct")),
                        n_iterations=r.get("n_iterations") or 0,
                        cost_usd=(r.get("cost") or {}).get("cost_usd", 0.0),
                        duration_ms=r.get("duration_ms") or 0,
                        required_check_statuses=r.get("required_checks") or {},
                        judge_reason=r.get("judge_reason"),
                        errors=r.get("errors") or [],
                        gt_score=r.get("gt_integrity_score"),
                        predicted_score=num.get(r["predicted"]),
                    ),
                )
            )

        blueprint_path = Path(blueprint_dir) / f"{bucket}.yaml"
        if not blueprint_path.exists():
            logger.warning(f"[update] {blueprint_path} not found — skipping.")
            continue
        blueprint = load_blueprint(blueprint_path)

        logger.info(f"[update] Updating '{bucket}' from {len(records)} outcome records…")
        result = updater.update(blueprint, records, extra_user_hint=_FEEDBACK_UPDATE_HINT)
        if result is None or result.updated_blueprint is None:
            logger.warning(f"[update] Updater produced nothing for '{bucket}' — original kept.")
            continue

        new_bp = result.updated_blueprint
        if new_bp.name != bucket:
            logger.info(f"[update] LLM renamed '{bucket}' → '{new_bp.name}'; restoring original name.")
            new_bp = new_bp.model_copy(update={"name": bucket})
        new_bp = enforce_iteration_floor(new_bp, route_counts.get(bucket, 0) / n_routed)

        with open(out_dir / f"{bucket}.yaml", "w", encoding="utf-8") as f:
            yaml.dump(new_bp.model_dump(by_alias=True), f, default_flow_style=False, allow_unicode=True)
        updated.append(bucket)
        logger.info(f"[update] '{bucket}' revised. Reasoning: {result.reasoning[:300]}")

    if not updated:
        shutil.rmtree(out_dir)
        raise SystemExit("No blueprints were updated — new dir removed.")

    state["previous_blueprint_dir"] = blueprint_dir
    state["blueprint_dir"] = str(out_dir)
    state["blueprint_dirs"].append(str(out_dir))
    state["updated_buckets"] = updated
    _save_state(args.workdir, state)
    print(f"\nUpdated {updated} → {out_dir}  (original {state['original_blueprint_dir']} untouched)")
    print("Next: python scripts/learning/feedback_loop.py validate --workdir", args.workdir)


# ---------------------------------------------------------------------------
# Round: validate (paired re-run of previously-run claims)
# ---------------------------------------------------------------------------


def cmd_validate(args: argparse.Namespace) -> None:
    state = _load_state(args.workdir)
    old_dir = _require(state, "previous_blueprint_dir", "run 'update' first")
    new_dir = _require(state, "blueprint_dir", "run 'update' first")
    buckets = args.blueprints or _require(state, "updated_buckets", "run 'update' first")

    old_results = _load_results(state, old_dir)
    num = _label_numeric_map()
    sample_ids = sorted(cid for cid, r in old_results.items() if r.get("blueprint_name") in buckets)
    if not sample_ids:
        raise SystemExit(f"No prior results for buckets {buckets}.")

    round_name = f"validate-{len(state['runs']) + 1:02d}"
    run_dir = _launch_run(state, args.workdir, round_name, sample_ids, args.concurrency)
    _record_run(state, args.workdir, round_name, run_dir, sample_ids)

    state = _load_state(args.workdir)
    new_results = _load_results(state, new_dir)
    paired = [cid for cid in sample_ids if cid in new_results]
    if not paired:
        raise SystemExit("Validation run produced no comparable results.")

    old_mse = sum(_sq_err(old_results[c], num) for c in paired) / len(paired)
    new_mse = sum(_sq_err(new_results[c], num) for c in paired) / len(paired)
    old_flips = sum(_is_flip(old_results[c], num) for c in paired)
    new_flips = sum(_is_flip(new_results[c], num) for c in paired)
    rerouted = sum(
        1 for c in paired if new_results[c].get("blueprint_name") != old_results[c].get("blueprint_name")
    )
    improved = sum(1 for c in paired if _sq_err(new_results[c], num) < _sq_err(old_results[c], num) - 1e-9)
    worsened = sum(1 for c in paired if _sq_err(new_results[c], num) > _sq_err(old_results[c], num) + 1e-9)

    verdict = "ACCEPT" if new_mse < old_mse else "REJECT (keep previous dir)"
    print(f"\n=== Paired validation on {len(paired)} claims ({buckets}) ===")
    print(f"  MSE   old={old_mse:.4f}  new={new_mse:.4f}  Δ={new_mse - old_mse:+.4f}")
    print(f"  flips old={old_flips}  new={new_flips}")
    print(f"  claims improved={improved} worsened={worsened} rerouted={rerouted}")
    print(f"  → {verdict}")

    state["last_validation"] = {
        "buckets": buckets,
        "n": len(paired),
        "old_mse": round(old_mse, 4),
        "new_mse": round(new_mse, 4),
        "old_flips": old_flips,
        "new_flips": new_flips,
        "rerouted": rerouted,
        "verdict": verdict,
    }
    if verdict.startswith("REJECT"):
        state["blueprint_dir"] = old_dir
        print(f"  blueprint_dir rolled back to {old_dir}; rejected dir kept at {new_dir} for inspection.")
    _save_state(args.workdir, state)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--workdir", type=Path, required=True, help="Feedback-loop state directory.")

    def report_args(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--flag-margin",
            type=float,
            default=0.08,
            help="Flag buckets with MSE >= pool + margin (default 0.08).",
        )
        p.add_argument(
            "--min-n", type=int, default=15, help="Minimum bucket sample size before flagging (default 15)."
        )
        p.add_argument(
            "--min-flips",
            type=int,
            default=3,
            help="Flag buckets with at least this many direction flips (default 3).",
        )

    def run_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--concurrency", type=int, default=None, help="Override run concurrency.")
        p.add_argument("--seed", type=int, default=42, help="Sampling seed (default 42).")

    p = sub.add_parser("route", help="Selector-only routing pass (no traces).")
    common(p)
    p.add_argument("--config", required=True, help="Benchmark config supplying models + blueprint dir.")
    p.add_argument(
        "--training-dir",
        type=Path,
        required=True,
        help="Training dataset directory (e.g. data/veritas_2025_with_fact_checks).",
    )
    p.add_argument("--blueprint-dir", default=None, help="Override config's blueprints.config_dir.")
    p.add_argument("--split", default="2025_train", help="Variant label for runs (default 2025_train).")
    p.add_argument("--workers", type=int, default=8, help="Parallel selector calls (default 8).")
    p.add_argument("--limit", type=int, default=None, help="Route only the first N claims (debug).")
    p.set_defaults(func=cmd_route)

    p = sub.add_parser("screen", help="Trace-run N claims per high-traffic bucket.")
    common(p)
    p.add_argument("--per-bucket", type=int, default=20, help="Claims per bucket (default 20).")
    p.add_argument(
        "--traffic-cover",
        type=float,
        default=0.85,
        help="Screen buckets covering this share of routed traffic (default 0.85).",
    )
    run_args(p)
    report_args(p)
    p.set_defaults(func=cmd_screen)

    p = sub.add_parser("confirm", help="Add claims for flagged (or named) buckets.")
    common(p)
    p.add_argument("--per-bucket", type=int, default=30, help="Additional claims per bucket (default 30).")
    p.add_argument("--blueprints", nargs="*", default=None, help="Buckets to confirm (default: flagged).")
    run_args(p)
    report_args(p)
    p.set_defaults(func=cmd_confirm)

    p = sub.add_parser("report", help="Recompute the report over accumulated results.")
    common(p)
    report_args(p)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("update", help="Revise flagged blueprints into a NEW directory.")
    common(p)
    p.add_argument("--blueprints", nargs="*", default=None, help="Buckets to update (default: flagged).")
    p.add_argument("--model", default="claude_4.8_opus", help="Updater LLM (default claude_4.8_opus).")
    p.add_argument("--max-tokens", type=int, default=20000)
    p.add_argument(
        "--max-records",
        type=int,
        default=40,
        help="Outcome records per blueprint sent to the updater (default 40).",
    )
    p.add_argument(
        "--out-dir", default=None, help="Explicit output dir (default <workdir>/blueprints/<name>-fbN)."
    )
    p.set_defaults(func=cmd_update)

    p = sub.add_parser("validate", help="Paired re-run of previously-run claims on the updated dir.")
    common(p)
    p.add_argument(
        "--blueprints", nargs="*", default=None, help="Buckets to validate (default: last updated)."
    )
    run_args(p)
    p.set_defaults(func=cmd_validate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
