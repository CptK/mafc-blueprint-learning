#!/usr/bin/env python3
"""Run the blueprint learning pipeline from a YAML config file.

Steps
-----
1. Load VeriTaS data and split into train / test sets.
2. Pre-extract article analyses for all training claims (cached to disk so
   re-runs are free).
3. Build the learning components (selector, fit assessor, updater, synthesizer,
   consolidator, pipeline) from the config.
4. Run the iterative learning loop, saving a blueprint snapshot after each epoch.
5. Write the final learned blueprints to the output directory.

Usage
-----
    python scripts/run_learning.py --config config/experiments/learning/learning_veritas.yaml
"""

from __future__ import annotations

import argparse
import dataclasses
import faulthandler
import json
import random
import resource
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import ezmm
import yaml
from ezmm.common.registry import ItemRegistry

from mafc.blueprints.registry import BlueprintRegistry
from mafc.blueprints.selector import BlueprintSelector
from mafc.common.logger import logger
from mafc.common.modeling import make_model
from mafc.eval.metrics import classification_block
from mafc.eval.run_config import BenchmarkRunConfig
from mafc.eval.single import build_fact_check_agent, compute_agent_fingerprint
from mafc.eval.veritas.benchmark import VeriTaS
from mafc.eval.veritas.metrics import VERDICT_TO_NUMERIC_3, VERDICT_TO_NUMERIC_7
from mafc.learning.article_analyzer import ArticleAnalyzer
from mafc.learning.blueprint_consolidator import BlueprintConsolidator
from mafc.learning.blueprint_fit_assessor import BlueprintFitAssessor
from mafc.learning.blueprint_updater import BlueprintUpdater
from mafc.learning.execution import (
    BlueprintExecutionCache,
    BlueprintExecutor,
    _MutableSingleBlueprintSelector,
)
from mafc.learning.learning_pipeline import EpochStats, LearningPipeline
from mafc.learning.models import ActionEvidenceLink, ArticleAnalysis, ClaimLearningRecord
from mafc.learning.new_blueprint_synthesizer import NewBlueprintSynthesizer
from mafc.learning.run_config import LearningRunConfig
from mafc.learning.scorecard import BlueprintScorecard
from mafc.learning.snapshot import restore_registry_in_place, snapshot_registry

faulthandler.enable()

# Raise the open-file-descriptor limit to avoid EMFILE errors under parallel load.
_soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, _hard), _hard))

# ezmm's ItemRegistry uses a single shared SQLite cursor which is not thread-safe.
# Prompt construction can trigger registry lookups when claim text contains media
# tokens, so all concurrent workers must serialize those calls.
_registry_lock = threading.RLock()
for _name in ("get", "get_by_path", "add_item", "get_cached", "update_file_path", "contains"):
    _orig = getattr(ItemRegistry, _name)

    def _make_locked(m):
        def _locked(self, *args, **kwargs):
            with _registry_lock:
                return m(self, *args, **kwargs)

        return _locked

    setattr(ItemRegistry, _name, _make_locked(_orig))

# ---------------------------------------------------------------------------
# Article analysis cache helpers
# ---------------------------------------------------------------------------


def _analysis_to_dict(a: ArticleAnalysis) -> dict:
    return dataclasses.asdict(a)


def _analysis_from_dict(d: dict) -> ArticleAnalysis:
    links = d.get("action_evidence_links")
    return ArticleAnalysis(
        claim_type=d["claim_type"],
        verdict_summary=d["verdict_summary"],
        key_evidence=d.get("key_evidence") or [],
        evidence_types=d.get("evidence_types") or [],
        action_evidence_links=([ActionEvidenceLink(**lnk) for lnk in links] if links else None),
        investigative_steps=d.get("investigative_steps"),
        search_queries=d.get("search_queries"),
        process_richness=d.get("process_richness", "result_only"),
        notes=d.get("notes"),
    )


def _load_analysis_cache(path: Path) -> dict[str, ArticleAnalysis]:
    if not path.exists():
        return {}
    with open(path) as f:
        raw: dict[str, dict] = json.load(f)
    return {claim_id: _analysis_from_dict(d) for claim_id, d in raw.items()}


def _save_analysis_cache(cache: dict[str, ArticleAnalysis], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({cid: _analysis_to_dict(a) for cid, a in cache.items()}, f, indent=2)


# ---------------------------------------------------------------------------
# Pre-extraction
# ---------------------------------------------------------------------------


def _extract_analyses(
    samples: list,
    cache: dict[str, ArticleAnalysis],
    analyzer: ArticleAnalyzer,
    cache_path: Path,
    workers: int = 4,
) -> dict[str, ArticleAnalysis]:
    """Run ArticleAnalyzer on all samples not already in cache.

    Saves the cache to disk after every completed analysis so progress is not
    lost if the process is interrupted.
    """
    pending = [s for s in samples if s.id not in cache and s.article_content]
    no_article = [s for s in samples if not s.article_content]

    if no_article:
        logger.warning(
            f"[Extraction] {len(no_article)} claim(s) have no article_content — "
            "they will have no article analysis."
        )
    if not pending:
        logger.info("[Extraction] All analyses already cached.")
        return cache

    logger.info(f"[Extraction] Extracting analyses for {len(pending)} claim(s) using {workers} workers...")

    def _analyze_one(sample):
        return sample.id, analyzer.analyze(
            article_content=sample.article_content,
            claim_text=str(sample.input).strip(),
            original_claim=sample.original_claim if sample.rectified else None,
            claim_id=sample.id,
        )

    completed = 0
    failures = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_analyze_one, s): s for s in pending}
        for future in as_completed(futures):
            sample = futures[future]
            try:
                claim_id, result = future.result()
            except Exception as exc:
                # Transient API failures (Gemini 5xx, rate limits, etc.) should not
                # abort the whole extraction. The failed sample stays out of the
                # cache so the next run picks it up via the `s.id not in cache` filter.
                logger.warning(
                    f"[Extraction] Analysis failed for claim {sample.id}: {type(exc).__name__}: {exc}"
                )
                failures += 1
                completed += 1
                continue
            if result is not None:
                cache[claim_id] = result
            completed += 1
            if completed % 10 == 0 or completed == len(pending):
                logger.info(
                    f"[Extraction] {completed}/{len(pending)} done"
                    + (f" ({failures} failed)" if failures else "")
                    + "."
                )
                _save_analysis_cache(cache, cache_path)
    if failures:
        logger.warning(
            f"[Extraction] {failures}/{len(pending)} samples failed extraction. "
            "Re-run with --analyses-cache pointing at the saved cache to retry."
        )

    _save_analysis_cache(cache, cache_path)
    logger.info(f"[Extraction] Done. Cache has {len(cache)} entries.")
    return cache


# ---------------------------------------------------------------------------
# Blueprint saving
# ---------------------------------------------------------------------------


def _run_dev_eval(
    executor: "BlueprintExecutor",
    selector: BlueprintSelector,
    dev_records: list[ClaimLearningRecord],
    dev_labels: dict[str, str],
    dev_gt_scores: dict[str, float],
    label_set: list[str],
    workers: int,
) -> tuple[dict, BlueprintScorecard]:
    """Score every dev record through (real selector → forced-blueprint executor).

    Mirrors production: ``selector.select`` runs without article_analysis so the
    selector sees what it would see at inference time, then the executor runs the
    blueprint the selector picked. Cache is shared with training so repeated dev
    claims across epochs are cheap unless the blueprint pool mutated.

    Returns a dict carrying both classification (accuracy, macro_f1) and
    regression (mse, mae) views. MSE/MAE are computed when records carry
    paired predicted_score / gt_score (set by the executor when
    label_to_numeric is configured).
    """
    dev_scorecard = BlueprintScorecard()
    y_true: list[str] = []
    y_pred: list[str] = []
    gt_scores: list[float] = []
    pred_scores: list[float] = []
    n_errored = 0
    total_cost_usd = 0.0
    n_total = len(dev_records)

    def _eval_one(rec: ClaimLearningRecord):
        cid = getattr(rec.claim, "id", None)
        if cid is None or cid not in dev_labels:
            return None
        selection = selector.select(rec.claim, article_analysis=None)
        blueprint = selection.selected_blueprint
        result = executor.run(
            rec.claim,
            blueprint,
            true_label=dev_labels[cid],
            claim_id=cid,
            gt_score=dev_gt_scores.get(cid),
        )
        rec.assigned_blueprint = blueprint.name
        rec.execution_result = result
        return result

    if workers <= 1:
        results = [_eval_one(rec) for rec in dev_records]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_eval_one, rec) for rec in dev_records]
            results = [f.result() for f in as_completed(futures)]

    for result in results:
        if result is None:
            continue
        dev_scorecard.record(result)
        total_cost_usd += result.cost_usd
        if result.predicted_label is None:
            n_errored += 1
            continue
        y_true.append(result.ground_truth)
        y_pred.append(result.predicted_label)
        if result.predicted_score is not None and result.gt_score is not None:
            gt_scores.append(result.gt_score)
            pred_scores.append(result.predicted_score)

    completed = len(y_true)
    metrics = {
        "n_total": n_total,
        "n_completed": completed,
        "n_errored": n_errored,
        "accuracy": None,
        "macro_f1": None,
        "mse": None,
        "mae": None,
        "n_scored": len(gt_scores),
        "avg_cost_usd": (total_cost_usd / n_total) if n_total else None,
    }
    if completed:
        block = classification_block(y_true, y_pred, label_set)
        metrics["accuracy"] = block["accuracy"]
        metrics["macro_f1"] = block["macro"]["f1"]
    if gt_scores:
        diffs = [(p - g) for p, g in zip(pred_scores, gt_scores)]
        metrics["mse"] = sum(d * d for d in diffs) / len(diffs)
        metrics["mae"] = sum(abs(d) for d in diffs) / len(diffs)
    return metrics, dev_scorecard


def _save_blueprints(registry: BlueprintRegistry, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for bp in registry.get_all():
        bp_dict = bp.model_dump(by_alias=True)
        path = directory / f"{bp.name}.yaml"
        with open(path, "w") as f:
            yaml.dump(bp_dict, f, default_flow_style=False, allow_unicode=True)
    logger.info(f"[Output] Saved {len(registry.get_all())} blueprint(s) to {directory}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the blueprint learning pipeline.")
    parser.add_argument("--config", required=True, metavar="PATH", help="Path to learning config YAML.")
    parser.add_argument(
        "--analyses-cache",
        metavar="PATH",
        help="Override path for the article analyses cache JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = LearningRunConfig.from_yaml(args.config)

    # Apply the configured console log level BEFORE anything else logs. The
    # default in mafc.common.logger is DEBUG, which dumps full action-node
    # prompts and responses per fact-check iteration — that's tens of MB on
    # any non-smoke run. ``logger.set_log_level`` only affects the stdout
    # handler; file handlers (and per-fact-check traces) keep their detail.
    logger.set_log_level(config.output.log_level.lower())  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(config.output.dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, run_dir / "config.yaml")
    ezmm.set_ezmm_path(run_dir / "temp")

    logger.info(f"[Setup] Run directory: {run_dir} (log_level={config.output.log_level})")

    # ------------------------------------------------------------------
    # Load data and split
    # ------------------------------------------------------------------
    benchmark = VeriTaS(
        data_path=config.data.data_path,
        variant=config.data.split,
        label_scheme=config.data.label_scheme,
    )
    samples = benchmark.data
    if config.data.first_n is not None:
        samples = samples[: config.data.first_n]

    rng = random.Random(config.data.seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    split = int(len(indices) * config.data.train_fraction)
    train_pool_samples = [samples[i] for i in indices[:split]]
    test_samples = [samples[i] for i in indices[split:]]

    # Carve a dev set off the (already-shuffled) training pool so dev
    # IDs are reproducible across runs with the same seed and never overlap
    # the test set.
    dev_n = int(round(len(train_pool_samples) * config.data.dev_fraction))
    if dev_n > 0 and config.data.dev_fraction > 0.0:
        dev_samples = train_pool_samples[:dev_n]
        train_samples = train_pool_samples[dev_n:]
    else:
        dev_samples = []
        train_samples = train_pool_samples

    logger.info(
        f"[Setup] Dataset: {len(samples)} total — "
        f"{len(train_samples)} train / {len(dev_samples)} dev / {len(test_samples)} test"
    )

    # Save the split so evaluation can use the same test set later.
    with open(run_dir / "splits.json", "w") as f:
        json.dump(
            {
                "train_ids": [s.id for s in train_samples],
                "dev_ids": [s.id for s in dev_samples],
                "test_ids": [s.id for s in test_samples],
                "seed": config.data.seed,
                "train_fraction": config.data.train_fraction,
                "dev_fraction": config.data.dev_fraction,
            },
            f,
            indent=2,
        )

    # ------------------------------------------------------------------
    # Article analysis pre-extraction
    # ------------------------------------------------------------------
    cache_path = Path(args.analyses_cache) if args.analyses_cache else run_dir / "article_analyses.json"
    # Load any existing cache (from a previous run or pre-built).
    analysis_cache = _load_analysis_cache(cache_path)

    model = make_model(
        config.model.name,
        temperature=config.model.temperature,
        max_response_length=config.model.max_response_length,
    )
    analyzer = ArticleAnalyzer(model)

    analysis_cache = _extract_analyses(
        train_samples,
        analysis_cache,
        analyzer,
        cache_path,
        workers=config.learning.workers,
    )

    # ------------------------------------------------------------------
    # Build ClaimLearningRecords
    # ------------------------------------------------------------------
    train_records = [
        ClaimLearningRecord(
            claim=s.input,
            article_analysis=analysis_cache.get(s.id),
        )
        for s in train_samples
    ]
    # Dev records intentionally carry no article_analysis
    # simulates production where no gold article is available at inference time.
    dev_records = [ClaimLearningRecord(claim=s.input, article_analysis=None) for s in dev_samples]
    # Ground-truth label lookup so the executor can score outcomes.
    train_labels: dict[str, str] = {s.id: s.label.value for s in train_samples}
    dev_labels: dict[str, str] = {s.id: s.label.value for s in dev_samples}
    # Continuous ground-truth integrity score (VeriTaS ensemble aggregation, in
    # [-1, +1]). Needed for MSE-gated rollback and score-error outcome bucketing.
    train_gt_scores: dict[str, float] = {
        s.id: s.gt_score for s in train_samples if getattr(s, "gt_score", None) is not None
    }
    dev_gt_scores: dict[str, float] = {
        s.id: s.gt_score for s in dev_samples if getattr(s, "gt_score", None) is not None
    }

    # ------------------------------------------------------------------
    # Build learning components
    # ------------------------------------------------------------------
    registry = BlueprintRegistry.from_path(config.blueprints.config_dir)
    selector = BlueprintSelector(
        model=make_model(
            config.model.name,
            temperature=config.model.temperature,
            max_response_length=config.blueprints.selector_max_response_length,
        ),
        registry=registry,
        default_blueprint_name=config.blueprints.default_blueprint,
    )
    # Outcome-aware mutators. Validated against execution.enabled below
    # so we can fail-fast with a clear error instead of silently disabling.
    if config.learning.use_execution_outcomes and not config.execution.enabled:
        raise ValueError(
            "learning.use_execution_outcomes=true requires execution.enabled=true "
            "(the updater and synthesizer read ClaimLearningRecord.execution_result, "
            "which is only populated when the executor runs)."
        )
    fit_assessor = BlueprintFitAssessor(model)
    updater = BlueprintUpdater(
        model,
        use_execution_outcomes=config.learning.use_execution_outcomes,
        outcome_error_threshold=config.learning.outcome_error_threshold,
    )
    generic_bp = registry.get(config.blueprints.default_blueprint)
    synthesizer = NewBlueprintSynthesizer(
        model=model,
        updater=updater,
        generic_blueprint=generic_bp,
        min_cluster_size=config.synthesizer.min_cluster_size,
        use_execution_outcomes=config.learning.use_execution_outcomes,
        outcome_error_threshold=config.learning.outcome_error_threshold,
    )
    consolidator = (
        BlueprintConsolidator(
            model=model,
            updater=updater,
            prune_threshold=config.consolidator.prune_threshold,
            protected_names=set(config.consolidator.protected_names),
        )
        if config.consolidator.enabled
        else None
    )
    # ------------------------------------------------------------------
    # execution feedback (observe-only)
    # ------------------------------------------------------------------
    executor: BlueprintExecutor | None = None
    scorecard: BlueprintScorecard | None = None
    label_set: list[str] | None = None
    if config.execution.enabled:
        if config.execution.agents is None or config.execution.blueprints is None:
            raise ValueError(
                "execution.enabled=true requires execution.agents and execution.blueprints "
                "sections in the config."
            )
        # Repackage as a BenchmarkRunConfig so we can reuse build_fact_check_agent + fingerprint.
        bench_cfg = BenchmarkRunConfig.model_validate(
            {
                "benchmark": {
                    "name": "veritas",
                    "split": config.data.split,
                    "label_scheme": config.data.label_scheme,
                    "data_path": config.data.data_path,
                },
                "agents": config.execution.agents.model_dump(),
                "blueprints": config.execution.blueprints.model_dump(),
                "run": config.execution.run.model_dump(),
            }
        )

        execution_trace_dir = run_dir / "execution_traces" if config.execution.write_traces else None
        if execution_trace_dir is not None:
            execution_trace_dir.mkdir(parents=True, exist_ok=True)
        execution_cache_dir = (
            Path(config.execution.cache_dir)
            if config.execution.cache_dir
            else Path(config.output.dir) / "execution_cache"
        )
        cache = BlueprintExecutionCache(
            root=execution_cache_dir,
            agent_fingerprint=compute_agent_fingerprint(bench_cfg),
        )
        logger.info(
            f"[Setup] Execution feedback ON: cache={execution_cache_dir} fingerprint={cache.fingerprint}"
        )

        def _agent_factory(forced_selector: _MutableSingleBlueprintSelector):
            agent = build_fact_check_agent(
                bench_cfg, benchmark, execution_trace_dir, cache_dir=run_dir / "search_cache"
            )
            # Replace the LLM-driven selector built by build_fact_check_agent with
            # our forced-blueprint adapter so the executor controls routing.
            agent.blueprint_selector = forced_selector  # type: ignore[assignment]
            return agent

        # Map predicted-label strings to their scalar positions so the executor
        # can populate ``ExecutionResult.predicted_score`` for MSE-based gating
        # and score-error outcome bucketing. VeriTaS-specific today; making it
        # data-driven from the benchmark
        label_to_numeric = VERDICT_TO_NUMERIC_7 if config.data.label_scheme == 7 else VERDICT_TO_NUMERIC_3

        executor = BlueprintExecutor(
            agent_factory=_agent_factory,
            registry=registry,
            cache=cache,
            default_blueprint_name=config.blueprints.default_blueprint,
            label_to_numeric=label_to_numeric,
        )
        scorecard = BlueprintScorecard()
        label_set = sorted({s.label.value for s in train_samples})

    def _post_select(rec: ClaimLearningRecord) -> None:
        if executor is None or rec.assigned_blueprint is None:
            return
        claim_id = getattr(rec.claim, "id", None)
        if claim_id is None or claim_id not in train_labels:
            return
        blueprint = registry.get(rec.assigned_blueprint)
        result = executor.run(
            rec.claim,
            blueprint,
            true_label=train_labels[claim_id],
            claim_id=claim_id,
            gt_score=train_gt_scores.get(claim_id),
        )
        rec.execution_result = result
        if scorecard is not None:
            scorecard.record(result)

    pipeline = LearningPipeline(
        registry=registry,
        selector=selector,
        fit_assessor=fit_assessor,
        updater=updater,
        synthesizer=synthesizer,
        consolidator=consolidator,
        update_threshold=config.learning.update_threshold,
        minibatch_size=config.learning.minibatch_size,
        max_epochs=config.learning.max_epochs,
        consolidate_every=config.learning.consolidate_every,
        use_article_analysis_for_selection=config.learning.use_article_analysis_for_selection,
        frozen_blueprint_names=(
            {bp.name for bp in registry.get_all()}
            if config.learning.freeze_all_blueprints
            else {config.blueprints.default_blueprint}
        ),
        workers=config.learning.workers,
        post_select_hook=_post_select if executor is not None else None,
        outcome_error_threshold=config.learning.outcome_error_threshold,
    )

    # ------------------------------------------------------------------
    # Rollback bookkeeping
    # ------------------------------------------------------------------
    rollback_enabled = config.learning.rollback_on_regression
    gate_metric = config.learning.gate_metric
    if gate_metric not in ("macro_f1", "mse"):
        raise ValueError(f"Unsupported learning.gate_metric: {gate_metric!r}. Expected 'macro_f1' or 'mse'.")
    if rollback_enabled:
        if not config.execution.enabled:
            raise ValueError(
                "learning.rollback_on_regression=true requires execution.enabled=true "
                "(rollback uses the dev metric produced by dev eval)."
            )
        if config.data.dev_fraction <= 0.0 or not dev_records:
            raise ValueError(
                "learning.rollback_on_regression=true requires data.dev_fraction>0 "
                "with a non-empty dev split."
            )
        if gate_metric == "mse" and not dev_gt_scores:
            raise ValueError(
                "learning.gate_metric='mse' requires continuous ground-truth scores on dev "
                "samples (none of the dev samples carry gt_score). VeriTaS exposes this; "
                "check data.label_scheme and the benchmark variant."
            )
    # Single dir reused across epochs — only the most recent pre-epoch state matters.
    rollback_snapshot_dir = run_dir / "rollback_snapshot"
    best_dev_macro_f1: float | None = None
    best_dev_mse: float | None = None

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    epoch_stats_log: list[dict] = []

    def on_epoch_begin(epoch: int, reg: BlueprintRegistry) -> None:
        if rollback_enabled:
            snapshot_registry(reg, rollback_snapshot_dir)

    def on_epoch_end(epoch: int, stats: EpochStats, reg: BlueprintRegistry) -> None:
        nonlocal best_dev_macro_f1, best_dev_mse
        # dev evaluation against the registry as it stands at end-of-epoch.
        if executor is not None and dev_records and label_set is not None:
            logger.info(f"[Dev] Evaluating {len(dev_records)} dev claims (epoch {epoch + 1})...")
            metrics, dev_scorecard = _run_dev_eval(
                executor=executor,
                selector=selector,
                dev_records=dev_records,
                dev_labels=dev_labels,
                dev_gt_scores=dev_gt_scores,
                label_set=label_set,
                workers=config.learning.workers,
            )
            stats.dev_macro_f1 = metrics["macro_f1"]
            stats.dev_accuracy = metrics["accuracy"]
            stats.dev_mse = metrics["mse"]
            stats.dev_mae = metrics["mae"]
            stats.dev_avg_cost_usd = metrics["avg_cost_usd"]
            stats.dev_n_completed = metrics["n_completed"]
            stats.dev_n_errored = metrics["n_errored"]
            dev_scorecard.save_json(
                run_dir / "dev_scorecard" / f"epoch_{epoch + 1:02d}.json", labels=label_set
            )
            mse_str = f"{stats.dev_mse:.4f}" if stats.dev_mse is not None else "n/a"
            mae_str = f"{stats.dev_mae:.4f}" if stats.dev_mae is not None else "n/a"
            logger.info(
                f"[Dev] epoch {epoch + 1} macro_f1={stats.dev_macro_f1} "
                f"accuracy={stats.dev_accuracy} mse={mse_str} mae={mae_str} "
                f"completed={stats.dev_n_completed}/{metrics['n_total']} "
                f"errored={stats.dev_n_errored}"
            )

        # Rollback decision. Direction depends on the configured gate
        # metric — macro_f1 is higher-is-better, MSE is lower-is-better.
        stats.gate_metric = gate_metric
        if rollback_enabled:
            margin = config.learning.rollback_margin
            if gate_metric == "macro_f1" and stats.dev_macro_f1 is not None:
                stats.dev_macro_f1_best_before = best_dev_macro_f1
                if best_dev_macro_f1 is not None and stats.dev_macro_f1 < best_dev_macro_f1 - margin:
                    restore_registry_in_place(reg, rollback_snapshot_dir)
                    stats.rolled_back = True
                    logger.warning(
                        f"[Rollback] epoch {epoch + 1} dev_macro_f1={stats.dev_macro_f1:.4f} "
                        f"< best={best_dev_macro_f1:.4f} - margin={margin} — registry restored."
                    )
                elif best_dev_macro_f1 is None or stats.dev_macro_f1 > best_dev_macro_f1:
                    best_dev_macro_f1 = stats.dev_macro_f1
            elif gate_metric == "mse" and stats.dev_mse is not None:
                stats.dev_mse_best_before = best_dev_mse
                if best_dev_mse is not None and stats.dev_mse > best_dev_mse + margin:
                    restore_registry_in_place(reg, rollback_snapshot_dir)
                    stats.rolled_back = True
                    logger.warning(
                        f"[Rollback] epoch {epoch + 1} dev_mse={stats.dev_mse:.4f} "
                        f"> best={best_dev_mse:.4f} + margin={margin} — registry restored."
                    )
                elif best_dev_mse is None or stats.dev_mse < best_dev_mse:
                    best_dev_mse = stats.dev_mse

        # Snapshots are written AFTER the rollback decision so blueprints/epoch_NN
        # always reflects the kept state (post-rollback if a rollback fired).
        if config.output.save_epoch_snapshots:
            _save_blueprints(reg, run_dir / "blueprints" / f"epoch_{epoch + 1:02d}")
        if scorecard is not None:
            scorecard.save_json(run_dir / "scorecard" / f"epoch_{epoch + 1:02d}.json", labels=label_set)

        # Log cumulative executor stats AFTER dev eval so the line reflects
        # both train and dev activity for this epoch.
        if executor is not None:
            stats_snapshot = executor.stats()
            logger.info(
                f"[Execution] epoch {epoch + 1} cache_hits={stats_snapshot.get('cache_hits', 0)} "
                f"executed={stats_snapshot.get('executed', 0)} "
                f"total_runs={stats_snapshot.get('total_runs', 0)}"
            )

        # Persist epoch_stats AFTER dev metrics + rollback decision land so the
        # file always reflects the full epoch summary.
        epoch_stats_log.append(dataclasses.asdict(stats))
        with open(run_dir / "epoch_stats.json", "w") as f:
            json.dump(epoch_stats_log, f, indent=2)

    logger.info("[Run] Starting learning pipeline...")
    all_stats = pipeline.run(
        train_records,
        on_epoch_end=on_epoch_end,
        on_epoch_begin=on_epoch_begin,
    )

    # ------------------------------------------------------------------
    # Save final blueprints + scorecard
    # ------------------------------------------------------------------
    _save_blueprints(registry, run_dir / "blueprints" / "final")
    if scorecard is not None:
        scorecard.save_json(run_dir / "scorecard" / "final.json", labels=label_set)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("[Summary] Learning complete.")
    logger.info(f"  Epochs run:        {len(all_stats)}")
    logger.info(f"  Final blueprints:  {len(registry.get_all())}")
    for bp in registry.get_all():
        logger.info(f"    - {bp.name}")
    logger.info(f"  Results saved to:  {run_dir}")
    logger.info("=" * 60)
    logger.info(
        f"[Next step] Evaluate learned blueprints on the test set using:\n"
        f"  The test IDs are in {run_dir / 'splits.json'}\n"
        f"  Learned blueprints are in {run_dir / 'blueprints' / 'final'}"
    )


if __name__ == "__main__":
    main()
