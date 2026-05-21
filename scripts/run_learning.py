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
    python scripts/run_learning.py --config config/experiments/learning_veritas.yaml
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
from mafc.eval.veritas.benchmark import VeriTaS
from mafc.learning.article_analyzer import ArticleAnalyzer
from mafc.learning.blueprint_consolidator import BlueprintConsolidator
from mafc.learning.blueprint_fit_assessor import BlueprintFitAssessor
from mafc.learning.blueprint_updater import BlueprintUpdater
from mafc.learning.learning_pipeline import EpochStats, LearningPipeline
from mafc.learning.models import ActionEvidenceLink, ArticleAnalysis, ClaimLearningRecord
from mafc.learning.new_blueprint_synthesizer import NewBlueprintSynthesizer
from mafc.learning.run_config import LearningRunConfig

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
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_analyze_one, s): s for s in pending}
        for future in as_completed(futures):
            claim_id, result = future.result()
            if result is not None:
                cache[claim_id] = result
            completed += 1
            if completed % 10 == 0 or completed == len(pending):
                logger.info(f"[Extraction] {completed}/{len(pending)} done.")
                _save_analysis_cache(cache, cache_path)

    _save_analysis_cache(cache, cache_path)
    logger.info(f"[Extraction] Done. Cache has {len(cache)} entries.")
    return cache


# ---------------------------------------------------------------------------
# Blueprint saving
# ---------------------------------------------------------------------------


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
    parser.add_argument(
        "--extraction-workers",
        type=int,
        default=4,
        metavar="N",
        help="Parallel workers for article analysis extraction (default: 4).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = LearningRunConfig.from_yaml(args.config)

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(config.output.dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, run_dir / "config.yaml")
    ezmm.set_ezmm_path(run_dir / "temp")

    logger.info(f"[Setup] Run directory: {run_dir}")

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
    train_samples = [samples[i] for i in indices[:split]]
    test_samples = [samples[i] for i in indices[split:]]

    logger.info(
        f"[Setup] Dataset: {len(samples)} total — " f"{len(train_samples)} train / {len(test_samples)} test"
    )

    # Save the split so evaluation can use the same test set later.
    with open(run_dir / "splits.json", "w") as f:
        json.dump(
            {
                "train_ids": [s.id for s in train_samples],
                "test_ids": [s.id for s in test_samples],
                "seed": config.data.seed,
                "train_fraction": config.data.train_fraction,
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
        workers=args.extraction_workers,
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
    fit_assessor = BlueprintFitAssessor(model)
    updater = BlueprintUpdater(model)
    generic_bp = registry.get(config.blueprints.default_blueprint)
    synthesizer = NewBlueprintSynthesizer(
        model=model,
        updater=updater,
        generic_blueprint=generic_bp,
        min_cluster_size=config.synthesizer.min_cluster_size,
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
        frozen_blueprint_names={config.blueprints.default_blueprint},
        workers=config.learning.workers,
    )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    epoch_stats_log: list[dict] = []

    def on_epoch_end(epoch: int, stats: EpochStats, reg: BlueprintRegistry) -> None:
        epoch_stats_log.append(dataclasses.asdict(stats))
        with open(run_dir / "epoch_stats.json", "w") as f:
            json.dump(epoch_stats_log, f, indent=2)
        if config.output.save_epoch_snapshots:
            _save_blueprints(reg, run_dir / "blueprints" / f"epoch_{epoch + 1:02d}")

    logger.info("[Run] Starting learning pipeline...")
    all_stats = pipeline.run(train_records, on_epoch_end=on_epoch_end)

    # ------------------------------------------------------------------
    # Save final blueprints
    # ------------------------------------------------------------------
    _save_blueprints(registry, run_dir / "blueprints" / "final")

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
