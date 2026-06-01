"""Blueprint learning pipeline.

Iteratively improves the blueprint pool against a training set of claims.

Loop structure
--------------
For each epoch:
  Shuffle the training records, then process in minibatches of size `minibatch_size`.

  Per minibatch:
    1. For each record:
       - Extract claim features (once; reused across epochs).
       - Run the blueprint selector to assign a blueprint.
       - Run the fit assessor to evaluate fit quality.
       - Route: needs_new_blueprint=True → unmatched_pool; otherwise → pending[blueprint].
    2. Flush any pending buffer that has reached `update_threshold` (mid-epoch update).
    3. Flush the unmatched pool if it has reached `min_cluster_size` (mid-epoch synthesis).

  End of epoch:
    - Flush all remaining pending buffers regardless of size.
    - Flush the unmatched pool regardless of size (synthesizer's min_cluster_size filter
      still applies per cluster, so tiny isolated groups are naturally discarded).
    - Run consolidation every `consolidate_every` epochs if a consolidator is provided:
      prune low-coverage blueprints, then merge overlapping ones.
    - Record EpochStats and check convergence.

Convergence
-----------
The loop stops when an epoch produces zero blueprint updates, zero new blueprints, and
zero consolidation changes, meaning the pool has stabilised, or when `max_epochs` is reached.

Registry updates
----------------
Updated blueprints are replaced in-place in the registry via BlueprintRegistry.replace().
New blueprints from the synthesizer are registered under their LLM-assigned name; if the
name collides with an existing blueprint a numeric suffix is appended.
"""

from __future__ import annotations

import random
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from mafc.blueprints.features import extract_claim_features
from mafc.blueprints.registry import BlueprintRegistry
from mafc.blueprints.selector import BlueprintSelector
from mafc.common.logger import logger
from mafc.learning.blueprint_consolidator import BlueprintConsolidator
from mafc.learning.blueprint_fit_assessor import BlueprintFitAssessor
from mafc.learning.blueprint_updater import BlueprintUpdater
from mafc.learning.models import ClaimLearningRecord
from mafc.learning.new_blueprint_synthesizer import NewBlueprintSynthesizer
from mafc.learning.outcomes import partition_by_outcome


@dataclass
class EpochStats:
    """Summary of one learning epoch."""

    epoch: int
    claims_processed: int
    unmatched_count: int
    """Claims flagged needs_new_blueprint=True during this epoch."""
    blueprints_updated: int
    """Successful updater calls (existing blueprints revised)."""
    blueprints_created: int
    """New blueprints added to the registry via the synthesizer."""
    early_flushes: int
    """Pending buffers flushed mid-epoch because they hit update_threshold."""
    blueprints_pruned: int = 0
    """Blueprints removed by the consolidator due to low coverage."""
    blueprints_merged: int = 0
    """Blueprint pairs collapsed by the consolidator."""

    # ---- Per-epoch dev evaluation ----
    # Populated by the script's on_epoch_end callback after the pipeline finishes
    # the epoch's own bookkeeping. None when no dev set is configured.
    dev_macro_f1: float | None = None
    dev_accuracy: float | None = None
    dev_mse: float | None = None
    """Mean squared error of predicted_score vs gt_score across dev claims.
    None when the executor was not configured with a label_to_numeric mapping
    or no dev claim produced a scored verdict."""
    dev_mae: float | None = None
    """Mean absolute error counterpart of dev_mse. Tracked for observability;
    only dev_mse is consulted by the rollback gate when gate_metric=mse."""
    dev_avg_cost_usd: float | None = None
    dev_n_completed: int = 0
    dev_n_errored: int = 0

    # ---- Rollback bookkeeping ----
    # Populated by the script's on_epoch_end callback when rollback is enabled.
    rolled_back: bool = False
    """True when this epoch's mutations were reverted because the gate metric
    regressed beyond ``rollback_margin``."""
    dev_macro_f1_best_before: float | None = None
    """Running best dev_macro_f1 *prior to* this epoch's dev eval. Populated
    only when gate_metric=macro_f1."""
    dev_mse_best_before: float | None = None
    """Running best dev_mse *prior to* this epoch's dev eval. Populated
    only when gate_metric=mse. Smaller is better."""
    gate_metric: str | None = None
    """Which metric this epoch's rollback decision consulted (mirrored from
    config so log readers don't need to cross-reference)."""

    # ---- Outcome-aware learning bookkeeping ----
    # Populated by the pipeline's flush logic when use_execution_outcomes=true.
    n_updates_with_failures: int = 0
    """Updater flushes this epoch where the input buffer contained ≥1
    record with an ``incorrect`` execution outcome."""
    n_updates_all_correct: int = 0
    """Updater flushes this epoch where the input buffer was 100% correct
    outcomes (no failures, no unknowns). A high ratio here suggests the
    blueprint has stabilised for its assigned claims."""
    synthesis_categories: dict[str, int] = field(default_factory=dict)
    """Counts of synthesizer category tags emitted this epoch
    (``fixes-failures`` / ``specializes-easy-cases`` / ``mixed`` /
    ``unspecified``)."""


@dataclass
class _State:
    """Mutable accumulator shared across minibatches within an epoch."""

    pending: dict[str, list[ClaimLearningRecord]] = field(default_factory=dict)
    unmatched_pool: list[ClaimLearningRecord] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


class LearningPipeline:
    """Iterative blueprint learning loop.

    Args:
        registry: Live blueprint registry — updated in place as blueprints improve.
        selector: Blueprint selector backed by the same registry.
        fit_assessor: Assesses how well the selected blueprint fits each claim.
        updater: Produces improved blueprints from batches of assigned claims.
        synthesizer: Clusters unmatched claims and creates new blueprints.
        update_threshold: Minimum pending-buffer size before a mid-epoch update fires.
        minibatch_size: Number of claims processed per minibatch.
        max_epochs: Hard cap on epochs.
        use_article_analysis_for_selection: Whether to pass article_analysis to the
            selector's LLM tie-break. Disable to simulate production conditions where
            no ground-truth article is available at selection time.
    """

    def __init__(
        self,
        registry: BlueprintRegistry,
        selector: BlueprintSelector,
        fit_assessor: BlueprintFitAssessor,
        updater: BlueprintUpdater,
        synthesizer: NewBlueprintSynthesizer,
        consolidator: BlueprintConsolidator | None = None,
        update_threshold: int = 3,
        minibatch_size: int = 20,
        max_epochs: int = 5,
        consolidate_every: int = 1,
        use_article_analysis_for_selection: bool = False,
        frozen_blueprint_names: set[str] | None = None,
        workers: int = 4,
        post_select_hook: Callable[[ClaimLearningRecord], None] | None = None,
        outcome_error_threshold: float | None = None,
    ) -> None:
        self.registry = registry
        self.selector = selector
        self.fit_assessor = fit_assessor
        self.updater = updater
        self.synthesizer = synthesizer
        self.consolidator = consolidator
        self.update_threshold = update_threshold
        self.minibatch_size = minibatch_size
        self.max_epochs = max_epochs
        self.consolidate_every = consolidate_every
        self.use_article_analysis_for_selection = use_article_analysis_for_selection
        self.frozen_blueprint_names: set[str] = frozen_blueprint_names or set()
        self.workers = workers
        self.post_select_hook = post_select_hook
        self.outcome_error_threshold: float | None = outcome_error_threshold
        """Threshold passed to ``partition_by_outcome`` when computing
        ``EpochStats.n_updates_with_failures`` / ``n_updates_all_correct``.
        Matches the threshold used by the updater so the bookkeeping reflects
        the same notion of "correct" as the actual mutator."""
        """Called from a worker thread after a record is assigned a blueprint but
        before fit assessment, with the mutated ``ClaimLearningRecord``."""

    def run(
        self,
        train_records: list[ClaimLearningRecord],
        on_epoch_end: Callable[[int, EpochStats, BlueprintRegistry], None] | None = None,
        on_epoch_begin: Callable[[int, BlueprintRegistry], None] | None = None,
    ) -> list[EpochStats]:
        """Run the learning loop and return per-epoch statistics.

        Args:
            train_records: Records to learn from. claim_features will be extracted
                and cached on first use; article_analysis should be pre-populated.
            on_epoch_end: Optional callback invoked after each epoch (including
                consolidation). Receives (epoch_index, stats, registry). Use for
                saving per-epoch snapshots.
            on_epoch_begin: Optional callback invoked before each epoch starts,
                with the registry in its pre-epoch state. Receives
                (epoch_index, registry). Used by rollback to capture
                the registry snapshot the epoch's mutations could be reverted to.
        """
        all_stats: list[EpochStats] = []

        for epoch in range(self.max_epochs):
            logger.info(f"[LearningPipeline] === Epoch {epoch + 1}/{self.max_epochs} ===")
            if on_epoch_begin is not None:
                on_epoch_begin(epoch, self.registry)
            state = _State()
            stats = EpochStats(
                epoch=epoch,
                claims_processed=0,
                unmatched_count=0,
                blueprints_updated=0,
                blueprints_created=0,
                early_flushes=0,
            )

            shuffled = train_records.copy()
            random.shuffle(shuffled)

            for batch_start in range(0, len(shuffled), self.minibatch_size):
                minibatch = shuffled[batch_start : batch_start + self.minibatch_size]
                self._process_minibatch(minibatch, state, stats)
                self._flush_pending(state, stats, epoch_end=False)

            # End of epoch: flush everything regardless of size
            self._flush_pending(state, stats, epoch_end=True)

            # Consolidation: prune and merge every `consolidate_every` epochs.
            if self.consolidator is not None and (epoch + 1) % self.consolidate_every == 0:
                c_result = self.consolidator.consolidate(self.registry, shuffled)
                stats.blueprints_pruned = len(c_result.pruned)
                stats.blueprints_merged = len(c_result.merged)

            all_stats.append(stats)
            logger.info(
                f"[LearningPipeline] Epoch {epoch + 1} done — "
                f"updated={stats.blueprints_updated} created={stats.blueprints_created} "
                f"pruned={stats.blueprints_pruned} merged={stats.blueprints_merged} "
                f"unmatched={stats.unmatched_count}/{stats.claims_processed}"
            )

            if on_epoch_end is not None:
                on_epoch_end(epoch, stats, self.registry)

            if self._converged(stats):
                logger.info("[LearningPipeline] Converged — no changes this epoch.")
                break

        return all_stats

    # ------------------------------------------------------------------
    # Per-minibatch processing
    # ------------------------------------------------------------------

    def _process_minibatch(
        self,
        records: list[ClaimLearningRecord],
        state: _State,
        stats: EpochStats,
    ) -> None:
        if self.workers <= 1:
            for rec in records:
                self._process_record(rec, state, stats)
            return
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [pool.submit(self._process_record, rec, state, stats) for rec in records]
            for future in as_completed(futures):
                future.result()

    def _process_record(
        self,
        rec: ClaimLearningRecord,
        state: _State,
        stats: EpochStats,
    ) -> None:
        # Extract claim features once and cache on the record.
        if rec.claim_features is None:
            rec.claim_features = extract_claim_features(rec.claim)

        article_analysis_for_selection = (
            rec.article_analysis if self.use_article_analysis_for_selection else None
        )
        selection = self.selector.select(
            rec.claim,
            article_analysis=article_analysis_for_selection,
        )
        rec.assigned_blueprint = selection.selected_blueprint.name

        if self.post_select_hook is not None:
            try:
                self.post_select_hook(rec)
            except Exception as exc:
                logger.warning(
                    f"[LearningPipeline] post_select_hook failed for "
                    f"claim={getattr(rec.claim, 'id', None)}: {type(exc).__name__}: {exc}"
                )

        fit_result = self.fit_assessor.assess(
            blueprint=selection.selected_blueprint,
            claim=rec.claim,
            claim_features=rec.claim_features,
            article_analysis=rec.article_analysis,
            claim_id=getattr(rec.claim, "id", None),
        )
        rec.fit_result = fit_result

        is_frozen = rec.assigned_blueprint in self.frozen_blueprint_names
        needs_new = fit_result is not None and fit_result.needs_new_blueprint
        with state.lock:
            stats.claims_processed += 1
            if needs_new or is_frozen:
                state.unmatched_pool.append(rec)
                stats.unmatched_count += 1
            else:
                bp_name = rec.assigned_blueprint
                state.pending.setdefault(bp_name, []).append(rec)

    # ------------------------------------------------------------------
    # Flushing logic
    # ------------------------------------------------------------------

    def _flush_pending(
        self,
        state: _State,
        stats: EpochStats,
        epoch_end: bool,
    ) -> None:
        # Flush blueprint-specific pending buffers.
        for bp_name in list(state.pending):
            buf = state.pending[bp_name]
            if not buf:
                continue
            should_flush = epoch_end or len(buf) >= self.update_threshold
            if not should_flush:
                continue

            # Bookkeeping: classify the buffer before flushing so we
            # can later attribute updates to failure-driven vs all-correct.
            correct, incorrect, _unknown = partition_by_outcome(
                buf, error_threshold=self.outcome_error_threshold
            )
            if incorrect:
                stats.n_updates_with_failures += 1
            elif correct and not _unknown:
                stats.n_updates_all_correct += 1

            blueprint = self.registry.get(bp_name)
            result = self.updater.update(blueprint, buf)
            if result is not None and result.updated_blueprint is not None:
                self.registry.replace(bp_name, result.updated_blueprint)
                stats.blueprints_updated += 1
                logger.debug(
                    f"[LearningPipeline] Updated '{bp_name}' "
                    f"({'epoch-end flush' if epoch_end else 'mid-epoch'}, n={len(buf)})."
                )
                if not epoch_end:
                    stats.early_flushes += 1

            state.pending[bp_name] = []

        # Flush unmatched pool.
        pool = state.unmatched_pool
        if not pool:
            return
        should_flush_pool = epoch_end or len(pool) >= self.synthesizer.min_cluster_size
        if not should_flush_pool:
            return

        synthesis_results = self.synthesizer.synthesize(pool)
        for syn in synthesis_results:
            name = self._unique_name(syn.blueprint.name)
            if name != syn.blueprint.name:
                # Patch the name if there was a collision.
                syn.blueprint = syn.blueprint.model_copy(update={"name": name})
            self.registry.register(syn.blueprint)
            stats.blueprints_created += 1
            stats.synthesis_categories[syn.category] = stats.synthesis_categories.get(syn.category, 0) + 1
            logger.info(
                f"[LearningPipeline] Created new blueprint '{name}' "
                f"(cluster='{syn.cluster_label}', size={syn.cluster_size}, "
                f"category={syn.category})."
            )

        state.unmatched_pool = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _converged(self, stats: EpochStats) -> bool:
        return (
            stats.blueprints_updated == 0
            and stats.blueprints_created == 0
            and stats.blueprints_pruned == 0
            and stats.blueprints_merged == 0
        )

    def _unique_name(self, name: str) -> str:
        """Return name unchanged if available, otherwise append a numeric suffix."""
        if not self.registry.contains(name):
            return name
        i = 2
        while self.registry.contains(f"{name}_{i}"):
            i += 1
        return f"{name}_{i}"
