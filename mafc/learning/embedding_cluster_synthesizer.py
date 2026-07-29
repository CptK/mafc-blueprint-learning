"""Embedding-based blueprint synthesizer.

Alternative to NewBlueprintSynthesizer that replaces LLM-based clustering with
embedding similarity + HDBSCAN. Suited for large batches where an LLM single-call
over hundreds of claims would hit context limits.

Pipeline
--------
1. Filter: discard records without article_analysis or with process_richness == "result_only".
2. Fingerprint: build a short strategy-focused text per record (claim_type, evidence_types,
   action sequence, steps, modality flags). Topic-neutral by design.
3. Embed: call OpenAI text-embedding-3-large in batches.
4. Cluster: L2-normalize embeddings, then run sklearn HDBSCAN (euclidean on normalized
   vectors == cosine distance). Points labelled -1 are noise and discarded.
5. Label: derive a snake_case label and rationale from each cluster's dominant features.
6. Synthesize: call BlueprintUpdater once per surviving cluster (same as NewBlueprintSynthesizer).

Interface
---------
Implements the same synthesize(records) -> list[BlueprintSynthesisResult] as
NewBlueprintSynthesizer, so it is a drop-in replacement.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from openai import OpenAI
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import normalize

from mafc.blueprints.models import Blueprint
from mafc.common.logger import logger
from mafc.learning.blueprint_updater import BlueprintUpdater
from mafc.learning.embedding_utils import (
    DEFAULT_EMBEDDING_MODEL,
    GOOD_RICHNESS,
    build_strategy_fingerprint,
    embed_all,
    label_cluster,
)
from mafc.learning.models import ClaimLearningRecord
from mafc.learning.new_blueprint_synthesizer import BlueprintSynthesisResult, _SYNTHESIS_HINT


def _strategy_fingerprint(rec: ClaimLearningRecord) -> str:
    """Build a strategy fingerprint from a ClaimLearningRecord."""
    assert rec.article_analysis is not None
    modality_flags: list[str] | None = None
    if rec.claim_features:
        flags = [
            k
            for k, v in rec.claim_features.model_dump().items()
            if isinstance(v, bool) and v and k != "has_claim_text"
        ]
        modality_flags = flags or None
    return build_strategy_fingerprint(rec.article_analysis, modality_flags)


# ---------------------------------------------------------------------------
# Cluster labelling
# ---------------------------------------------------------------------------


@dataclass
class _EmbeddingCluster:
    label: str
    rationale: str
    record_indices: list[int]


def _label_cluster(records: list[ClaimLearningRecord], cluster_idx: int) -> tuple[str, str]:
    analyses = [r.article_analysis for r in records if r.article_analysis is not None]
    return label_cluster(analyses, cluster_idx)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class EmbeddingClusterSynthesizer:
    """Clusters records via embedding similarity and synthesizes one blueprint per cluster.

    Drop-in replacement for NewBlueprintSynthesizer. Uses OpenAI
    text-embedding-3-large + sklearn HDBSCAN instead of an LLM clustering call.

    Args:
        updater: BlueprintUpdater used to synthesize each cluster's blueprint.
        generic_blueprint: Template blueprint passed to the updater.
        min_cluster_size: Minimum number of records for a cluster to produce a blueprint.
            Passed directly to HDBSCAN and used as a post-filter.
        min_samples: HDBSCAN min_samples (controls noise sensitivity). Defaults to
            min_cluster_size when None.
        cluster_selection_epsilon: HDBSCAN epsilon for merging nearby micro-clusters.
            0.0 (default) applies no merging.
    """

    def __init__(
        self,
        updater: BlueprintUpdater,
        generic_blueprint: Blueprint,
        min_cluster_size: int = 3,
        min_samples: int | None = None,
        cluster_selection_epsilon: float = 0.0,
    ) -> None:
        self.updater = updater
        self.generic_blueprint = generic_blueprint
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples if min_samples is not None else min_cluster_size
        self.cluster_selection_epsilon = cluster_selection_epsilon

        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("openai_api_key")
        if not api_key:
            raise ValueError("Missing OpenAI API key. Set OPENAI_API_KEY in the environment or config/.env.")
        self._client = OpenAI(api_key=api_key, timeout=120)

    def synthesize(self, records: list[ClaimLearningRecord]) -> list[BlueprintSynthesisResult]:
        """Embed, cluster, and synthesize blueprints for the given records.

        Records without article_analysis or with process_richness == "result_only"
        are silently discarded before embedding.
        """
        eligible = self._filter(records)
        if not eligible:
            logger.info("[EmbeddingClusterSynthesizer] No eligible records after filtering.")
            return []

        logger.info(
            f"[EmbeddingClusterSynthesizer] {len(eligible)}/{len(records)} records eligible "
            f"(filtered {len(records) - len(eligible)} without analysis or result_only richness)."
        )

        clusters = self._cluster(eligible)
        if not clusters:
            logger.info("[EmbeddingClusterSynthesizer] No clusters found (all noise).")
            return []

        results: list[BlueprintSynthesisResult] = []
        for cluster in clusters:
            cluster_records = [eligible[i] for i in cluster.record_indices]
            logger.info(
                f"[EmbeddingClusterSynthesizer] Synthesizing blueprint for cluster "
                f"'{cluster.label}' ({len(cluster_records)} records)."
            )
            update_result = self.updater.update(
                self.generic_blueprint,
                cluster_records,
                extra_user_hint=_SYNTHESIS_HINT,
            )
            if update_result is None or update_result.updated_blueprint is None:
                logger.warning(
                    f"[EmbeddingClusterSynthesizer] Updater returned no blueprint "
                    f"for cluster '{cluster.label}', skipping."
                )
                continue

            results.append(
                BlueprintSynthesisResult(
                    blueprint=update_result.updated_blueprint,
                    cluster_label=cluster.label,
                    cluster_rationale=cluster.rationale,
                    cluster_size=len(cluster_records),
                    update_result=update_result,
                    category="unspecified",
                )
            )

        return results

    # ------------------------------------------------------------------

    def _filter(self, records: list[ClaimLearningRecord]) -> list[ClaimLearningRecord]:
        return [
            r
            for r in records
            if r.article_analysis is not None and r.article_analysis.process_richness in GOOD_RICHNESS
        ]

    def _cluster(self, records: list[ClaimLearningRecord]) -> list[_EmbeddingCluster]:
        fingerprints = [_strategy_fingerprint(r) for r in records]

        logger.info(
            f"[EmbeddingClusterSynthesizer] Embedding {len(fingerprints)} strategy fingerprints "
            f"via {DEFAULT_EMBEDDING_MODEL}."
        )
        raw, _ = embed_all(
            fingerprints,
            self._client,
            model=DEFAULT_EMBEDDING_MODEL,
            label="EmbeddingClusterSynthesizer",
        )

        X = normalize(raw, norm="l2")

        clusterer = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric="euclidean",
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            store_centers="centroid",
        )
        labels: np.ndarray = clusterer.fit_predict(X)

        unique_labels = sorted(set(labels.tolist()))
        n_noise = int((labels == -1).sum())
        n_clusters = len([label for label in unique_labels if label >= 0])
        logger.info(f"[EmbeddingClusterSynthesizer] HDBSCAN: {n_clusters} clusters, {n_noise} noise points.")

        clusters: list[_EmbeddingCluster] = []
        for cluster_idx in unique_labels:
            if cluster_idx < 0:
                continue
            indices = [i for i, lbl in enumerate(labels.tolist()) if lbl == cluster_idx]
            cluster_records = [records[i] for i in indices]
            label, rationale = _label_cluster(cluster_records, cluster_idx)
            clusters.append(
                _EmbeddingCluster(
                    label=label,
                    rationale=rationale,
                    record_indices=indices,
                )
            )
            logger.debug(
                f"[EmbeddingClusterSynthesizer] Cluster {cluster_idx}: "
                f"label='{label}' size={len(indices)}"
            )

        return clusters
