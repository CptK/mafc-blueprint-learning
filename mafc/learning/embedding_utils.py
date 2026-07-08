"""Shared embedding utilities for strategy fingerprinting and OpenAI embedding calls.

Used by both EmbeddingClusterSynthesizer (learning pipeline) and build_embeddings.py
(offline preprocessing script).
"""

from __future__ import annotations

import numpy as np
from openai import OpenAI

from mafc.common.logger import logger
from mafc.learning.models import ArticleAnalysis

from collections import Counter

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
EMBED_BATCH_SIZE = 512

# process_richness values that carry enough strategy signal to embed
GOOD_RICHNESS: frozenset[str] = frozenset({"full", "partial"})

# Cost per 1M tokens in USD for known OpenAI embedding models.
COST_PER_M_TOKENS: dict[str, float] = {
    "text-embedding-3-large": 0.13,
    "text-embedding-3-small": 0.02,
    "text-embedding-ada-002": 0.10,
}


def label_cluster(analyses: list[ArticleAnalysis], cluster_idx: int) -> tuple[str, str]:
    """Derive a snake_case label and rationale from the dominant features of a cluster.

    The label uses the most common decisive action for the strategy component;
    falls back to the most common evidence type when no decisive actions are present.

    Returns (label, rationale).
    """
    claim_types = Counter(a.claim_type for a in analyses)
    dominant_type = claim_types.most_common(1)[0][0] if claim_types else f"cluster_{cluster_idx}"

    decisive: list[str] = []
    for a in analyses:
        if a.action_evidence_links:
            decisive.extend(lnk.action for lnk in a.action_evidence_links if lnk.was_decisive)

    all_evidence: list[str] = []
    for a in analyses:
        all_evidence.extend(a.evidence_types)

    top_decisive = [ev for ev, _ in Counter(decisive).most_common(1)]
    top_evidence = [ev for ev, _ in Counter(all_evidence).most_common(1)]
    strategy = top_decisive[0] if top_decisive else (top_evidence[0] if top_evidence else None)

    label = "_".join([dominant_type] + ([strategy] if strategy else []))

    top3_decisive = [ev for ev, _ in Counter(decisive).most_common(3)]
    top3_evidence = [ev for ev, _ in Counter(all_evidence).most_common(3)]
    rationale = (
        f"Cluster of {len(analyses)} claims dominated by type '{dominant_type}'. "
        f"Most common decisive actions: {', '.join(top3_decisive) if top3_decisive else 'none'}. "
        f"Most common evidence types: {', '.join(top3_evidence) if top3_evidence else 'none'}."
    )
    return label, rationale


def build_strategy_fingerprint(
    analysis: ArticleAnalysis,
    modality_flags: list[str] | None = None,
) -> str:
    """Build a short strategy-focused text for embedding.

    Strips topic vocabulary; keeps only the verification-strategy signal.
    ``modality_flags`` is an optional list of boolean ClaimFeature names that
    are true for this claim (e.g. ["has_image", "is_multimodal"]).
    """
    # Extract decisive actions upfront — they drive blueprint strategy more than
    # supporting steps do, so they get their own line and influence evidence ordering.
    decisive_actions: list[str] = []
    if analysis.action_evidence_links:
        decisive_actions = [lnk.action for lnk in analysis.action_evidence_links if lnk.was_decisive]

    parts: list[str] = [f"claim_type: {analysis.claim_type}"]

    if decisive_actions:
        parts.append(f"decisive_actions: {', '.join(decisive_actions)}")

    if analysis.evidence_types:
        # Put decisive evidence types first so they carry more weight in the embedding.
        decisive_set = set(decisive_actions)
        ordered = [e for e in analysis.evidence_types if e in decisive_set] + [
            e for e in analysis.evidence_types if e not in decisive_set
        ]
        parts.append(f"evidence_types: {', '.join(ordered)}")

    if analysis.action_evidence_links:
        actions = [lnk.action for lnk in analysis.action_evidence_links]
        parts.append(f"action_sequence: {', '.join(actions)}")

    if analysis.investigative_steps:
        trimmed = [s[:120] for s in analysis.investigative_steps[:5]]
        parts.append(f"steps: {'; '.join(trimmed)}")

    parts.append(f"process_richness: {analysis.process_richness}")

    if modality_flags:
        parts.append(f"modality_flags: {', '.join(modality_flags)}")

    return "\n".join(parts)


def pick_diverse_representatives(X: np.ndarray, n: int) -> list[int]:
    """Select ``n`` row indices covering both the core and the spread of a point set.

    Half the picks are the points nearest the centroid (the cluster's "core"),
    the other half are chosen by farthest-point sampling seeded with that core,
    so outlying subgroups are represented. Centroid-only sampling hides cluster
    heterogeneity — a synthesis LLM shown only near-centroid claims can never
    detect that a cluster mixes several verification strategies.
    """
    if len(X) <= n:
        return list(range(len(X)))

    centroid = X.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    dists_to_centroid = np.linalg.norm(X - centroid, axis=1)

    n_core = max(1, n // 2)
    selected: list[int] = np.argsort(dists_to_centroid)[:n_core].tolist()

    # Farthest-point sampling for the remainder: greedily add the point with the
    # largest distance to its nearest already-selected point.
    min_dist = np.min(np.linalg.norm(X[:, None, :] - X[selected][None, :, :], axis=2), axis=1)
    min_dist[selected] = -np.inf
    while len(selected) < n:
        idx = int(np.argmax(min_dist))
        selected.append(idx)
        min_dist = np.minimum(min_dist, np.linalg.norm(X - X[idx], axis=1))
        min_dist[idx] = -np.inf
    return selected


def split_oversized_clusters(
    X: np.ndarray,
    cluster_indices: list[list[int]],
    max_frac: float,
    min_cluster_size: int,
    n_total: int | None = None,
    max_depth: int = 3,
) -> list[tuple[list[int], int]]:
    """Recursively split clusters holding more than ``max_frac`` of all points.

    A blueprint synthesized from a mega-cluster becomes a shallow catch-all that
    wins routing for half the traffic (the eom_new regression), so no cluster may
    exceed the cap. Splitting first retries HDBSCAN (leaf method) on the subset;
    if that yields fewer than two sub-clusters, KMeans with just enough parts to
    get under the cap is used instead. HDBSCAN noise points are reassigned to the
    nearest sub-cluster rather than dropped — they were validly clustered at the
    top level.

    Args:
        X: Point matrix in the same space used for the original clustering.
        cluster_indices: One list of row indices per input cluster.
        max_frac: Maximum fraction of ``n_total`` a cluster may hold; <= 0 disables.
        min_cluster_size: HDBSCAN min_cluster_size for sub-clustering attempts.
        n_total: Denominator for the cap; defaults to the sum of cluster sizes.
        max_depth: Recursion limit per input cluster.

    Returns:
        (indices, parent_position) tuples — parent_position is the index of the
        originating cluster in ``cluster_indices`` so callers can track provenance.
    """
    from sklearn.cluster import HDBSCAN, KMeans

    if n_total is None:
        n_total = sum(len(ids) for ids in cluster_indices)
    if max_frac <= 0 or n_total == 0:
        return [(ids, pos) for pos, ids in enumerate(cluster_indices)]
    cap = max_frac * n_total

    def _split(indices: list[int], depth: int) -> list[list[int]]:
        if len(indices) <= cap or depth >= max_depth:
            return [indices]
        sub = X[indices]
        labels = HDBSCAN(
            min_cluster_size=min(min_cluster_size, max(2, len(indices) // 2)),
            min_samples=min(min_cluster_size, max(2, len(indices) // 2)),
            metric="euclidean",
            cluster_selection_method="leaf",
        ).fit_predict(sub)

        parts: dict[int, list[int]] = {}
        noise: list[int] = []
        for row, lbl in enumerate(labels.tolist()):
            (parts.setdefault(lbl, []) if lbl >= 0 else noise).append(row)

        if len(parts) < 2:
            k = max(2, int(np.ceil(len(indices) / cap)))
            km_labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(sub)
            parts = {}
            for row, lbl in enumerate(km_labels.tolist()):
                parts.setdefault(int(lbl), []).append(row)
            noise = []

        if noise:
            centroids = {lbl: sub[rows].mean(axis=0) for lbl, rows in parts.items()}
            lbls = list(centroids)
            cents = np.stack([centroids[lbl] for lbl in lbls])
            for row in noise:
                nearest = int(np.argmin(np.linalg.norm(cents - sub[row], axis=1)))
                parts[lbls[nearest]].append(row)

        result: list[list[int]] = []
        for rows in parts.values():
            result.extend(_split([indices[r] for r in rows], depth + 1))
        return result

    out: list[tuple[list[int], int]] = []
    for pos, ids in enumerate(cluster_indices):
        for part in _split(list(ids), 0):
            out.append((part, pos))
    return out


def _embed_batch(texts: list[str], client: OpenAI, model: str) -> tuple[list[list[float]], int]:
    """Embed one batch. Returns (vectors, total_tokens)."""
    response = client.embeddings.create(model=model, input=texts)
    ordered = sorted(response.data, key=lambda d: d.index)
    total_tokens = response.usage.total_tokens if response.usage else 0
    return [d.embedding for d in ordered], total_tokens


def embed_all(
    texts: list[str],
    client: OpenAI,
    model: str = DEFAULT_EMBEDDING_MODEL,
    label: str = "",
) -> tuple[np.ndarray, int]:
    """Embed all texts in batches. Returns (array, total_tokens).

    Logs INFO-level progress after each batch and prints total token count
    and estimated cost on completion.
    """
    prefix = f"[{label}] " if label else ""
    vectors: list[list[float]] = []
    total_tokens = 0
    n_batches = (len(texts) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE

    for batch_idx, start in enumerate(range(0, len(texts), EMBED_BATCH_SIZE)):
        batch = texts[start : start + EMBED_BATCH_SIZE]
        vecs, tokens = _embed_batch(batch, client, model)
        vectors.extend(vecs)
        total_tokens += tokens
        done = min(start + EMBED_BATCH_SIZE, len(texts))
        logger.info(
            f"{prefix}Embedded {done}/{len(texts)} (batch {batch_idx + 1}/{n_batches}, {tokens} tokens)."
        )

    cost = total_tokens / 1_000_000 * COST_PER_M_TOKENS.get(model, 0.0)
    cost_str = f"  Estimated cost: ${cost:.4f}." if cost > 0 else ""
    logger.info(f"{prefix}Done. Total tokens: {total_tokens}.{cost_str}")

    return np.array(vectors, dtype=np.float32), total_tokens
