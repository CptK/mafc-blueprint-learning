#!/usr/bin/env python3
"""Cluster embeddings and write clusters.json for one or more dataset directories.

Reads embeddings.json (output of build_embeddings.py) and article_analyses.json,
clusters via HDBSCAN, and writes clusters.json next to the source files.

Optional: pass --reduce-dims N to run UMAP before HDBSCAN. This is strongly
recommended for high-dimensional embeddings (e.g. text-embedding-3-large at 3072
dims) — HDBSCAN's density estimation degrades in high dimensions and most points
end up as noise without prior dimensionality reduction.

clusters.json format:
    {
      "embedding_model": "text-embedding-3-large",
      "umap_params": { "n_components": 50, ... },   // only present when --reduce-dims used
      "hdbscan_params": { "min_cluster_size": 5, ... },
      "n_total": 600,
      "n_clustered": 480,
      "n_noise": 120,
      "n_clusters": 12,
      "clusters": [
        {
          "label": "media_authenticity_reverse_image_search",
          "rationale": "...",
          "size": 34,
          "claim_ids": ["123", "456", ...]
        },
        ...
      ]
    }

Usage
-----
    python scripts/build_clusters.py \\
        --data-dir data/veritas_2024_q1_with_fact_checks \\
        --reduce-dims 50 --min-cluster-size 5
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mafc.common.logger import logger
from mafc.learning.analysis_io import load_analyses
from mafc.learning.embedding_utils import (
    DEFAULT_EMBEDDING_MODEL,
    label_cluster,
    split_oversized_clusters,
)

_CLUSTERS_FILENAME = "clusters.json"


def _load_embeddings(path: Path) -> dict[str, dict[str, list[float]]]:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _process_dir(
    data_dir: Path,
    embedding_model: str,
    reduce_dims: int | None,
    umap_neighbors: int,
    umap_min_dist: float,
    min_cluster_size: int,
    min_samples: int,
    epsilon: float,
    cluster_selection_method: str,
    merge_same_label: bool,
    max_cluster_frac: float,
    force: bool,
) -> None:
    out_path = data_dir / _CLUSTERS_FILENAME
    if out_path.exists() and not force:
        logger.info(f"[{data_dir.name}] clusters.json already exists — skipping (use --force to recompute).")
        return

    embeddings_path = data_dir / "embeddings.json"
    if not embeddings_path.exists():
        logger.warning(f"[{data_dir.name}] embeddings.json not found — run build_embeddings.py first.")
        return

    analyses_path = data_dir / "article_analyses.json"
    if not analyses_path.exists():
        logger.warning(
            f"[{data_dir.name}] article_analyses.json not found — run build_article_analyses.py first."
        )
        return

    all_embeddings = _load_embeddings(embeddings_path)
    analyses = load_analyses(analyses_path)

    eligible_ids = [cid for cid, model_vecs in all_embeddings.items() if embedding_model in model_vecs]

    if not eligible_ids:
        logger.warning(
            f"[{data_dir.name}] No embeddings found for model '{embedding_model}'. "
            f"Run build_embeddings.py --embedding-model {embedding_model} first."
        )
        return

    X = np.array([all_embeddings[cid][embedding_model] for cid in eligible_ids], dtype=np.float32)
    X = normalize(X, norm="l2")

    # --- Optional UMAP reduction ---
    umap_params: dict | None = None
    if reduce_dims is not None:
        try:
            import logging

            logging.getLogger("numba").setLevel(logging.WARNING)
            logging.getLogger("umap").setLevel(logging.WARNING)
            import umap as umap_lib
        except ImportError:
            raise SystemExit("umap-learn is not installed. Run: pip install umap-learn")

        umap_params = {
            "n_components": reduce_dims,
            "n_neighbors": umap_neighbors,
            "min_dist": umap_min_dist,
            "metric": "euclidean",
            "random_state": 42,
        }
        logger.info(
            f"[{data_dir.name}] Reducing {X.shape[1]}→{reduce_dims} dims with UMAP "
            f"(n_neighbors={umap_neighbors}, min_dist={umap_min_dist})…"
        )
        reducer = umap_lib.UMAP(**umap_params)
        X = reducer.fit_transform(X)
        logger.info(f"[{data_dir.name}] UMAP done.")

    # --- HDBSCAN ---
    logger.info(
        f"[{data_dir.name}] Clustering {len(eligible_ids)} points "
        f"({'→'.join([str(d) for d in ([3072 if reduce_dims else X.shape[1]], [reduce_dims]) if reduce_dims] or [str(X.shape[1])])} dims) "
        f"with HDBSCAN (method={cluster_selection_method}, min_cluster_size={min_cluster_size})."
    )
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_epsilon=epsilon,
        cluster_selection_method=cluster_selection_method,
        copy=True,
    )
    labels: np.ndarray = clusterer.fit_predict(X)

    unique_labels = sorted(set(labels.tolist()))
    cluster_indices = [label for label in unique_labels if label >= 0]
    n_noise = int((labels == -1).sum())
    logger.info(f"[{data_dir.name}] {len(cluster_indices)} clusters, {n_noise} noise points.")

    index_lists = [
        [i for i, lbl in enumerate(labels.tolist()) if lbl == cluster_idx] for cluster_idx in cluster_indices
    ]

    # Split clusters exceeding max_cluster_frac of the clustered points: a blueprint
    # synthesized from a mega-cluster degrades into a shallow catch-all at eval time.
    n_clustered_total = len(eligible_ids) - n_noise
    split_lists = split_oversized_clusters(
        X,
        index_lists,
        max_frac=max_cluster_frac,
        min_cluster_size=min_cluster_size,
        n_total=n_clustered_total,
    )
    n_split = len(split_lists) - len(index_lists)
    if n_split > 0:
        logger.info(
            f"[{data_dir.name}] Split oversized clusters (> {max_cluster_frac:.0%} of "
            f"{n_clustered_total} clustered points): {len(index_lists)} → {len(split_lists)} clusters."
        )

    parent_was_split = collections.Counter(parent_pos for _, parent_pos in split_lists)
    parent_labels: dict[int, str] = {}
    for parent_pos, indices in enumerate(index_lists):
        if parent_was_split[parent_pos] > 1:
            parent_analyses = [analyses[eligible_ids[i]] for i in indices if eligible_ids[i] in analyses]
            parent_labels[parent_pos], _ = label_cluster(parent_analyses, parent_pos)

    # Label each (possibly split) cluster by its dominant strategy features.
    labeled_parts: list[dict] = []
    for cluster_pos, (indices, parent_pos) in enumerate(split_lists):
        claim_ids = [eligible_ids[i] for i in indices]
        cluster_analyses = [analyses[cid] for cid in claim_ids if cid in analyses]
        base_label, _ = label_cluster(cluster_analyses, cluster_pos)
        labeled_parts.append(
            {
                "base_label": base_label,
                "claim_ids": claim_ids,
                "parent_pos": parent_pos,
            }
        )

    # Repack split siblings that share the same strategy label, up to the cap.
    # Leaf sub-clustering shatters a mega-cluster into many shards; shards with an
    # identical label follow the same verification strategy and would only yield
    # near-duplicate blueprints, so they are greedily recombined while staying
    # below max_cluster_frac. Shards with distinct labels stay separate.
    if n_split > 0:
        cap = max_cluster_frac * n_clustered_total
        buckets: dict[tuple[int, str], dict] = {}
        packed: list[dict] = []
        for part in labeled_parts:
            key = (part["parent_pos"], part["base_label"])
            bucket = buckets.get(key)
            if (
                parent_was_split[part["parent_pos"]] > 1
                and bucket is not None
                and len(bucket["claim_ids"]) + len(part["claim_ids"]) <= cap
            ):
                bucket["claim_ids"].extend(part["claim_ids"])
            else:
                buckets[key] = part
                packed.append(part)
        if len(packed) < len(labeled_parts):
            logger.info(
                f"[{data_dir.name}] Repacked same-strategy split shards: "
                f"{len(labeled_parts)} → {len(packed)} clusters."
            )
        labeled_parts = packed

    # First pass: build a record per cluster keeping its base label.
    raw_clusters = []
    for cluster_pos, part in enumerate(labeled_parts):
        claim_ids = part["claim_ids"]
        cluster_analyses = [analyses[cid] for cid in claim_ids if cid in analyses]
        base_label, rationale = label_cluster(cluster_analyses, cluster_pos)
        raw_clusters.append(
            {
                "base_label": base_label,
                "rationale": rationale,
                "claim_ids": claim_ids,
                "parent_label": parent_labels.get(part["parent_pos"]),
            }
        )

    if merge_same_label:
        # Merge all HDBSCAN clusters that share the same base label into one,
        # combining their claims. The rationale of the largest contributor wins.
        # Children of a split mega-cluster keep a unique key — merging them back
        # together would undo the max_cluster_frac cap.
        merged: dict[str, dict] = {}
        for i, rc in enumerate(raw_clusters):
            merge_key = rc["base_label"] if rc["parent_label"] is None else f"{rc['base_label']}#split{i}"
            existing = merged.get(merge_key)
            if existing is None:
                merged[merge_key] = {
                    "label": rc["base_label"],
                    "rationale": rc["rationale"],
                    "_rationale_size": len(rc["claim_ids"]),
                    "claim_ids": list(rc["claim_ids"]),
                }
            else:
                existing["claim_ids"].extend(rc["claim_ids"])
                if len(rc["claim_ids"]) > existing["_rationale_size"]:
                    existing["rationale"] = rc["rationale"]
                    existing["_rationale_size"] = len(rc["claim_ids"])
        seen_merged: dict[str, int] = {}
        clusters = []
        for m in merged.values():
            count = seen_merged.get(m["label"], 0) + 1
            seen_merged[m["label"]] = count
            clusters.append(
                {
                    "label": m["label"] if count == 1 else f"{m['label']}_{count}",
                    "rationale": m["rationale"],
                    "size": len(m["claim_ids"]),
                    "claim_ids": m["claim_ids"],
                }
            )
    else:
        # Deduplicate: append _2, _3, … when the same base label appears multiple times
        seen_labels: dict[str, int] = {}
        clusters = []
        for rc in raw_clusters:
            count = seen_labels.get(rc["base_label"], 0) + 1
            seen_labels[rc["base_label"]] = count
            lbl = rc["base_label"] if count == 1 else f"{rc['base_label']}_{count}"
            entry = {
                "label": lbl,
                "rationale": rc["rationale"],
                "size": len(rc["claim_ids"]),
                "claim_ids": rc["claim_ids"],
            }
            if rc["parent_label"] is not None:
                entry["parent_label"] = rc["parent_label"]
            clusters.append(entry)

    clusters.sort(key=lambda c: c["size"], reverse=True)

    for c in clusters:
        logger.info(f"[{data_dir.name}] Cluster '{c['label']}': {c['size']} claims.")

    output: dict = {
        "embedding_model": embedding_model,
    }
    if umap_params is not None:
        output["umap_params"] = umap_params
    output.update(
        {
            "hdbscan_params": {
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "cluster_selection_epsilon": epsilon,
                "cluster_selection_method": cluster_selection_method,
            },
            "max_cluster_frac": max_cluster_frac,
            "n_total": len(eligible_ids),
            "n_clustered": len(eligible_ids) - n_noise,
            "n_noise": n_noise,
            "n_clusters": len(clusters),
            "clusters": clusters,
        }
    )

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(
        f"[{data_dir.name}] Saved {len(clusters)} clusters "
        f"({len(eligible_ids) - n_noise}/{len(eligible_ids)} claims clustered) to {out_path}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data-dir",
        nargs="+",
        required=True,
        metavar="PATH",
        help="One or more dataset directories containing embeddings.json.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"Which model's embeddings to cluster (default: {DEFAULT_EMBEDDING_MODEL}).",
    )

    umap_group = parser.add_argument_group("UMAP (dimensionality reduction)")
    umap_group.add_argument(
        "--reduce-dims",
        type=int,
        default=None,
        metavar="N",
        help="Reduce embeddings to N dims with UMAP before clustering. "
        "Strongly recommended for high-dimensional embeddings (e.g. 50). "
        "Requires umap-learn (pip install umap-learn).",
    )
    umap_group.add_argument(
        "--umap-neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors — larger values = more global structure (default: 15).",
    )
    umap_group.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.0,
        help="UMAP min_dist — use 0.0 for clustering (keeps clusters tight). Default: 0.0.",
    )

    hdbscan_group = parser.add_argument_group("HDBSCAN")
    hdbscan_group.add_argument(
        "--min-cluster-size",
        type=int,
        default=5,
        help="Minimum claims for a cluster to survive (default: 5).",
    )
    hdbscan_group.add_argument(
        "--min-samples",
        type=int,
        default=None,
        help="HDBSCAN min_samples — controls noise sensitivity. Defaults to --min-cluster-size.",
    )
    hdbscan_group.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="HDBSCAN cluster_selection_epsilon — merges nearby micro-clusters (default: 0.0).",
    )
    hdbscan_group.add_argument(
        "--cluster-selection-method",
        default="leaf",
        choices=["eom", "leaf"],
        help="'leaf' = finer clusters; 'eom' = fewer larger clusters (default: leaf).",
    )

    parser.add_argument(
        "--max-cluster-frac",
        type=float,
        default=0.2,
        help="Maximum fraction of clustered claims a single cluster may hold; larger "
        "clusters are recursively split (HDBSCAN leaf, KMeans fallback). A blueprint "
        "synthesized from a mega-cluster degrades into a shallow catch-all. "
        "Set 0 to disable (default: 0.2).",
    )
    parser.add_argument(
        "--merge-same-label",
        action="store_true",
        help="Merge all clusters sharing the same base label (e.g. event_claim_official_records, "
        "event_claim_official_records_2, …) into a single cluster, combining their claims.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Recompute even if clusters.json already exists."
    )
    args = parser.parse_args()

    min_samples = args.min_samples if args.min_samples is not None else args.min_cluster_size

    for raw_path in args.data_dir:
        data_dir = Path(raw_path)
        if not data_dir.is_dir():
            logger.error(f"Not a directory: {data_dir}")
            continue
        _process_dir(
            data_dir,
            embedding_model=args.embedding_model,
            reduce_dims=args.reduce_dims,
            umap_neighbors=args.umap_neighbors,
            umap_min_dist=args.umap_min_dist,
            min_cluster_size=args.min_cluster_size,
            min_samples=min_samples,
            epsilon=args.epsilon,
            cluster_selection_method=args.cluster_selection_method,
            merge_same_label=args.merge_same_label,
            max_cluster_frac=args.max_cluster_frac,
            force=args.force,
        )


if __name__ == "__main__":
    main()
