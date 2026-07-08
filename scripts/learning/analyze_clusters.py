#!/usr/bin/env python3
"""Analyze and visualize clustering results.

Reads clusters.json, embeddings.json, and claims.json from a dataset directory
and produces:
  - cluster_visualization.png  — 2D UMAP scatter plot colored by cluster
  - cluster_summary.txt        — N representative claims per cluster

The 2D projection is computed fresh (independent of the clustering UMAP), using
the same n_neighbors stored in clusters.json when available.

Usage
-----
    python scripts/analyze_clusters.py \\
        --data-dir data/veritas_2025_q1_with_fact_checks \\
        --samples 3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mafc.common.logger import logger
from mafc.learning.analysis_io import load_analyses
from mafc.learning.embedding_utils import DEFAULT_EMBEDDING_MODEL, build_strategy_fingerprint
from mafc.learning.models import ArticleAnalysis

_FIG_FILENAME = "cluster_visualization.png"
_SUMMARY_FILENAME = "cluster_summary.txt"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _load_claims_by_id(data_dir: Path) -> dict[str, str]:
    """Return {str(claim_id): claim_text}."""
    raw = _load_json(data_dir / "claims.json")
    return {str(c["id"]): c["text"] for c in raw["claims"]}


# ---------------------------------------------------------------------------
# 2-D UMAP projection
# ---------------------------------------------------------------------------

def _project_2d(X: np.ndarray, n_neighbors: int) -> np.ndarray:
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("umap").setLevel(logging.WARNING)
    try:
        import umap as umap_lib
    except ImportError:
        raise SystemExit("umap-learn is not installed. Run: pip install umap-learn")

    logger.info(f"Projecting {X.shape[0]} points to 2D with UMAP (n_neighbors={n_neighbors})…")
    reducer = umap_lib.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,   # slightly spread out for readability vs 0.0 for clustering
        metric="euclidean",
        random_state=42,
    )
    return reducer.fit_transform(X)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _plot(
    coords_2d: np.ndarray,
    point_labels: list[str],   # cluster label or "noise" per point
    cluster_names: list[str],  # ordered unique cluster names (no "noise")
    out_path: Path,
) -> None:
    cmap = plt.colormaps["tab20"]
    color_map = {name: cmap(i / max(len(cluster_names) - 1, 1)) for i, name in enumerate(cluster_names)}

    fig, ax = plt.subplots(figsize=(14, 10))

    # Noise first (background)
    noise_mask = np.array([l == "noise" for l in point_labels])
    if noise_mask.any():
        ax.scatter(
            coords_2d[noise_mask, 0], coords_2d[noise_mask, 1],
            c="lightgrey", s=8, alpha=0.4, linewidths=0, label="noise",
        )

    # Clusters
    for name in cluster_names:
        mask = np.array([l == name for l in point_labels])
        color = color_map[name]
        ax.scatter(
            coords_2d[mask, 0], coords_2d[mask, 1],
            c=[color], s=18, alpha=0.7, linewidths=0,
        )
        # Label at centroid
        cx, cy = coords_2d[mask, 0].mean(), coords_2d[mask, 1].mean()
        short = name if len(name) <= 32 else name[:30] + "…"
        ax.annotate(
            short, (cx, cy),
            fontsize=6.5, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6, lw=0),
        )

    ax.set_title("Cluster visualization (2D UMAP)", fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])

    # Compact legend
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map[n],
                   markersize=7, label=n)
        for n in cluster_names
    ]
    if noise_mask.any():
        handles.append(
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="lightgrey",
                       markersize=7, label=f"noise ({noise_mask.sum()})")
        )
    ax.legend(handles=handles, fontsize=7, loc="upper left",
              bbox_to_anchor=(1.01, 1), borderaxespad=0, framealpha=0.8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved visualization to {out_path}.")


# ---------------------------------------------------------------------------
# Cluster summary
# ---------------------------------------------------------------------------

def _representative_claim_ids(
    cluster_claim_ids: list[str],
    id_to_idx: dict[str, int],
    coords_2d: np.ndarray,
    n: int,
) -> list[str]:
    """Return IDs of the N claims closest to the cluster centroid in 2D space."""
    indices = [id_to_idx[cid] for cid in cluster_claim_ids if cid in id_to_idx]
    if not indices:
        return cluster_claim_ids[:n]
    pts = coords_2d[indices]
    centroid = pts.mean(axis=0)
    dists = np.linalg.norm(pts - centroid, axis=1)
    closest = np.argsort(dists)[:n]
    return [cluster_claim_ids[i] for i in closest]


def _write_summary(
    clusters: list[dict],
    claims_by_id: dict[str, str],
    analyses_by_id: dict[str, ArticleAnalysis],
    id_to_idx: dict[str, int],
    coords_2d: np.ndarray,
    n_samples: int,
    out_path: Path,
) -> None:
    lines: list[str] = []
    for cluster in clusters:
        lines.append(f"{'=' * 70}")
        lines.append(f"{cluster['label']}  ({cluster['size']} claims)")
        lines.append(cluster["rationale"])
        lines.append("")

        rep_ids = _representative_claim_ids(cluster["claim_ids"], id_to_idx, coords_2d, n_samples)
        for i, cid in enumerate(rep_ids, 1):
            text = claims_by_id.get(cid, f"[claim {cid} not found]")
            wrapped = textwrap.fill(text, width=90, subsequent_indent="   ")
            lines.append(f"  {i}. {wrapped}")

            analysis = analyses_by_id.get(cid)
            if analysis is not None:
                fingerprint = build_strategy_fingerprint(analysis)
                for fp_line in fingerprint.splitlines():
                    lines.append(f"     | {fp_line}")
            lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Saved cluster summary to {out_path}.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _process_dir(data_dir: Path, embedding_model: str, n_samples: int) -> None:
    for filename in ("clusters.json", "embeddings.json", "claims.json"):
        if not (data_dir / filename).exists():
            logger.error(f"[{data_dir.name}] {filename} not found.")
            return

    cluster_data = _load_json(data_dir / "clusters.json")
    all_embeddings: dict[str, dict[str, list[float]]] = _load_json(data_dir / "embeddings.json")
    claims_by_id = _load_claims_by_id(data_dir)
    analyses_by_id = load_analyses(data_dir / "article_analyses.json")

    used_model = cluster_data.get("embedding_model", embedding_model)
    clusters: list[dict] = cluster_data["clusters"]

    # Ordered list of all claim IDs that have an embedding for this model
    eligible_ids = [cid for cid, vecs in all_embeddings.items() if used_model in vecs]
    id_to_idx = {cid: i for i, cid in enumerate(eligible_ids)}

    # Build per-point label array
    clustered_ids: set[str] = {cid for c in clusters for cid in c["claim_ids"]}
    cid_to_label: dict[str, str] = {cid: c["label"] for c in clusters for cid in c["claim_ids"]}
    point_labels = [cid_to_label.get(cid, "noise") for cid in eligible_ids]
    cluster_names = [c["label"] for c in clusters]   # preserves size-sorted order

    # 2D projection
    X = np.array([all_embeddings[cid][used_model] for cid in eligible_ids], dtype=np.float32)
    X = normalize(X, norm="l2")
    n_neighbors = (cluster_data.get("umap_params") or {}).get("n_neighbors", 15)
    coords_2d = _project_2d(X, n_neighbors=n_neighbors)

    _plot(coords_2d, point_labels, cluster_names, data_dir / _FIG_FILENAME)
    _write_summary(clusters, claims_by_id, analyses_by_id, id_to_idx, coords_2d, n_samples, data_dir / _SUMMARY_FILENAME)

    logger.info(
        f"[{data_dir.name}] Done. {len(clusters)} clusters, "
        f"{len(clustered_ids)}/{len(eligible_ids)} points clustered."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-dir", nargs="+", required=True, metavar="PATH",
        help="One or more dataset directories containing clusters.json.",
    )
    parser.add_argument(
        "--embedding-model", default=DEFAULT_EMBEDDING_MODEL,
        help=f"Embedding model key to use (default: {DEFAULT_EMBEDDING_MODEL}). "
             "Overridden by the model stored in clusters.json.",
    )
    parser.add_argument(
        "--samples", type=int, default=3,
        help="Number of representative claims to show per cluster (default: 3).",
    )
    args = parser.parse_args()

    for raw_path in args.data_dir:
        data_dir = Path(raw_path)
        if not data_dir.is_dir():
            logger.error(f"Not a directory: {data_dir}")
            continue
        _process_dir(data_dir, embedding_model=args.embedding_model, n_samples=args.samples)


if __name__ == "__main__":
    main()
