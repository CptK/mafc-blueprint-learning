#!/usr/bin/env python3
"""Synthesize a complete, routing-consistent blueprint set from clusters.json.

Stages (all automatic — no manual curation step):
  1. Per cluster: select representatives covering the cluster's core AND edges
     (half nearest-centroid, half farthest-point sampling), then call
     BlueprintUpdater with the generic blueprint as template.
  2. If the LLM flags should_split, the cluster is bisected via KMeans on its
     embeddings and each half is synthesized separately (one level deep).
  3. Consolidation: prune-free merge pass over the synthesized pool — strategic
     near-duplicates and topical variants of the same strategy are merged, with
     the merged blueprint keeping the larger parent iteration budget.
  4. Iteration floor: blueprints expected to serve >= 10% of traffic get
     max_iterations >= 4.
  5. Contrast pass: one LLM call over the whole pool rewrites descriptions and
     selector hints for mutual exclusivity (routing is LLM-tiebreak over these
     fields; overlapping wording lets one catch-all absorb the traffic).
  6. The generic fallback blueprint is copied alongside, so --out-dir is a
     complete config_dir for benchmark runs.

A synthesis_log.json is written to --out-dir with per-cluster reasoning,
split/merge decisions, and contrast notes.

Usage
-----
    python scripts/learning/build_blueprints.py \\
        --data-dir data/veritas_2025_with_fact_checks \\
        --generic-blueprint config/blueprints/generic.yaml \\
        --out-dir out/learned_blueprints \\
        --model claude_4.8_opus \\
        --samples 15
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from mafc.blueprints.loader import load_blueprint
from mafc.blueprints.registry import BlueprintRegistry
from mafc.common.claim import Claim
from mafc.common.logger import logger
from mafc.common.modeling import make_model
from mafc.learning.analysis_io import load_analyses
from mafc.learning.blueprint_consolidator import BlueprintConsolidator
from mafc.learning.blueprint_contrast import BlueprintContrastPass, enforce_iteration_floor
from mafc.learning.blueprint_updater import BlueprintUpdater
from mafc.learning.embedding_utils import (
    DEFAULT_EMBEDDING_MODEL,
    label_cluster,
    pick_diverse_representatives,
)
from mafc.learning.models import ClaimLearningRecord
from mafc.learning.new_blueprint_synthesizer import _SYNTHESIS_HINT

_LOG_FILENAME = "synthesis_log.json"
_MAX_SPLIT_DEPTH = 1

_CLUSTER_CONTEXT_HINT = """

CLUSTER CONTEXT: this cluster contains {size} claims — {share:.0%} of the training set. \
The representative claims below were sampled to cover the cluster's spread: some sit at \
its core, some at its edges. If they reveal materially different verification strategies, \
set should_split=true and describe the subgroups rather than forcing one blueprint to \
cover all of them. Size the investigation budget to the traffic this blueprint will \
serve: a cluster holding 10% or more of the training set needs max_iterations of at \
least 4 and a corroboration layer in the verification graph.\
"""


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _load_claims_by_id(data_dir: Path) -> dict[str, str]:
    raw = _load_json(data_dir / "claims.json")
    return {str(c["id"]): c["text"] for c in raw["claims"]}


def _vectors_for(
    claim_ids: list[str],
    all_embeddings: dict[str, dict[str, list[float]]],
    embedding_model: str,
) -> tuple[list[str], np.ndarray]:
    """Return (ids, L2-normalized vectors) for the claims that have embeddings."""
    ids = [
        cid for cid in claim_ids
        if cid in all_embeddings and embedding_model in all_embeddings[cid]
    ]
    if not ids:
        return [], np.empty((0, 0), dtype=np.float32)
    X = np.array([all_embeddings[cid][embedding_model] for cid in ids], dtype=np.float32)
    return ids, normalize(X, norm="l2")


def _pick_representatives(
    cluster_claim_ids: list[str],
    all_embeddings: dict[str, dict[str, list[float]]],
    embedding_model: str,
    n: int,
) -> list[str]:
    """Representatives covering core and spread of the cluster (see pick_diverse_representatives)."""
    ids, X = _vectors_for(cluster_claim_ids, all_embeddings, embedding_model)
    if not ids:
        return cluster_claim_ids[:n]
    if len(ids) <= n:
        return ids
    return [ids[i] for i in pick_diverse_representatives(X, n)]


def _bisect_cluster(
    claim_ids: list[str],
    all_embeddings: dict[str, dict[str, list[float]]],
    embedding_model: str,
) -> list[list[str]] | None:
    """Split a cluster into two halves via KMeans on its embeddings. None when infeasible."""
    ids, X = _vectors_for(claim_ids, all_embeddings, embedding_model)
    if len(ids) < 4:
        return None
    labels = KMeans(n_clusters=2, n_init=10, random_state=42).fit_predict(X)
    halves = [
        [ids[i] for i in range(len(ids)) if labels[i] == k]
        for k in (0, 1)
    ]
    if min(len(h) for h in halves) < 2:
        return None
    # Claims without embeddings still belong to the cluster; keep them in the larger half.
    missing = [cid for cid in claim_ids if cid not in set(ids)]
    if missing:
        halves[0 if len(halves[0]) >= len(halves[1]) else 1].extend(missing)
    return halves


def _save_blueprint(bp, out_dir: Path) -> Path:
    out_path = out_dir / f"{bp.name}.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(bp.model_dump(by_alias=True), f, default_flow_style=False, allow_unicode=True)
    return out_path


def _unique_name(name: str, taken: set[str]) -> str:
    if name not in taken:
        return name
    i = 2
    while f"{name}_{i}" in taken:
        i += 1
    return f"{name}_{i}"


def _synthesize_cluster(
    label: str,
    claim_ids: list[str],
    updater: BlueprintUpdater,
    generic_bp,
    all_embeddings: dict,
    embedding_model: str,
    analyses_by_id: dict,
    claims_by_id: dict[str, str],
    n_samples: int,
    n_clustered_total: int,
    log_entries: list[dict],
    depth: int = 0,
) -> list[tuple[object, list[str], list[ClaimLearningRecord]]]:
    """Synthesize blueprint(s) for one cluster, acting on should_split up to _MAX_SPLIT_DEPTH.

    Returns (blueprint, claim_ids, representative_records) tuples — one per resulting
    blueprint (two or more when the cluster was split).
    """
    size = len(claim_ids)
    share = size / max(n_clustered_total, 1)

    logger.info(f"[{label}] Selecting {n_samples} representatives from {size} claims…")
    rep_ids = _pick_representatives(claim_ids, all_embeddings, embedding_model, n_samples)
    records = [
        ClaimLearningRecord(
            claim=Claim(claims_by_id.get(cid, ""), id=cid),
            article_analysis=analyses_by_id.get(cid),
        )
        for cid in rep_ids
    ]

    hint = _SYNTHESIS_HINT + _CLUSTER_CONTEXT_HINT.format(size=size, share=share)
    logger.info(f"[{label}] Synthesizing blueprint from {len(records)} representatives…")
    result = updater.update(generic_bp, records, extra_user_hint=hint)

    if result is None:
        logger.warning(f"[{label}] Updater returned nothing — skipping.")
        log_entries.append({
            "cluster_label": label, "cluster_size": size,
            "n_representatives": len(records), "status": "failed",
        })
        return []

    if result.should_split and depth < _MAX_SPLIT_DEPTH:
        logger.info(f"[{label}] should_split=True: {result.split_rationale} — bisecting cluster.")
        halves = _bisect_cluster(claim_ids, all_embeddings, embedding_model)
        if halves is not None:
            outputs: list = []
            for k, half in enumerate(halves):
                half_analyses = [analyses_by_id[cid] for cid in half if cid in analyses_by_id]
                child_base, _ = label_cluster(half_analyses, k)
                child_label = f"{label}/split{k + 1}_{child_base}"
                outputs.extend(_synthesize_cluster(
                    child_label, half, updater, generic_bp, all_embeddings, embedding_model,
                    analyses_by_id, claims_by_id, n_samples, n_clustered_total,
                    log_entries, depth=depth + 1,
                ))
            if outputs:
                log_entries.append({
                    "cluster_label": label, "cluster_size": size,
                    "status": "split", "split_rationale": result.split_rationale,
                })
                return outputs
        logger.warning(f"[{label}] Split requested but bisection/synthesis failed — keeping unsplit blueprint.")

    if result.updated_blueprint is None:
        logger.warning(f"[{label}] No blueprint produced (should_split without usable split) — skipping.")
        log_entries.append({
            "cluster_label": label, "cluster_size": size,
            "n_representatives": len(records), "status": "failed",
            "split_rationale": result.split_rationale,
        })
        return []

    log_entries.append({
        "cluster_label": label,
        "cluster_size": size,
        "n_representatives": len(records),
        "status": "ok",
        "blueprint_name": result.updated_blueprint.name,
        "should_split": result.should_split,
        "split_rationale": result.split_rationale,
        "reasoning": result.reasoning,
    })
    return [(result.updated_blueprint, claim_ids, records)]


def _process_dir(
    data_dir: Path,
    generic_blueprint_path: Path,
    out_dir: Path,
    model_name: str,
    max_tokens: int,
    n_samples: int,
    consolidate: bool,
    contrast: bool,
    force: bool,
) -> None:
    for filename in ("clusters.json", "embeddings.json", "article_analyses.json", "claims.json"):
        if not (data_dir / filename).exists():
            logger.error(f"[{data_dir.name}] {filename} not found — run prerequisite scripts first.")
            return

    existing = list(out_dir.glob("*.yaml"))
    if existing:
        if not force:
            logger.error(
                f"[{data_dir.name}] {out_dir} already contains {len(existing)} blueprint(s). "
                "The pool passes (consolidation/contrast) must see the full set, so partial "
                "reuse is not supported — use --force or a fresh --out-dir."
            )
            return
        # A fresh run may name its blueprints differently; stale files would
        # pollute the config_dir, so --force clears the previous set entirely.
        for path in existing:
            path.unlink()
        logger.info(f"[{data_dir.name}] --force: removed {len(existing)} existing blueprint file(s).")

    cluster_data = _load_json(data_dir / "clusters.json")
    embedding_model: str = cluster_data.get("embedding_model", DEFAULT_EMBEDDING_MODEL)
    clusters: list[dict] = cluster_data["clusters"]
    n_clustered_total: int = cluster_data.get("n_clustered") or sum(c["size"] for c in clusters)

    all_embeddings: dict[str, dict[str, list[float]]] = _load_json(data_dir / "embeddings.json")
    analyses_by_id = load_analyses(data_dir / "article_analyses.json")
    claims_by_id = _load_claims_by_id(data_dir)

    generic_bp = load_blueprint(generic_blueprint_path)
    model = make_model(model_name, max_response_length=max_tokens)
    updater = BlueprintUpdater(model=model)

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Stage 1+2: per-cluster synthesis with split handling ---
    log_entries: list[dict] = []
    pool: list = []           # Blueprint objects, uniquely named
    sizes: dict[str, int] = {}       # blueprint name -> cluster size
    reps: dict[str, list[ClaimLearningRecord]] = {}  # blueprint name -> representative records

    for cluster in clusters:
        outputs = _synthesize_cluster(
            cluster["label"], cluster["claim_ids"], updater, generic_bp,
            all_embeddings, embedding_model, analyses_by_id, claims_by_id,
            n_samples, n_clustered_total, log_entries,
        )
        for bp, cluster_claim_ids, records in outputs:
            name = _unique_name(bp.name, set(sizes))
            if name != bp.name:
                bp = bp.model_copy(update={"name": name})
            pool.append(bp)
            sizes[name] = len(cluster_claim_ids)
            reps[name] = records

    if not pool:
        logger.error(f"[{data_dir.name}] No blueprints synthesized — aborting.")
        return

    # --- Stage 3: consolidation (merge strategic near-duplicates / topical variants) ---
    consolidation_summary: list[dict] = []
    if consolidate and len(pool) >= 2:
        registry = BlueprintRegistry(pool)
        all_records: list[ClaimLearningRecord] = []
        for name, records in reps.items():
            for rec in records:
                rec.assigned_blueprint = name
                all_records.append(rec)
        max_cluster_frac = cluster_data.get("max_cluster_frac") or 0
        consolidator = BlueprintConsolidator(
            model=model,
            updater=updater,
            prune_threshold=0,
            merge_size_lookup=dict(sizes),
            max_merged_size=int(max_cluster_frac * n_clustered_total) if max_cluster_frac > 0 else None,
        )
        c_result = consolidator.consolidate(registry, all_records)
        for detail in c_result.merge_details:
            sizes[detail["kept"]] = sizes.pop(detail["base"], 0) + sizes.pop(detail["removed"], 0)
            reps[detail["kept"]] = reps.pop(detail["base"], []) + reps.pop(detail["removed"], [])
        consolidation_summary = c_result.merge_details
        pool = registry.get_all()
        logger.info(
            f"[{data_dir.name}] Consolidation: {len(c_result.merged)} merge(s), "
            f"{len(pool)} blueprints remain."
        )

    # --- Stage 4: iteration floor by expected traffic share ---
    pool = [
        enforce_iteration_floor(bp, sizes.get(bp.name, 0) / max(n_clustered_total, 1))
        for bp in pool
    ]

    # --- Stage 5: contrast pass (descriptions/selector hints partition the claim space) ---
    if contrast and len(pool) >= 2:
        shares = {name: size / max(n_clustered_total, 1) for name, size in sizes.items()}
        pool = BlueprintContrastPass(model).run(pool, shares)

    # --- Stage 6: save pool + generic fallback ---
    for bp in pool:
        saved_path = _save_blueprint(bp, out_dir)
        logger.info(f"[{data_dir.name}] Saved '{bp.name}' to {saved_path}.")
    if generic_bp.name not in {bp.name for bp in pool}:
        _save_blueprint(generic_bp, out_dir)

    log_path = out_dir / _LOG_FILENAME
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "clusters": log_entries,
                "consolidation_merges": consolidation_summary,
                "final_blueprints": [
                    {"name": bp.name, "cluster_size": sizes.get(bp.name)} for bp in pool
                ],
            },
            f, indent=2, ensure_ascii=False,
        )

    logger.info(
        f"[{data_dir.name}] Done. {len(pool)} blueprints (+ generic) written to {out_dir}. "
        f"Log: {log_path}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data-dir", nargs="+", required=True, metavar="PATH",
        help="One or more dataset directories containing clusters.json.",
    )
    parser.add_argument(
        "--generic-blueprint", required=True, metavar="PATH",
        help="Path to the generic/template blueprint (YAML or JSON).",
    )
    parser.add_argument(
        "--out-dir", required=True, metavar="PATH",
        help="Directory to write synthesized blueprint YAML files and synthesis_log.json.",
    )
    parser.add_argument(
        "--model", default="claude_4.8_opus",
        help="LLM for synthesis — shorthand from config/available_models.csv (default: claude_4.8_opus).",
    )
    parser.add_argument(
        "--samples", type=int, default=15,
        help="Representative claims per cluster to send to the LLM (default: 15).",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=20000,
        help="Max response tokens for the LLM — blueprint JSON easily exceeds the 2048 default (default: 20000).",
    )
    parser.add_argument(
        "--no-consolidate", action="store_true",
        help="Skip the pool-level merge pass for strategic near-duplicates.",
    )
    parser.add_argument(
        "--no-contrast", action="store_true",
        help="Skip the pool-level description/selector-hint differentiation pass.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite an out-dir that already contains blueprints.",
    )
    args = parser.parse_args()

    generic_blueprint_path = Path(args.generic_blueprint)
    if not generic_blueprint_path.is_file():
        logger.error(f"Generic blueprint not found: {generic_blueprint_path}")
        sys.exit(1)

    out_dir = Path(args.out_dir)

    for raw_path in args.data_dir:
        data_dir = Path(raw_path)
        if not data_dir.is_dir():
            logger.error(f"Not a directory: {data_dir}")
            continue
        _process_dir(
            data_dir=data_dir,
            generic_blueprint_path=generic_blueprint_path,
            out_dir=out_dir,
            model_name=args.model,
            max_tokens=args.max_tokens,
            n_samples=args.samples,
            consolidate=not args.no_consolidate,
            contrast=not args.no_contrast,
            force=args.force,
        )


if __name__ == "__main__":
    main()
