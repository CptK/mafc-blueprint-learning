#!/usr/bin/env python3
"""Fit the embedding probe that routes claims to blueprints, and save it with the pool.

Blueprints are synthesized from clusters of ground-truth fact-check articles, but at
run time only the claim is available. This fits a multinomial logistic regression from
claim embedding -> blueprint, using each claim's cluster (via clusters.json and the
pool's synthesis_log.json) as the label. Every comparison is claim-to-claim; article
embeddings are never involved.

Writes `selector_probe.json` into the blueprint directory, where BlueprintSelector
picks it up when `blueprints.selection_method` is `embedding_probe` or `hybrid`.

Usage
-----
    python scripts/learning/train_probe.py \\
        --data-dir data/veritas_2025_with_fact_checks \\
        --blueprint-dir out/learned_blueprints/2025/eom_v4
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from mafc.blueprints.probe import PROBE_FILENAME, BlueprintProbe
from mafc.learning.embedding_utils import DEFAULT_EMBEDDING_MODEL
from scripts.learning.cluster_separability import load_claim_embeddings
from scripts.learning.eval_routing import build_target_map, load_claim_texts


def main() -> None:
    """Fit the probe and write it next to the blueprints."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--blueprint-dir", type=Path, required=True)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--refresh", action="store_true", help="Ignore the embedding cache.")
    args = parser.parse_args()

    claim_to_blueprint, _ = build_target_map(args.data_dir, args.blueprint_dir)
    claim_texts = load_claim_texts(args.data_dir)

    ids = sorted(cid for cid in claim_to_blueprint if claim_texts.get(cid))
    labels = np.array([claim_to_blueprint[cid] for cid in ids])

    # Classes with too few examples cannot be cross-validated and would be predicted
    # unreliably; drop them so the probe simply never routes there and the tie-break
    # keeps handling those claims.
    counts = Counter(labels)
    usable = {name for name, count in counts.items() if count >= args.folds * 2}
    dropped = {name: counts[name] for name in counts if name not in usable}
    if dropped:
        print(f"Dropping classes with too few claims to fit: {dropped}")
    keep = np.array([label in usable for label in labels])
    ids = [cid for cid, k in zip(ids, keep) if k]
    labels = labels[keep]

    if len(usable) < 2:
        raise SystemExit(f"Need at least 2 trainable blueprint classes, got {len(usable)}.")

    embeddings = load_claim_embeddings(
        args.data_dir, ids, [claim_texts[cid] for cid in ids], args.embedding_model, args.refresh
    )
    features = normalize(embeddings)

    print(f"\nFitting on {len(labels)} claims across {len(usable)} blueprints.")
    for name, count in counts.most_common():
        print(f"  {name[:66]:68s} n={count}")

    estimator = LogisticRegression(max_iter=2000, class_weight="balanced")
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=0)
    out_of_fold = cross_val_predict(estimator, features, labels, cv=cv, n_jobs=-1)
    accuracy = float((out_of_fold == labels).mean())
    majority = max(counts[name] for name in usable) / len(labels)
    print(f"\nCross-validated routing accuracy: {accuracy:.1%}  (majority baseline {majority:.1%})")
    for name in sorted(usable):
        mask = labels == name
        print(f"  {name[:66]:68s} recall={(out_of_fold[mask] == name).mean():5.1%}")

    estimator.fit(features, labels)
    probe = BlueprintProbe(
        classes=list(estimator.classes_),
        coefficients=estimator.coef_,
        intercepts=estimator.intercept_,
        embedding_model=args.embedding_model,
    )
    out_path = args.blueprint_dir / PROBE_FILENAME
    probe.save(out_path)
    print(f"\nWrote {out_path}")

    # Round-trip so a broken artifact fails here rather than mid-eval.
    reloaded = BlueprintProbe.load(out_path)
    check = reloaded.predict(features[0])
    assert check.blueprint_name == estimator.predict(features[:1])[0], "probe round-trip mismatch"
    print(f"Round-trip verified (first claim -> {check.blueprint_name}, {check.confidence:.2f}).")


if __name__ == "__main__":
    main()
