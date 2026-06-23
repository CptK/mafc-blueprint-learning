#!/usr/bin/env python3
"""Trace -> feature-table extractor CLI.

Joins execution traces under one or more ``out/<run>/traces`` dirs with the
ground-truth ``claims.json`` (for ``target = abs(integrity.score)``) and writes a
feature table (CSV always; parquet when an engine is installed).

Embedding features are off by default for a fast structured-only table; pass
``--embeddings`` to add justification-embedding + evidence-dispersion features
(requires the OpenAI client / network).

Usage
-----
    # fast structured-only table from existing 2026 traces (dry run)
    python scripts/training/build_features.py \\
        --traces out/generic-only/traces \\
        --claims data/veritas_2025_q4_with_fact_checks \\
        --out out/training/features_dryrun

    # with embeddings
    python scripts/training/build_features.py \\
        --traces out/<run>/traces --claims data/veritas_2025_q4_with_fact_checks \\
        --embeddings --out out/training/features
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from mafc.common.logger import logger
from mafc.training.claims_io import load_many, resolve_claims_paths
from mafc.training.dataset import build_dataframe, build_meta_table, save_table
from mafc.training.features import FeatureExtractorConfig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", nargs="+", required=True, help="one or more traces dirs")
    ap.add_argument("--claims", nargs="+", required=True, help="claims.json files or dirs")
    ap.add_argument("--embeddings", action="store_true", help="add embedding features")
    ap.add_argument("--embedding-model", default="text-embedding-3-large")
    ap.add_argument("--out", type=Path, required=True, help="output path stem")
    args = ap.parse_args()

    claim_paths = resolve_claims_paths([Path(p) for p in args.claims])
    if not claim_paths:
        ap.error("no valid claims.json found in --claims")
    claims_by_id = load_many(claim_paths)
    logger.info(f"Loaded {len(claims_by_id)} claims.")

    cfg = FeatureExtractorConfig(
        include_embeddings=args.embeddings,
        embedding_model=args.embedding_model,
    )
    df = build_dataframe([Path(t) for t in args.traces], claims_by_id, cfg)
    if df.empty:
        logger.error("No rows produced; nothing written.")
        sys.exit(1)
    written = save_table(df, args.out)

    meta = build_meta_table([Path(t) for t in args.traces], claims_by_id)
    meta_stem = Path(str(args.out) + "_meta")
    written += save_table(meta, meta_stem)
    logger.info(f"Wrote: {', '.join(str(p) for p in written)}")


if __name__ == "__main__":
    main()
