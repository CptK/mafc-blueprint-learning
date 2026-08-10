#!/usr/bin/env python3
"""Stratified, boundary-weighted sampler CLI for calibration training data.

Selects a claim subset (stratified by integrity direction, oversampling the
certain/rather-certain boundary region) and writes:

  - <out>.sample_ids.yaml   a YAML fragment with ``benchmark.sample_ids`` ready to
                            paste into an experiment config.
  - <out>.manifest.csv      id, score, direction, stratum, weight for analysis.
  - <out>.manifest.json     the same manifest as JSON.

Usage
-----
    python scripts/training/sample_claims.py \\
        --claims data/veritas_2025_q2_with_fact_checks \\
                 data/veritas_2025_q3_with_fact_checks \\
                 data/veritas_2025_q4_with_fact_checks \\
        --target-n 600 --hard-weight 3.0 --seed 0 \\
        --out out/training/sample_q2q4
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import yaml

from mafc.common.logger import logger
from mafc.training.claims_io import load_many, resolve_claims_paths
from mafc.training.sampler import SamplerConfig, sample, stratum_counts

DEFAULT_CLAIMS = [
    "data/veritas_2025_q2_with_fact_checks",
    "data/veritas_2025_q3_with_fact_checks",
    "data/veritas_2025_q4_with_fact_checks",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--claims", nargs="+", default=DEFAULT_CLAIMS, help="claims.json files or their containing dirs"
    )
    ap.add_argument("--target-n", type=int, default=None, help="number of claims to select")
    ap.add_argument("--hard-band", type=float, nargs=2, default=(0.5, 1.0), help="|score| band to oversample")
    ap.add_argument("--hard-weight", type=float, default=3.0)
    ap.add_argument("--easy-weight", type=float, default=1.0)
    ap.add_argument("--unknown-weight", type=float, default=1.0)
    ap.add_argument("--no-balance-directions", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True, help="output path stem")
    args = ap.parse_args()

    paths = resolve_claims_paths([Path(p) for p in args.claims])
    if not paths:
        ap.error("no valid claims.json found in --claims")
    claims = list(load_many(paths).values())
    logger.info(f"Loaded {len(claims)} claims from {len(paths)} file(s).")

    cfg = SamplerConfig(
        target_n=args.target_n,
        hard_band=tuple(args.hard_band),
        hard_weight=args.hard_weight,
        easy_weight=args.easy_weight,
        unknown_weight=args.unknown_weight,
        seed=args.seed,
        balance_directions=not args.no_balance_directions,
    )
    selected = sample(claims, cfg)
    logger.info(f"Selected {len(selected)} claims. Strata: {stratum_counts(selected)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    ids = [s.id for s in selected]
    yaml_path = out.with_suffix(".sample_ids.yaml")
    yaml_path.write_text(
        yaml.safe_dump({"benchmark": {"sample_ids": ids}}, sort_keys=False), encoding="utf-8"
    )

    manifest = [
        {"id": s.id, "score": s.score, "direction": s.direction, "stratum": s.stratum, "weight": s.weight}
        for s in selected
    ]
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with out.with_suffix(".manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "score", "direction", "stratum", "weight"])
        w.writeheader()
        w.writerows(manifest)

    logger.info(f"Wrote {yaml_path}, {out.with_suffix('.manifest.csv')}, {out.with_suffix('.manifest.json')}")


if __name__ == "__main__":
    main()
