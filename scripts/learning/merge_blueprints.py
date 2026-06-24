#!/usr/bin/env python3
"""Merge a directory of blueprints into one large strategy tree.

Loads every blueprint in a directory, folds them together by recursively
aligning their verification graphs branch-by-branch (see
mafc.learning.merge_blueprints), and writes the single merged blueprint to YAML.

Usage
-----
    python scripts/learning/merge_blueprints.py \\
        --blueprints-dir config/blueprints \\
        --out out/merged_blueprint.yaml \\
        --model claude_4.8_opus \\
        --name merged

The merge is greedy and order-dependent at borderline matches; the blueprints in
--seed-first are placed first as the spine (default: generic, the most general).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import config.globals  # noqa: F401  -- loads config/.env (API keys) on import
from mafc.blueprints.loader import load_blueprints
from mafc.common.logger import logger
from mafc.common.modeling import make_model
from mafc.learning.merge_blueprints import BlueprintTreeMerger


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--blueprints-dir", required=True, metavar="PATH",
        help="Directory containing the blueprint YAML/JSON files to merge.",
    )
    parser.add_argument(
        "--out", "-o", required=True, metavar="PATH",
        help="Path to write the merged blueprint YAML.",
    )
    parser.add_argument(
        "--model", default="claude_4.8_opus",
        help="LLM for branch matching — shorthand from config/available_models.csv "
        "(default: claude_4.8_opus).",
    )
    parser.add_argument(
        "--name", default="merged",
        help="Name field for the merged blueprint (default: merged).",
    )
    parser.add_argument(
        "--description", default="Merged strategy tree consolidating multiple blueprints.",
        help="Description field for the merged blueprint.",
    )
    parser.add_argument(
        "--seed-first", default="generic", metavar="NAMES",
        help="Comma-separated blueprint names to place first as the merge spine "
        "(default: generic). Names not present are ignored.",
    )
    parser.add_argument(
        "--no-reconcile", action="store_true",
        help="Skip the final pass that merges sibling branches split apart by merge order.",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=4096,
        help="Max response tokens for the LLM seams (default: 4096).",
    )
    args = parser.parse_args()

    blueprints_dir = Path(args.blueprints_dir)
    if not blueprints_dir.is_dir():
        logger.error(f"Not a directory: {blueprints_dir}")
        sys.exit(1)

    blueprints = load_blueprints(blueprints_dir)
    if not blueprints:
        logger.error(f"No blueprints found in {blueprints_dir}")
        sys.exit(1)
    logger.info(f"Loaded {len(blueprints)} blueprint(s): {', '.join(bp.name for bp in blueprints)}")

    seed_first = tuple(n.strip() for n in args.seed_first.split(",") if n.strip())
    model = make_model(args.model, max_response_length=args.max_tokens)

    merger = BlueprintTreeMerger(
        model=model,
        seed_first=seed_first,
        reconcile=not args.no_reconcile,
    )
    result = merger.merge(
        blueprints, name=args.name, description=args.description, progress=True
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    blueprint_yaml = yaml.dump(
        result.blueprint.model_dump(by_alias=True),
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )
    out_path.write_text(blueprint_yaml, encoding="utf-8")

    n_nodes = len(result.blueprint.verification_graph.nodes)
    logger.info(
        f"Merged {len(blueprints)} blueprint(s) into '{args.name}' "
        f"({len(result.tree.entries)} router branch(es), {n_nodes} nodes) -> {out_path}"
    )


if __name__ == "__main__":
    main()
