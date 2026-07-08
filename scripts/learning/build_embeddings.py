#!/usr/bin/env python3
"""Build embeddings.json for one or more dataset directories.

Reads article_analyses.json (output of build_article_analyses.py), filters to
claims with process_richness in {full, partial}, builds a strategy fingerprint
per claim, embeds them via the OpenAI embedding API, and writes embeddings.json.

embeddings.json format:
    { "<claim_id>": { "<model_name>": [float, ...] }, ... }

Multiple embedding models can coexist — running with a different --embedding-model
adds a new key without touching existing ones.

Incremental: claims already present under the target model key are skipped.

Usage
-----
    python scripts/build_embeddings.py \\
        --data-dir data/veritas_2024_q1_with_fact_checks \\
                   data/veritas_2024_q2_with_fact_checks \\
        --embedding-model text-embedding-3-large
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mafc.common.logger import logger
from mafc.learning.analysis_io import load_analyses
from mafc.learning.embedding_utils import (
    COST_PER_M_TOKENS,
    GOOD_RICHNESS,
    build_strategy_fingerprint,
    embed_all,
)

import os
from openai import OpenAI

_EMBEDDINGS_FILENAME = "embeddings.json"


def _load_embeddings(path: Path) -> dict[str, dict[str, list[float]]]:
    """Load existing embeddings.json. Returns {} if not found."""
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _save_embeddings(embeddings: dict[str, dict[str, list[float]]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(embeddings, f)


def _modality_flags_from_claim(claim: dict) -> list[str]:
    """Derive boolean modality feature names from a raw claim dict."""
    media = claim.get("media") or []
    has_image = any(m.get("type") == "image" for m in media)
    has_video = any(m.get("type") == "video" for m in media)
    flags = []
    if has_image:
        flags.append("has_image")
    if has_video:
        flags.append("has_video")
    if has_image or has_video:
        flags.append("is_multimodal")
    return flags


def _load_claims_by_id(data_dir: Path) -> dict[str, dict]:
    """Load claims.json and return a {claim_id: claim_dict} mapping."""
    claims_path = data_dir / "claims.json"
    if not claims_path.exists():
        return {}
    with open(claims_path) as f:
        raw = json.load(f)
    return {str(c["id"]): c for c in raw["claims"]}


def _process_dir(
    data_dir: Path,
    embedding_model: str,
    client: OpenAI,
    force: bool,
) -> None:
    analyses_path = data_dir / "article_analyses.json"
    if not analyses_path.exists():
        logger.warning(f"[{data_dir.name}] article_analyses.json not found — run build_article_analyses.py first.")
        return

    analyses = load_analyses(analyses_path)
    claims_by_id = _load_claims_by_id(data_dir)

    out_path = data_dir / _EMBEDDINGS_FILENAME
    embeddings: dict[str, dict[str, list[float]]] = {} if force else _load_embeddings(out_path)

    # Filter to claims with sufficient process richness
    eligible = {
        cid: a for cid, a in analyses.items()
        if a.process_richness in GOOD_RICHNESS
    }
    discarded = len(analyses) - len(eligible)

    # Skip claims already embedded under this model
    pending_ids = [
        cid for cid in eligible
        if cid not in embeddings or embedding_model not in embeddings[cid]
    ]

    logger.info(
        f"[{data_dir.name}] {len(eligible)}/{len(analyses)} eligible "
        f"({discarded} discarded as result_only), "
        f"{len(pending_ids)} pending for '{embedding_model}'."
    )

    if not pending_ids:
        logger.info(f"[{data_dir.name}] Nothing to embed.")
        return

    # Build fingerprints
    fingerprints = [
        build_strategy_fingerprint(
            eligible[cid],
            modality_flags=_modality_flags_from_claim(claims_by_id[cid]) if cid in claims_by_id else None,
        )
        for cid in pending_ids
    ]

    # Embed
    logger.info(f"[{data_dir.name}] Embedding {len(fingerprints)} fingerprints via '{embedding_model}'…")
    vectors, total_tokens = embed_all(fingerprints, client, model=embedding_model, label=data_dir.name)

    cost = total_tokens / 1_000_000 * COST_PER_M_TOKENS.get(embedding_model, 0.0)
    cost_str = f"  Cost: ${cost:.4f}." if cost > 0 else " (unknown model price)."
    logger.info(f"[{data_dir.name}] {total_tokens} tokens used.{cost_str}")

    # Merge into output dict
    for cid, vec in zip(pending_ids, vectors.tolist()):
        embeddings.setdefault(cid, {})[embedding_model] = vec

    _save_embeddings(embeddings, out_path)
    logger.info(f"[{data_dir.name}] Saved {len(pending_ids)} new embeddings to {out_path}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-dir", nargs="+", required=True, metavar="PATH",
        help="One or more dataset directories containing article_analyses.json.",
    )
    parser.add_argument(
        "--embedding-model", default="text-embedding-3-large",
        help="OpenAI embedding model (default: text-embedding-3-large).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-embed all claims, ignoring existing embeddings for this model.",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("openai_api_key")
    if not api_key:
        raise SystemExit("Missing OPENAI_API_KEY. Set it in the environment or config/.env.")
    client = OpenAI(api_key=api_key, timeout=120)

    for raw_path in args.data_dir:
        data_dir = Path(raw_path)
        if not data_dir.is_dir():
            logger.error(f"Not a directory: {data_dir}")
            continue
        _process_dir(data_dir, embedding_model=args.embedding_model, client=client, force=args.force)


if __name__ == "__main__":
    main()
