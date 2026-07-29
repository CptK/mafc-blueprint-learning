#!/usr/bin/env python3
"""Build article_analyses.json for one or more dataset directories.

Reads claims.json, runs ArticleAnalyzer on each claim that has article_content,
and writes article_analyses.json next to claims.json.

Incremental: claims already present in article_analyses.json are skipped, so
the script can be interrupted and resumed freely.

Usage
-----
    python scripts/build_article_analyses.py \\
        --data-dir data/veritas_2024_q1_with_fact_checks \\
                   data/veritas_2024_q2_with_fact_checks \\
        --model gemini_3.5_flash \\
        --workers 4
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mafc.common.logger import logger
from mafc.common.modeling import make_model
from mafc.learning.analysis_io import load_analyses, save_analyses
from mafc.learning.article_analyzer import ArticleAnalyzer
from mafc.learning.models import ArticleAnalysis

_SAVE_EVERY = 10  # persist after every N completions
_MAX_RETRIES = 3
_RETRY_BACKOFF = 5  # seconds; doubles on each retry


def _load_claims(data_dir: Path) -> list[dict]:
    claims_path = data_dir / "claims.json"
    if not claims_path.exists():
        raise FileNotFoundError(f"claims.json not found in {data_dir}")
    with open(claims_path) as f:
        raw = json.load(f)
    return raw["claims"]


def _process_dir(
    data_dir: Path,
    analyzer: ArticleAnalyzer,
    workers: int,
    force: bool,
) -> None:
    out_path = data_dir / "article_analyses.json"

    claims = _load_claims(data_dir)
    analyses: dict[str, ArticleAnalysis] = {} if force else load_analyses(out_path)

    pending = [c for c in claims if c.get("article_content") and str(c["id"]) not in analyses]
    no_article = [c for c in claims if not c.get("article_content")]

    if no_article:
        logger.warning(f"[{data_dir.name}] {len(no_article)} claims have no article_content — skipped.")
    if not pending:
        logger.info(f"[{data_dir.name}] All {len(analyses)} analyses already present.")
        return

    logger.info(
        f"[{data_dir.name}] Analyzing {len(pending)} claims "
        f"({len(analyses)} already cached) with {workers} workers…"
    )

    lock = threading.Lock()
    completed = 0
    failures = 0

    def _analyze(claim: dict) -> tuple[str, ArticleAnalysis | None]:
        claim_id = str(claim["id"])
        delay = _RETRY_BACKOFF
        for attempt in range(1, _MAX_RETRIES + 1):
            result = analyzer.analyze(
                article_content=claim["article_content"],
                claim_text=claim["text"],
                original_claim=claim.get("original_claim") if claim.get("rectified") else None,
                claim_id=claim_id,
            )
            if result is not None:
                return claim_id, result
            if attempt < _MAX_RETRIES:
                logger.warning(
                    f"[{data_dir.name}] claim={claim_id} parse failed "
                    f"(attempt {attempt}/{_MAX_RETRIES}), retrying in {delay}s…"
                )
                time.sleep(delay)
                delay *= 2
        return claim_id, None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_analyze, c): c for c in pending}
        for future in as_completed(futures):
            claim = futures[future]
            nonlocal_completed = 0
            try:
                claim_id, result = future.result()
            except Exception as exc:
                logger.warning(
                    f"[{data_dir.name}] Analysis failed for claim {claim['id']}: "
                    f"{type(exc).__name__}: {exc}"
                )
                with lock:
                    failures += 1
                    completed += 1
                    nonlocal_completed = completed
            else:
                with lock:
                    if result is not None:
                        analyses[claim_id] = result
                    completed += 1
                    nonlocal_completed = completed

            if nonlocal_completed % _SAVE_EVERY == 0 or nonlocal_completed == len(pending):
                with lock:
                    save_analyses(analyses, out_path)
                logger.info(
                    f"[{data_dir.name}] {nonlocal_completed}/{len(pending)}"
                    + (f" ({failures} failed)" if failures else "")
                    + "."
                )

    save_analyses(analyses, out_path)
    logger.info(
        f"[{data_dir.name}] Done. {len(analyses)} analyses saved to {out_path}"
        + (f" ({failures} failures — re-run to retry)" if failures else "")
        + "."
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
        help="One or more dataset directories containing claims.json.",
    )
    parser.add_argument(
        "--model",
        default="gemini_3.5_flash",
        help="LLM model specifier for ArticleAnalyzer (default: gemini_3.5_flash).",
    )
    parser.add_argument(
        "--max-response-length",
        type=int,
        default=20000,
        help="Max output tokens for the LLM (default: 20000). The analysis JSON can be "
        "long — the 2048-token model default is too low and causes truncation.",
    )
    parser.add_argument("--workers", type=int, default=10, help="Parallel workers (default: 10).")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-analyze all claims, ignoring any existing article_analyses.json.",
    )
    args = parser.parse_args()

    model = make_model(args.model, temperature=0.0, max_response_length=args.max_response_length)
    analyzer = ArticleAnalyzer(model)

    for raw_path in args.data_dir:
        data_dir = Path(raw_path)
        if not data_dir.is_dir():
            logger.error(f"Not a directory: {data_dir}")
            continue
        _process_dir(data_dir, analyzer, workers=args.workers, force=args.force)


if __name__ == "__main__":
    main()
