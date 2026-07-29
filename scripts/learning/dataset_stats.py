#!/usr/bin/env python3
"""Print dataset statistics for one or more data directories.

Reads claims.json, article_analyses.json, and embeddings.json (if present)
and reports coverage and distributions.

Usage
-----
    python scripts/dataset_stats.py --data-dir data/veritas_2025_q1_with_fact_checks
    python scripts/dataset_stats.py --data-dir data/veritas_2024_q*_with_fact_checks
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mafc.learning.analysis_io import load_analyses
from mafc.learning.embedding_utils import GOOD_RICHNESS


def _bar(value: int, total: int, width: int = 30) -> str:
    filled = round(width * value / total) if total else 0
    return (
        f"[{'█' * filled}{'░' * (width - filled)}] {value:>5} ({100 * value / total:5.1f}%)"
        if total
        else "n/a"
    )


def _print_counter(counter: Counter, total: int, top_n: int = 10) -> None:
    for value, count in counter.most_common(top_n):
        label = str(value) if value else "(none)"
        print(f"    {label:<40} {_bar(count, total)}")


def report_dir(data_dir: Path) -> None:
    print(f"\n{'═' * 70}")
    print(f"  {data_dir.name}")
    print(f"{'═' * 70}")

    # --- Claims ---
    claims_path = data_dir / "claims.json"
    if not claims_path.exists():
        print("  claims.json not found — skipping.")
        return

    with open(claims_path) as f:
        claims = json.load(f)["claims"]

    n_claims = len(claims)
    n_with_article = sum(1 for c in claims if c.get("article_content"))
    n_with_media = sum(1 for c in claims if c.get("media"))
    lang_counter = Counter(c.get("language") for c in claims)

    print(f"\n  Claims total:          {n_claims}")
    print(f"  With article content:  {n_with_article}  ({100 * n_with_article / n_claims:.1f}%)")
    print(f"  With media:            {n_with_media}  ({100 * n_with_media / n_claims:.1f}%)")

    print("\n  Language distribution (top 10):")
    _print_counter(lang_counter, n_claims)

    # --- Article analyses ---
    analyses_path = data_dir / "article_analyses.json"
    if not analyses_path.exists():
        print("\n  article_analyses.json not found — run build_article_analyses.py first.")
        return

    analyses = load_analyses(analyses_path)
    n_analyzed = len(analyses)

    richness_counter = Counter(a.process_richness for a in analyses.values())
    claim_type_counter = Counter(a.claim_type for a in analyses.values())
    evidence_counter: Counter = Counter()
    for a in analyses.values():
        evidence_counter.update(a.evidence_types)

    decisive_occurrences: Counter = Counter()  # raw count of decisive link occurrences
    decisive_claims: Counter = Counter()  # distinct claims where action was decisive
    n_with_any_decisive = 0
    n_with_links = 0
    for a in analyses.values():
        if a.action_evidence_links:
            n_with_links += 1
            decisive = [lnk.action for lnk in a.action_evidence_links if lnk.was_decisive]
            if decisive:
                n_with_any_decisive += 1
                decisive_occurrences.update(decisive)
                decisive_claims.update(set(decisive))

    n_eligible = sum(1 for a in analyses.values() if a.process_richness in GOOD_RICHNESS)

    print(
        f"\n  Analyses coverage:     {n_analyzed}/{n_with_article}  ({100 * n_analyzed / n_with_article:.1f}% of claims with article)"
    )
    print(
        f"  Eligible for embed:    {n_eligible}/{n_analyzed}  ({100 * n_eligible / n_analyzed:.1f}% — process_richness in {{full, partial}})"
    )

    print("\n  Process richness:")
    _print_counter(richness_counter, n_analyzed)

    print("\n  Claim type:")
    _print_counter(claim_type_counter, n_analyzed)

    print("\n  Evidence types (top 10, multi-label):")
    _print_counter(evidence_counter, sum(evidence_counter.values()))

    print("\n  Decisive actions:")
    print(
        f"    action_evidence_links present: {n_with_links}/{n_analyzed}  ({100 * n_with_links / n_analyzed:.1f}%)"
    )
    if n_with_links:
        print(
            f"    at least one decisive action:  {n_with_any_decisive}/{n_with_links}  ({100 * n_with_any_decisive / n_with_links:.1f}% of those with links)"
        )
    if decisive_claims:
        print(f"\n    % of claims where action was decisive (distinct claims, denominator = {n_analyzed}):")
        _print_counter(decisive_claims, n_analyzed)
        print(
            f"\n    share of total decisive occurrences (multi-label, denominator = {sum(decisive_occurrences.values())} occurrences):"
        )
        _print_counter(decisive_occurrences, sum(decisive_occurrences.values()))

    # --- Embeddings ---
    embeddings_path = data_dir / "embeddings.json"
    if not embeddings_path.exists():
        print("\n  embeddings.json not found — run build_embeddings.py first.")
        return

    with open(embeddings_path) as f:
        embeddings = json.load(f)

    models_present: Counter = Counter()
    for entry in embeddings.values():
        models_present.update(entry.keys())

    print("\n  Embeddings coverage (per model):")
    for model_name, count in models_present.most_common():
        print(f"    {model_name:<40} {_bar(count, n_eligible)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        nargs="+",
        required=True,
        metavar="PATH",
        help="One or more dataset directories.",
    )
    args = parser.parse_args()

    for raw_path in args.data_dir:
        for data_dir in sorted(Path(".").glob(raw_path)) or [Path(raw_path)]:
            if data_dir.is_dir():
                report_dir(data_dir)
            else:
                print(f"Not a directory: {data_dir}")


if __name__ == "__main__":
    main()
