#!/usr/bin/env python3
"""Build a single ``Strategy.md`` playbook by sequentially folding fact-check analyses.

The fact-checker baseline: instead of a routed pool of structured blueprints, one
free-text playbook distilled from professional fact-checks. This script owns the
running document and folds the corpus into it one batch at a time (one LLM call per
batch), a single pass by default. Multiple passes (epochs) and resuming from a saved
document are supported.

Inputs (per --data-dir, same layout as the blueprint scripts):
  claims.json            {"claims": [{"id", "text"}, ...]}
  article_analyses.json  {claim_id: ArticleAnalysis dict}

Outputs (in --out-dir): strategy.md (latest), strategy_epoch{N}.md (per-epoch
snapshots), fold_log.jsonl (per-batch changelog), state.json (resume metadata).

Usage
-----
    # single pass over the corpus
    python scripts/strategy/build_strategy.py \\
        --data-dir data/veritas_2025_q4_with_fact_checks \\
        --out-dir out/strategy_baseline \\
        --model claude_4.8_opus --batch-size 10 --max-words 2000

    # continue for 2 more passes from a finished run
    python scripts/strategy/build_strategy.py \\
        --data-dir data/veritas_2025_q4_with_fact_checks \\
        --out-dir out/strategy_baseline \\
        --resume-from out/strategy_baseline/strategy_epoch1.md \\
        --start-epoch 1 --epochs 2
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from mafc.common.claim import Claim
from mafc.common.logger import logger
from mafc.common.modeling import make_model
from mafc.learning.analysis_io import load_analyses
from mafc.learning.models import ClaimLearningRecord
from mafc.single_file_strategy.checkpoint import (
    STRATEGY_FILENAME,
    RunState,
    append_fold_log,
    load_document,
    load_state,
    reset_fold_log,
    save_state,
    write_document,
)
from mafc.single_file_strategy.synthesizer import DEFAULT_SKELETON, StrategySynthesizer

# process_richness values worth folding for methodology. "result_only" articles
# state conclusions without process, so they carry little transferable method.
_GOOD_RICHNESS = {"full", "partial"}


def _load_records(data_dirs: list[Path], richness: str, limit: int | None) -> list[ClaimLearningRecord]:
    records: list[ClaimLearningRecord] = []
    for data_dir in data_dirs:
        claims_path = data_dir / "claims.json"
        analyses_path = data_dir / "article_analyses.json"
        if not claims_path.exists():
            logger.error(f"[{data_dir.name}] claims.json not found — skipping.")
            continue
        analyses_by_id = load_analyses(analyses_path)
        raw = json.loads(claims_path.read_text(encoding="utf-8"))
        for c in raw["claims"]:
            cid = str(c["id"])
            analysis = analyses_by_id.get(cid)
            if richness == "good" and (analysis is None or analysis.process_richness not in _GOOD_RICHNESS):
                continue
            records.append(ClaimLearningRecord(claim=Claim(c["text"], id=cid), article_analysis=analysis))

    if limit is not None:
        records = records[:limit]
    return records


def _batches(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data-dir",
        nargs="+",
        required=True,
        metavar="PATH",
        help="Dataset dir(s) with claims.json and article_analyses.json.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        metavar="PATH",
        help="Run directory for strategy.md, snapshots, log, and state.json.",
    )
    parser.add_argument(
        "--model", default="claude_4.8_opus", help="LLM shorthand or specifier (default: claude_4.8_opus)."
    )
    parser.add_argument(
        "--batch-size", type=int, default=10, help="Fact-check analyses folded per LLM call (default: 10)."
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Passes over the corpus (default: 1 — single pass)."
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=2000,
        help="Soft length target the model aims for; not enforced (default: 2000).",
    )
    parser.add_argument(
        "--consolidate-every",
        type=int,
        default=5,
        help="Run a quality consolidation pass every N folds, plus once at the end. "
        "0 disables consolidation (default: 5).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=20000,
        help="Max LLM response tokens; the doc can be long (default: 20000).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Shuffle seed; epoch e uses seed+e (default: 42)."
    )
    parser.add_argument(
        "--richness",
        choices=["good", "all"],
        default="good",
        help="'good' keeps full/partial-process articles; 'all' keeps everything (default: good).",
    )
    parser.add_argument(
        "--resume-from",
        metavar="PATH",
        help="Existing strategy .md to continue from instead of starting empty.",
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=0,
        help="0-based epoch index to start numbering at when resuming (default: 0).",
    )
    parser.add_argument(
        "--seed-skeleton",
        action="store_true",
        help="Start from the recommended skeleton instead of an empty document.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a crashed/stopped run in --out-dir from its last checkpoint "
        "(reads state.json + strategy.md; reuses the saved run parameters).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap number of records (quick smoke tests).")
    args = parser.parse_args()

    data_dirs = [Path(p) for p in args.data_dir]
    out_dir = Path(args.out_dir)

    records = _load_records(data_dirs, args.richness, args.limit)
    if not records:
        logger.error("No eligible records loaded. Check --data-dir and --richness.")
        sys.exit(1)
    logger.info(f"Loaded {len(records)} records (richness={args.richness}).")

    # Resolve the run parameters and starting position. A --resume run reuses the
    # parameters and position saved in out_dir; a fresh run uses the CLI args.
    if args.resume:
        state = load_state(out_dir)
        if state is None:
            logger.error(f"--resume given but no state.json found in {out_dir}.")
            sys.exit(1)
        doc = load_document(out_dir / STRATEGY_FILENAME)
        if state.n_records != len(records):
            logger.warning(
                f"Resume record count ({len(records)}) differs from the saved run "
                f"({state.n_records}); batch boundaries may not line up. Use the same "
                f"--data-dir/--richness/--limit as the original run."
            )
        eff_seed, eff_batch_size, eff_max_words = state.seed, state.batch_size, state.max_words
        eff_epochs, eff_start_epoch, model_name = state.epochs_planned, state.start_epoch, state.model
        resume_epoch, resume_batch = state.resume_epoch_index, state.resume_batch_index
        logger.info(
            f"Resuming run in {out_dir} at epoch {resume_epoch + 1} batch {resume_batch + 1} "
            f"({len(doc.split())} words, {state.total_folds} folds done)."
        )
    else:
        # A fresh run starts the fold log clean; a --resume-from warm start keeps appending.
        if not args.resume_from:
            reset_fold_log(out_dir)
        if args.resume_from:
            doc = load_document(Path(args.resume_from))
            logger.info(f"Warm-starting from {args.resume_from} ({len(doc.split())} words).")
        elif args.seed_skeleton:
            doc = DEFAULT_SKELETON
            logger.info("Starting from the recommended skeleton.")
        else:
            doc = ""
            logger.info("Starting from an empty document.")
        eff_seed, eff_batch_size, eff_max_words = args.seed, args.batch_size, args.max_words
        eff_epochs, eff_start_epoch, model_name = args.epochs, args.start_epoch, args.model
        resume_epoch, resume_batch = args.start_epoch, 0
        state = RunState(
            model=args.model,
            batch_size=args.batch_size,
            max_words=args.max_words,
            seed=args.seed,
            n_records=len(records),
            epochs_planned=args.epochs,
            start_epoch=args.start_epoch,
            resumed_from=args.resume_from,
            resume_epoch_index=args.start_epoch,
        )

    model = make_model(model_name, max_response_length=args.max_tokens)
    synth = StrategySynthesizer(model=model, max_words=eff_max_words)

    def consolidate(current_doc: str, epoch: int, batch: int, label: str) -> str:
        """Run one consolidation pass, log it, and return the cleaned document."""
        words_before = len(current_doc.split())
        c_result = synth.consolidate(current_doc)
        new_doc = c_result.strategy_md
        state.total_consolidations += 1
        append_fold_log(
            out_dir,
            {
                "epoch": epoch,
                "batch": batch,
                "kind": "consolidate",
                "n": 0,
                "ok": c_result.ok,
                "doc_words": len(new_doc.split()),
                "changelog": c_result.changelog,
            },
        )
        logger.info(
            f"[{label}] consolidated {words_before} -> {len(new_doc.split())} words"
            f"{'' if c_result.ok else ' [FAILED(parse), unchanged]'}"
        )
        return new_doc

    consolidated_last = False
    for epoch in range(resume_epoch, eff_start_epoch + eff_epochs):
        rng = random.Random(eff_seed + epoch)
        shuffled = records[:]
        rng.shuffle(shuffled)
        batches = _batches(shuffled, eff_batch_size)
        start_b = resume_batch if epoch == resume_epoch else 0
        logger.info(
            f"=== Epoch {epoch + 1} (index {epoch}) — {len(batches)} batches"
            f"{f', resuming at {start_b + 1}' if start_b else ''} ==="
        )

        for b in range(start_b, len(batches)):
            batch = batches[b]
            result = synth.fold(doc, batch)
            doc = result.strategy_md
            state.total_folds += 1
            if not result.ok:
                state.failed_folds += 1

            append_fold_log(
                out_dir,
                {
                    "epoch": epoch,
                    "batch": b,
                    "kind": "fold",
                    "n": len(batch),
                    "ok": result.ok,
                    "doc_words": len(doc.split()),
                    "changelog": result.changelog,
                },
            )
            status = "ok" if result.ok else "FAILED(parse)"
            logger.info(
                f"[epoch {epoch + 1} batch {b + 1}/{len(batches)}] {status} — " f"{len(doc.split())} words"
            )
            consolidated_last = False

            # Periodic consolidation: a scheduled quality cleanup, not budget-driven.
            if args.consolidate_every > 0 and state.total_folds % args.consolidate_every == 0:
                doc = consolidate(doc, epoch, b, f"epoch {epoch + 1} batch {b + 1}/{len(batches)}")
                consolidated_last = True

            # Checkpoint after every batch: write the doc and advance the resume
            # position to the next work item, so a crash loses at most one fold.
            if b + 1 < len(batches):
                state.resume_epoch_index, state.resume_batch_index = epoch, b + 1
            else:
                state.resume_epoch_index, state.resume_batch_index = epoch + 1, 0
            write_document(out_dir, doc)
            save_state(out_dir, state)

        snapshot = write_document(out_dir, doc, epoch_index=epoch)
        state.epochs_completed += 1
        state.last_epoch_index = epoch
        save_state(out_dir, state)
        logger.info(f"Epoch {epoch + 1} snapshot: {snapshot}")

    # Final consolidation so the tail folds get cleaned (skip if one just ran).
    if args.consolidate_every > 0 and not consolidated_last:
        doc = consolidate(doc, state.last_epoch_index or 0, -1, "final")

    write_document(out_dir, doc)
    save_state(out_dir, state)
    logger.info(
        f"Done. {state.total_folds} folds ({state.failed_folds} parse failures), "
        f"{state.total_consolidations} consolidation passes. "
        f"Final document: {out_dir / 'strategy.md'} ({len(doc.split())} words)."
    )


if __name__ == "__main__":
    main()
