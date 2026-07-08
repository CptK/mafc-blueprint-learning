"""Assemble a feature table (pandas DataFrame) from traces + claims.

This is the glue between ``trace_io`` / ``features`` and the training harness. It
joins each trace to its claim by id, builds structured features, optionally appends
embedding features (justification embedding + evidence dispersion / evidence-vs-claim
cosine), and writes the table to parquet (when an engine is installed) and CSV.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from mafc.common.logger import logger
from mafc.training.claims_io import ClaimRecord
from mafc.training.features import (
    FeatureExtractorConfig,
    FeatureRow,
    dispersion_stats,
    evidence_vs_claim_cosine,
    extract_row,
)
from mafc.training.trace_io import discover_traces, load_normalised


def build_feature_rows(
    trace_dirs: list[Path],
    claims_by_id: dict[str, ClaimRecord],
    cfg: FeatureExtractorConfig,
) -> list[FeatureRow]:
    """Extract structured feature rows for every trace that joins to a claim."""
    rows: list[FeatureRow] = []
    seen: set[str] = set()
    for trace_dir in trace_dirs:
        for cid, path in discover_traces(trace_dir).items():
            if cid in seen:
                continue
            claim = claims_by_id.get(cid)
            if claim is None:
                continue
            norm = load_normalised(path)
            if norm is None:
                continue
            rows.append(extract_row(norm, claim, cfg))
            seen.add(cid)
    return rows


def _attach_embeddings(rows: list[FeatureRow], cfg: FeatureExtractorConfig) -> None:
    """Embed justifications, claims and evidence, then add embedding features in place.

    One OpenAI embedding pass over the concatenation of all texts; vectors are then
    sliced back per row. Requires the OpenAI client (network). No-op if a row has no
    text. Mutates ``row.features``.
    """
    from openai import OpenAI

    from mafc.training.embedding_features import just_embedding_features

    texts: list[str] = []
    spans: list[tuple[int, int, int, int]] = []  # claim_idx, just_idx, ev_start, ev_end
    for row in rows:
        claim_idx = len(texts)
        texts.append(row.claim_text or "")
        just_idx = len(texts)
        texts.append(row.justification_text or "")
        ev_start = len(texts)
        texts.extend(row.evidence_texts[: cfg.max_evidence_for_dispersion])
        ev_end = len(texts)
        spans.append((claim_idx, just_idx, ev_start, ev_end))

    from mafc.learning.embedding_utils import embed_all

    # The OpenAI embeddings endpoint rejects empty strings; replace blanks with a
    # placeholder so batch indices stay aligned with ``spans``.
    texts = [t if (t and t.strip()) else "[empty]" for t in texts]

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("openai_api_key")
    if not api_key:
        raise SystemExit(
            "Missing OpenAI API key. Set OPENAI_API_KEY or openai_api_key in the environment or config/.env."
        )
    client = OpenAI(api_key=api_key, timeout=120)
    vectors, _ = embed_all(texts, client, model=cfg.embedding_model, label="features")

    for row, (ci, ji, es, ee) in zip(rows, spans):
        claim_vec = vectors[ci]
        just_vec = vectors[ji]
        ev_vecs = vectors[es:ee]
        row.features.update(dispersion_stats(ev_vecs))
        row.features.update(evidence_vs_claim_cosine(ev_vecs, claim_vec))
        row.features.update(just_embedding_features(just_vec))


def rows_to_dataframe(rows: list[FeatureRow]) -> pd.DataFrame:
    records = []
    for r in rows:
        rec: dict = {"claim_id": r.claim_id, "target": r.target}
        rec.update(r.features)
        records.append(rec)
    return pd.DataFrame.from_records(records)


# Meta columns are NOT model features: they carry evaluation-only ground truth
# (signed score) and the judge's verbalized baseline label. Prefixed so the
# trainer's column selection (which excludes only id/target) never picks them up.
META_PREFIX = "meta__"


def build_meta_table(trace_dirs: list[Path], claims_by_id: dict[str, ClaimRecord]) -> pd.DataFrame:
    """Per-claim evaluation metadata: signed score + judge verbalized 7-class label.

    Kept separate from the feature table to guarantee no ground-truth leakage into
    training. Joined back by ``claim_id`` only at evaluation time.
    """
    from mafc.training.trace_io import discover_traces, load_normalised

    records = []
    seen: set[str] = set()
    for trace_dir in trace_dirs:
        for cid, path in discover_traces(trace_dir).items():
            if cid in seen:
                continue
            claim = claims_by_id.get(cid)
            if claim is None:
                continue
            norm = load_normalised(path)
            if norm is None:
                continue
            records.append(
                {
                    "claim_id": cid,
                    f"{META_PREFIX}gt_score": claim.integrity_score,
                    f"{META_PREFIX}judge_label": norm.judge_label,
                    f"{META_PREFIX}date": claim.date,
                }
            )
            seen.add(cid)
    return pd.DataFrame.from_records(records)


def build_dataframe(
    trace_dirs: list[Path],
    claims_by_id: dict[str, ClaimRecord],
    cfg: FeatureExtractorConfig,
) -> pd.DataFrame:
    rows = build_feature_rows(trace_dirs, claims_by_id, cfg)
    if not rows:
        logger.warning("No trace/claim joins produced — feature table is empty.")
        return pd.DataFrame()
    if cfg.include_embeddings:
        _attach_embeddings(rows, cfg)
    df = rows_to_dataframe(rows)
    logger.info(f"Built feature table: {len(df)} rows x {df.shape[1]} columns.")
    return df


def save_table(df: pd.DataFrame, out_stem: Path) -> list[Path]:
    """Write CSV always; parquet when an engine is available. Returns written paths."""
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    csv_path = out_stem.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    written.append(csv_path)
    parquet_path = out_stem.with_suffix(".parquet")
    try:
        df.to_parquet(parquet_path, index=False)
        written.append(parquet_path)
    except (ImportError, ValueError) as exc:
        logger.info(f"Skipping parquet (no engine installed): {exc}")
    return written
