"""Trace -> feature-table extractor for the magnitude regressor.

Joins normalised traces with the ground-truth ``claims.json`` purely to attach the
training ``target = abs(integrity.score)``. No feature is derived from ground truth
(no ``true_label``, no signed score) — only the regression target uses it.

Feature tiers (each degrades gracefully when a trace lacks the underlying field):

- Tier 1  agreement & sufficiency: evidence-embedding dispersion, evidence-vs-claim
  cosine, evidence counts / useful ratio / distinct source domains, search-struggle
  (iterations, max-iter hit, evidence growth, delegated tasks, retrieval-failure
  rate, error count, runtime).
- Tier 2  judge hedging: justification embedding, hedge-lexicon counts (DE+EN),
  justification length, judge output tokens, repair-fired flag, errors-present flag.
- Tier 3  difficulty priors: claim_features (from blueprint selection), modality,
  language, claim length, date recency vs evidence dates, blueprint name.
- Conditioning: the judge's predicted direction (categorical feature).

Embedding features are toggleable so a fast structured-only table can be produced
first (``include_embeddings=False``).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

from mafc.training.claims_io import ClaimRecord
from mafc.training.labels import direction_of_label
from mafc.training.trace_io import NormalisedTrace

# Target column name and the column carrying the categorical judge direction.
TARGET_COL = "target"
ID_COL = "claim_id"
JUDGE_DIRECTION_COL = "judge_direction"


@dataclass
class FeatureExtractorConfig:
    include_embeddings: bool = False
    embedding_model: str = "text-embedding-3-large"
    # Cap evidence considered for pairwise dispersion to bound the O(n^2) cost.
    max_evidence_for_dispersion: int = 20


@dataclass
class FeatureRow:
    claim_id: str
    target: float
    features: dict[str, float | str | None] = field(default_factory=dict)
    # Texts deferred for optional batched embedding.
    justification_text: str | None = None
    evidence_texts: list[str] = field(default_factory=list)
    claim_text: str = ""


_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def _domain(source: str | None) -> str | None:
    if not source:
        return None
    try:
        netloc = urlparse(source).netloc.lower()
    except ValueError:
        return None
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc or None


def _parse_year_month(date: str | None) -> tuple[int, int] | None:
    if not date:
        return None
    m = _DATE_RE.search(date)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _months_between(a: tuple[int, int], b: tuple[int, int]) -> int:
    return (a[0] - b[0]) * 12 + (a[1] - b[1])


def _structured_features(
    trace: NormalisedTrace, claim: ClaimRecord, cfg: FeatureExtractorConfig
) -> dict[str, float | str | None]:
    f: dict[str, float | str | None] = {}

    # Lean, blueprint-independent feature set. Features tied to the blueprint set
    # (blueprint_name, the selector's claim_features, hit_max_iterations) and ones
    # that empirically carried no signal (hedge lexicon, redundant counts/lengths)
    # were removed — see out/training/REPORT.md §Feature pruning.

    # --- Conditioning: judge's predicted direction (sign comes from the judge) ---
    f[JUDGE_DIRECTION_COL] = (
        direction_of_label(trace.judge_label) if trace.judge_label else None
    )

    # --- Evidence sufficiency ---
    ev = trace.evidence
    n_ev = len(ev)
    n_useful = sum(1 for e in ev if e.is_useful)
    domains = {d for e in ev if (d := _domain(e.source))}
    f["evidence_count"] = float(trace.evidence_count or n_ev)
    f["useful_ratio"] = (n_useful / n_ev) if n_ev else 0.0
    f["n_distinct_domains"] = float(len(domains))

    # --- Search struggle (behavioural difficulty proxies) ---
    f["n_iterations"] = float(trace.n_iterations)
    growth = trace.evidence_growth
    f["evidence_growth_total"] = float(growth[-1] - growth[0]) if len(growth) >= 2 else 0.0
    f["runtime_seconds"] = (
        float(trace.runtime_seconds) if trace.runtime_seconds is not None else math.nan
    )

    # --- Judge output shape ---
    f["justification_char_len"] = float(len(trace.judge_justification or ""))
    f["judge_output_tokens"] = (
        float(trace.judge_output_tokens)
        if trace.judge_output_tokens is not None
        else math.nan
    )

    # --- Difficulty priors (claim-intrinsic) ---
    f["has_media"] = float(claim.has_media)
    f["language"] = claim.language
    f["claim_char_len"] = float(len(claim.text))

    # date recency: claim date vs latest evidence date mentioned in takeaways
    claim_ym = _parse_year_month(claim.date)
    ev_yms = []
    for e in ev:
        ym = _parse_year_month(e.takeaways_text or "")
        if ym is not None:
            ev_yms.append(ym)
    if claim_ym is not None and ev_yms:
        latest = max(ev_yms)
        f["claim_vs_evidence_months"] = float(_months_between(claim_ym, latest))
    else:
        f["claim_vs_evidence_months"] = math.nan

    return f


def extract_row(
    trace: NormalisedTrace, claim: ClaimRecord, cfg: FeatureExtractorConfig
) -> FeatureRow:
    """Build one feature row (structured features + deferred texts for embeddings)."""
    feats = _structured_features(trace, claim, cfg)
    evidence_texts = [e.takeaways_text for e in trace.evidence if e.takeaways_text]
    return FeatureRow(
        claim_id=claim.id,
        target=claim.magnitude,
        features=feats,
        justification_text=trace.judge_justification,
        evidence_texts=evidence_texts,
        claim_text=claim.text,
    )


# ---------------------------------------------------------------------------
# Embedding features (optional) — pure numpy given precomputed vectors.
# ---------------------------------------------------------------------------


def _cosine(a, b) -> float:
    import numpy as np

    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def dispersion_stats(vectors) -> dict[str, float]:
    """Mean/var/max of pairwise cosine *distance* over evidence embeddings.

    Returns NaNs when fewer than 2 vectors are available.
    """
    import numpy as np

    vectors = np.asarray(vectors, dtype=np.float32)
    n = len(vectors)
    if n < 2:
        return {
            "emb_disp_mean": math.nan,
            "emb_disp_var": math.nan,
            "emb_disp_max": math.nan,
        }
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = vectors / norms
    sims = unit @ unit.T
    iu = np.triu_indices(n, k=1)
    dists = 1.0 - sims[iu]
    return {
        "emb_disp_mean": float(np.mean(dists)),
        "emb_disp_var": float(np.var(dists)),
        "emb_disp_max": float(np.max(dists)),
    }


def evidence_vs_claim_cosine(evidence_vecs, claim_vec) -> dict[str, float]:
    import numpy as np

    evidence_vecs = np.asarray(evidence_vecs, dtype=np.float32)
    if len(evidence_vecs) == 0 or claim_vec is None:
        return {"emb_ev_claim_mean": math.nan, "emb_ev_claim_max": math.nan}
    sims = [_cosine(v, claim_vec) for v in evidence_vecs]
    return {"emb_ev_claim_mean": float(np.mean(sims)), "emb_ev_claim_max": float(np.max(sims))}
