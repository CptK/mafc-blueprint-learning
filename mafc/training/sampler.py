"""Stratified, boundary-weighted training-set sampler.

Selects a claim subset for collecting calibration training data. The selection is:

1. Stratified by integrity *direction* (intact / unknown / compromised), so each
   side of the decision is represented.
2. Boundary-weighted within each stratum: claims in the ``certain / rather-certain``
   confusion region (``|score| in [0.5, 1.0]``) are oversampled; easy far / trivial
   cases are undersampled. The weights are configurable.

The output is directly consumable as ``benchmark.sample_ids`` in an experiment YAML,
plus a manifest (id, score, direction, stratum, weight) for analysis. Deterministic
given a seed.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass

from mafc.training.claims_io import ClaimRecord
from mafc.training.labels import HARD_BAND, direction_of_score


@dataclass(frozen=True)
class SampledClaim:
    id: str
    score: float
    direction: str
    stratum: str  # "<direction>/<hard|easy>"
    weight: float


@dataclass
class SamplerConfig:
    """Tunable sampling knobs."""

    target_n: int | None = None  # None -> keep all (weighted ordering only)
    hard_band: tuple[float, float] = HARD_BAND
    hard_weight: float = 3.0  # oversampling factor for the boundary region
    easy_weight: float = 1.0  # weight for the rest (undersampled relative to hard)
    unknown_weight: float = 1.0  # extra factor applied to the whole unknown stratum
    seed: int = 0
    balance_directions: bool = True  # equalise the 3 direction strata when sampling


def _stratum(rec: ClaimRecord, cfg: SamplerConfig) -> tuple[str, str, float]:
    """Return (direction, stratum_key, base_weight) for a claim."""
    direction = direction_of_score(rec.integrity_score)
    lo, hi = cfg.hard_band
    is_hard = lo <= rec.magnitude <= hi
    band = "hard" if is_hard else "easy"
    weight = cfg.hard_weight if is_hard else cfg.easy_weight
    if direction == "unknown":
        weight *= cfg.unknown_weight
    return direction, f"{direction}/{band}", weight


def _weighted_sample_without_replacement(
    items: list[SampledClaim], k: int, rng: random.Random
) -> list[SampledClaim]:
    """Weighted sampling without replacement (Efraimidis-Spirakis A-Res).

    Each item gets key ``u ** (1/weight)`` with ``u`` uniform; the top-k keys are
    the sample. Higher-weight items are more likely to surface. Deterministic for
    a fixed ``rng``.
    """
    if k >= len(items):
        return list(items)
    keyed = [(rng.random() ** (1.0 / max(it.weight, 1e-9)), it) for it in items]
    keyed.sort(key=lambda t: t[0], reverse=True)
    return [it for _, it in keyed[:k]]


def build_pool(records: list[ClaimRecord], cfg: SamplerConfig) -> list[SampledClaim]:
    """Annotate every claim with its stratum and weight (no selection yet)."""
    pool: list[SampledClaim] = []
    for rec in records:
        direction, stratum, weight = _stratum(rec, cfg)
        pool.append(
            SampledClaim(
                id=rec.id,
                score=rec.integrity_score,
                direction=direction,
                stratum=stratum,
                weight=weight,
            )
        )
    return pool


def sample(records: list[ClaimRecord], cfg: SamplerConfig) -> list[SampledClaim]:
    """Select claims per the config. Deterministic for a fixed seed.

    If ``target_n`` is None, returns the full pool sorted by stratum/score.
    Otherwise performs weighted sampling-without-replacement. When
    ``balance_directions`` is set, the per-direction budget is equalised first,
    then boundary weights drive the within-direction pick.
    """
    rng = random.Random(cfg.seed)
    pool = build_pool(records, cfg)

    if cfg.target_n is None or cfg.target_n >= len(pool):
        return sorted(pool, key=lambda s: (s.direction, -abs(s.score), s.id))

    if not cfg.balance_directions:
        selected = _weighted_sample_without_replacement(pool, cfg.target_n, rng)
        return sorted(selected, key=lambda s: (s.direction, -abs(s.score), s.id))

    by_dir: dict[str, list[SampledClaim]] = {}
    for s in pool:
        by_dir.setdefault(s.direction, []).append(s)

    present = [d for d in by_dir if by_dir[d]]
    base = cfg.target_n // len(present)
    remainder = cfg.target_n - base * len(present)
    # Distribute the remainder to the largest strata for determinism.
    order = sorted(present, key=lambda d: (-len(by_dir[d]), d))

    selected: list[SampledClaim] = []
    leftover_budget = 0
    for i, d in enumerate(order):
        budget = base + (1 if i < remainder else 0)
        avail = by_dir[d]
        take = min(budget, len(avail))
        leftover_budget += budget - take
        selected.extend(_weighted_sample_without_replacement(avail, take, rng))

    # Re-allocate any unfilled budget (small strata) to the remaining claims.
    if leftover_budget > 0:
        chosen_ids = {s.id for s in selected}
        rest = [s for s in pool if s.id not in chosen_ids]
        selected.extend(_weighted_sample_without_replacement(rest, leftover_budget, rng))

    return sorted(selected, key=lambda s: (s.direction, -abs(s.score), s.id))


def stratum_counts(samples: list[SampledClaim]) -> dict[str, int]:
    """Count selected claims per stratum (for logging / tests)."""
    return dict(Counter(s.stratum for s in samples))
