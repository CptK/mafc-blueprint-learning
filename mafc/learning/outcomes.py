"""Outcome-bucketing helper for outcome-aware learning.

Partitioning a ``ClaimLearningRecord`` by its ``execution_result`` answers
"was the blueprint actually right on this claim?" — the signal the updater
and synthesizer consult to bias their revisions toward fixing
failures without regressing successes.

The three buckets are stable across modes:

- ``correct``  — execution ran, produced a verdict, and the verdict either
                 matches the ground truth (label-equality mode) OR has a
                 score error within ``error_threshold`` (score-error mode).
- ``incorrect``— execution ran, produced a verdict, but the verdict failed
                 the corresponding match criterion.
- ``unknown``  — execution did not run, errored before producing a verdict,
                 or (in score-error mode) lacks numeric ground-truth or
                 predicted scores. Treated as information-free.

Mode is selected by the ``error_threshold`` argument. ``None`` keeps the
original strict-equality behaviour. A float switches to score-error mode,
which respects the ordinal structure of labels like VeriTaS integrity
(off-by-one-bin is a near miss, off-by-three-bins is a miss).
"""

from __future__ import annotations

from typing import Literal

from mafc.learning.models import ClaimLearningRecord

OutcomeBucket = Literal["correct", "incorrect", "unknown"]


def outcome_bucket(rec: ClaimLearningRecord, error_threshold: float | None = None) -> OutcomeBucket:
    """Classify a record's execution outcome.

    Args:
        rec: The learning record.
        error_threshold: ``None`` → strict label equality. Float → score-error
            mode: ``correct`` iff ``abs(predicted_score - gt_score) <= threshold``.
            In score-error mode, missing numeric scores yield ``unknown``.
    """
    er = rec.execution_result
    if er is None or er.predicted_label is None:
        return "unknown"
    if error_threshold is None:
        return "correct" if er.correct else "incorrect"
    if er.predicted_score is None or er.gt_score is None:
        return "unknown"
    return "correct" if abs(er.predicted_score - er.gt_score) <= error_threshold else "incorrect"


def partition_by_outcome(
    records: list[ClaimLearningRecord],
    error_threshold: float | None = None,
) -> tuple[list[ClaimLearningRecord], list[ClaimLearningRecord], list[ClaimLearningRecord]]:
    """Return ``(correct, incorrect, unknown)`` partitions of ``records``."""
    correct: list[ClaimLearningRecord] = []
    incorrect: list[ClaimLearningRecord] = []
    unknown: list[ClaimLearningRecord] = []
    for rec in records:
        bucket = outcome_bucket(rec, error_threshold=error_threshold)
        if bucket == "correct":
            correct.append(rec)
        elif bucket == "incorrect":
            incorrect.append(rec)
        else:
            unknown.append(rec)
    return correct, incorrect, unknown


def category_from_outcomes(
    n_correct: int, n_incorrect: int, n_unknown: int
) -> Literal["fixes-failures", "specializes-easy-cases", "mixed", "unspecified"]:
    """Tag a synthesizer cluster from its outcome distribution.

    ``unspecified`` is reserved for the outcomes-off path. With outcomes on,
    a cluster is tagged ``fixes-failures`` when incorrect outcomes dominate,
    ``specializes-easy-cases`` when correct outcomes dominate, and ``mixed``
    otherwise (including the all-unknown case, since we have no signal).
    """
    n_signal = n_correct + n_incorrect
    if n_signal == 0:
        return "mixed"
    if n_incorrect > n_correct:
        return "fixes-failures"
    if n_correct > n_incorrect:
        return "specializes-easy-cases"
    return "mixed"
