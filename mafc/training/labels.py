"""Shared score <-> direction <-> 7-class helpers for the training subsystem.

Everything here is built on top of the canonical mappings in
``mafc.eval.veritas.labels`` / ``mafc.eval.veritas.metrics`` so the regressor's
view of the label space stays consistent with the evaluator. The continuous
prediction is binned to a 7-class label by the same ``label_from_signed_score``
that defines the ground-truth labels — no training-local cut-points.
"""

from __future__ import annotations

import math

from mafc.eval.veritas.labels import THRESHOLDS_7, Veritas7Label
from mafc.eval.veritas.metrics import COARSEN_7_TO_3, VERDICT_TO_NUMERIC_7

# Direction sign of a continuous integrity score. The 7->3 coarsening implies an
# "unknown" band of |score| < 1/3 (intact/compromised require |score| >= 1/3).
UNKNOWN_BAND = 1 / 3

# Boundary-weighting region for the sampler: the certain/rather-certain confusion
# lives where |score| in [0.5, 1.0]. Oversample there.
HARD_BAND: tuple[float, float] = (0.5, 1.0)


def direction_of_score(score: float, unknown_band: float = UNKNOWN_BAND) -> str:
    """Map a continuous integrity score to {intact, unknown, compromised}."""
    if score >= unknown_band:
        return "intact"
    if score <= -unknown_band:
        return "compromised"
    return "unknown"


def sign_of_direction(direction: str) -> int:
    """+1 for intact, -1 for compromised, 0 for unknown/unrecognised."""
    if direction == "intact":
        return 1
    if direction == "compromised":
        return -1
    return 0


def direction_of_label(label: str) -> str:
    """Coarsen any 7-class (or 3-class) label string to its direction."""
    return COARSEN_7_TO_3.get(label, label)


def label_from_signed_score(signed_score: float) -> str:
    """Bin a signed score in [-1, 1] to a 7-class label using THRESHOLDS_7."""
    for upper, label in THRESHOLDS_7:
        if signed_score < upper:
            return label.value
    return Veritas7Label.INTACT_CERTAIN.value


def numeric_7(label: str) -> float:
    """Signed numeric value of a 7-class label (delegates to the evaluator map)."""
    return VERDICT_TO_NUMERIC_7.get(label, 0.0 if label == "unknown" else math.nan)
