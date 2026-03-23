"""VeriTaS-specific evaluation metrics.

Uses the generic primitives from ``mafc.eval.metrics`` and adds VeriTaS
domain logic: numeric label mappings, 7→3 coarsening, and integrity-score
based regression metrics.
"""

from __future__ import annotations

from typing import Any

from mafc.eval.metrics import classification_block, regression_metrics, format_confusion_matrix

# ---------------------------------------------------------------------------
# Label ordering (matches Veritas3Label / Veritas7Label .value strings)
# ---------------------------------------------------------------------------

LABELS_3 = ["intact", "unknown", "compromised"]

LABELS_7 = [
    "intact (certain)",
    "intact (rather certain)",
    "intact (rather uncertain)",
    "unknown",
    "compromised (rather uncertain)",
    "compromised (rather certain)",
    "compromised (certain)",
]

# ---------------------------------------------------------------------------
# Numeric mappings & coarsening
# ---------------------------------------------------------------------------

VERDICT_TO_NUMERIC_3: dict[str, float] = {
    "intact": 1.0,
    "unknown": 0.0,
    "compromised": -1.0,
}

VERDICT_TO_NUMERIC_7: dict[str, float] = {
    "intact (certain)": 1.0,
    "intact (rather certain)": 2 / 3,
    "intact (rather uncertain)": 1 / 3,
    "unknown": 0.0,
    "compromised (rather uncertain)": -1 / 3,
    "compromised (rather certain)": -2 / 3,
    "compromised (certain)": -1.0,
}

COARSEN_7_TO_3: dict[str, str] = {
    "intact (certain)": "intact",
    "intact (rather certain)": "intact",
    "intact (rather uncertain)": "unknown",
    "unknown": "unknown",
    "compromised (rather uncertain)": "unknown",
    "compromised (rather certain)": "compromised",
    "compromised (certain)": "compromised",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _regression_from_results(
    results: list[dict[str, Any]],
    verdict_to_numeric: dict[str, float],
    pred_field: str = "predicted",
) -> dict[str, Any]:
    """Extract paired (gt_integrity_score, pred_numeric) from result dicts and compute MSE/MAE."""
    gt_scores: list[float] = []
    pred_scores: list[float] = []
    for r in results:
        pred_label = r.get(pred_field)
        if pred_label is None:
            continue
        gt_score = r.get("gt_integrity_score")
        pred_numeric = verdict_to_numeric.get(pred_label)
        if gt_score is None or pred_numeric is None:
            continue
        try:
            gt_scores.append(float(gt_score))
            pred_scores.append(pred_numeric)
        except (TypeError, ValueError):
            pass
    return regression_metrics(gt_scores, pred_scores)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def compute_veritas_metrics(results: list[dict[str, Any]], label_scheme: int = 3) -> dict[str, Any]:
    """Compute all VeriTaS metrics from a list of per-claim result dicts.

    Each result dict must contain:
        ground_truth    (str)          – lowercase label value
        predicted       (str | None)   – lowercase label value, or None
        gt_integrity_score (float | None) – raw GT integrity score for MSE/MAE

    Returns a JSON-serialisable metrics dict.
    """
    labels = LABELS_7 if label_scheme == 7 else LABELS_3
    verdict_to_numeric = VERDICT_TO_NUMERIC_7 if label_scheme == 7 else VERDICT_TO_NUMERIC_3

    scored = [(r["ground_truth"], r["predicted"]) for r in results if r.get("predicted") is not None]
    if not scored:
        return {}

    y_true = [t for t, _ in scored]
    y_pred = [p for _, p in scored]

    metrics = classification_block(y_true, y_pred, labels)
    reg = _regression_from_results(results, verdict_to_numeric)
    if reg:
        metrics.update(reg)

    # For 7-class runs, also compute coarsened 3-bin metrics
    if label_scheme == 7:
        y_true_c = [COARSEN_7_TO_3.get(t, t) for t in y_true]
        y_pred_c = [COARSEN_7_TO_3.get(p, p) for p in y_pred]
        coarsened_results = [
            (
                {**r, "_pred_c": COARSEN_7_TO_3.get(r["predicted"], r["predicted"])}
                if r.get("predicted") is not None
                else r
            )
            for r in results
        ]
        coarsened = classification_block(y_true_c, y_pred_c, LABELS_3)
        reg_c = _regression_from_results(coarsened_results, VERDICT_TO_NUMERIC_3, pred_field="_pred_c")
        if reg_c:
            coarsened.update(reg_c)
        metrics["coarsened_3class"] = coarsened

    return metrics


def format_veritas_metrics_report(metrics: dict[str, Any], label_scheme: int = 3) -> str:
    """Render a human-readable VeriTaS metrics report."""
    if not metrics:
        return "No metrics available."
    lines: list[str] = ["=== Classification Metrics ==="]
    lines.append(f"  Accuracy:    {metrics['accuracy']:.2%}")
    lines.append(f"  Macro F1:    {metrics['macro']['f1']:.4f}")
    lines.append(f"  Weighted F1: {metrics['weighted']['f1']:.4f}")
    if "mse" in metrics:
        lines += [
            "",
            "=== Regression Metrics ===",
            f"  MSE: {metrics['mse']:.4f}   MAE: {metrics['mae']:.4f}   (n={metrics['n']})",
        ]
    lines += ["", "=== Per-class ==="]
    for label, m in metrics["per_class"].items():
        lines.append(
            f"  {label:<35s}  P={m['precision']:.3f}  R={m['recall']:.3f}"
            f"  F1={m['f1']:.3f}  n={m['support']}"
        )
    lines += ["", format_confusion_matrix(metrics["confusion_matrix"])]
    coarsened = metrics.get("coarsened_3class")
    if coarsened:
        lines += [
            "",
            f"--- Coarsened 3-class (from {label_scheme}-class) ---",
            f"  Accuracy: {coarsened['accuracy']:.2%}   Macro F1: {coarsened['macro']['f1']:.4f}",
        ]
        if "mse" in coarsened:
            lines.append(
                f"  MSE: {coarsened['mse']:.4f}   MAE: {coarsened['mae']:.4f}   (n={coarsened['n']})"
            )
        lines += [
            "",
            format_confusion_matrix(
                coarsened["confusion_matrix"], title="Confusion Matrix (coarsened 3-class)"
            ),
        ]
    return "\n".join(lines)
