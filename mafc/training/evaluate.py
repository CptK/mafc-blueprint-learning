"""Evaluation + calibration reporting for the magnitude regressor.

Reuses ``mafc.eval.veritas.metrics`` for the 7-class + coarsened 3-class +
regression block, then adds:

- ECE (expected calibration error) over the magnitude prediction, with a binned
  reliability-diagram data dump.
- A focused ``certain <-> rather-certain`` confusion report, per side, comparing the
  regressor-reconstructed label against the judge's own verbalized 7-class label
  (the baseline already present in each trace).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mafc.eval.veritas.metrics import compute_veritas_metrics
from mafc.training.labels import numeric_7


@dataclass
class ReliabilityBin:
    lo: float
    hi: float
    count: int
    mean_pred: float
    mean_true: float


def expected_calibration_error(
    pred_magnitude: np.ndarray, true_magnitude: np.ndarray, n_bins: int = 10
) -> tuple[float, list[ReliabilityBin]]:
    """ECE for a magnitude regressor: |mean_pred - mean_true| weighted per bin.

    Bins are over the predicted magnitude in [0, 1]. Returns (ece, bins) where each
    bin carries the reliability-diagram coordinates.
    """
    pred = np.asarray(pred_magnitude, dtype=float)
    true = np.asarray(true_magnitude, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[ReliabilityBin] = []
    n = len(pred)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (pred >= lo) & (pred < hi) if i < n_bins - 1 else (pred >= lo) & (pred <= hi)
        count = int(mask.sum())
        if count == 0:
            bins.append(ReliabilityBin(float(lo), float(hi), 0, float("nan"), float("nan")))
            continue
        mp = float(pred[mask].mean())
        mt = float(true[mask].mean())
        ece += (count / n) * abs(mp - mt)
        bins.append(ReliabilityBin(float(lo), float(hi), count, mp, mt))
    return float(ece), bins


def reliability_dump(bins: list[ReliabilityBin]) -> list[dict]:
    return [
        {
            "lo": b.lo,
            "hi": b.hi,
            "count": b.count,
            "mean_pred": b.mean_pred,
            "mean_true": b.mean_true,
        }
        for b in bins
    ]


# ---------------------------------------------------------------------------
# certain <-> rather-certain focus
# ---------------------------------------------------------------------------

_CERTAIN = {"intact (certain)", "compromised (certain)"}
_RATHER_CERTAIN = {"intact (rather certain)", "compromised (rather certain)"}


def certain_confusion(y_true: list[str], y_pred: list[str]) -> dict:
    """2x2 certain<->rather-certain confusion, per side and overall.

    Only rows whose *true* label is certain or rather-certain are counted, since
    that is the band the regressor is meant to fix.
    """

    def _side_block(side: str | None) -> dict:
        c_to_c = c_to_rc = rc_to_c = rc_to_rc = 0
        for t, p in zip(y_true, y_pred):
            if side is not None and not t.startswith(side):
                continue
            if t in _CERTAIN:
                if p in _CERTAIN:
                    c_to_c += 1
                elif p in _RATHER_CERTAIN:
                    c_to_rc += 1
            elif t in _RATHER_CERTAIN:
                if p in _CERTAIN:
                    rc_to_c += 1
                elif p in _RATHER_CERTAIN:
                    rc_to_rc += 1
        n = c_to_c + c_to_rc + rc_to_c + rc_to_rc
        confused = c_to_rc + rc_to_c
        return {
            "certain->certain": c_to_c,
            "certain->rather_certain": c_to_rc,
            "rather_certain->certain": rc_to_c,
            "rather_certain->rather_certain": rc_to_rc,
            "n": n,
            "confusion_rate": (confused / n) if n else None,
        }

    return {
        "overall": _side_block(None),
        "intact": _side_block("intact"),
        "compromised": _side_block("compromised"),
    }


# ---------------------------------------------------------------------------
# Top-level evaluation
# ---------------------------------------------------------------------------


def _results_payload(
    y_true: list[str], y_pred: list[str], gt_scores: list[float]
) -> list[dict]:
    return [
        {"ground_truth": t, "predicted": p, "gt_integrity_score": s}
        for t, p, s in zip(y_true, y_pred, gt_scores)
    ]


def evaluate(
    y_true_7: list[str],
    y_pred_7: list[str],
    gt_scores: list[float],
    pred_magnitude: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Full metric block for one predictor (regressor or baseline).

    ``pred_magnitude`` is the magnitude implied by ``y_pred_7`` (``|numeric_7|``)
    for the baseline, or the regressor's raw output. ECE uses |gt_score| as the
    target magnitude.
    """
    payload = _results_payload(y_true_7, y_pred_7, gt_scores)
    metrics = compute_veritas_metrics(payload, label_scheme=7)
    true_mag = np.abs(np.asarray(gt_scores, dtype=float))
    ece, bins = expected_calibration_error(pred_magnitude, true_mag, n_bins)
    metrics["ece"] = ece
    metrics["reliability_diagram"] = reliability_dump(bins)
    metrics["certain_confusion"] = certain_confusion(y_true_7, y_pred_7)
    return metrics


def baseline_magnitudes(judge_labels: list[str]) -> np.ndarray:
    """Magnitude the judge implicitly assigned via its verbalized 7-class label."""
    return np.array([abs(numeric_7(lbl)) for lbl in judge_labels], dtype=float)


def _sign(direction: str) -> float:
    return 1.0 if direction == "intact" else -1.0 if direction == "compromised" else 0.0


def continuous_signed_metrics(
    judge_directions: list[str], magnitudes: np.ndarray, gt_scores: list[float]
) -> dict:
    """MSE/MAE of the **continuous** deliverable: ``sign(judge) × magnitude`` in [-1, 1].

    This is the VeriTaS-scored quantity — a continuous prediction against the
    continuous GT integrity score, with no discretization to the 7 fixed values.
    """
    sign = np.array([_sign(d) for d in judge_directions], dtype=float)
    pred = sign * np.asarray(magnitudes, dtype=float)
    gt = np.asarray(gt_scores, dtype=float)
    err = gt - pred
    return {
        "mse": float(np.mean(err**2)),
        "mae": float(np.mean(np.abs(err))),
        "n": int(len(gt)),
    }


def baseline_continuous_metrics(judge_labels: list[str], gt_scores: list[float]) -> dict:
    """MSE/MAE if the system emitted the judge's discretized 7-value as its number."""
    pred = np.array([numeric_7(lbl) for lbl in judge_labels], dtype=float)
    gt = np.asarray(gt_scores, dtype=float)
    err = gt - pred
    return {"mse": float(np.mean(err**2)), "mae": float(np.mean(np.abs(err))), "n": int(len(gt))}


def compare_reports(baseline: dict, model: dict) -> str:
    """Human-readable before/after focused on calibration + certain confusion."""
    lines = ["=== Calibration: baseline (judge verbalized) vs regressor ==="]
    lines.append(
        f"  ECE       baseline={baseline.get('ece'):.4f}   regressor={model.get('ece'):.4f}"
    )
    lines.append(
        f"  Accuracy  baseline={baseline.get('accuracy'):.2%}   "
        f"regressor={model.get('accuracy'):.2%}"
    )
    if "mse" in baseline and "mse" in model:
        lines.append(
            f"  MSE       baseline={baseline['mse']:.4f}   regressor={model['mse']:.4f}"
        )
    lines.append("")
    lines.append("=== certain <-> rather-certain confusion rate (lower is better) ===")
    for side in ("overall", "intact", "compromised"):
        b = baseline["certain_confusion"][side]
        m = model["certain_confusion"][side]
        br = b["confusion_rate"]
        mr = m["confusion_rate"]
        bs = f"{br:.2%}" if br is not None else " n/a "
        ms = f"{mr:.2%}" if mr is not None else " n/a "
        lines.append(f"  {side:<12s} baseline={bs} (n={b['n']})   regressor={ms} (n={m['n']})")
    return "\n".join(lines)
