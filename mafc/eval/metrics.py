"""Generic, benchmark-agnostic metric primitives.

All functions operate on plain Python lists/dicts and know nothing about
any specific benchmark.  Benchmark-specific logic (label ordering, numeric
mappings, coarsening) lives in the benchmark's own metrics module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------


def confusion_matrix(y_true: list[str], y_pred: list[str], labels: list[str]) -> list[list[int]]:
    """Return an N×N confusion matrix (rows=actual, cols=predicted)."""
    return [[sum(1 for t, p in zip(y_true, y_pred) if t == tl and p == pl) for pl in labels] for tl in labels]


def format_confusion_matrix(cm_dict: dict[str, Any], title: str = "Confusion Matrix") -> str:
    """Render a confusion matrix dict as a plain-text ASCII table."""
    labels = cm_dict["labels"]
    matrix = cm_dict["matrix"]
    col_w = max(max(len(label) for label in labels), 5) + 2
    row_w = max(len(label) for label in labels)
    lines = [f"{title} (rows=actual, cols=predicted)"]
    header = " " * (row_w + 3) + "".join(f"{label:>{col_w}}" for label in labels)
    lines.append(header)
    lines.append("-" * len(header))
    for i, row_label in enumerate(labels):
        row = "".join(f"{v:>{col_w}d}" for v in matrix[i])
        lines.append(f"  {row_label:<{row_w}} {row}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------


def per_class_metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict[str, Any]:
    """Compute per-class precision, recall, F1, and support."""
    out: dict[str, Any] = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        out[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(1 for t in y_true if t == label),
        }
    return out


def macro_average(per_class: dict[str, Any], labels: list[str]) -> dict[str, float]:
    n = len(labels)
    if n == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {
        "precision": round(sum(per_class[label]["precision"] for label in labels) / n, 4),
        "recall": round(sum(per_class[label]["recall"] for label in labels) / n, 4),
        "f1": round(sum(per_class[label]["f1"] for label in labels) / n, 4),
    }


def weighted_average(per_class: dict[str, Any], labels: list[str], total: int) -> dict[str, float]:
    if total == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {
        "precision": round(
            sum(per_class[label]["precision"] * per_class[label]["support"] for label in labels) / total, 4
        ),
        "recall": round(
            sum(per_class[label]["recall"] * per_class[label]["support"] for label in labels) / total, 4
        ),
        "f1": round(sum(per_class[label]["f1"] * per_class[label]["support"] for label in labels) / total, 4),
    }


def classification_block(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict[str, Any]:
    """Build a self-contained classification metrics block."""
    cm = confusion_matrix(y_true, y_pred, labels)
    pc = per_class_metrics(y_true, y_pred, labels)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return {
        "accuracy": round(correct / len(y_true), 4) if y_true else None,
        "macro": macro_average(pc, labels),
        "weighted": weighted_average(pc, labels, len(y_true)),
        "per_class": pc,
        "confusion_matrix": {"labels": labels, "matrix": cm},
    }


# ---------------------------------------------------------------------------
# Regression metrics (MSE / MAE)
# ---------------------------------------------------------------------------


def regression_metrics(gt_scores: list[float], pred_scores: list[float]) -> dict[str, Any]:
    """Compute MSE and MAE between paired numeric scores."""
    if not gt_scores:
        return {}
    n = len(gt_scores)
    mse = sum((g - p) ** 2 for g, p in zip(gt_scores, pred_scores)) / n
    mae = sum(abs(g - p) for g, p in zip(gt_scores, pred_scores)) / n
    return {"mse": round(mse, 4), "mae": round(mae, 4), "n": n}


# ---------------------------------------------------------------------------
# Confusion matrix plot
# ---------------------------------------------------------------------------


def save_confusion_matrix_png(
    cm_dict: dict[str, Any],
    path: Path,
    title: str = "Confusion Matrix",
    subtitle: str = "",
) -> None:
    """Save a confusion matrix as a PNG heatmap image."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = cm_dict["labels"]
    matrix = np.array(cm_dict["matrix"], dtype=int)
    n = len(labels)

    # Normalise per row (true class) for colour intensity; show raw counts as text
    row_sums = matrix.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = np.where(row_sums > 0, matrix / row_sums, 0.0)

    fig_w = max(6, n * 1.2)
    fig_h = max(5, n * 1.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-normalised proportion")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    tick_labels = [label.replace(" (", "\n(") for label in labels]
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(tick_labels, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual", fontsize=10)

    full_title = f"{title}\n{subtitle}" if subtitle else title
    ax.set_title(full_title, fontsize=11, pad=12)

    # Annotate cells with raw count (top) and row-normalised proportion (bottom, smaller)
    thresh = 0.5
    for i in range(n):
        for j in range(n):
            count = matrix[i, j]
            prop = norm[i, j]
            color = "white" if prop > thresh else "black"
            ax.text(j, i - 0.15, str(count), ha="center", va="center", fontsize=9, color=color)
            ax.text(j, i + 0.22, f"({prop:.2f})", ha="center", va="center", fontsize=7, color=color)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
