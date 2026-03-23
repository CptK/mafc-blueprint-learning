import pytest

from mafc.eval.veritas.metrics import (
    VERDICT_TO_NUMERIC_3,
    _regression_from_results,
    compute_veritas_metrics,
    format_veritas_metrics_report,
)


def _r(gt: str, pred: str | None, integrity_score: float | None = None) -> dict:
    return {"ground_truth": gt, "predicted": pred, "gt_integrity_score": integrity_score}


# ---------------------------------------------------------------------------
# compute_veritas_metrics
# ---------------------------------------------------------------------------


def test_compute_veritas_metrics_empty_list():
    assert compute_veritas_metrics([]) == {}


def test_compute_veritas_metrics_no_valid_predictions():
    results = [_r("intact", None), _r("compromised", None)]
    assert compute_veritas_metrics(results) == {}


def test_compute_veritas_metrics_3class_accuracy():
    results = [
        _r("intact", "intact"),
        _r("compromised", "compromised"),
        _r("intact", "compromised"),  # wrong
    ]
    metrics = compute_veritas_metrics(results, label_scheme=3)
    assert metrics["accuracy"] == pytest.approx(2 / 3, abs=0.001)
    assert "per_class" in metrics
    assert "confusion_matrix" in metrics
    assert "coarsened_3class" not in metrics


def test_compute_veritas_metrics_3class_perfect():
    results = [_r("intact", "intact"), _r("unknown", "unknown"), _r("compromised", "compromised")]
    metrics = compute_veritas_metrics(results, label_scheme=3)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro"]["f1"] == 1.0


def test_compute_veritas_metrics_7class_includes_coarsened():
    results = [
        _r("intact (certain)", "intact (certain)"),
        _r("compromised (certain)", "compromised (certain)"),
    ]
    metrics = compute_veritas_metrics(results, label_scheme=7)
    assert "coarsened_3class" in metrics
    coarsened = metrics["coarsened_3class"]
    assert coarsened["accuracy"] == 1.0


def test_compute_veritas_metrics_7class_coarsening_maps_uncertain_to_unknown():
    # "intact (rather uncertain)" and "compromised (rather uncertain)" coarsen to "unknown"
    results = [
        _r("intact (rather uncertain)", "intact (rather uncertain)"),
    ]
    metrics = compute_veritas_metrics(results, label_scheme=7)
    coarsened = metrics["coarsened_3class"]
    assert coarsened["per_class"]["unknown"]["support"] == 1


def test_compute_veritas_metrics_regression_fields_present():
    results = [
        _r("intact", "intact", integrity_score=0.9),
        _r("compromised", "compromised", integrity_score=-0.9),
    ]
    metrics = compute_veritas_metrics(results, label_scheme=3)
    assert "mse" in metrics
    assert "mae" in metrics
    assert metrics["mse"] == pytest.approx(0.01, abs=0.001)


def test_compute_veritas_metrics_regression_skipped_without_scores():
    results = [_r("intact", "intact"), _r("compromised", "compromised")]
    metrics = compute_veritas_metrics(results, label_scheme=3)
    # No gt_integrity_score → no regression fields
    assert "mse" not in metrics


# ---------------------------------------------------------------------------
# _regression_from_results
# ---------------------------------------------------------------------------


def test_regression_from_results_valid_pair():
    results = [_r("intact", "intact", integrity_score=0.8)]
    # gt=0.8, pred_numeric=1.0 → mse=(0.8-1.0)^2=0.04, mae=0.2
    result = _regression_from_results(results, VERDICT_TO_NUMERIC_3)
    assert result["n"] == 1
    assert result["mse"] == pytest.approx(0.04, abs=0.001)
    assert result["mae"] == pytest.approx(0.2, abs=0.001)


def test_regression_from_results_skips_none_predicted():
    results = [_r("intact", None, integrity_score=0.8)]
    result = _regression_from_results(results, VERDICT_TO_NUMERIC_3)
    assert result == {}


def test_regression_from_results_skips_none_score():
    results = [_r("intact", "intact", integrity_score=None)]
    result = _regression_from_results(results, VERDICT_TO_NUMERIC_3)
    assert result == {}


def test_regression_from_results_skips_unknown_label():
    results = [_r("intact", "not-a-label", integrity_score=0.5)]
    result = _regression_from_results(results, VERDICT_TO_NUMERIC_3)
    assert result == {}


# ---------------------------------------------------------------------------
# format_veritas_metrics_report
# ---------------------------------------------------------------------------


def test_format_veritas_metrics_report_empty():
    report = format_veritas_metrics_report({})
    assert "No metrics" in report


def test_format_veritas_metrics_report_contains_key_sections():
    results = [_r("intact", "intact"), _r("compromised", "compromised")]
    metrics = compute_veritas_metrics(results, label_scheme=3)
    report = format_veritas_metrics_report(metrics)
    assert "Accuracy" in report
    assert "Macro F1" in report
    assert "Confusion Matrix" in report
    assert "Per-class" in report


def test_format_veritas_metrics_report_with_regression():
    results = [
        _r("intact", "intact", integrity_score=0.9),
        _r("compromised", "compromised", integrity_score=-0.9),
    ]
    metrics = compute_veritas_metrics(results, label_scheme=3)
    report = format_veritas_metrics_report(metrics)
    assert "Regression" in report
    assert "MSE" in report


def test_format_veritas_metrics_report_7class_includes_coarsened_section():
    results = [
        _r("intact (certain)", "intact (certain)"),
        _r("compromised (certain)", "compromised (certain)"),
    ]
    metrics = compute_veritas_metrics(results, label_scheme=7)
    report = format_veritas_metrics_report(metrics, label_scheme=7)
    assert "Coarsened" in report or "coarsened" in report
