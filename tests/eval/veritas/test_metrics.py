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


def test_compute_veritas_metrics_reports_the_flip_decomposition():
    results = [
        _r("intact", "intact", integrity_score=0.9),
        _r("intact", "compromised", integrity_score=0.9),  # direction flip
    ]
    metrics = compute_veritas_metrics(results, label_scheme=3)
    assert metrics["flips"] == 1
    assert metrics["flip_rate"] == pytest.approx(0.5)
    assert metrics["n_excl_flips"] == 1
    # the surviving claim is gt=0.9 vs pred=1.0
    assert metrics["mse_excl_flips"] == pytest.approx(0.01, abs=1e-4)
    assert metrics["mse"] > metrics["mse_excl_flips"]


def test_compute_veritas_metrics_counts_an_abstention_as_a_flip():
    """Predicting unknown on a directional claim misses the direction like a reversal."""
    results = [
        _r("intact", "intact", integrity_score=0.9),
        _r("intact", "unknown", integrity_score=0.9),
    ]
    metrics = compute_veritas_metrics(results, label_scheme=3)
    assert metrics["flips"] == 1
    assert metrics["flips_opposite"] == 0
    assert metrics["flips_neutral"] == 1


def test_compute_veritas_metrics_flip_band_follows_the_label_scheme():
    """7-class calls direction outside ±1/6; 3-class only outside ±1/3."""
    m7 = compute_veritas_metrics(
        [_r("intact (certain)", "intact (certain)", integrity_score=0.9)], label_scheme=7
    )
    assert m7["flip_deadband"] == pytest.approx(1 / 6)
    m3 = compute_veritas_metrics([_r("intact", "intact", integrity_score=0.9)], label_scheme=3)
    assert m3["flip_deadband"] == pytest.approx(1 / 3)


def test_compute_veritas_metrics_flip_fields_reach_the_report():
    results = [
        _r("intact", "intact", integrity_score=0.9),
        _r("intact", "compromised", integrity_score=0.9),
    ]
    report = format_veritas_metrics_report(
        compute_veritas_metrics(results, label_scheme=3), label_scheme=3
    )
    assert "Direction flips" in report
    assert "MSE excl. flips" in report


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


def test_regression_prefers_the_unsnapped_score_over_the_label():
    """Ground truth is continuous, so the aggregate is scored, not its nearest label."""
    results = [{**_r("intact", "intact", integrity_score=0.8), "predicted_score": 0.75}]
    result = _regression_from_results(results, VERDICT_TO_NUMERIC_3)
    # 0.75 rather than the label's 1.0: mse=(0.8-0.75)^2, mae=0.05
    assert result["mse"] == pytest.approx(0.0025, abs=1e-6)
    assert result["mae"] == pytest.approx(0.05, abs=1e-6)


def test_regression_falls_back_to_the_label_without_a_score():
    """Single-sample runs carry no aggregate; the two are identical there anyway."""
    with_none = [{**_r("intact", "intact", integrity_score=0.8), "predicted_score": None}]
    assert _regression_from_results(with_none, VERDICT_TO_NUMERIC_3)["mse"] == pytest.approx(0.04)
    assert _regression_from_results([_r("intact", "intact", 0.8)], VERDICT_TO_NUMERIC_3)[
        "mse"
    ] == pytest.approx(0.04)


def test_regression_ignores_the_score_when_the_label_is_missing():
    """A score without a verdict is not a prediction."""
    results = [{**_r("intact", None, integrity_score=0.8), "predicted_score": 0.75}]
    assert _regression_from_results(results, VERDICT_TO_NUMERIC_3) == {}


def test_coarsened_regression_stays_on_the_label_scale():
    """The 3-bin block measures the coarsened verdict, so the 7-class aggregate
    must not leak into it."""
    results = [
        {
            "ground_truth": "intact (certain)",
            "predicted": "intact (certain)",
            "predicted_score": 0.6,
            "gt_integrity_score": 1.0,
        }
    ]
    metrics = compute_veritas_metrics(results, label_scheme=7)
    assert metrics["mse"] == pytest.approx(0.16, abs=1e-6)  # (1.0 - 0.6)^2, un-snapped
    assert metrics["coarsened_3class"]["mse"] == pytest.approx(0.0, abs=1e-6)  # label-based


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
