import numpy as np

from mafc.training.evaluate import (
    baseline_magnitudes,
    certain_confusion,
    evaluate,
    expected_calibration_error,
)


def test_ece_zero_for_perfect_calibration() -> None:
    pred = np.array([0.05, 0.25, 0.55, 0.95])
    true = np.array([0.05, 0.25, 0.55, 0.95])
    ece, bins = expected_calibration_error(pred, true, n_bins=10)
    assert ece == 0.0
    # bins with no samples carry NaN means
    assert any(b.count == 0 for b in bins)


def test_ece_positive_for_miscalibration() -> None:
    pred = np.array([0.9, 0.9, 0.9, 0.9])
    true = np.array([0.1, 0.1, 0.1, 0.1])
    ece, _ = expected_calibration_error(pred, true, n_bins=10)
    assert abs(ece - 0.8) < 1e-9


def test_certain_confusion_counts_per_side() -> None:
    y_true = [
        "intact (certain)",
        "intact (certain)",
        "intact (rather certain)",
        "compromised (certain)",
    ]
    y_pred = [
        "intact (certain)",
        "intact (rather certain)",  # confused
        "intact (rather certain)",
        "compromised (rather certain)",  # confused
    ]
    cc = certain_confusion(y_true, y_pred)
    assert cc["intact"]["certain->certain"] == 1
    assert cc["intact"]["certain->rather_certain"] == 1
    assert cc["compromised"]["certain->rather_certain"] == 1
    assert cc["overall"]["n"] == 4


def test_baseline_magnitudes() -> None:
    mags = baseline_magnitudes(
        ["intact (certain)", "compromised (rather certain)", "unknown"]
    )
    assert abs(mags[0] - 1.0) < 1e-9
    assert abs(mags[1] - 2 / 3) < 1e-9
    assert mags[2] == 0.0


def test_evaluate_produces_expected_keys() -> None:
    y_true = ["intact (certain)", "compromised (certain)", "unknown"]
    y_pred = ["intact (rather certain)", "compromised (certain)", "unknown"]
    gt = [1.0, -1.0, 0.0]
    mag = np.array([0.7, 0.95, 0.0])
    m = evaluate(y_true, y_pred, gt, mag)
    assert "ece" in m and "reliability_diagram" in m and "certain_confusion" in m
    assert "accuracy" in m
