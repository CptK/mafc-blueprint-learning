import pytest

from mafc.eval.metrics import (
    classification_block,
    confusion_matrix,
    direction_metrics,
    format_confusion_matrix,
    macro_average,
    per_class_metrics,
    regression_metrics,
    weighted_average,
)

# ---------------------------------------------------------------------------
# confusion_matrix
# ---------------------------------------------------------------------------


def test_confusion_matrix_perfect_prediction():
    cm = confusion_matrix(["a", "b", "a"], ["a", "b", "a"], ["a", "b"])
    assert cm == [[2, 0], [0, 1]]


def test_confusion_matrix_all_wrong():
    cm = confusion_matrix(["a", "a"], ["b", "b"], ["a", "b"])
    assert cm == [[0, 2], [0, 0]]


def test_confusion_matrix_empty_inputs():
    cm = confusion_matrix([], [], ["a", "b"])
    assert cm == [[0, 0], [0, 0]]


def test_confusion_matrix_mixed():
    y_true = ["a", "b", "a", "b"]
    y_pred = ["a", "a", "a", "b"]
    cm = confusion_matrix(y_true, y_pred, ["a", "b"])
    # row a: 2 correct, 0 wrong; row b: 1 wrong (pred a), 1 correct
    assert cm == [[2, 0], [1, 1]]


# ---------------------------------------------------------------------------
# per_class_metrics
# ---------------------------------------------------------------------------


def test_per_class_metrics_perfect():
    pc = per_class_metrics(["a", "b", "a"], ["a", "b", "a"], ["a", "b"])
    assert pc["a"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 2}
    assert pc["b"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 1}


def test_per_class_metrics_zero_division():
    # label "b" never appears in y_true or y_pred
    pc = per_class_metrics(["a"], ["a"], ["a", "b"])
    assert pc["b"]["precision"] == 0.0
    assert pc["b"]["recall"] == 0.0
    assert pc["b"]["f1"] == 0.0
    assert pc["b"]["support"] == 0


def test_per_class_metrics_false_positives():
    # "a" predicted for everything, "b" never predicted
    pc = per_class_metrics(["a", "b"], ["a", "a"], ["a", "b"])
    assert pc["a"]["precision"] == pytest.approx(0.5)
    assert pc["a"]["recall"] == 1.0
    assert pc["b"]["recall"] == 0.0


# ---------------------------------------------------------------------------
# macro_average
# ---------------------------------------------------------------------------


def test_macro_average_empty_labels():
    assert macro_average({}, []) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_macro_average_two_classes():
    pc = {
        "a": {"precision": 1.0, "recall": 0.5, "f1": 0.6667, "support": 2},
        "b": {"precision": 0.5, "recall": 1.0, "f1": 0.6667, "support": 1},
    }
    result = macro_average(pc, ["a", "b"])
    assert result["precision"] == pytest.approx(0.75, abs=0.001)
    assert result["recall"] == pytest.approx(0.75, abs=0.001)


# ---------------------------------------------------------------------------
# weighted_average
# ---------------------------------------------------------------------------


def test_weighted_average_zero_total():
    assert weighted_average({}, [], 0) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_weighted_average_skews_toward_high_support():
    pc = {
        "a": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 9},
        "b": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 1},
    }
    result = weighted_average(pc, ["a", "b"], 10)
    assert result["precision"] == pytest.approx(0.9)
    assert result["f1"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# classification_block
# ---------------------------------------------------------------------------


def test_classification_block_accuracy():
    y_true = ["a", "b", "a", "b"]
    y_pred = ["a", "a", "a", "b"]
    block = classification_block(y_true, y_pred, ["a", "b"])
    assert block["accuracy"] == pytest.approx(0.75)
    assert "macro" in block
    assert "weighted" in block
    assert "per_class" in block
    assert "confusion_matrix" in block


def test_classification_block_empty_inputs():
    block = classification_block([], [], ["a", "b"])
    assert block["accuracy"] is None


def test_classification_block_perfect():
    block = classification_block(["a", "b"], ["a", "b"], ["a", "b"])
    assert block["accuracy"] == 1.0
    assert block["macro"]["f1"] == 1.0


# ---------------------------------------------------------------------------
# regression_metrics
# ---------------------------------------------------------------------------


def test_regression_metrics_perfect():
    result = regression_metrics([1.0, 0.0, -1.0], [1.0, 0.0, -1.0])
    assert result["mse"] == 0.0
    assert result["mae"] == 0.0
    assert result["n"] == 3


def test_regression_metrics_empty():
    assert regression_metrics([], []) == {}


def test_regression_metrics_known_values():
    # gt=[1.0, 0.0], pred=[0.0, 0.0] → mse=0.5, mae=0.5
    result = regression_metrics([1.0, 0.0], [0.0, 0.0])
    assert result["mse"] == pytest.approx(0.5)
    assert result["mae"] == pytest.approx(0.5)
    assert result["n"] == 2


# ---------------------------------------------------------------------------
# direction_metrics
# ---------------------------------------------------------------------------


def test_direction_metrics_empty():
    assert direction_metrics([], [], 1 / 6) == {}


def test_direction_metrics_counts_hedges_as_flips_too():
    gt = [0.9, -0.9, 0.9, 0.9]
    pred = [-0.5, 0.5, 0.1, 1.0]  # reversed, reversed, hedged into the band, correct
    m = direction_metrics(gt, pred, 1 / 6)
    assert m["flips"] == 3
    assert m["flips_opposite"] == 2
    assert m["flips_neutral"] == 1
    assert m["flip_rate"] == pytest.approx(0.75)
    assert m["n_excl_flips"] == 1


def test_direction_metrics_excludes_flips_from_mse():
    gt = [1.0, 1.0]
    pred = [-1.0, 0.5]
    m = direction_metrics(gt, pred, 1 / 6)
    # only the (1.0, 0.5) pair survives: se=0.25
    assert m["mse_excl_flips"] == pytest.approx(0.25)
    assert m["mae_excl_flips"] == pytest.approx(0.5)
    assert m["mse_flips_only"] == pytest.approx(4.0)
    assert m["flip_se_share"] == pytest.approx(4.0 / 4.25, abs=1e-4)


def test_direction_metrics_no_flips_leaves_mse_unchanged():
    gt = [1.0, -1.0, 0.0]
    pred = [0.5, -0.5, 0.0]  # third pair is neutral on both sides, so no mismatch
    m = direction_metrics(gt, pred, 1 / 6)
    assert m["flips"] == 0
    assert m["flip_se_share"] == 0.0
    assert m["mse_excl_flips"] == pytest.approx(0.5 / 3, abs=1e-4)
    assert "mse_flips_only" not in m


def test_direction_metrics_deadband_widens_the_neutral_zone():
    gt, pred = [0.9], [-0.25]
    assert direction_metrics(gt, pred, 1 / 6)["flips_opposite"] == 1
    # -0.25 is inside ±1/3, so it stops being a reversal and becomes a hedge
    wide = direction_metrics(gt, pred, 1 / 3)
    assert wide["flips_opposite"] == 0
    assert wide["flips"] == 1


# ---------------------------------------------------------------------------
# format_confusion_matrix
# ---------------------------------------------------------------------------


def test_format_confusion_matrix_contains_title_and_labels():
    cm_dict = {"labels": ["a", "b"], "matrix": [[2, 0], [1, 1]]}
    text = format_confusion_matrix(cm_dict, title="My CM")
    assert "My CM" in text
    assert "a" in text
    assert "b" in text


def test_format_confusion_matrix_contains_counts():
    cm_dict = {"labels": ["x"], "matrix": [[42]]}
    text = format_confusion_matrix(cm_dict)
    assert "42" in text
