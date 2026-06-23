from mafc.training.labels import (
    direction_of_score,
    label_from_signed_score,
    sign_of_direction,
)


def test_direction_of_score_unknown_band() -> None:
    assert direction_of_score(0.9) == "intact"
    assert direction_of_score(1 / 3) == "intact"  # boundary is inclusive
    assert direction_of_score(0.33) == "unknown"  # 0.33 < 1/3
    assert direction_of_score(0.0) == "unknown"
    assert direction_of_score(-1 / 3) == "compromised"  # boundary is inclusive
    assert direction_of_score(-0.33) == "unknown"  # -0.33 > -1/3
    assert direction_of_score(-1.0) == "compromised"


def test_sign_of_direction() -> None:
    assert sign_of_direction("intact") == 1
    assert sign_of_direction("compromised") == -1
    assert sign_of_direction("unknown") == 0


def test_label_from_signed_score_matches_thresholds() -> None:
    assert label_from_signed_score(1.0) == "intact (certain)"
    assert label_from_signed_score(0.7) == "intact (rather certain)"
    assert label_from_signed_score(0.0) == "unknown"
    assert label_from_signed_score(-0.7) == "compromised (rather certain)"
    assert label_from_signed_score(-1.0) == "compromised (certain)"


def test_label_from_signed_score_binning_boundaries() -> None:
    # The continuous prediction is binned by this same function (cuts ±1/6,±1/2,±5/6).
    assert label_from_signed_score(0.95) == "intact (certain)"
    assert label_from_signed_score(0.6) == "intact (rather certain)"
    assert label_from_signed_score(0.05) == "unknown"
    assert label_from_signed_score(-0.95) == "compromised (certain)"
