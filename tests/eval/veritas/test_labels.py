from mafc.eval.veritas.labels import (
    CLASS_MAPPING_3,
    CLASS_MAPPING_7,
    THRESHOLDS_3,
    THRESHOLDS_7,
    Veritas3Label,
    Veritas7Label,
)


def test_veritas3_label_values():
    assert Veritas3Label.INTACT.value == "intact"
    assert Veritas3Label.COMPROMISED.value == "compromised"
    assert Veritas3Label.UNKNOWN.value == "unknown"


def test_veritas7_label_values():
    assert Veritas7Label.INTACT_CERTAIN.value == "intact (certain)"
    assert Veritas7Label.INTACT_RATHER_CERTAIN.value == "intact (rather certain)"
    assert Veritas7Label.INTACT_RATHER_UNCERTAIN.value == "intact (rather uncertain)"
    assert Veritas7Label.UNKNOWN.value == "unknown"
    assert Veritas7Label.COMPROMISED_RATHER_UNCERTAIN.value == "compromised (rather uncertain)"
    assert Veritas7Label.COMPROMISED_RATHER_CERTAIN.value == "compromised (rather certain)"
    assert Veritas7Label.COMPROMISED_CERTAIN.value == "compromised (certain)"


def test_class_mapping_3_covers_all_labels():
    assert set(CLASS_MAPPING_3.values()) == set(Veritas3Label)


def test_class_mapping_7_covers_all_labels():
    assert set(CLASS_MAPPING_7.values()) == set(Veritas7Label)


def test_class_mapping_3_keys_are_title_case():
    for key in CLASS_MAPPING_3:
        assert key[0].isupper(), f"Expected title-case key, got: {key!r}"


def test_thresholds_3_intact_above_compromised():
    assert THRESHOLDS_3["intact"] > THRESHOLDS_3["compromised"]


def test_thresholds_7_bounds_are_ascending():
    finite_bounds = [bound for bound, _ in THRESHOLDS_7 if bound != float("inf")]
    assert finite_bounds == sorted(finite_bounds)


def test_thresholds_7_has_seven_entries():
    assert len(THRESHOLDS_7) == 7


def test_thresholds_7_ends_with_inf_intact_certain():
    last_bound, last_label = THRESHOLDS_7[-1]
    assert last_bound == float("inf")
    assert last_label == Veritas7Label.INTACT_CERTAIN
