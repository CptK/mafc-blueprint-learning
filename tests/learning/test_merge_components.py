from __future__ import annotations

from mafc.learning.blueprint_consolidator import BlueprintConsolidator, _MergeGroup


def _groups(*pairs: tuple[str, str]) -> list[_MergeGroup]:
    return [_MergeGroup(blueprints=[a, b], rationale="r") for a, b in pairs]


def test_clique_becomes_one_component():
    """The observed case: one hub paired with six others is ONE family, not six pairs."""
    components = BlueprintConsolidator._merge_components(
        _groups(("hub", "a"), ("hub", "b"), ("hub", "c"), ("hub", "d"))
    )
    assert len(components) == 1
    assert set(components[0]) == {"hub", "a", "b", "c", "d"}


def test_transitive_pairs_join_one_component():
    components = BlueprintConsolidator._merge_components(_groups(("a", "b"), ("b", "c"), ("c", "d")))
    assert len(components) == 1
    assert set(components[0]) == {"a", "b", "c", "d"}


def test_disjoint_families_stay_separate():
    components = BlueprintConsolidator._merge_components(_groups(("a", "b"), ("c", "d")))
    assert len(components) == 2
    assert {frozenset(c) for c in components} == {frozenset({"a", "b"}), frozenset({"c", "d"})}


def test_component_base_is_first_mentioned():
    """The base drives the merge, so ordering must be predictable."""
    components = BlueprintConsolidator._merge_components(_groups(("hub", "a"), ("hub", "b")))
    assert components[0][0] == "hub"


def test_no_groups_yields_no_components():
    assert BlueprintConsolidator._merge_components([]) == []


def test_regression_2025_pool_collapses_to_two_families():
    """The eight merge groups the detector reported on the 2025 pool."""
    components = BlueprintConsolidator._merge_components(
        _groups(
            ("media_origin_and_context", "recontextualized_media"),
            ("media_origin_and_context", "media_provenance_and_context"),
            ("media_origin_and_context", "origin_and_context_verification"),
            ("media_origin_and_context", "origin_and_context_verification_2"),
            ("media_origin_and_context", "recontextualized_media_and_quote"),
            ("media_origin_and_context", "media_origin_and_institutional_records"),
            ("media_origin_and_authenticity", "origin_and_context_verification_2"),
            ("statistical_and_official_record", "scientific_medical_claims"),
        )
    )
    assert len(components) == 2
    sizes = sorted(len(c) for c in components)
    assert sizes == [2, 8]
    # 8 merges, not the 2 the pairwise-disjoint loop performed.
    assert sum(len(c) - 1 for c in components) == 8
