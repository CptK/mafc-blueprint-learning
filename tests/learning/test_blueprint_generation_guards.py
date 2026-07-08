"""Tests for the mechanical guards in end-to-end blueprint generation.

These guards encode the lessons from the eom_new regression (run 0706-150314):
a mega-cluster became a shallow catch-all blueprint that absorbed 47% of eval
traffic with a smaller iteration budget than niche blueprints, and overlapping
descriptions let the LLM-tiebreak selector route arbitrarily.
"""

from __future__ import annotations

import numpy as np
import pytest

from mafc.blueprints.models import (
    Blueprint,
    BlueprintAction,
    BlueprintActionNode,
    BlueprintEntryConditions,
    BlueprintPolicyConstraints,
    BlueprintRequiredCheck,
    BlueprintSelectorHints,
    BlueprintVerificationGraph,
)
from mafc.learning.blueprint_consolidator import apply_merge_budget_guard
from mafc.learning.blueprint_contrast import BlueprintContrastPass, _ContrastRevision, enforce_iteration_floor
from mafc.learning.embedding_utils import pick_diverse_representatives, split_oversized_clusters


def _make_blueprint(name: str = "bp", max_iterations: int = 3, description: str = "Neutral verification of claims.") -> Blueprint:
    return Blueprint(
        name=name,
        description=description,
        entry_conditions=BlueprintEntryConditions(),
        selector_hints=BlueprintSelectorHints(),
        policy_constraints=BlueprintPolicyConstraints(
            allowed_actions=["web_search"], max_iterations=max_iterations
        ),
        required_checks=[BlueprintRequiredCheck(id="c1", description="A check was made.")],
        verification_graph=BlueprintVerificationGraph(
            start_node="n1",
            nodes=[
                BlueprintActionNode(
                    id="n1",
                    type="actions",
                    actions=[BlueprintAction(action="web_search", intent="find sources", query_guidance="search")],
                    transition=[],
                ),
            ],
        ),
    )


# ---------------------------------------------------------------------------
# split_oversized_clusters
# ---------------------------------------------------------------------------


def _blob(center: np.ndarray, n: int, seed: int, scale: float = 0.05) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return center + rng.normal(scale=scale, size=(n, center.shape[0]))


def test_mega_cluster_is_split_below_cap() -> None:
    # One cluster with two clear sub-blobs holding 80% of points, one small cluster.
    big = np.vstack([_blob(np.array([0.0, 0.0]), 200, 1), _blob(np.array([5.0, 5.0]), 200, 2)])
    small = _blob(np.array([-5.0, 5.0]), 100, 3)
    X = np.vstack([big, small])
    clusters = [list(range(400)), list(range(400, 500))]

    result = split_oversized_clusters(X, clusters, max_frac=0.25, min_cluster_size=10)

    assert len(result) >= 3
    cap = 0.25 * 500
    for indices, _parent in result:
        assert len(indices) <= cap
    # No points lost or duplicated.
    all_indices = sorted(i for indices, _ in result for i in indices)
    assert all_indices == list(range(500))
    # Provenance: children of the big cluster point to position 0.
    for indices, parent in result:
        expected_parent = 0 if indices[0] < 400 else 1
        assert parent == expected_parent


def test_small_clusters_pass_through_unchanged() -> None:
    X = np.vstack([_blob(np.array([0.0, 0.0]), 50, 1), _blob(np.array([5.0, 5.0]), 50, 2)])
    clusters = [list(range(50)), list(range(50, 100))]
    result = split_oversized_clusters(X, clusters, max_frac=0.6, min_cluster_size=10)
    assert [(sorted(ids), p) for ids, p in result] == [(list(range(50)), 0), (list(range(50, 100)), 1)]


def test_split_disabled_with_zero_frac() -> None:
    X = _blob(np.array([0.0, 0.0]), 100, 1)
    clusters = [list(range(100))]
    result = split_oversized_clusters(X, clusters, max_frac=0.0, min_cluster_size=10)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# pick_diverse_representatives
# ---------------------------------------------------------------------------


def test_representatives_cover_outlying_subgroup() -> None:
    # 90 points in a dense core, 10 in a far-away satellite. Centroid-nearest
    # sampling would pick only core points; the FPS half must reach the satellite.
    core = _blob(np.array([0.0, 0.0]), 90, 1)
    satellite = _blob(np.array([10.0, 10.0]), 10, 2)
    X = np.vstack([core, satellite])

    picked = pick_diverse_representatives(X, 10)

    assert len(picked) == len(set(picked)) == 10
    assert any(i >= 90 for i in picked), "no representative from the outlying subgroup"
    assert any(i < 90 for i in picked), "no representative from the core"


def test_representatives_small_input_returns_all() -> None:
    X = _blob(np.array([0.0, 0.0]), 5, 1)
    assert pick_diverse_representatives(X, 10) == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Iteration budget guards
# ---------------------------------------------------------------------------


def test_iteration_floor_raises_high_traffic_blueprint() -> None:
    bp = _make_blueprint(max_iterations=3)
    result = enforce_iteration_floor(bp, expected_share=0.4)
    assert result.policy_constraints.max_iterations == 4


def test_iteration_floor_ignores_low_traffic_blueprint() -> None:
    bp = _make_blueprint(max_iterations=2)
    result = enforce_iteration_floor(bp, expected_share=0.05)
    assert result.policy_constraints.max_iterations == 2


def test_merge_budget_guard_keeps_parent_max() -> None:
    merged = _make_blueprint("merged", max_iterations=3)
    base = _make_blueprint("base", max_iterations=4)
    removed = _make_blueprint("removed", max_iterations=3)
    result = apply_merge_budget_guard(merged, base, removed)
    assert result.policy_constraints.max_iterations == 4

    already_fine = _make_blueprint("merged2", max_iterations=4)
    assert apply_merge_budget_guard(already_fine, base, removed) is already_fine


def test_merge_size_veto_blocks_oversized_merge() -> None:
    from mafc.blueprints.registry import BlueprintRegistry
    from mafc.learning.blueprint_consolidator import BlueprintConsolidator, ConsolidationResult, _MergeGroup

    a = _make_blueprint("a")
    b = _make_blueprint("b")
    registry = BlueprintRegistry([a, b])
    consolidator = BlueprintConsolidator(
        model=object(),
        updater=object(),
        prune_threshold=0,
        merge_size_lookup={"a": 500, "b": 400},
        max_merged_size=731,
    )
    executed: list[tuple[str, str]] = []
    consolidator._execute_merge = lambda *args: executed.append((args[2], args[3]))  # type: ignore[method-assign]
    consolidator._detect_merges = lambda *args: [_MergeGroup(blueprints=["a", "b"], rationale="same")]  # type: ignore[method-assign]

    consolidator._merge(registry, {}, ConsolidationResult())
    assert executed == [], "merge exceeding the size cap must be vetoed"

    consolidator.merge_size_lookup = {"a": 300, "b": 400}
    consolidator._merge(registry, {}, ConsolidationResult())
    assert executed == [("a", "b")], "merge within the cap must proceed"


# ---------------------------------------------------------------------------
# Contrast pass revision application
# ---------------------------------------------------------------------------


@pytest.fixture()
def contrast_pass() -> BlueprintContrastPass:
    return BlueprintContrastPass(model=object())  # model unused by _apply_revision


def test_contrast_revision_applied(contrast_pass: BlueprintContrastPass) -> None:
    bp = _make_blueprint("media_bp")
    revision = _ContrastRevision(
        name="media_bp",
        description="Handles video/image claims needing origin tracing; unlike text_bp, requires media.",
        selector_hints={"positive": {"features": ["has_video"], "examples": ["A clip said to show event X."]}},
    )
    revised = contrast_pass._apply_revision(bp, revision)
    assert revised.description.startswith("Handles video/image claims")
    assert revised.selector_hints.positive.features == ["has_video"]
    # Strategy content untouched.
    assert revised.verification_graph == bp.verification_graph
    assert revised.policy_constraints == bp.policy_constraints


def test_contrast_revision_rejected_when_presuppositional(contrast_pass: BlueprintContrastPass) -> None:
    bp = _make_blueprint("media_bp")
    revision = _ContrastRevision(
        name="media_bp",
        description="Targets authentic media shared with a false context to expose the mismatch.",
        selector_hints={},
    )
    revised = contrast_pass._apply_revision(bp, revision)
    assert revised.description == bp.description


def test_contrast_revision_missing_keeps_original(contrast_pass: BlueprintContrastPass) -> None:
    bp = _make_blueprint("media_bp")
    assert contrast_pass._apply_revision(bp, None) is bp
