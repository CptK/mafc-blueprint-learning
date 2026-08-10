"""Tests for eliding the router when a merge collapses to a single lane.

A pairwise merge of two redundant blueprints — the consolidation case — folds both
onto one branch. The router over that single branch decides nothing, yet it consumes
an iteration and a synthesis call on every run and lengthens the longest path, so the
load-time budget guard raises max_iterations to pay for it.
"""

from __future__ import annotations

import json

from mafc.blueprints.topology import longest_path_nodes
from mafc.learning.merge_blueprints.merger import BlueprintTreeMerger

from .test_merge_routing_descriptions import ScriptedModel, _blueprint


def _merged_pair(match_index: int | None) -> "object":
    """Merge two blueprints, scripting whether the matcher folds them onto one lane."""
    outputs = [json.dumps({"match_index": match_index, "rationale": "scripted"})]
    if match_index is not None:
        outputs += [
            json.dumps({"pairs": [], "unmatched_signal": []}),  # branch alignment
            json.dumps({"description": "Both."}),  # description fold
        ]
    merger = BlueprintTreeMerger(ScriptedModel(outputs), reconcile=False, consolidate=False, sharpen=False)
    base = _blueprint("base", "Verifies AI-generated media.")
    other = _blueprint("other", "Verifies AI-generated media.")
    return merger.merge([base, other], name="merged").blueprint


def test_single_lane_merge_starts_at_the_lane_not_a_router() -> None:
    bp = _merged_pair(match_index=0)

    node_ids = {node.id for node in bp.verification_graph.nodes}
    assert "router" not in node_ids
    assert bp.verification_graph.start_node == "base/n1"


def test_single_lane_merge_costs_no_extra_iteration() -> None:
    """Without elision the router adds a node to every path, and the budget with it."""
    bp = _merged_pair(match_index=0)

    assert longest_path_nodes(bp) == 1
    assert all(node.id.startswith("base/") for node in bp.verification_graph.nodes)


def test_multi_lane_merge_keeps_the_router() -> None:
    bp = _merged_pair(match_index=None)

    assert bp.verification_graph.start_node == "router"
    router = next(node for node in bp.verification_graph.nodes if node.id == "router")
    assert len(router.transition) == 2


def test_force_single_branch_folds_without_consulting_the_matcher() -> None:
    """The merge detector already established redundancy; re-deciding it can only lose."""
    model = ScriptedModel(
        [
            json.dumps({"pairs": [], "unmatched_signal": []}),  # branch alignment
            json.dumps({"description": "Both."}),  # description fold
        ]
    )
    merger = BlueprintTreeMerger(
        model, reconcile=False, consolidate=False, sharpen=False, force_single_branch=True
    )
    base = _blueprint("base", "Verifies AI-generated media.")
    other = _blueprint("other", "Verifies statistical claims from official records.")

    bp = merger.merge([base, other], name="merged").blueprint

    # Descriptions share nothing, so an unforced matcher would open a second branch.
    assert "router" not in {node.id for node in bp.verification_graph.nodes}
    assert bp.verification_graph.start_node == "base/n1"
    assert not any("match" in call.lower() and "index" in call.lower() for call in model.calls)


def test_single_lane_merge_keeps_both_parents_checks() -> None:
    """Eliding the router must not drop the checks it would have led to."""
    bp = _merged_pair(match_index=0)

    assert sorted(check.id for check in bp.required_checks) == ["base_check", "other_check"]
    refs = {node.id: node.activates_checks for node in bp.verification_graph.nodes if node.activates_checks}
    assert sorted(refs["base/n1"]) == ["base_check", "other_check"]
