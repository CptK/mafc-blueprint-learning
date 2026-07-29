"""Tests for node-scoped required checks (schema, runtime activation, merge).

The merged eom_v3 tree imposed the union of all lanes' checks (27) on every
claim. Checks now attach to nodes and activate only when the execution path
reaches them, so a claim carries exactly the checks its path owes.
"""

from __future__ import annotations

import json

import yaml

from mafc.agents.fact_check.models import CheckStatus, FactCheckSessionState
from mafc.blueprints.models import (
    Blueprint,
    BlueprintActionNode,
    BlueprintEntryConditions,
    BlueprintPolicyConstraints,
    BlueprintRequiredCheck,
    BlueprintSynthesisNode,
    BlueprintVerificationGraph,
)
from mafc.learning.merge_blueprints.consolidate import CheckConsolidator
from mafc.learning.merge_blueprints.merger import BlueprintTreeMerger

from .test_merge_routing_descriptions import ScriptedModel, _blueprint


def _check(cid: str) -> BlueprintRequiredCheck:
    return BlueprintRequiredCheck(id=cid, description=f"{cid} verified.")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_node_check_refs_roundtrip_yaml() -> None:
    bp = Blueprint(
        name="bp",
        description="Neutral verification.",
        entry_conditions=BlueprintEntryConditions(),
        policy_constraints=BlueprintPolicyConstraints(allowed_actions=["web_search"], max_iterations=2),
        required_checks=[_check("global_check"), _check("lane_check")],
        verification_graph=BlueprintVerificationGraph(
            start_node="n1",
            nodes=[
                BlueprintSynthesisNode(
                    id="n1", type="synthesis", transition=[], activates_checks=["lane_check"]
                ),
            ],
        ),
    )
    dumped = yaml.safe_load(yaml.dump(bp.model_dump(by_alias=True)))
    restored = Blueprint.model_validate(dumped)
    assert restored.verification_graph.nodes[0].activates_checks == ["lane_check"]
    assert restored.node_scoped_check_ids() == {"lane_check"}


def test_node_refs_to_undefined_checks_rejected() -> None:
    import pytest

    with pytest.raises(ValueError, match="undefined check"):
        Blueprint(
            name="bp",
            description="d",
            entry_conditions=BlueprintEntryConditions(),
            policy_constraints=BlueprintPolicyConstraints(allowed_actions=[], max_iterations=2),
            required_checks=[_check("global_check")],
            verification_graph=BlueprintVerificationGraph(
                start_node="n1",
                nodes=[
                    BlueprintSynthesisNode(
                        id="n1", type="synthesis", transition=[], activates_checks=["missing"]
                    ),
                ],
            ),
        )


def test_node_checks_default_empty_for_legacy_blueprints() -> None:
    node = BlueprintActionNode.model_validate(
        {"id": "n1", "type": "actions", "actions": [], "transition": []}
    )
    assert node.activates_checks == []


# ---------------------------------------------------------------------------
# Runtime activation
# ---------------------------------------------------------------------------


def _state(**kwargs) -> FactCheckSessionState:
    bp = Blueprint(
        name="bp",
        description="d",
        entry_conditions=BlueprintEntryConditions(),
        policy_constraints=BlueprintPolicyConstraints(allowed_actions=[], max_iterations=2),
        required_checks=[_check("global_check"), _check("lane_check")],
        verification_graph=BlueprintVerificationGraph(
            start_node="n1",
            nodes=[
                BlueprintSynthesisNode(id="n1", type="synthesis", transition=[]),
                BlueprintSynthesisNode(
                    id="lane",
                    type="synthesis",
                    transition=[],
                    activates_checks=["lane_check", "global_check"],
                ),
            ],
        ),
    )
    return FactCheckSessionState(
        selected_blueprint=bp,
        current_node_id="n1",
        node_layers={"n1": 0, "lane": 1},
        max_layer=1,
        required_check_status={"global_check": CheckStatus.UNCHECKED},
        required_check_defs={"global_check": _check("global_check")},
        **kwargs,
    )


def test_activation_adds_node_checks_once() -> None:
    state = _state()
    lane_node = next(n for n in state.selected_blueprint.verification_graph.nodes if n.id == "lane")
    added = state.activate_node_checks(lane_node)
    assert added == ["lane_check"]  # global_check already active, not re-added
    assert state.required_check_status["lane_check"] == CheckStatus.UNCHECKED
    assert state.required_check_defs["lane_check"].description == "lane_check verified."

    # Second visit (converging path): no re-add, no status reset.
    state.required_check_status["lane_check"] = CheckStatus.SUPPORTED
    assert state.activate_node_checks(lane_node) == []
    assert state.required_check_status["lane_check"] == CheckStatus.SUPPORTED


# ---------------------------------------------------------------------------
# Merge pipeline
# ---------------------------------------------------------------------------


def test_merge_attaches_checks_to_lane_entries_not_globally() -> None:
    model = ScriptedModel(
        [
            json.dumps({"match_index": 0, "rationale": "same"}),  # statements -> quotes branch
            json.dumps({"pairs": [], "unmatched_signal": [0]}),  # branch alignment
            json.dumps({"description": "Quotes and statements."}),  # description fold
        ]
    )
    merger = BlueprintTreeMerger(model, reconcile=False, consolidate=False, sharpen=False)
    generic = _blueprint("generic", "Generic fallback.", gated=False)
    quotes = _blueprint("quotes", "Verifies attributed quotes.")
    statements = _blueprint("statements", "Verifies official statements.")

    result = merger.merge([generic, quotes, statements], name="merged")
    bp = result.blueprint

    # Definitions all sit at the root; every one is referenced by a lane entry
    # (no global-only checks in this pool), so none is active from the start.
    assert sorted(c.id for c in bp.required_checks) == [
        "generic_check",
        "quotes_check",
        "statements_check",
    ]
    refs_by_node = {n.id: n.activates_checks for n in bp.verification_graph.nodes if n.activates_checks}
    assert refs_by_node["generic/n1"] == ["generic_check"]
    assert sorted(refs_by_node["quotes/n1"]) == ["quotes_check", "statements_check"]
    assert len(refs_by_node) == 2
    assert bp.node_scoped_check_ids() == {"generic_check", "quotes_check", "statements_check"}


def test_colliding_check_ids_with_different_descriptions_are_renamed() -> None:
    from mafc.blueprints.models import BlueprintCondition
    from mafc.learning.merge_blueprints.tree import EntryBranch, MergedStrategyTree, MergeNode

    tree = MergedStrategyTree()
    tree.allowed_actions = ["web_search"]
    node_a = MergeNode(
        "a/n1", "synthesis", checks=[BlueprintRequiredCheck(id="src", description="Version A.")]
    )
    node_b = MergeNode(
        "b/n1", "synthesis", checks=[BlueprintRequiredCheck(id="src", description="Version B.")]
    )
    gate = BlueprintEntryConditions(any=[BlueprintCondition(feature="has_claim_text", op="==", value=True)])
    tree.entries = [EntryBranch("a", gate, node_a, "A lane."), EntryBranch("b", gate, node_b, "B lane.")]

    bp = tree.to_blueprint("merged", "d")
    ids = sorted(c.id for c in bp.required_checks)
    assert ids == ["src", "src_2"]
    refs = {n.id: n.activates_checks for n in bp.verification_graph.nodes if n.activates_checks}
    assert refs["a/n1"] == ["src"]
    assert refs["b/n1"] == ["src_2"]


def test_check_consolidator_escape_only_when_requested() -> None:
    model = ScriptedModel([json.dumps({"groups": []}), json.dumps({"groups": []})])
    consolidator = CheckConsolidator(model)
    checks = [_check("a"), _check("b")]
    plain = consolidator.consolidate(checks)
    assert all("UNCHECKED" not in c.description for c in plain)
    escaped = consolidator.consolidate(checks, add_applicability_escape=True)
    assert all("Mark UNCHECKED" in c.description for c in escaped)
