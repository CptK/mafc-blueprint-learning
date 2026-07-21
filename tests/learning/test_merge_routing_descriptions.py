"""Tests for description-carrying router branches in the blueprint tree merge.

The eom_v3 merge emitted router conditions rendered from boolean entry-condition
gates — tautological for every text claim — and matched fold targets on those
same gates, silently dissolving semantically distinct lanes. Branch identity now
lives in a routing description maintained across merge events.
"""

from __future__ import annotations

import json

from mafc.blueprints.models import (
    Blueprint,
    BlueprintAction,
    BlueprintActionNode,
    BlueprintCondition,
    BlueprintEntryConditions,
    BlueprintPolicyConstraints,
    BlueprintRequiredCheck,
    BlueprintSelectorHints,
    BlueprintVerificationGraph,
)
from mafc.common.modeling.message import Message
from mafc.common.modeling.model import Model, Response
from mafc.learning.merge_blueprints.consolidate import _with_applicability_escape
from mafc.learning.merge_blueprints.matching import BranchMatcher
from mafc.learning.merge_blueprints.merger import BlueprintTreeMerger
from mafc.learning.merge_blueprints.tree import (
    EntryBranch,
    MergedStrategyTree,
    MergeNode,
    routing_description,
)


class ScriptedModel(Model):
    """Returns queued responses; records prompts."""

    def __init__(self, outputs: list[str]):
        super().__init__(specifier="OPENAI:gpt-5-mini-2025-08-07")
        self.outputs = list(outputs)
        self.calls: list[str] = []

    def _do_generate(self, messages: list[Message]) -> Response:
        self.calls.append("\n".join(str(m.content) for m in messages))
        return Response(text=self.outputs.pop(0) if self.outputs else "{}", total_cost=0.0)


def _blueprint(name: str, description: str, examples: list[str] | None = None, gated: bool = True) -> Blueprint:
    conditions = (
        BlueprintEntryConditions(any=[BlueprintCondition(feature="has_claim_text", op="==", value=True)])
        if gated
        else BlueprintEntryConditions()
    )
    hints = BlueprintSelectorHints.model_validate(
        {"positive": {"features": [], "examples": examples or []}}
    )
    return Blueprint(
        name=name,
        description=description,
        entry_conditions=conditions,
        selector_hints=hints,
        policy_constraints=BlueprintPolicyConstraints(allowed_actions=["web_search"], max_iterations=3),
        required_checks=[BlueprintRequiredCheck(id=f"{name}_check", description=f"{name} was checked.")],
        verification_graph=BlueprintVerificationGraph(
            start_node="n1",
            nodes=[
                BlueprintActionNode(
                    id="n1",
                    type="actions",
                    actions=[BlueprintAction(action="web_search", intent=f"verify {name}", query_guidance="q")],
                    transition=[],
                ),
            ],
        ),
    )


def test_routing_description_combines_description_and_examples() -> None:
    bp = _blueprint("quotes", "Verifies attributed quotes.", ["Politician X said Y.", "CEO Z claimed W.", "third"])
    text = routing_description(bp)
    assert text.startswith("Verifies attributed quotes.")
    assert "Typical claims: Politician X said Y. | CEO Z claimed W." in text
    assert "third" not in text  # capped at two examples


def test_router_emits_descriptions_with_fallback_last() -> None:
    tree = MergedStrategyTree()
    tree.allowed_actions = ["web_search"]
    fallback = EntryBranch("generic", BlueprintEntryConditions(), MergeNode("g/n1", "synthesis"), description="")
    quotes = EntryBranch(
        "quotes",
        BlueprintEntryConditions(any=[BlueprintCondition(feature="has_claim_text", op="==", value=True)]),
        MergeNode("q/n1", "synthesis"),
        description="Claims attributing a quote to a named person.",
    )
    tree.entries = [fallback, quotes]  # fallback deliberately first

    bp = tree.to_blueprint("merged", "desc")
    router = next(n for n in bp.verification_graph.nodes if n.id == "router")
    conditions = [t.if_ for t in router.transition]
    assert conditions[0] == "Claims attributing a quote to a named person."
    assert "only if none of the other branches fits" in conditions[-1]
    targets = [t.to for t in router.transition]
    assert targets == ["q/n1", "g/n1"]


def test_merge_seeds_folds_and_matches_on_descriptions() -> None:
    # Scripted seams. generic (fallback) and quotes open branches without model
    # calls (no non-fallback candidates yet); statements then triggers, in order:
    # entry match -> branch alignment during the fold -> description fold.
    model = ScriptedModel(
        [
            json.dumps({"match_index": 0, "rationale": "same strategy"}),
            json.dumps({"pairs": [], "unmatched_signal": [0]}),
            json.dumps({"description": "Handles quotes AND statements."}),
        ]
    )
    merger = BlueprintTreeMerger(model, reconcile=False, consolidate=False, sharpen=False)
    generic = _blueprint("generic", "Generic fallback.", gated=False)
    quotes = _blueprint("quotes", "Verifies attributed quotes.", ["X said Y."])
    statements = _blueprint("statements", "Verifies official statements.")

    result = merger.merge([generic, quotes, statements], name="merged")

    tree_by_label = {e.label: e for e in result.tree.entries}
    assert set(tree_by_label) == {"generic", "quotes"}
    assert tree_by_label["quotes"].description == "Handles quotes AND statements."
    assert tree_by_label["generic"].is_fallback
    # match_entry prompts carried descriptions, not boolean gates
    entry_prompts = [c for c in model.calls if "Existing branches" in c]
    assert all("has_claim_text" not in p for p in entry_prompts)
    assert any("Verifies attributed quotes." in p for p in entry_prompts)


def test_sharpen_router_applies_only_on_count_match() -> None:
    good = json.dumps({"descriptions": ["sharp A", "sharp B"]})
    bad = json.dumps({"descriptions": ["only one"]})
    matcher = BranchMatcher(ScriptedModel([good]))
    assert matcher.sharpen_router(["a", "b"]) == ["sharp A", "sharp B"]
    matcher_bad = BranchMatcher(ScriptedModel([bad, bad]))  # second for repair attempt
    assert matcher_bad.sharpen_router(["a", "b"]) == ["a", "b"]


def test_applicability_escape_added_once() -> None:
    checks = [
        BlueprintRequiredCheck(id="a", description="Link destination was traced."),
        BlueprintRequiredCheck(id="b", description="Record checked. (Mark UNCHECKED when no record.)"),
    ]
    out = _with_applicability_escape(checks)
    assert "Mark UNCHECKED" in out[0].description
    assert out[1].description.count("UNCHECKED") == 1
