from __future__ import annotations

from mafc.blueprints.gate_validation import validate_entry_gates
from mafc.blueprints.models import Blueprint
from mafc.blueprints.semantic_features import SemanticFeatureExtractor
from mafc.common.modeling.message import Message
from mafc.common.modeling.model import Model, Response


class FixedFeatureModel(Model):
    """Returns the same semantic labelling for every claim."""

    def __init__(self, **flags: str):
        super().__init__(specifier="OPENAI:gpt-5-mini-2025-08-07")
        defaults = {
            "asserts_place_or_date": "no",
            "asserts_identity": "no",
            "asserts_synthetic_origin": "no",
            "asserts_recontextualization": "no",
            "is_document_screenshot": "no",
            "is_quote_attribution": "no",
            "is_statistical": "no",
            "is_scientific_medical": "no",
        }
        defaults.update(flags)
        self.payload = defaults

    def _do_generate(self, messages: list[Message]) -> Response:
        import json

        return Response(text=json.dumps(self.payload), total_cost=0.0)


def _blueprint(*conditions: dict) -> Blueprint:
    return Blueprint.model_validate(
        {
            "name": "bp",
            "description": "d",
            "entry_conditions": {"all": list(conditions)},
            "verification_graph": {
                "start_node": "s",
                "nodes": [{"id": "s", "type": "synthesis", "transition": []}],
            },
        }
    )


CLAIMS = ["Unemployment fell to 4% last quarter.", "GDP grew 2% in Q3.", "Inflation hit 3%."]


def test_inverted_gate_is_dropped():
    """The observed bug: a statistics blueprint gated on asserts_synthetic_origin."""
    blueprint = _blueprint(
        {"feature": "asserts_synthetic_origin", "op": "==", "value": True},
    )
    extractor = SemanticFeatureExtractor(FixedFeatureModel(is_statistical="yes"))

    result = validate_entry_gates(blueprint, CLAIMS, extractor)

    assert result.repaired
    assert result.blueprint.entry_conditions.all == []
    assert result.coverage_before == 0.0
    assert result.coverage_after == 1.0


def test_correct_gate_is_kept():
    blueprint = _blueprint({"feature": "is_statistical", "op": "==", "value": True})
    extractor = SemanticFeatureExtractor(FixedFeatureModel(is_statistical="yes"))

    result = validate_entry_gates(blueprint, CLAIMS, extractor)

    assert not result.repaired
    assert len(result.blueprint.entry_conditions.all) == 1
    assert result.coverage_after == 1.0


def test_only_the_offending_condition_is_dropped():
    blueprint = _blueprint(
        {"feature": "is_statistical", "op": "==", "value": True},
        {"feature": "asserts_synthetic_origin", "op": "==", "value": True},
    )
    extractor = SemanticFeatureExtractor(FixedFeatureModel(is_statistical="yes"))

    result = validate_entry_gates(blueprint, CLAIMS, extractor)

    kept = [c.feature for c in result.blueprint.entry_conditions.all]
    assert kept == ["is_statistical"]
    assert [c.feature for c in result.dropped] == ["asserts_synthetic_origin"]


def test_undetermined_features_do_not_trigger_a_drop():
    """Tri-state unknown already passes at runtime, so it must not look like exclusion."""
    blueprint = _blueprint({"feature": "is_statistical", "op": "==", "value": True})
    extractor = SemanticFeatureExtractor(FixedFeatureModel(is_statistical="unknown"))

    result = validate_entry_gates(blueprint, CLAIMS, extractor)

    assert not result.repaired


def test_original_blueprint_is_not_mutated():
    blueprint = _blueprint({"feature": "asserts_synthetic_origin", "op": "==", "value": True})
    extractor = SemanticFeatureExtractor(FixedFeatureModel())

    result = validate_entry_gates(blueprint, CLAIMS, extractor)

    assert len(blueprint.entry_conditions.all) == 1
    assert result.blueprint is not blueprint


def test_no_claims_is_a_no_op():
    blueprint = _blueprint({"feature": "is_statistical", "op": "==", "value": True})
    result = validate_entry_gates(blueprint, [], SemanticFeatureExtractor(FixedFeatureModel()))
    assert not result.repaired
