from __future__ import annotations

from mafc.blueprints.features import evaluate_entry_conditions, extract_claim_features
from mafc.blueprints.models import BlueprintCondition, BlueprintEntryConditions
from mafc.blueprints.semantic_features import SemanticFeatureExtractor
from mafc.common.modeling.message import Message
from mafc.common.modeling.model import Model, Response


class StubModel(Model):
    def __init__(self, output: str):
        super().__init__(specifier="OPENAI:gpt-5-mini-2025-08-07")
        self.output = output
        self.calls = 0

    def _do_generate(self, messages: list[Message]) -> Response:
        self.calls += 1
        return Response(text=self.output, total_cost=0.0)


def _gate(feature: str, value: bool) -> BlueprintEntryConditions:
    return BlueprintEntryConditions(all=[BlueprintCondition(feature=feature, op="==", value=value)])


def test_semantic_features_default_to_undetermined():
    features = extract_claim_features("A claim with no extractor run.")
    assert features.asserts_synthetic_origin is None
    assert features.is_statistical is None


def test_undetermined_semantic_feature_never_eliminates():
    """The safety property: a failed extraction must degrade to the tie-break."""
    features = extract_claim_features("Some claim.")
    for value in (True, False):
        matched, reasons = evaluate_entry_conditions(features, _gate("asserts_synthetic_origin", value))
        assert matched, reasons


def test_determined_semantic_feature_does_eliminate():
    features = extract_claim_features("Some claim.", {"asserts_synthetic_origin": False})
    matched, _ = evaluate_entry_conditions(features, _gate("asserts_synthetic_origin", True))
    assert not matched

    matched, _ = evaluate_entry_conditions(features, _gate("asserts_synthetic_origin", False))
    assert matched


def test_structural_feature_still_eliminates_when_absent():
    """Undetermined-passes applies only to semantic features, not structural ones."""
    features = extract_claim_features("Text-only claim.")
    matched, _ = evaluate_entry_conditions(features, _gate("has_image", True))
    assert not matched


def test_extractor_maps_tristate_values():
    model = StubModel(
        '{"asserts_place_or_date":"yes","asserts_identity":"no",'
        '"asserts_synthetic_origin":"unknown","asserts_recontextualization":"no",'
        '"is_document_screenshot":"no","is_quote_attribution":"no",'
        '"is_statistical":"no","is_scientific_medical":"no"}'
    )
    values = SemanticFeatureExtractor(model).extract("Esta foto fue tomada en Madrid en 2024.")
    assert values["asserts_place_or_date"] is True
    assert values["asserts_identity"] is False
    assert values["asserts_synthetic_origin"] is None


def test_extractor_fails_open_on_bad_json():
    values = SemanticFeatureExtractor(StubModel("not json at all")).extract("A claim.")
    assert values == {}

    features = extract_claim_features("A claim.", values)
    matched, _ = evaluate_entry_conditions(features, _gate("asserts_identity", True))
    assert matched


def test_unknown_feature_names_are_ignored():
    features = extract_claim_features("A claim.", {"not_a_feature": True})
    assert not hasattr(features, "not_a_feature")


def test_extractor_retries_before_giving_up():
    class FlakyModel(StubModel):
        def _do_generate(self, messages):
            self.calls += 1
            text = "" if self.calls == 1 else self.output
            return Response(text=text, total_cost=0.0)

    model = FlakyModel(
        '{"asserts_place_or_date":"yes","asserts_identity":"no",'
        '"asserts_synthetic_origin":"no","asserts_recontextualization":"no",'
        '"is_document_screenshot":"no","is_quote_attribution":"no",'
        '"is_statistical":"no","is_scientific_medical":"no"}'
    )
    values = SemanticFeatureExtractor(model).extract("Bir iddia.")
    assert model.calls == 2
    assert values["asserts_place_or_date"] is True
