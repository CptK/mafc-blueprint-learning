from __future__ import annotations

import json
from unittest.mock import MagicMock

from mafc.blueprints.models import (
    Blueprint,
    BlueprintAction,
    BlueprintActionNode,
    BlueprintEntryConditions,
    BlueprintPolicyConstraints,
    BlueprintRequiredCheck,
    BlueprintSelectorHints,
    BlueprintSynthesisNode,
    BlueprintVerificationGraph,
    ClaimFeatures,
)
from mafc.learning.blueprint_fit_assessor import (
    BlueprintFitAssessor,
    _parse_fit_response,
)
from mafc.learning.models import ActionEvidenceLink, ArticleAnalysis

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_blueprint(
    name: str = "media_location",
    description: str = "Investigate location-oriented image claims.",
    allowed_actions: list[str] | None = None,
) -> Blueprint:
    return Blueprint(
        name=name,
        description=description,
        entry_conditions=BlueprintEntryConditions(),
        selector_hints=BlueprintSelectorHints(),
        policy_constraints=BlueprintPolicyConstraints(
            allowed_actions=allowed_actions or ["reverse_image_search", "geolocation"],
            max_iterations=3,
        ),
        required_checks=[
            BlueprintRequiredCheck(id="confirm_location", description="Verified location matches claim."),
        ],
        verification_graph=BlueprintVerificationGraph(
            start_node="find_origin",
            nodes=[
                BlueprintActionNode(
                    id="find_origin",
                    type="actions",
                    actions=[
                        BlueprintAction(
                            action="reverse_image_search",
                            intent="Find the original context of the image.",
                            query_guidance="Search using the main subject of the image.",
                        )
                    ],
                    transition=[],
                ),
                BlueprintSynthesisNode(id="synth", type="synthesis", transition=[]),
            ],
        ),
    )


def _make_claim_features(has_image: bool = True) -> ClaimFeatures:
    return ClaimFeatures(
        has_claim_text=True,
        text_length=40,
        has_image=has_image,
        image_count=1 if has_image else 0,
        has_video=False,
        video_count=0,
        is_multimodal=has_image,
        has_url=False,
        has_date=False,
        has_question=True,
    )


def _make_article_analysis(**overrides) -> ArticleAnalysis:
    base = dict(
        claim_type="media_authenticity",
        verdict_summary="The image was taken in Greece, not in a conflict zone.",
        key_evidence=["Reverse image search confirmed location.", "EXIF data matches Greece."],
        evidence_types=["reverse_image_search", "geolocation"],
        action_evidence_links=[
            ActionEvidenceLink(
                action="reverse_image_search",
                finding="Image originates from a 2019 tourism campaign.",
                query_or_input="image of ruins",
                was_decisive=True,
            )
        ],
        investigative_steps=["Ran reverse image search.", "Cross-referenced EXIF metadata."],
        search_queries=["Greece ruins tourism 2019"],
        process_richness="full",
        notes=None,
    )
    base.update(overrides)
    return ArticleAnalysis(**base)  # type: ignore[arg-type]


def _make_model(response_text: str) -> MagicMock:
    model = MagicMock()
    response = MagicMock()
    response.text = response_text
    model.generate.return_value = response
    return model


def _valid_llm_response(**overrides) -> str:
    base = {
        "fit_level": "good",
        "needs_new_blueprint": False,
        "covered_capabilities": ["reverse_image_search", "geolocation"],
        "missing_capabilities": [],
        "reason": "The blueprint's approach matches the claim's verification needs.",
    }
    base.update(overrides)
    return json.dumps(base)


# ---------------------------------------------------------------------------
# _parse_fit_response
# ---------------------------------------------------------------------------


def test_parse_valid_good_response() -> None:
    result = _parse_fit_response(_valid_llm_response())
    assert result is not None
    assert result.fit_level == "good"
    assert result.needs_new_blueprint is False
    assert result.covered_capabilities == ["reverse_image_search", "geolocation"]
    assert result.missing_capabilities == []
    assert "blueprint" in result.reason.lower()


def test_parse_valid_poor_response_with_missing_capabilities() -> None:
    result = _parse_fit_response(
        _valid_llm_response(
            fit_level="poor",
            needs_new_blueprint=True,
            covered_capabilities=[],
            missing_capabilities=["metadata_analysis", "expert_consultation"],
            reason="Blueprint lacks metadata analysis.",
        )
    )
    assert result is not None
    assert result.fit_level == "poor"
    assert result.needs_new_blueprint is True
    assert "metadata_analysis" in result.missing_capabilities


def test_parse_strips_json_fences() -> None:
    fenced = "```json\n" + _valid_llm_response() + "\n```"
    result = _parse_fit_response(fenced)
    assert result is not None
    assert result.fit_level == "good"


def test_parse_extracts_json_from_surrounding_text() -> None:
    wrapped = "Here is my assessment:\n" + _valid_llm_response() + "\nDone."
    result = _parse_fit_response(wrapped)
    assert result is not None
    assert result.fit_level == "good"


def test_parse_invalid_json_returns_none() -> None:
    assert _parse_fit_response("not json") is None
    assert _parse_fit_response("") is None
    assert _parse_fit_response("{incomplete") is None


def test_parse_invalid_fit_level_returns_none() -> None:
    bad = _valid_llm_response(fit_level="excellent")
    assert _parse_fit_response(bad) is None


def test_parse_empty_capability_lists_are_accepted() -> None:
    result = _parse_fit_response(_valid_llm_response(covered_capabilities=[], missing_capabilities=[]))
    assert result is not None
    assert result.covered_capabilities == []
    assert result.missing_capabilities == []


# ---------------------------------------------------------------------------
# BlueprintFitAssessor.assess — prompt construction
# ---------------------------------------------------------------------------


def test_assess_includes_claim_text_in_prompt() -> None:
    model = _make_model(_valid_llm_response())
    assessor = BlueprintFitAssessor(model)
    assessor.assess(
        blueprint=_make_blueprint(),
        claim="Where was this image taken?",
        claim_features=_make_claim_features(),
    )
    prompt = model.generate.call_args[0][0][1].content.data[0]
    assert "Where was this image taken?" in prompt


def test_assess_includes_blueprint_yaml_in_prompt() -> None:
    model = _make_model(_valid_llm_response())
    assessor = BlueprintFitAssessor(model)
    assessor.assess(
        blueprint=_make_blueprint(name="media_location"),
        claim="Where was this image taken?",
        claim_features=_make_claim_features(),
    )
    prompt = model.generate.call_args[0][0][1].content.data[0]
    assert "media_location" in prompt
    assert "reverse_image_search" in prompt
    assert "Find the original context of the image" in prompt


def test_assess_includes_article_analysis_when_provided() -> None:
    model = _make_model(_valid_llm_response())
    assessor = BlueprintFitAssessor(model)
    analysis = _make_article_analysis()
    assessor.assess(
        blueprint=_make_blueprint(),
        claim="Where was this image taken?",
        claim_features=_make_claim_features(),
        article_analysis=analysis,
    )
    prompt = model.generate.call_args[0][0][1].content.data[0]
    assert "media_authenticity" in prompt
    assert "reverse_image_search" in prompt
    assert "Greece" in prompt


def test_assess_omits_article_analysis_section_when_not_provided() -> None:
    model = _make_model(_valid_llm_response())
    assessor = BlueprintFitAssessor(model)
    assessor.assess(
        blueprint=_make_blueprint(),
        claim="Where was this image taken?",
        claim_features=_make_claim_features(),
        article_analysis=None,
    )
    prompt = model.generate.call_args[0][0][1].content.data[0]
    assert "ARTICLE ANALYSIS" not in prompt
    assert "No ground-truth article analysis" in prompt


def test_assess_includes_claim_features_yaml_in_prompt() -> None:
    model = _make_model(_valid_llm_response())
    assessor = BlueprintFitAssessor(model)
    assessor.assess(
        blueprint=_make_blueprint(),
        claim="Where was this image taken?",
        claim_features=_make_claim_features(has_image=True),
    )
    prompt = model.generate.call_args[0][0][1].content.data[0]
    assert "has_image" in prompt


# ---------------------------------------------------------------------------
# BlueprintFitAssessor.assess — result handling
# ---------------------------------------------------------------------------


def test_assess_returns_parsed_result_on_valid_response() -> None:
    model = _make_model(_valid_llm_response())
    assessor = BlueprintFitAssessor(model)
    result = assessor.assess(
        blueprint=_make_blueprint(),
        claim="Where was this image taken?",
        claim_features=_make_claim_features(),
    )
    assert result is not None
    assert result.fit_level == "good"
    assert result.needs_new_blueprint is False
    assert model.generate.call_count == 1


def test_assess_attaches_llm_prompt_and_raw_response() -> None:
    raw = _valid_llm_response()
    model = _make_model(raw)
    assessor = BlueprintFitAssessor(model)
    result = assessor.assess(
        blueprint=_make_blueprint(),
        claim="Where was this image taken?",
        claim_features=_make_claim_features(),
    )
    assert result is not None
    assert result.llm_prompt is not None
    assert "Where was this image taken?" in result.llm_prompt
    assert result.llm_raw_response == raw


def test_assess_attempts_repair_on_invalid_json() -> None:
    call_count = 0
    valid = _valid_llm_response(fit_level="partial", needs_new_blueprint=True)

    def fake_generate(messages):
        nonlocal call_count
        call_count += 1
        resp = MagicMock()
        resp.text = "not json" if call_count == 1 else valid
        return resp

    model = MagicMock()
    model.generate.side_effect = fake_generate
    assessor = BlueprintFitAssessor(model)
    result = assessor.assess(
        blueprint=_make_blueprint(),
        claim="Where was this image taken?",
        claim_features=_make_claim_features(),
    )
    assert result is not None
    assert result.fit_level == "partial"
    assert call_count == 2


def test_assess_returns_none_when_repair_also_fails(monkeypatch) -> None:
    model = _make_model("still not json")
    warnings: list[str] = []
    monkeypatch.setattr("mafc.learning.blueprint_fit_assessor.logger.warning", warnings.append)

    assessor = BlueprintFitAssessor(model)
    result = assessor.assess(
        blueprint=_make_blueprint(),
        claim="Where was this image taken?",
        claim_features=_make_claim_features(),
    )
    assert result is None
    assert any("Failed to parse" in w for w in warnings)
