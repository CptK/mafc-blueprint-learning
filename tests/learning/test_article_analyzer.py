import json
from unittest.mock import MagicMock


from mafc.learning.article_analyzer import (
    _parse_article_analysis,
    ArticleAnalyzer,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_valid_payload(**overrides) -> dict:
    base = {
        "claim_type": "media_authenticity",
        "verdict_summary": "The video shows a fire in Ghana, not Nigeria.",
        "key_evidence": ["Reverse image search linked to Ghana school fire."],
        "evidence_types": ["reverse_image_search", "web_search"],
        "action_evidence_links": [
            {
                "action": "reverse_image_search",
                "finding": "Video originates from Ghana, 2019.",
                "query_or_input": "video keyframes",
                "was_decisive": True,
            }
        ],
        "investigative_steps": ["Ran reverse image search on keyframes."],
        "search_queries": ["Ghana school fire 2019"],
        "process_richness": "full",
        "notes": None,
    }
    base.update(overrides)
    return base


def _make_model(response_text: str) -> MagicMock:
    model = MagicMock()
    response = MagicMock()
    response.text = response_text
    model.generate.return_value = response
    return model


# ---------------------------------------------------------------------------
# _parse_article_analysis
# ---------------------------------------------------------------------------


def test_parse_valid_full_payload() -> None:
    payload = _make_valid_payload()
    result = _parse_article_analysis(json.dumps(payload))
    assert result is not None
    assert result.claim_type == "media_authenticity"
    assert result.process_richness == "full"
    assert result.verdict_summary == "The video shows a fire in Ghana, not Nigeria."
    assert result.key_evidence == ["Reverse image search linked to Ghana school fire."]
    assert result.evidence_types == ["reverse_image_search", "web_search"]
    assert result.investigative_steps == ["Ran reverse image search on keyframes."]
    assert result.search_queries == ["Ghana school fire 2019"]
    assert result.notes is None


def test_parse_action_evidence_links() -> None:
    payload = _make_valid_payload()
    result = _parse_article_analysis(json.dumps(payload))
    assert result is not None
    assert result.action_evidence_links is not None
    assert len(result.action_evidence_links) == 1
    link = result.action_evidence_links[0]
    assert link.action == "reverse_image_search"
    assert link.finding == "Video originates from Ghana, 2019."
    assert link.query_or_input == "video keyframes"
    assert link.was_decisive is True


def test_parse_null_action_evidence_links() -> None:
    payload = _make_valid_payload(action_evidence_links=None)
    result = _parse_article_analysis(json.dumps(payload))
    assert result is not None
    assert result.action_evidence_links is None


def test_parse_null_investigative_steps_and_queries() -> None:
    payload = _make_valid_payload(investigative_steps=None, search_queries=None)
    result = _parse_article_analysis(json.dumps(payload))
    assert result is not None
    assert result.investigative_steps is None
    assert result.search_queries is None


def test_parse_result_only_process_richness() -> None:
    payload = _make_valid_payload(
        process_richness="result_only",
        action_evidence_links=None,
        investigative_steps=None,
        search_queries=None,
    )
    result = _parse_article_analysis(json.dumps(payload))
    assert result is not None
    assert result.process_richness == "result_only"


def test_parse_unknown_process_richness_falls_back_to_result_only() -> None:
    payload = _make_valid_payload(process_richness="unknown_value")
    result = _parse_article_analysis(json.dumps(payload))
    assert result is not None
    assert result.process_richness == "result_only"


def test_parse_missing_claim_type_falls_back_to_other() -> None:
    payload = _make_valid_payload()
    del payload["claim_type"]
    result = _parse_article_analysis(json.dumps(payload))
    assert result is not None
    assert result.claim_type == "other"


def test_parse_malformed_action_evidence_link_falls_back_to_none() -> None:
    payload = _make_valid_payload(action_evidence_links=[{"bad_key": "no action or finding"}])
    result = _parse_article_analysis(json.dumps(payload))
    assert result is not None
    assert result.action_evidence_links is None


def test_parse_strips_json_fences() -> None:
    payload = _make_valid_payload()
    fenced = "```json\n" + json.dumps(payload) + "\n```"
    result = _parse_article_analysis(fenced)
    assert result is not None
    assert result.claim_type == "media_authenticity"


def test_parse_extracts_json_from_surrounding_text() -> None:
    payload = _make_valid_payload()
    wrapped = "Here is my analysis:\n" + json.dumps(payload) + "\nThat's all."
    result = _parse_article_analysis(wrapped)
    assert result is not None
    assert result.claim_type == "media_authenticity"


def test_parse_invalid_json_returns_none() -> None:
    assert _parse_article_analysis("not json at all") is None
    assert _parse_article_analysis("") is None
    assert _parse_article_analysis("{incomplete") is None


# ---------------------------------------------------------------------------
# ArticleAnalyzer.analyze — prompt construction
# ---------------------------------------------------------------------------


def test_analyze_includes_claim_text_in_prompt() -> None:
    model = _make_model(json.dumps(_make_valid_payload()))
    analyzer = ArticleAnalyzer(model)
    analyzer.analyze("Some article.", claim_text="The video shows X.")

    prompt_text = model.generate.call_args[0][0][1].content.data[0]
    assert "The video shows X." in prompt_text


def test_analyze_strips_media_tokens_from_article() -> None:
    model = _make_model(json.dumps(_make_valid_payload()))
    analyzer = ArticleAnalyzer(model)
    analyzer.analyze(
        "Article text <image:123> more text <video:456> end.",
        claim_text="Some claim.",
    )
    prompt_text = model.generate.call_args[0][0][1].content.data[0]
    assert "<image:123>" not in prompt_text
    assert "<video:456>" not in prompt_text
    assert "Article text" in prompt_text
    assert "more text" in prompt_text


def test_analyze_includes_rectification_note_when_original_claim_given() -> None:
    model = _make_model(json.dumps(_make_valid_payload()))
    analyzer = ArticleAnalyzer(model)
    analyzer.analyze(
        "Some article.",
        claim_text="The video was AI-generated.",
        original_claim="The video shows a real event.",
    )
    prompt_text = model.generate.call_args[0][0][1].content.data[0]
    assert "The video shows a real event." in prompt_text
    assert "corrected version" in prompt_text


def test_analyze_omits_rectification_note_when_no_original_claim() -> None:
    model = _make_model(json.dumps(_make_valid_payload()))
    analyzer = ArticleAnalyzer(model)
    analyzer.analyze("Some article.", claim_text="The video shows X.")

    prompt_text = model.generate.call_args[0][0][1].content.data[0]
    assert "corrected version" not in prompt_text


# ---------------------------------------------------------------------------
# ArticleAnalyzer.analyze — model interaction and repair
# ---------------------------------------------------------------------------


def test_analyze_returns_parsed_result_on_valid_response() -> None:
    model = _make_model(json.dumps(_make_valid_payload()))
    analyzer = ArticleAnalyzer(model)
    result = analyzer.analyze("Some article.", claim_text="The claim.")
    assert result is not None
    assert result.claim_type == "media_authenticity"
    assert model.generate.call_count == 1


def test_analyze_attempts_repair_on_invalid_json(monkeypatch) -> None:
    valid_json = json.dumps(_make_valid_payload())
    call_count = 0

    def fake_generate(messages):
        nonlocal call_count
        call_count += 1
        resp = MagicMock()
        resp.text = "not json" if call_count == 1 else valid_json
        return resp

    model = MagicMock()
    model.generate.side_effect = fake_generate
    analyzer = ArticleAnalyzer(model)
    result = analyzer.analyze("Some article.", claim_text="The claim.")

    assert result is not None
    assert call_count == 2


def test_analyze_returns_none_when_repair_also_fails(monkeypatch) -> None:
    model = _make_model("still not json")
    warnings: list[str] = []
    monkeypatch.setattr("mafc.learning.article_analyzer.logger.warning", warnings.append)

    analyzer = ArticleAnalyzer(model)
    result = analyzer.analyze("Some article.", claim_text="The claim.")

    assert result is None
    assert any("Failed to parse" in w for w in warnings)
