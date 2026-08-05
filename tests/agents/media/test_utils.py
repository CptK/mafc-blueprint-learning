from __future__ import annotations

from ezmm import MultimodalSequence

from mafc.agents.media.utils import build_evidences_from_tool_result
from mafc.common.media_referent import extract_referent_digest
from mafc.tools.geolocate.geolocate import Geolocate, GeolocationResults
from mafc.tools.tool_result import ToolResult
from mafc.tools.web_search.common import Query, WebSource
from mafc.tools.web_search.google_vision import (
    EXACT_MATCH_NOTE,
    PARTIAL_MATCH_NOTE,
    UNKNOWN_MATCH_NOTE,
    GoogleRisResults,
)
from mafc.tools.web_search.reverse_image_search import ReverseImageSearch

MEDIA_REF = "<<image:1>>"

# The aggregate block the RIS tool puts in `takeaways`: entity tags describe Google's
# index neighbourhood, not the media. It must never reach an Evidence.
AGGREGATE_TAKEAWAYS = (
    "**Reverse Image Search Results**\n\n"
    "Possible identified entities:\n- light-hearted\n- event\n- Festival\n\n"
    "Best guess about the topic of the image: tire, event, crowd, toddler."
)


def _ris_result(sources: list[WebSource], takeaways: MultimodalSequence | None = None) -> ToolResult:
    return ToolResult(
        raw=GoogleRisResults(
            sources=sources,
            query=Query(text="seed"),
            entities={"light-hearted": 0.9, "Festival": 0.8},
            best_guess_labels=["tire", "event", "crowd", "toddler"],
        ),
        action=ReverseImageSearch(MEDIA_REF),
        takeaways=takeaways,
    )


def _geo_result(text: str = "Most likely: Greece") -> ToolResult:
    return ToolResult(
        raw=GeolocationResults(text=text, most_likely_location="Greece", top_k_locations=["Greece"]),
        action=Geolocate(MEDIA_REF),
        takeaways=MultimodalSequence(text),
    )


def _exact(url: str, title: str | None = None) -> WebSource:
    return WebSource(reference=url, title=title, preview=EXACT_MATCH_NOTE)


def _partial(url: str, title: str | None = None) -> WebSource:
    return WebSource(reference=url, title=title, preview=PARTIAL_MATCH_NOTE)


def _unknown(url: str, title: str | None = None) -> WebSource:
    return WebSource(reference=url, title=title, preview=UNKNOWN_MATCH_NOTE)


# --- RIS with confirmed matches ---


def test_ris_with_multiple_confirmed_sources_produces_one_evidence_per_source() -> None:
    sources = [_exact("https://example.com/a", "A"), _partial("https://example.com/b", "B")]
    evidences = build_evidences_from_tool_result(_ris_result(sources), MEDIA_REF)

    assert len(evidences) == 2
    assert evidences[0].source == "https://example.com/a"
    assert evidences[1].source == "https://example.com/b"


def test_ris_with_single_confirmed_source_produces_one_evidence() -> None:
    evidences = build_evidences_from_tool_result(_ris_result([_exact("https://example.com/a")]), MEDIA_REF)

    assert len(evidences) == 1
    assert evidences[0].source == "https://example.com/a"


def test_confirmed_evidence_carries_only_its_own_match_line() -> None:
    """Regression: every per-source evidence used to carry the whole aggregate block,
    so one API call looked like N independent findings to the judge."""
    sources = [_exact("https://example.com/a", "A"), _partial("https://example.com/b", "B")]
    takeaways = MultimodalSequence(AGGREGATE_TAKEAWAYS)
    evidences = build_evidences_from_tool_result(_ris_result(sources, takeaways=takeaways), MEDIA_REF)

    first, second = (str(e.takeaways) for e in evidences)
    # Each item mentions its own page and match type, and not the other page's.
    assert "https://example.com/a" in first and "https://example.com/b" not in first
    assert "https://example.com/b" in second and "https://example.com/a" not in second
    assert "EXACT" in first
    assert "PARTIAL" in second


def test_confirmed_evidence_never_carries_entity_tags() -> None:
    sources = [_exact("https://example.com/a"), _exact("https://example.com/b")]
    takeaways = MultimodalSequence(AGGREGATE_TAKEAWAYS)
    evidences = build_evidences_from_tool_result(_ris_result(sources, takeaways=takeaways), MEDIA_REF)

    for evidence in evidences:
        blob = f"{evidence.raw}\n{evidence.takeaways}"
        for tag in ("light-hearted", "Festival", "best guess", "Best guess", "toddler"):
            assert tag not in blob


def test_confirmed_evidences_stay_useful_and_keep_the_action() -> None:
    result = _ris_result([_exact("https://example.com/a")])
    evidences = build_evidences_from_tool_result(result, MEDIA_REF)

    assert evidences[0].is_useful()
    assert evidences[0].action is result.action


def test_unconfirmed_sources_are_dropped_alongside_confirmed_ones() -> None:
    sources = [
        _unknown("https://lookalike.example/one"),
        _exact("https://example.com/a"),
        _unknown("https://lookalike.example/two"),
    ]
    evidences = build_evidences_from_tool_result(_ris_result(sources), MEDIA_REF)

    assert [e.source for e in evidences] == ["https://example.com/a"]


# --- RIS with no confirmed match (the trace-102031 failure mode) ---


def test_all_unknown_precision_collapses_to_single_negative_evidence() -> None:
    sources = [_unknown(f"https://lookalike.example/{i}") for i in range(6)]
    takeaways = MultimodalSequence(AGGREGATE_TAKEAWAYS)
    evidences = build_evidences_from_tool_result(_ris_result(sources, takeaways=takeaways), MEDIA_REF)

    assert len(evidences) == 1
    assert evidences[0].source == MEDIA_REF


def test_negative_evidence_carries_no_entity_tags_or_lookalike_urls() -> None:
    sources = [_unknown(f"https://lookalike.example/{i}") for i in range(6)]
    takeaways = MultimodalSequence(AGGREGATE_TAKEAWAYS)
    evidences = build_evidences_from_tool_result(_ris_result(sources, takeaways=takeaways), MEDIA_REF)

    blob = f"{evidences[0].raw}\n{evidences[0].takeaways}"
    for tag in ("light-hearted", "Festival", "best guess", "Best guess", "toddler"):
        assert tag not in blob
    assert "lookalike.example" not in blob


def test_negative_evidence_stays_visible_to_the_fact_check_planner() -> None:
    """`takeaways=None` would make the planner drop this line entirely
    (fact_check.prompts._planner_summary_for_evidence) and flip is_useful() to False."""
    evidences = build_evidences_from_tool_result(
        _ris_result([_unknown("https://lookalike.example/one")]), MEDIA_REF
    )

    assert evidences[0].takeaways is not None
    assert str(evidences[0].takeaways).strip()
    assert evidences[0].is_useful()


def test_negative_evidence_states_the_absence_and_the_lookalike_count() -> None:
    sources = [_unknown(f"https://lookalike.example/{i}") for i in range(6)]
    evidences = build_evidences_from_tool_result(_ris_result(sources), MEDIA_REF)

    text = str(evidences[0].takeaways)
    assert "No provenance could be established" in text
    assert "6 visually-similar page(s)" in text


def test_negative_evidence_sets_no_match_reported_in_referent_digest() -> None:
    """The negative note must keep media_referent's marker wording verbatim."""
    evidences = build_evidences_from_tool_result(
        _ris_result([_unknown("https://lookalike.example/one")]), MEDIA_REF
    )
    digest = extract_referent_digest(evidences)

    assert digest.ris_ran
    assert digest.no_match_reported
    assert not digest.exact
    assert not digest.partial


def test_confirmed_matches_are_classified_by_the_referent_digest() -> None:
    sources = [_exact("https://example.com/a"), _partial("https://example.com/b")]
    evidences = build_evidences_from_tool_result(_ris_result(sources), MEDIA_REF)
    digest = extract_referent_digest(evidences)

    assert digest.classify("https://example.com/a") == "exact"
    assert digest.classify("https://example.com/b") == "partial"
    assert not digest.no_match_reported


def test_confirmed_matches_store_referent_status_on_the_evidence() -> None:
    """Match precision is recorded structurally, not left to be re-parsed from prose."""
    sources = [_exact("https://example.com/a"), _partial("https://example.com/b")]
    evidences = build_evidences_from_tool_result(_ris_result(sources), MEDIA_REF)

    assert {e.source: e.referent for e in evidences} == {
        "https://example.com/a": "exact",
        "https://example.com/b": "partial",
    }


def test_urls_ending_in_a_parenthesis_stay_classifiable() -> None:
    """Wikipedia-style URLs survive the structured path.

    The legacy text parser strips trailing punctuation from the URLs it scrapes
    out of prose, which truncates a closing ')' that is part of the path. The key
    it stored could then never match the evidence item's real source, so the page
    silently went untagged. Reading the source directly removes the ambiguity.
    """
    url = "https://en.wikipedia.org/wiki/C_(New_York_City_Subway_service)"
    evidences = build_evidences_from_tool_result(_ris_result([_exact(url)]), MEDIA_REF)

    assert extract_referent_digest(evidences).classify(url) == "exact"


def test_stored_referent_survives_a_digest_built_without_match_text() -> None:
    """The structured path alone reproduces the digest, with the prose stripped out."""
    sources = [_exact("https://example.com/a"), _partial("https://example.com/b")]
    evidences = build_evidences_from_tool_result(_ris_result(sources), MEDIA_REF)
    for evidence in evidences:
        evidence.raw = MultimodalSequence("(match note removed)")
        evidence.takeaways = MultimodalSequence("(match note removed)")

    digest = extract_referent_digest(evidences)
    assert digest.classify("https://example.com/a") == "exact"
    assert digest.classify("https://example.com/b") == "partial"


# --- RIS with no sources at all ---


def test_ris_with_no_sources_yields_one_negative_evidence_with_media_reference() -> None:
    evidences = build_evidences_from_tool_result(_ris_result([]), MEDIA_REF)

    assert len(evidences) == 1
    assert evidences[0].source == MEDIA_REF
    text = str(evidences[0].takeaways)
    assert "No provenance could be established" in text
    assert "No pages containing this media were found" in text


def test_ris_with_no_sources_drops_entity_tags() -> None:
    takeaways = MultimodalSequence(AGGREGATE_TAKEAWAYS)
    evidences = build_evidences_from_tool_result(_ris_result([], takeaways=takeaways), MEDIA_REF)

    blob = f"{evidences[0].raw}\n{evidences[0].takeaways}"
    for tag in ("light-hearted", "Festival", "toddler"):
        assert tag not in blob


# --- Non-RIS tool (geolocation) ---


def test_geo_result_produces_single_evidence_with_media_reference() -> None:
    evidences = build_evidences_from_tool_result(_geo_result(), MEDIA_REF)

    assert len(evidences) == 1
    assert evidences[0].source == MEDIA_REF


def test_geo_result_preserves_takeaways() -> None:
    evidences = build_evidences_from_tool_result(_geo_result("Most likely: Greece"), MEDIA_REF)

    assert evidences[0].takeaways is not None
    assert "Greece" in str(evidences[0].takeaways)


def test_geo_result_action_matches_original_tool_result() -> None:
    result = _geo_result()
    evidences = build_evidences_from_tool_result(result, MEDIA_REF)

    assert evidences[0].action is result.action
