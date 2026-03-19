from __future__ import annotations

from ezmm import MultimodalSequence

from mafc.agents.media.utils import build_evidences_from_tool_result
from mafc.tools.geolocate.geolocate import Geolocate, GeolocationResults
from mafc.tools.tool_result import ToolResult
from mafc.tools.web_search.common import Query, WebSource
from mafc.tools.web_search.google_vision import GoogleRisResults
from mafc.tools.web_search.reverse_image_search import ReverseImageSearch

MEDIA_REF = "<<image:1>>"


def _ris_result(sources: list[WebSource], takeaways: MultimodalSequence | None = None) -> ToolResult:
    return ToolResult(
        raw=GoogleRisResults(sources=sources, query=Query(text="seed"), entities={}, best_guess_labels=[]),
        action=ReverseImageSearch(MEDIA_REF),
        takeaways=takeaways,
    )


def _geo_result(text: str = "Most likely: Greece") -> ToolResult:
    return ToolResult(
        raw=GeolocationResults(text=text, most_likely_location="Greece", top_k_locations=["Greece"]),
        action=Geolocate(MEDIA_REF),
        takeaways=MultimodalSequence(text),
    )


# --- RIS with sources ---


def test_ris_with_multiple_sources_produces_one_evidence_per_source() -> None:
    sources = [
        WebSource(reference="https://example.com/a", title="A"),
        WebSource(reference="https://example.com/b", title="B"),
    ]
    evidences = build_evidences_from_tool_result(_ris_result(sources), MEDIA_REF)

    assert len(evidences) == 2
    assert evidences[0].source == "https://example.com/a"
    assert evidences[1].source == "https://example.com/b"


def test_ris_with_single_source_produces_one_evidence() -> None:
    sources = [WebSource(reference="https://example.com/a", title="A")]
    evidences = build_evidences_from_tool_result(_ris_result(sources), MEDIA_REF)

    assert len(evidences) == 1
    assert evidences[0].source == "https://example.com/a"


def test_ris_with_takeaways_includes_takeaways_text_in_each_evidence() -> None:
    sources = [
        WebSource(reference="https://example.com/a", title="A"),
        WebSource(reference="https://example.com/b", title="B"),
    ]
    takeaways = MultimodalSequence("found on two sites")
    evidences = build_evidences_from_tool_result(_ris_result(sources, takeaways=takeaways), MEDIA_REF)

    assert all(e.takeaways is not None for e in evidences)
    assert all("found on two sites" in str(e.takeaways) for e in evidences)


def test_ris_source_action_matches_original_tool_result() -> None:
    sources = [WebSource(reference="https://example.com/a", title="A")]
    result = _ris_result(sources)
    evidences = build_evidences_from_tool_result(result, MEDIA_REF)

    assert evidences[0].action is result.action


# --- RIS with no sources (fallback) ---


def test_ris_with_no_sources_falls_back_to_single_evidence_with_media_reference() -> None:
    evidences = build_evidences_from_tool_result(_ris_result([]), MEDIA_REF)

    assert len(evidences) == 1
    assert evidences[0].source == MEDIA_REF


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
