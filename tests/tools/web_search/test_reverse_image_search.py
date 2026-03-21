from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from ezmm import Image
from ezmm.common.registry import item_registry

from mafc.tools.web_search.common import Query
from mafc.tools.web_search.google_vision import GoogleRisResults, GoogleVisionAPI
from mafc.tools.web_search.reverse_image_search import ReverseImageSearch, ReverseImageSearchTool

ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets"


class FakeVisionAPI(GoogleVisionAPI):
    def __init__(self, result: GoogleRisResults):
        # Skip GoogleVisionAPI.__init__ to avoid real API connection
        self.result = result
        self.queries: list[Query] = []

    def search(self, query: Query) -> GoogleRisResults:
        self.queries.append(query)
        return self.result


def test_reverse_image_search_tool_performs_registered_media_lookup() -> None:
    image = Image(file_path=ASSETS_DIR / "Greece.jpeg")
    item_registry.add_item(image)
    expected = GoogleRisResults(
        sources=[],
        query=Query(text="seed", media=cast(Any, object())),
        entities={"Mountain": 0.8},
        best_guess_labels=["Alps"],
    )
    api = FakeVisionAPI(expected)
    tool = ReverseImageSearchTool(api=api)

    out = tool.perform(ReverseImageSearch(image.reference))

    assert out.raw is expected
    assert out.takeaways is not None
    assert "Reverse Image Search Results" in str(out.takeaways)
    assert len(api.queries) == 1
    assert api.queries[0].media is not None


def test_reverse_image_search_tool_returns_empty_result_for_missing_media() -> None:
    tool = ReverseImageSearchTool(
        api=FakeVisionAPI(
            GoogleRisResults(
                sources=[],
                query=Query(text="seed", media=cast(Any, object())),
                entities={},
                best_guess_labels=[],
            )
        )
    )

    out = tool.perform(ReverseImageSearch("<image:999999>"))

    assert isinstance(out.raw, GoogleRisResults)
    raw = out.raw
    assert raw.sources == []
    assert raw.entities == {}
    assert raw.best_guess_labels == []
    assert out.takeaways is None


def test_reverse_image_search_tool_returns_none_summary_for_empty_results() -> None:
    image = Image(file_path=ASSETS_DIR / "Greece.jpeg")
    item_registry.add_item(image)
    empty_result = GoogleRisResults(
        sources=[],
        query=Query(text="seed", media=cast(Any, object())),
        entities={},
        best_guess_labels=[],
    )
    tool = ReverseImageSearchTool(api=FakeVisionAPI(empty_result))

    out = tool.perform(ReverseImageSearch(image.reference))

    assert out.raw is empty_result
    assert out.takeaways is None
