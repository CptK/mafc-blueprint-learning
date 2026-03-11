from datetime import date, datetime

import pytest
from ezmm import MultimodalSequence, Video

from mafc.tools.web_search.common import Query, SearchMode, SearchResults, Source, WebSource


def test_query_requires_text_or_image() -> None:
    with pytest.raises(AssertionError, match="at least one"):
        Query()


def test_query_helpers_and_time_properties() -> None:
    query = Query(
        text="hello",
        search_mode=SearchMode.NEWS,
        limit=3,
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 2),
    )

    assert query.has_text() is True
    assert query.has_media() is False
    assert query.has_image() is False
    assert query.has_video() is False
    assert query.start_time == datetime(2025, 1, 1, 0, 0, 0)
    assert query.end_time == datetime(2025, 1, 2, 23, 59, 59, 999999)
    assert query == Query(
        text="hello",
        search_mode=SearchMode.NEWS,
        limit=3,
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 2),
    )
    assert len({query, Query(text="hello")}) == 2


def test_query_accepts_video_media() -> None:
    query = Query(media=Video(binary_data=b"video-bytes"))

    assert query.has_media() is True
    assert query.has_image() is False
    assert query.has_video() is True


def test_source_relevance_and_string() -> None:
    unloaded = Source(reference="ref://x")
    assert unloaded.is_relevant() is None
    assert "not yet loaded" in str(unloaded)

    irrelevant = Source(
        reference="ref://y",
        content=MultimodalSequence("full"),
        takeaways=MultimodalSequence("NONE"),
    )
    assert irrelevant.is_relevant() is False
    assert "Content: full" in str(irrelevant)

    relevant = Source(
        reference="ref://z",
        content=MultimodalSequence("full"),
        takeaways=MultimodalSequence("key points"),
    )
    assert relevant.is_relevant() is True
    assert "Takeaways: key points" in str(relevant)


def test_web_source_string_equality_and_hash() -> None:
    web_source = WebSource(
        reference="https://example.com/a",
        title="Title",
        release_date=date(2024, 5, 3),
        preview="Preview",
        content=MultimodalSequence("body"),
    )

    text = str(web_source)
    assert "Web Source https://example.com/a" in text
    assert "Title: Title" in text
    assert "Release Date: May 03, 2024" in text
    assert "Preview" in text
    assert "Content: body" in text

    assert web_source == WebSource(reference="https://example.com/a")
    assert len({web_source, WebSource(reference="https://example.com/a")}) == 1


def test_search_results_rendering() -> None:
    query = Query(text="q")
    empty = SearchResults(sources=[], query=query)
    assert str(empty) == "No search results found."

    non_empty = SearchResults(sources=[Source(reference="ref://1")], query=query)
    assert "**Search Results**" in str(non_empty)
    assert "Source ref://1" in str(non_empty)
