"""Tests for the answer-key retrieval guard.

Both defences are pinned here because each covers a hole the other cannot: the
date bound misses same-day and undated sources, and the blocklist only knows the
one URL the harness hands it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from mafc.common.source_guard import filter_blocked_sources, is_blocked, normalize_blocked
from mafc.tools.web_search.common import Query, WebSource


@dataclass
class _Src:
    reference: str


REVIEW = "https://www.boomlive.in/fact-check/some-claim-30381"


def test_normalization_survives_scheme_and_www_differences():
    """1 of 274 measured leaks differed from the recorded review URL by only 'www.'."""
    blocked = normalize_blocked([REVIEW])
    assert is_blocked("http://boomlive.in/fact-check/some-claim-30381", blocked)
    assert is_blocked("https://www.boomlive.in/fact-check/some-claim-30381/", blocked)
    assert is_blocked(REVIEW + "?utm_source=x", blocked)


def test_other_pages_on_the_same_site_are_not_blocked():
    """The guard must key on the article, not the outlet."""
    blocked = normalize_blocked([REVIEW])
    assert not is_blocked("https://www.boomlive.in/fact-check/a-different-claim-99", blocked)


def test_empty_blocklist_and_missing_url_are_inert():
    assert not is_blocked(REVIEW, set())
    assert not is_blocked(None, normalize_blocked([REVIEW]))
    assert normalize_blocked(None) == set()
    assert normalize_blocked([]) == set()


def test_filter_drops_only_the_blocked_source():
    sources = [_Src("https://example.com/a"), _Src(REVIEW), _Src("https://example.com/b")]
    kept = filter_blocked_sources(sources, normalize_blocked([REVIEW]))
    assert [s.reference for s in kept] == ["https://example.com/a", "https://example.com/b"]


def test_filter_passes_everything_through_without_a_blocklist():
    sources = [_Src("https://example.com/a"), _Src(REVIEW)]
    assert len(filter_blocked_sources(sources, set())) == 2


def test_filter_handles_none_sources():
    assert filter_blocked_sources(None, normalize_blocked([REVIEW])) == []


def test_filter_logs_when_it_fires(caplog):
    """A guard that never reports cannot be distinguished from one that is not wired up."""
    filter_blocked_sources([_Src(REVIEW)], normalize_blocked([REVIEW]), context="RIS")
    assert any("Blocked 1 answer-key source" in r.message for r in caplog.records)


# --- the date fencepost ----------------------------------------------------


def _parse(api_result: dict, end_date: date | None):
    from mafc.tools.web_search.serper import SerperAPI

    api = SerperAPI.__new__(SerperAPI)  # no API key needed for parsing
    api.gl, api.hl, api.tbs = "us", "en", None
    query = Query(text="q", end_date=end_date)
    return api._parse_sources({"organic": [api_result]}, query)


def test_source_published_on_the_cutoff_day_is_excluded():
    """VeriTaS falls back to the review date, so same-day IS the answer key."""
    assert _parse({"link": "https://x.test/a", "date": "Jan 8, 2026"}, date(2026, 1, 8)) == []


def test_source_published_before_the_cutoff_is_kept():
    kept = _parse({"link": "https://x.test/a", "date": "Jan 7, 2026"}, date(2026, 1, 8))
    assert [s.reference for s in kept] == ["https://x.test/a"]


def test_undated_source_is_excluded_when_a_cutoff_is_set():
    assert _parse({"link": "https://x.test/a"}, date(2026, 1, 8)) == []


def test_undated_source_is_kept_without_a_cutoff():
    kept = _parse({"link": "https://x.test/a"}, None)
    assert [s.reference for s in kept] == ["https://x.test/a"]


# --- the guard must reach the EVIDENCE, not merely a `sources` attribute -------


def test_tool_perform_filters_before_summarizing():
    """Regression: takeaways are rendered FROM the result.

    Filtering after ``perform`` returns leaves the blocked URL in the takeaways
    text even once it is gone from ``sources`` -- and the takeaways are what
    becomes evidence. An earlier version of this guard read ``.sources`` off the
    ToolResult (where it does not exist), silently did nothing, and let 10/101
    answer keys through a live run.
    """
    from mafc.tools.web_search.common import Query
    from mafc.tools.web_search.google_vision import GoogleRisResults
    from mafc.tools.web_search.reverse_image_search import ReverseImageSearch, ReverseImageSearchTool

    class _FakeRis(ReverseImageSearchTool):
        def _perform(self, action):
            return GoogleRisResults(
                sources=[
                    WebSource(reference="https://example.com/ok", title="ok"),
                    WebSource(reference=REVIEW, title="the answer key"),
                ],
                query=Query(text="q"),
                entities={},
                best_guess_labels=[],
            )

    tool = _FakeRis.__new__(_FakeRis)
    tool.actions = [ReverseImageSearch]

    unguarded = tool.perform(ReverseImageSearch.__new__(ReverseImageSearch))
    assert REVIEW in str(unguarded.takeaways), "fixture must actually surface the URL"

    guarded = tool.perform(
        ReverseImageSearch.__new__(ReverseImageSearch), blocked_urls=normalize_blocked([REVIEW])
    )
    assert REVIEW not in str(guarded.takeaways)  # the text that becomes evidence
    assert REVIEW not in str(guarded.raw)  # the rendered raw result
    assert [s.reference for s in guarded.raw.sources] == ["https://example.com/ok"]
