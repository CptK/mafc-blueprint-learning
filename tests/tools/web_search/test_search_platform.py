import sqlite3
from datetime import date

from mafc.tools.web_search.common import Query, SearchMode, SearchResults, Source
from mafc.tools.web_search.search_platform import RemoteSearchPlatform, SearchPlatform


class DummySearchPlatform(SearchPlatform):
    name = "dummy_search"
    description = "d"

    def _call_api(self, query: Query) -> SearchResults:
        return SearchResults(sources=[Source(reference=f"ref://{query.text}")], query=query)


class DummyRemote(RemoteSearchPlatform):
    name = "dummy_remote"
    description = "d"

    def __init__(self, **kwargs):
        self.api_calls = 0
        super().__init__(**kwargs)

    def _call_api(self, query: Query) -> SearchResults:
        self.api_calls += 1
        return SearchResults(sources=[Source(reference=f"ref://{self.api_calls}")], query=query)


def test_search_platform_tracks_search_stats() -> None:
    platform = DummySearchPlatform()
    out = platform.search("abc")

    assert out is not None
    assert out.sources[0].reference == "ref://abc"
    assert platform.stats == {"Searches (API Calls)": 1}

    platform.reset()
    assert platform.stats == {"Searches (API Calls)": 0}


def test_remote_platform_uses_cache_and_tracks_hits(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("mafc.tools.web_search.search_platform.temp_dir", tmp_path)
    platform = DummyRemote(activate_cache=True)
    query = Query(text="hello")
    try:
        out1 = platform.search(query)
        out2 = platform.search(query)

        assert out1 is not None and out2 is not None
        assert out1.sources[0].reference == "ref://1"
        assert out2.sources[0].reference == "ref://1"
        assert platform.api_calls == 1
        assert platform.stats["Cache hits"] == 1
        assert platform.stats["Searches (API Calls)"] == 1
    finally:
        platform.close()


def test_remote_platform_cache_write_error_is_counted(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("mafc.tools.web_search.search_platform.temp_dir", tmp_path)
    platform = DummyRemote(activate_cache=True)

    class BrokenCursor:
        def execute(self, *args, **kwargs):
            raise sqlite3.OperationalError

    platform.cur = BrokenCursor()

    try:
        platform._add_to_cache(Query(text="q"), SearchResults(sources=[], query=Query(text="q")))
        assert platform.n_cache_write_errors == 1
    finally:
        platform.close()


def test_cache_key_is_stable_for_equivalent_queries(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("mafc.tools.web_search.search_platform.temp_dir", tmp_path)
    platform = DummyRemote(activate_cache=True)

    q1 = Query(
        text="x",
        search_mode=SearchMode.NEWS,
        limit=5,
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 2),
    )
    q2 = Query(
        text="x",
        search_mode=SearchMode.NEWS,
        limit=5,
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 2),
    )

    try:
        assert platform._cache_key(q1) == platform._cache_key(q2)
    finally:
        platform.close()
