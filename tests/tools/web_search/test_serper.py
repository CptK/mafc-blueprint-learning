from datetime import date

import pytest
import requests

from mafc.tools.web_search.common import Query
from mafc.tools.web_search.serper import (
    GoogleSearchResults,
    SerperAPI,
    _parse_answer_box,
    _parse_knowledge_graph,
    filter_unique_results_by_domain,
)


def test_google_search_results_string_and_repr() -> None:
    results = GoogleSearchResults(
        sources=[],
        query=Query(text="q"),
        answer="A",
        knowledge_graph="K",
    )
    assert str(results) == "No search results found."
    assert "has_answer=True" in repr(results)


def test_resolve_api_key_fallback(monkeypatch) -> None:
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    monkeypatch.setenv("serper_api_key", "k")
    assert SerperAPI().serper_api_key == "k"


def test_search_builds_tbs_and_calls_parser(monkeypatch) -> None:
    api = SerperAPI()
    api.serper_api_key = "k"

    captured = {}

    def fake_call(search_term, **kwargs):
        captured["search_term"] = search_term
        captured["kwargs"] = kwargs
        return {"organic": []}

    monkeypatch.setattr(api, "_call_serper_api", fake_call)
    monkeypatch.setattr(api, "_parse_results", lambda response, query: ("a", "kg", []))

    out = api.search(Query(text="term", end_date=date(2024, 2, 3)))

    assert out is not None
    assert captured["search_term"] == "term"
    assert captured["kwargs"]["search_type"] == "search"
    assert captured["kwargs"]["tbs"] == "cdr:1,cd_min:1/1/1900,cd_max:02/03/2024"


def test_call_serper_api_retries_timeout_then_succeeds(monkeypatch) -> None:
    api = SerperAPI()
    api.serper_api_key = "k"
    calls = {"n": 0}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"ok": True}

    def fake_post(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.exceptions.Timeout()
        return FakeResponse()

    monkeypatch.setattr("mafc.tools.web_search.serper.requests.post", fake_post)
    monkeypatch.setattr("mafc.tools.web_search.serper.time.sleep", lambda s: None)
    monkeypatch.setattr("mafc.tools.web_search.serper.random.uniform", lambda a, b: 1.5)

    out = api._call_serper_api("q", max_retries=2)
    assert out == {"ok": True}
    assert calls["n"] == 2


def test_call_serper_api_raises_for_credit_exhaustion(monkeypatch) -> None:
    api = SerperAPI()
    api.serper_api_key = "k"

    class FakeResponse:
        status_code = 400

        def json(self):
            return {"message": "Not enough credits"}

        def raise_for_status(self):
            return None

    monkeypatch.setattr("mafc.tools.web_search.serper.requests.post", lambda *args, **kwargs: FakeResponse())

    with pytest.raises(RuntimeError, match="No Serper API credits"):
        api._call_serper_api("q", max_retries=1)


def test_parse_helpers() -> None:
    answer = _parse_answer_box(
        {
            "answerBox": {
                "answer": "A",
                "snippet": "S",
                "link": "L",
                "snippetHighlighted": ["H"],
            }
        }
    )
    assert answer == "A\nS\nL\n['H']"

    knowledge_graph = _parse_knowledge_graph(
        {
            "knowledgeGraph": {
                "title": "T",
                "type": "Person",
                "description": "D",
                "attributes": {"Born": "1970"},
            }
        }
    )
    assert knowledge_graph == "T\nType: Person\nD\nBorn: 1970"


def test_filter_unique_results_by_domain() -> None:
    results = [
        {"link": "https://www.example.com/1"},
        {"link": "https://m.example.com/2"},
        {"link": "https://another.org/x"},
        {"title": "missing link"},
    ]
    filtered = filter_unique_results_by_domain(results)
    assert filtered == [
        {"link": "https://www.example.com/1"},
        {"link": "https://another.org/x"},
    ]


def test_parse_sources_uses_limit_and_date_parsing() -> None:
    api = SerperAPI()
    response = {
        "organic": [
            {"link": "https://a.example.com/1", "title": "A", "date": "Jan 10, 2024"},
            {"link": "https://b.example.com/2", "title": "B", "date": "bad"},
        ]
    }
    sources = api._parse_sources(response, Query(text="q", limit=1))

    assert len(sources) == 1
    assert sources[0].reference == "https://a.example.com/1"
    assert sources[0].release_date == date(2024, 1, 10)
    assert sources[0].title == "A"


def test_parse_sources_applies_strict_end_date_filter() -> None:
    api = SerperAPI()
    response = {
        "organic": [
            {"link": "https://a.example.com/1", "title": "A", "date": "Jan 10, 2024"},
            {"link": "https://b.example.com/2", "title": "B", "date": "Feb 10, 2024"},
            {"link": "https://c.example.com/3", "title": "C", "date": "bad"},
        ]
    }
    sources = api._parse_sources(response, Query(text="q", end_date=date(2024, 1, 31)))

    assert len(sources) == 1
    assert sources[0].reference == "https://a.example.com/1"
