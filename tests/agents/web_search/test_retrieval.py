from __future__ import annotations

from datetime import date

from mafc.agents.common import AgentStatus
from mafc.agents.web_search.agent import WebSearchAgent
from mafc.tools.web_search.common import Query, SearchResults, WebSource

from tests.agents.web_search.helpers import (
    FakeRetriever,
    FakeSearchTool,
    SequencedModel,
    make_search_result,
    make_session,
)


def test_collects_query_search_errors() -> None:
    class BrokenSearchTool(FakeSearchTool):
        def search(self, query: Query) -> SearchResults:
            raise RuntimeError("search down")

    planner = SequencedModel(outputs=['{"queries":["q1"],"done":true}'])
    summarizer = SequencedModel(outputs=["summary"])
    agent = WebSearchAgent(
        main_model=planner,
        summarization_model=summarizer,
        search_tool=BrokenSearchTool({}),
    )

    out = agent.run(make_session("Any task"))

    assert out.result is None
    assert out.session.status == AgentStatus.FAILED
    assert any("Search failed for query 'q1'" in e for e in out.errors)


def test_collects_retrieval_errors_but_still_produces_result() -> None:
    planner = SequencedModel(outputs=['{"queries":["q1"],"done":true}'])
    summarizer = SequencedModel(outputs=["summary"])
    search_tool = FakeSearchTool({"q1": make_search_result("q1", ["https://a.example.com"])})
    retriever = FakeRetriever({"https://a.example.com": None})
    agent = WebSearchAgent(
        main_model=planner,
        summarization_model=summarizer,
        search_tool=search_tool,
        retriever=retriever,
    )

    out = agent.run(make_session("Any task"))

    assert out.result is not None
    assert out.errors == ["Failed to retrieve content from https://a.example.com"]
    assert len(out.evidences) == 1
    assert out.evidences[0].source == "https://a.example.com"


def test_passes_end_date_to_query() -> None:
    planner = SequencedModel(outputs=['{"queries":["q1"],"done":true}'])
    summarizer = SequencedModel(outputs=["summary"])
    search_tool = FakeSearchTool(
        {
            "q1": SearchResults(
                sources=[
                    WebSource(reference="https://a.example.com", title="A", release_date=date(2024, 1, 10)),
                    WebSource(reference="https://b.example.com", title="B", release_date=date(2024, 2, 10)),
                    WebSource(reference="https://c.example.com", title="C", release_date=None),
                ],
                query=Query(text="q1"),
            )
        }
    )
    retriever = FakeRetriever(
        {
            "https://a.example.com": "older content",
            "https://b.example.com": "newer content",
            "https://c.example.com": "undated content",
        }
    )
    cutoff = date(2024, 1, 31)
    agent = WebSearchAgent(
        main_model=planner,
        summarization_model=summarizer,
        search_tool=search_tool,
        retriever=retriever,
    )

    session = make_session("Any task")
    session.cutoff_date = cutoff
    out = agent.run(session)

    assert out.result is not None
    assert search_tool.query_objects[0].end_date == cutoff
    assert sorted(retriever.calls) == sorted(
        ["https://a.example.com", "https://b.example.com", "https://c.example.com"]
    )


def test_filters_sources_with_model_when_many_candidates() -> None:
    planner = SequencedModel(
        outputs=[
            '{"queries":["q1"],"done":true}',
            '{"selected_urls":["https://b.example.com","https://d.example.com"]}',
        ]
    )
    summarizer = SequencedModel(outputs=["summary"])
    urls = [f"https://{c}.example.com" for c in "abcdef"]
    search_tool = FakeSearchTool({"q1": make_search_result("q1", urls)})
    retriever = FakeRetriever({url: f"content:{url}" for url in urls})
    agent = WebSearchAgent(
        main_model=planner,
        summarization_model=summarizer,
        search_tool=search_tool,
        retriever=retriever,
        max_results_per_query=5,
    )

    out = agent.run(make_session("Any task"))

    assert out.result is not None
    assert sorted(retriever.calls) == sorted(["https://b.example.com", "https://d.example.com"])


def test_filters_sources_globally_across_queries() -> None:
    planner = SequencedModel(
        outputs=[
            '{"queries":["q1","q2"],"done":true}',
            '{"selected_urls":["https://c.example.com","https://e.example.com"]}',
        ]
    )
    summarizer = SequencedModel(outputs=["summary"])
    search_tool = FakeSearchTool(
        {
            "q1": make_search_result(
                "q1", ["https://a.example.com", "https://b.example.com", "https://c.example.com"]
            ),
            "q2": make_search_result(
                "q2", ["https://d.example.com", "https://e.example.com", "https://f.example.com"]
            ),
        }
    )
    retriever = FakeRetriever({f"https://{c}.example.com": f"content:{c}" for c in "abcdef"})
    agent = WebSearchAgent(
        main_model=planner,
        summarization_model=summarizer,
        search_tool=search_tool,
        retriever=retriever,
        max_results_per_query=5,
    )

    out = agent.run(make_session("Any task"))

    assert out.result is not None
    assert sorted(retriever.calls) == sorted(["https://c.example.com", "https://e.example.com"])


def test_already_seen_urls_not_retrieved_again_on_second_run() -> None:
    planner = SequencedModel(
        outputs=[
            '{"queries":["q1"],"done":true}',
            '{"queries":["q1"],"done":true}',
        ]
    )
    summarizer = SequencedModel(outputs=["summary1", "final1", "final2"])
    search_tool = FakeSearchTool({"q1": make_search_result("q1", ["https://a.example.com"])})
    retriever = FakeRetriever({"https://a.example.com": "content"})
    agent = WebSearchAgent(
        main_model=planner,
        summarization_model=summarizer,
        search_tool=search_tool,
        retriever=retriever,
    )
    session = make_session("Any task")

    agent.run(session)
    calls_after_first = len(retriever.calls)

    agent.run(session)

    # URL was already in session.evidences — retriever must not be called for it again
    assert len(retriever.calls) == calls_after_first
