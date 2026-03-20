from __future__ import annotations

from mafc.agents.common import AgentStatus
from mafc.agents.web_search.agent import WebSearchAgent

from tests.agents.web_search.helpers import (
    FakeRetriever,
    FakeSearchTool,
    SequencedModel,
    make_search_result,
    make_session,
)


def test_handles_invalid_planner_output_falls_back_to_task_text() -> None:
    planner = SequencedModel(outputs=["not-json"])
    search_tool = FakeSearchTool({})
    agent = WebSearchAgent(main_model=planner, search_tool=search_tool, retriever=FakeRetriever({}))

    out = agent.run(make_session("Any task"))

    assert out.result is None
    assert out.session.status == AgentStatus.FAILED
    assert "Falling back to the original task text" in out.errors[0]
    assert search_tool.queries == ["Any task"]


def test_parses_json_embedded_in_surrounding_text() -> None:
    planner = SequencedModel(outputs=['Plan:\n{"queries":["q1"],"done":true}\nThanks'])
    summarizer = SequencedModel(outputs=["summary"])
    search_tool = FakeSearchTool({"q1": make_search_result("q1", ["https://a.example.com"])})
    retriever = FakeRetriever({"https://a.example.com": "content"})
    agent = WebSearchAgent(
        main_model=planner,
        summarization_model=summarizer,
        search_tool=search_tool,
        retriever=retriever,
    )

    out = agent.run(make_session("Any task"))

    assert out.result is not None
    assert out.errors == []
    assert search_tool.queries == ["q1"]


def test_repairs_non_json_planner_output() -> None:
    planner = SequencedModel(
        outputs=[
            "Ich suche jetzt nach passenden Quellen.",
            '{"queries":["q1"],"done":true}',
        ]
    )
    summarizer = SequencedModel(outputs=["summary"])
    search_tool = FakeSearchTool({"q1": make_search_result("q1", ["https://a.example.com"])})
    retriever = FakeRetriever({"https://a.example.com": "content"})
    agent = WebSearchAgent(
        main_model=planner,
        summarization_model=summarizer,
        search_tool=search_tool,
        retriever=retriever,
    )

    out = agent.run(make_session("Any task"))

    assert out.result is not None
    assert out.errors == []
    assert search_tool.queries == ["q1"]


def test_done_with_no_queries_terminates_without_searching() -> None:
    # Planner signals done=true but provides no queries → loop terminates immediately
    planner = SequencedModel(outputs=['{"queries":[],"done":true}'])
    search_tool = FakeSearchTool({})
    agent = WebSearchAgent(main_model=planner, search_tool=search_tool, retriever=FakeRetriever({}))

    out = agent.run(make_session("Any task"))

    assert search_tool.queries == []
    assert out.result is None
    assert out.session.status == AgentStatus.FAILED
