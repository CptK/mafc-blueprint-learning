from __future__ import annotations

from mafc.agents.web_search.agent import WebSearchAgent
from mafc.common.evidence import Evidence
from mafc.common.modeling.prompt import Prompt

from tests.agents.web_search.helpers import (
    DummyAction,
    FakeRetriever,
    FakeSearchTool,
    SequencedModel,
    make_search_result,
    make_session,
)


def test_falls_back_when_summary_is_failure_text() -> None:
    planner = SequencedModel(outputs=['{"queries":["q1"],"done":true}'])
    summarizer = SequencedModel(outputs=["Failed to generate a response.", "Failed to generate a response."])
    search_tool = FakeSearchTool({"q1": make_search_result("q1", ["https://a.example.com"])})
    retriever = FakeRetriever({"https://a.example.com": "retrieved content"})
    agent = WebSearchAgent(
        main_model=planner,
        summarization_model=summarizer,
        search_tool=search_tool,
        retriever=retriever,
    )

    out = agent.run(make_session("Any task"))

    assert out.result is None
    assert len(out.evidences) == 0


def test_synthesize_from_evidences_returns_string() -> None:
    summarizer = SequencedModel(outputs=['{"synthesis":"The event happened in Greece."}'])
    agent = WebSearchAgent(
        main_model=SequencedModel(outputs=[]),
        summarization_model=summarizer,
        search_tool=FakeSearchTool({}),
        retriever=FakeRetriever({}),
    )
    evidences = [
        Evidence(
            raw=Prompt(text="Greece article text"),
            action=DummyAction(),
            source="https://example.com",
            takeaways=Prompt(text="The event happened in Greece."),
        )
    ]

    result = agent.synthesize_from_evidences("Where did it happen?", evidences)

    assert isinstance(result, str)
    assert "Greece" in result
