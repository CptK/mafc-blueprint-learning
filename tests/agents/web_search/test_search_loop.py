from __future__ import annotations

from mafc.agents.common import AgentMessageType, AgentStatus
from mafc.agents.web_search.agent import WebSearchAgent
from mafc.common.modeling.prompt import Prompt

from tests.agents.web_search.helpers import (
    FakeRetriever,
    FakeSearchTool,
    SequencedModel,
    make_search_result,
    make_session,
)


def test_iterative_loop_runs_queries_and_synthesizes() -> None:
    planner = SequencedModel(
        outputs=[
            '{"queries":["Samrat Choudhary helicopter fall election campaign"],"done":false}',
            '{"queries":[],"done":true}',
        ]
    )
    summarizer = SequencedModel(outputs=["Summary step 1", "Final synthesis"])
    retriever = FakeRetriever({"https://a.example.com": "Retrieved page content A"})
    search_tool = FakeSearchTool(
        {
            "Samrat Choudhary helicopter fall election campaign": make_search_result(
                "Samrat Choudhary helicopter fall election campaign",
                ["https://a.example.com"],
            )
        }
    )
    agent = WebSearchAgent(
        main_model=planner,
        summarization_model=summarizer,
        search_tool=search_tool,
        retriever=retriever,
        max_iterations=4,
    )

    out = agent.run(make_session("Retrieve incidents during the election campaign of Samrat Choudhary."))

    assert out.result is not None
    assert out.session.status == AgentStatus.COMPLETED
    assert len(out.messages) == 1
    assert len(out.evidences) == 1
    assert out.evidences[0].source == "https://a.example.com"
    assert out.messages[0].evidences == out.evidences
    assert "Final synthesis" in str(out.result)
    assert out.errors == []
    assert search_tool.queries == ["Samrat Choudhary helicopter fall election campaign"]
    assert retriever.calls == ["https://a.example.com"]


def test_uses_prior_session_context_for_follow_up() -> None:
    planner = SequencedModel(
        outputs=[
            '{"queries":["winner baden-wuerttemberg landtagswahl"],"done":true}',
            '{"queries":[],"done":true}',
        ]
    )
    summarizer = SequencedModel(outputs=["Win summary", "Election synthesis", "Seat distribution summary"])
    retriever = FakeRetriever({"https://a.example.com": "Parliament result article"})
    search_tool = FakeSearchTool(
        {
            "winner baden-wuerttemberg landtagswahl": make_search_result(
                "winner baden-wuerttemberg landtagswahl",
                ["https://a.example.com"],
            )
        }
    )
    agent = WebSearchAgent(
        main_model=planner,
        summarization_model=summarizer,
        search_tool=search_tool,
        retriever=retriever,
    )
    session = make_session("Wer hat die letzte Landtagswahl in Baden-Wuerttemberg gewonnen?")

    first = agent.run(session)
    assert first.result is not None
    assert len(session.evidences) == 1

    session.goal = Prompt(text="Wie ist die Sitzverteilung im neuen Parlament?")
    second = agent.run(session)

    assert second.result is not None
    assert second.session.status == AgentStatus.COMPLETED
    assert second.errors == []
    assert "Accepted evidence:" in planner.calls[1]
    assert "https://a.example.com" in planner.calls[1]
    assert "Election synthesis" in planner.calls[1]
    assert len(second.evidences) == 1
    assert len([m for m in session.messages if m.message_type == AgentMessageType.RESULT]) == 2


def test_loop_stops_at_max_iterations() -> None:
    # Planner always returns done=false — the loop stops only because max_iterations is reached
    planner = SequencedModel(
        outputs=[
            '{"queries":["q1"],"done":false}',
            '{"queries":["q2"],"done":false}',
        ]
    )
    summarizer = SequencedModel(outputs=["summary1", "summary2", "final synthesis"])
    search_tool = FakeSearchTool(
        {
            "q1": make_search_result("q1", ["https://a.example.com"]),
            "q2": make_search_result("q2", ["https://b.example.com"]),
        }
    )
    retriever = FakeRetriever({"https://a.example.com": "content a", "https://b.example.com": "content b"})
    agent = WebSearchAgent(
        main_model=planner,
        summarization_model=summarizer,
        search_tool=search_tool,
        retriever=retriever,
        max_iterations=2,
    )

    out = agent.run(make_session("Any task"))

    assert len(planner.calls) == 2  # exactly max_iterations planner invocations
    assert out.result is not None
    assert out.session.status == AgentStatus.COMPLETED
    assert len(out.evidences) == 2
