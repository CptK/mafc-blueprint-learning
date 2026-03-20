from __future__ import annotations

from mafc.agents.common import AgentStatus
from mafc.agents.web_search.agent import WebSearchAgent

from tests.agents.web_search.helpers import FakeRetriever, FakeSearchTool, SequencedModel, make_session


def test_stop_signal_aborts_before_execution() -> None:
    agent = WebSearchAgent(
        main_model=SequencedModel(outputs=[]),
        search_tool=FakeSearchTool({}),
        retriever=FakeRetriever({}),
    )
    agent._should_stop = True

    out = agent.run(make_session("Any task"))

    assert out.result is None
    assert out.session.status == AgentStatus.FAILED
    assert any("stopped" in e for e in out.errors)


def test_empty_instruction_returns_failed_status() -> None:
    agent = WebSearchAgent(
        main_model=SequencedModel(outputs=[]),
        search_tool=FakeSearchTool({}),
        retriever=FakeRetriever({}),
    )

    out = agent.run(make_session("   "))

    assert out.result is None
    assert out.session.status == AgentStatus.FAILED
    assert any("prompt" in e.lower() or "empty" in e.lower() for e in out.errors)
