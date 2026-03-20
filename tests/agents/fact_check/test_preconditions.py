from __future__ import annotations

from mafc.agents import AgentSession, AgentStatus
from mafc.agents.fact_check.agent import FactCheckAgent
from mafc.common.claim import Claim
from mafc.common.modeling.prompt import Prompt

from tests.agents.fact_check.helpers import (
    FakeWorkerAgent,
    SequencedModel,
    make_registry,
    make_selector,
    registered_image,
)


def test_no_claim_and_empty_goal_returns_failed(tmp_path) -> None:
    registry = make_registry(tmp_path)
    agent = FactCheckAgent(
        model=SequencedModel(outputs=[]),
        blueprint_selector=make_selector(registry),
    )
    session = AgentSession(id="fact-check:empty", goal=Prompt(text="   "))

    result = agent.run(session)

    assert result.result is None
    assert result.session.status == AgentStatus.FAILED
    assert any("claim" in e.lower() or "goal" in e.lower() for e in result.errors)


def test_resolves_claim_from_nonempty_goal(tmp_path) -> None:
    registry = make_registry(tmp_path, include_media=False)
    planner = SequencedModel(
        outputs=[
            '{"decision_type":"delegate","rationale":"search","tasks":[{"task_id":"t1","agent_type":"web_search","instruction":"Find evidence."}]}',
            "Evidence summary.",
            "Final answer from goal.",
        ]
    )
    agent = FactCheckAgent(
        model=planner,
        blueprint_selector=make_selector(registry),
        delegation_agents={"web_search": [FakeWorkerAgent("Web result.", "https://example.com")]},
    )
    # No claim set — agent should construct one from the goal text
    session = AgentSession(id="fact-check:goal-claim", goal=Prompt(text="Is the sky blue?"))

    result = agent.run(session)

    assert result.result is not None
    assert result.session.status == AgentStatus.COMPLETED
    assert session.claim is not None
    assert "sky" in str(session.claim).lower()


def test_stop_signal_aborts_during_iteration(tmp_path) -> None:
    registry = make_registry(tmp_path)
    agent = FactCheckAgent(
        model=SequencedModel(outputs=[]),
        blueprint_selector=make_selector(registry),
        delegation_agents={"media": [FakeWorkerAgent("unused", "image://unused")]},
    )
    agent._should_stop = True
    image = registered_image()
    session = AgentSession(
        id="fact-check:stop",
        goal=Prompt(text="Fact-check claim"),
        claim=Claim("This image shows Athens.", image),
    )

    result = agent.run(session)

    assert result.result is None
    assert result.session.status == AgentStatus.FAILED
    assert any("stopped" in e.lower() or "stop" in e.lower() for e in result.errors)
