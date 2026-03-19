from __future__ import annotations

from mafc.agents.common import AgentSession, AgentStatus
from mafc.agents.judge.agent import JudgeAgent
from mafc.common.modeling.prompt import Prompt

from tests.agents.judge.helpers import (
    CLASS_DEFINITIONS,
    SequencedModel,
    make_session,
    make_session_no_claim,
)


def test_missing_claim_aborts() -> None:
    agent = JudgeAgent(model=SequencedModel(outputs=[]), class_definitions=CLASS_DEFINITIONS)
    session = make_session_no_claim()

    out = agent.run(session)

    assert out.result is None
    assert out.session.status == AgentStatus.FAILED
    assert any("claim" in e.lower() for e in out.errors)


def test_missing_evidence_aborts() -> None:
    agent = JudgeAgent(model=SequencedModel(outputs=[]), class_definitions=CLASS_DEFINITIONS)
    session = AgentSession(
        id="judge:test",
        goal=Prompt(text="Judge the claim."),
        claim=make_session("The event happened.").claim,
        evidences=[],
    )

    out = agent.run(session)

    assert out.result is None
    assert out.session.status == AgentStatus.FAILED
    assert any("evidence" in e.lower() for e in out.errors)


def test_unparseable_output_after_repair_aborts() -> None:
    agent = JudgeAgent(
        model=SequencedModel(outputs=["not-json", "still-not-json"]),
        class_definitions=CLASS_DEFINITIONS,
    )

    out = agent.run(make_session())

    assert out.result is None
    assert out.session.status == AgentStatus.FAILED
    assert any("invalid output" in e.lower() for e in out.errors)
    assert out.evidences != []  # evidences are preserved in the result


def test_unknown_label_aborts() -> None:
    agent = JudgeAgent(
        model=SequencedModel(outputs=['{"label":"maybe","justification":"Unclear."}'] * 2),
        class_definitions=CLASS_DEFINITIONS,
    )

    out = agent.run(make_session())

    assert out.result is None
    assert out.session.status == AgentStatus.FAILED
    assert any("unknown label" in e.lower() for e in out.errors)
    assert out.evidences != []
