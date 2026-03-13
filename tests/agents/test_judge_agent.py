from __future__ import annotations

from mafc.agents.common import AgentSession, AgentStatus
from mafc.agents.judge.agent import JudgeAgent
from mafc.common.action import Action
from mafc.common.claim import Claim
from mafc.common.evidence import Evidence
from mafc.common.label import BaseLabel
from mafc.common.modeling.message import Message
from mafc.common.modeling.model import Model, Response
from mafc.common.modeling.prompt import Prompt


class DummyLabel(BaseLabel):
    TRUE = "true"
    FALSE = "false"


class DummyAction(Action):
    name = "dummy_action"

    def __init__(self):
        self._save_parameters(locals())


class SequencedModel(Model):
    def __init__(self, outputs: list[str]):
        super().__init__(specifier="OPENAI:gpt-5-mini-2025-08-07")
        self.outputs = outputs

    def generate(self, messages: list[Message]) -> Response:
        text = self.outputs.pop(0) if self.outputs else ""
        return Response(text=text, total_cost=0.0)


def test_judge_agent_predicts_label_and_sets_claim_fields() -> None:
    agent = JudgeAgent(
        model=SequencedModel(
            outputs=['{"label":"false","justification":"The evidence contradicts the claim."}']
        ),
        class_definitions={
            DummyLabel.TRUE: "The claim is supported.",
            DummyLabel.FALSE: "The claim is contradicted.",
        },
        extra_judge_rules="Prefer FALSE when evidence clearly contradicts the claim.",
    )
    claim = Claim("The event happened in 2025.")
    session = AgentSession(
        id="judge:1",
        goal=Prompt(text="Judge the claim."),
        claim=claim,
        evidences=[
            Evidence(
                raw=Prompt(text="Raw evidence"),
                action=DummyAction(),
                source="https://example.com",
                takeaways=Prompt(text="The event did not happen in 2025."),
            )
        ],
    )

    result = agent.run(session)

    assert result.result is not None
    assert result.session.status == AgentStatus.COMPLETED
    assert claim.verdict == DummyLabel.FALSE
    assert claim.justification is not None


def test_judge_agent_fails_without_evidence() -> None:
    agent = JudgeAgent(
        model=SequencedModel(outputs=['{"label":"true","justification":"unsupported"}']),
        class_definitions={DummyLabel.TRUE: "supported"},
    )
    session = AgentSession(
        id="judge:2",
        goal=Prompt(text="Judge the claim."),
        claim=Claim("Any claim"),
    )

    result = agent.run(session)

    assert result.result is None
    assert result.session.status == AgentStatus.FAILED
