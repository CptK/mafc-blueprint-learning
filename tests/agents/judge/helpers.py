from __future__ import annotations

from collections.abc import Mapping

from mafc.agents.common import AgentSession
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
    UNCERTAIN = "uncertain"


CLASS_DEFINITIONS: Mapping[BaseLabel, str] = {
    DummyLabel.TRUE: "The claim is supported by evidence.",
    DummyLabel.FALSE: "The claim is contradicted by evidence.",
    DummyLabel.UNCERTAIN: "Evidence is insufficient to decide.",
}


class DummyAction(Action):
    name = "dummy_action"

    def __init__(self):
        self._save_parameters(locals())


class SequencedModel(Model):
    def __init__(self, outputs: list[str]):
        super().__init__(specifier="OPENAI:gpt-5-mini-2025-08-07")
        self.outputs = outputs
        self.calls: list[list[Message]] = []

    def generate(self, messages: list[Message]) -> Response:
        self.calls.append(messages)
        text = self.outputs.pop(0) if self.outputs else ""
        return Response(text=text, total_cost=0.0)


def make_evidence(summary: str = "The event did not happen in 2025.") -> Evidence:
    return Evidence(
        raw=Prompt(text="Raw evidence text"),
        action=DummyAction(),
        source="https://example.com",
        takeaways=Prompt(text=summary),
    )


def make_session(
    claim_text: str = "The event happened in 2025.", *, evidences: list[Evidence] | None = None
) -> AgentSession:
    return AgentSession(
        id="judge:test",
        goal=Prompt(text="Judge the claim."),
        claim=Claim(claim_text),
        evidences=evidences or [make_evidence()],
    )


def make_session_no_claim(*, evidences: list[Evidence] | None = None) -> AgentSession:
    return AgentSession(
        id="judge:test",
        goal=Prompt(text="Judge the claim."),
        claim=None,
        evidences=evidences or [make_evidence()],
    )
