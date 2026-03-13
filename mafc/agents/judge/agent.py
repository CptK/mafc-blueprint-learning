import json
from collections.abc import Mapping
from pathlib import Path

from ezmm import MultimodalSequence
from pydantic import BaseModel, ConfigDict, ValidationError

from mafc.agents.agent import Agent, AgentResult
from mafc.agents.common import AgentSession
from mafc.agents.judge.tracing import JudgeTraceRecorder
from mafc.common.claim import Claim
from mafc.common.evidence import Evidence
from mafc.common.label import BaseLabel
from mafc.common.modeling.message import Message, MessageRole
from mafc.common.modeling.model import Model
from mafc.common.modeling.prompt import Prompt
from mafc.common.trace import TraceScope
from mafc.utils.parsing import extract_json_object


class JudgeDecisionPayload(BaseModel):
    """Validated JSON payload returned by the judge model."""

    model_config = ConfigDict(extra="forbid")

    label: str
    justification: str


class JudgeAgent(Agent):
    """Predict a benchmark label from a claim and accepted evidence."""

    name = "JudgeAgent"
    description = "Assigns one benchmark label from grounded claim evidence."
    allowed_tools = []

    def __init__(
        self,
        model: Model,
        class_definitions: Mapping[BaseLabel, str],
        extra_judge_rules: str | None = None,
        n_workers: int = 1,
        agent_id: str | None = None,
        trace_dir: str | Path | None = None,
    ):
        """Initialize the judge with benchmark class definitions and optional extra rules."""
        super().__init__(model, n_workers=n_workers, agent_id=agent_id)
        self.class_definitions = dict(class_definitions)
        self.extra_judge_rules = extra_judge_rules
        self._labels_by_value = {label.value: label for label in self.class_definitions}
        self.trace_dir = trace_dir

    def run(self, session: AgentSession, trace_scope=None) -> AgentResult:
        """Judge the claim label using only accepted evidence."""
        self._mark_running(session)
        scope = (
            trace_scope.child_scope("judge_run", key=session.id, metadata={"agent": self.name})
            if trace_scope is not None
            else TraceScope.root(
                scope_type="judge_run",
                trace_id=session.id,
                trace_dir=self.trace_dir,
                key=session.id,
                metadata={"agent": self.name},
            )
        )
        trace = JudgeTraceRecorder(self.trace_dir, session, self.name, trace_scope=scope)
        trace.record_class_definitions({str(k): v for k, v in self.class_definitions.items()})

        claim = session.claim
        if claim is None:
            self._mark_failed(session)
            result = AgentResult(
                session=session,
                result=None,
                errors=["Judge session requires a claim."],
                status=session.status,
            )
            trace.record_error("missing_claim", result.errors[0])
            trace.finalize(session=session, result=result, errors=result.errors)
            result.trace = trace.trace
            return result
        if not session.evidences:
            self._mark_failed(session)
            result = AgentResult(
                session=session,
                result=None,
                errors=["Judge session requires accepted evidence."],
                status=session.status,
            )
            trace.record_error("missing_evidence", result.errors[0])
            trace.finalize(session=session, result=result, errors=result.errors)
            result.trace = trace.trace
            return result

        messages = self._build_messages(claim, session.evidences)
        trace.record_prompt_messages(messages)
        response_text = self.model.generate(messages).text.strip()
        trace.record_model_response(response_text)

        parsed = self._parse_response(response_text)
        if parsed is None:
            self._mark_failed(session)
            result = AgentResult(
                session=session,
                result=None,
                evidences=list(session.evidences),
                errors=["Judge returned invalid output."],
                status=session.status,
            )
            trace.record_error("parse_response", result.errors[0])
            trace.finalize(session=session, result=result, errors=result.errors)
            result.trace = trace.trace
            return result

        label = self._labels_by_value.get(parsed.label)
        if label is None:
            self._mark_failed(session)
            result = AgentResult(
                session=session,
                result=None,
                evidences=list(session.evidences),
                errors=[f"Judge returned unknown label '{parsed.label}'."],
                status=session.status,
            )
            trace.record_error("unknown_label", result.errors[0])
            trace.finalize(session=session, result=result, errors=result.errors)
            result.trace = trace.trace
            return result

        trace.record_decision(parsed.label, parsed.justification)
        claim.verdict = label
        claim.justification = MultimodalSequence(parsed.justification)
        result_text = MultimodalSequence(f"Label: {label.value}\nJustification: {parsed.justification}")
        result_message = self.make_result_message(session, result_text, list(session.evidences))
        session.messages.append(result_message)
        self._mark_completed(session)
        result = AgentResult(
            session=session,
            result=result_text,
            messages=[result_message],
            evidences=list(session.evidences),
            status=session.status,
        )
        trace.finalize(session=session, result=result, errors=[])
        result.trace = trace.trace
        return result

    def synthesize_from_evidences(self, instruction: str, evidences: list[Evidence]) -> str:
        """Synthesize a label-focused response from evidence."""
        claim = Claim(instruction)
        response_text = self.model.generate(self._build_messages(claim, evidences)).text.strip()
        parsed = self._parse_response(response_text)
        if parsed is None:
            return response_text
        return f"Label: {parsed.label}\nJustification: {parsed.justification}"

    def _build_messages(self, claim: Claim, evidences: list[Evidence]) -> list[Message]:
        """Build judge messages with benchmark schema and accepted evidence."""
        label_lines = [
            f"- {label.value}: {definition}" for label, definition in self.class_definitions.items()
        ]
        evidence_lines = []
        for evidence in evidences:
            summary = (
                str(evidence.takeaways).strip()
                if evidence.takeaways is not None
                else str(evidence.raw).strip()
            )
            if summary:
                evidence_lines.append(f"- Source: {evidence.source}\n  Summary: {summary}")

        system_text = (
            "You are a benchmark judging agent.\n"
            "Predict exactly one allowed benchmark label.\n"
            "Use only the accepted evidence provided below.\n"
            "If evidence is limited or mixed, prefer the appropriate uncertainty label.\n"
        )
        if self.extra_judge_rules:
            system_text += f"\nAdditional benchmark rules:\n{self.extra_judge_rules.strip()}\n"

        user_text = (
            f"Claim:\n{claim.describe()}\n\n"
            f"Allowed labels:\n{chr(10).join(label_lines)}\n\n"
            f"Accepted evidence:\n{chr(10).join(evidence_lines)}\n\n"
            "Return strict JSON only with schema:\n"
            '{"label":"one allowed label","justification":"short grounded justification"}'
        )
        return [
            Message(role=MessageRole.SYSTEM, content=Prompt(text=system_text)),
            Message(role=MessageRole.USER, content=Prompt(text=user_text)),
        ]

    def _parse_response(self, response_text: str) -> JudgeDecisionPayload | None:
        """Parse the judge model response into a validated label decision."""
        text = response_text.strip()
        if text.startswith("```"):
            lines = [line for line in text.splitlines() if not line.startswith("```")]
            text = "\n".join(lines).strip()
        try:
            return JudgeDecisionPayload.model_validate(json.loads(extract_json_object(text)))
        except (json.JSONDecodeError, ValidationError, ValueError):
            return None
