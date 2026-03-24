import json
from collections.abc import Mapping
from pathlib import Path

from ezmm import MultimodalSequence
from pydantic import BaseModel, ConfigDict, ValidationError

from mafc.agents.agent import Agent, AgentResult, format_evidence_block
from mafc.agents.common import AgentSession
from mafc.agents.judge.tracing import JudgeTraceRecorder
from mafc.common.claim import Claim
from mafc.common.evidence import Evidence
from mafc.common.label import BaseLabel
from mafc.common.modeling.message import Message, MessageRole
from mafc.common.modeling.model import Model
from mafc.common.modeling.prompt import Prompt
from mafc.utils.media import deduplicate_media
from mafc.utils.parsing import extract_json_object, strip_json_fences, try_parse_with_repair


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
        trace = self._setup_trace(session, trace_scope)
        trace.record_class_definitions({str(k): v for k, v in self.class_definitions.items()})

        if session.claim is None:
            return self._abort(session, trace, "missing_claim", "Judge session requires a claim.")
        if not session.evidences:
            return self._abort(
                session, trace, "missing_evidence", "Judge session requires accepted evidence."
            )

        messages = self._build_messages(session.claim, session.evidences)
        trace.record_prompt_messages(messages)
        _judge_resp = self.model.generate(messages)
        response_text = _judge_resp.text.strip()
        trace.add_usage(_judge_resp, self.model.name)
        trace.record_model_response(response_text)

        parsed = self._parse_with_repair(response_text, trace)
        if parsed is None:
            return self._abort(
                session,
                trace,
                "parse_response",
                "Judge returned invalid output.",
                evidences=list(session.evidences),
            )

        label = self._labels_by_value.get(parsed.label)
        if label is None:
            return self._abort(
                session,
                trace,
                "unknown_label",
                f"Judge returned unknown label '{parsed.label}'.",
                evidences=list(session.evidences),
            )

        return self._succeed(session, trace, parsed, label)

    def _setup_trace(self, session: AgentSession, trace_scope) -> JudgeTraceRecorder:
        scope = self._build_trace_scope("judge_run", session, trace_scope)
        return JudgeTraceRecorder(self.trace_dir, session, self.name, trace_scope=scope)

    def _abort(
        self,
        session: AgentSession,
        trace: JudgeTraceRecorder,
        error_key: str,
        error_msg: str,
        evidences: list[Evidence] | None = None,
    ) -> AgentResult:
        """Fail and record a trace error. Used for all judge failure paths."""
        self._mark_failed(session)
        result = AgentResult(
            session=session,
            result=None,
            evidences=evidences or [],
            errors=[error_msg],
            status=session.status,
        )
        trace.record_error(error_key, error_msg)
        trace.finalize(session=session, result=result, errors=result.errors)
        result.trace = trace.trace
        return result

    def _parse_with_repair(
        self, response_text: str, trace: JudgeTraceRecorder
    ) -> JudgeDecisionPayload | None:
        """Parse the model response; attempt a repair call if the first parse fails."""
        repair_prefix = (
            "Convert the following judge response to strict JSON with schema:\n"
            '{"label": "one allowed label", "justification": "short grounded justification"}\n'
            f"Allowed labels: {', '.join(k.value for k in self.class_definitions)}\n"
            "Only return JSON."
        )
        parsed, repair_text = try_parse_with_repair(
            response_text, self._parse_response, self.model, repair_prefix, trace
        )
        if repair_text is not None:
            trace.record_repair(
                prompt=f"{repair_prefix}\n\nResponse:\n{response_text}",
                response_text=repair_text,
            )
        return parsed

    def _succeed(
        self,
        session: AgentSession,
        trace: JudgeTraceRecorder,
        parsed: JudgeDecisionPayload,
        label: BaseLabel,
    ) -> AgentResult:
        assert session.claim is not None
        trace.record_decision(parsed.label, parsed.justification)
        session.claim.verdict = label
        session.claim.justification = MultimodalSequence(parsed.justification)
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
        evidence_lines = [
            block for evidence in evidences if (block := format_evidence_block(evidence)) is not None
        ]

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
            Message(role=MessageRole.USER, content=deduplicate_media(Prompt(text=user_text))),
        ]

    def _parse_response(self, response_text: str) -> JudgeDecisionPayload | None:
        """Parse the judge model response into a validated label decision."""
        try:
            return JudgeDecisionPayload.model_validate(
                json.loads(extract_json_object(strip_json_fences(response_text.strip())))
            )
        except (json.JSONDecodeError, ValidationError, ValueError):
            return None
