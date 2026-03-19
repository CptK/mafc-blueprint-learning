from __future__ import annotations

import json
from typing import cast

from ezmm import MultimodalSequence
from ezmm.common.items import Image, Video

from mafc.agents.agent import Agent, AgentResult
from mafc.agents.common import AgentSession
from mafc.agents.media.planner import plan_media_tools
from mafc.agents.media.utils import build_evidences_from_tool_result
from mafc.agents.media.tracing import MediaTraceRecorder
from mafc.common.trace import TraceScope
from mafc.utils.media import extract_media_items
from mafc.utils.parsing import extract_json_object
from mafc.common.evidence import Evidence
from mafc.common.logger import logger
from mafc.common.modeling.message import Message, MessageRole
from mafc.common.modeling.model import Model, Response
from mafc.common.modeling.prompt import Prompt
from mafc.tools.geolocate.geolocate import Geolocate, Geolocator
from mafc.tools.tool import Tool
from mafc.tools.tool_result import ToolResult
from mafc.tools.web_search.google_search import GoogleSearchPlatform
from mafc.tools.web_search.reverse_image_search import ReverseImageSearch, ReverseImageSearchTool


class MediaAgent(Agent):
    name = "MediaAgent"
    description = (
        "Analyzes image or video that is already attached to the claim or task session. "
        "Use it for reverse image search, geolocation, and other inspection of existing media. "
        "Do not use it to discover new media on the web when no image or video is provided."
    )

    allowed_tools = cast(list[type[Tool]], [ReverseImageSearchTool, Geolocator, GoogleSearchPlatform])

    def __init__(
        self,
        model: Model,
        n_workers: int = 1,
        summarization_model: Model | None = None,
        ris_tool: ReverseImageSearchTool | None = None,
        geolocator: Geolocator | None = None,
        agent_id: str | None = None,
        trace_dir: str | None = None,
    ):
        super().__init__(model, n_workers=n_workers, agent_id=agent_id)
        self.summarization_model = summarization_model or model
        self.ris_tool = ris_tool or ReverseImageSearchTool()
        self.geolocator = geolocator or Geolocator()
        self.trace_dir = trace_dir

    def run(self, session: AgentSession, trace_scope=None) -> AgentResult:
        self._mark_running(session)
        trace = self._setup_trace(session, trace_scope)

        instruction = str(session.goal).strip()
        prior_context = self.build_prior_context(session)

        if self._should_stop:
            return self._abort(session, trace, "stop_signal", "Agent was stopped before execution started.")
        if not instruction:
            return self._abort(session, trace, "empty_instruction", "Task prompt is empty.")

        errors: list[str] = []
        media_items = extract_media_items(session.goal)
        trace.record_media_items([item.reference for item in media_items])
        if not media_items:
            return self._abort(
                session, trace, "no_media_items", "Task does not contain any image or video item."
            )
        if len(media_items) > 1:
            errors.append("Task contains multiple media items. Only the first item is processed for now.")

        self._collect_tool_evidences(session, instruction, prior_context, media_items[0], errors, trace)

        if not session.evidences:
            return self._fail(session, trace, errors, evidences=[])

        synthesis, relevant_evidences, synthesis_resp = self._synthesize_with_relevant_evidences(
            instruction, session.evidences
        )
        if synthesis_resp is not None:
            trace.add_usage(synthesis_resp, self.summarization_model.name)
        if synthesis.strip():
            trace.record_synthesis(synthesis, len(session.evidences))

        if not synthesis.strip():
            return self._fail(session, trace, errors, evidences=list(relevant_evidences))

        return self._succeed(session, trace, synthesis, relevant_evidences, errors)

    def _setup_trace(self, session: AgentSession, trace_scope) -> MediaTraceRecorder:
        scope = (
            trace_scope.child_scope("media_run", key=session.id, metadata={"agent": self.name})
            if trace_scope is not None
            else TraceScope.root(
                scope_type="media_run",
                trace_id=session.id,
                trace_dir=self.trace_dir,
                key=session.id,
                metadata={"agent": self.name},
            )
        )
        return MediaTraceRecorder(self.trace_dir, session, self.name, trace_scope=scope)

    def _abort(
        self, session: AgentSession, trace: MediaTraceRecorder, error_key: str, error_msg: str
    ) -> AgentResult:
        """Fail due to a precondition not being met. Records a trace error."""
        self._mark_failed(session)
        result = AgentResult(session=session, result=None, errors=[error_msg], status=session.status)
        trace.record_error(error_key, error_msg)
        trace.finalize(session=session, result=result, errors=result.errors)
        result.trace = trace.trace
        return result

    def _fail(
        self, session: AgentSession, trace: MediaTraceRecorder, errors: list[str], evidences: list[Evidence]
    ) -> AgentResult:
        """Fail after execution has started (no evidences or empty synthesis)."""
        self._mark_failed(session)
        result = AgentResult(
            session=session, result=None, evidences=evidences, errors=errors, status=session.status
        )
        trace.finalize(session=session, result=result, errors=errors)
        result.trace = trace.trace
        return result

    def _collect_tool_evidences(
        self,
        session: AgentSession,
        instruction: str,
        prior_context: str,
        media_item: Image | Video,
        errors: list[str],
        trace: MediaTraceRecorder,
    ) -> None:
        for tool_name, tool_result in self._run_selected_tools(
            instruction, prior_context, media_item, errors, trace
        ):
            trace.record_tool_result(tool_name, tool_result)
            for evidence in build_evidences_from_tool_result(tool_result, media_item.reference):
                if evidence not in session.evidences:
                    session.evidences.append(evidence)
        trace.record_evidences(session.evidences)

    def _succeed(
        self,
        session: AgentSession,
        trace: MediaTraceRecorder,
        synthesis: str,
        relevant_evidences: list[Evidence],
        errors: list[str],
    ) -> AgentResult:
        result_text = MultimodalSequence(synthesis)
        result_message = self.make_result_message(session, result_text, list(relevant_evidences))
        session.messages.append(result_message)
        self._mark_completed(session)
        result = AgentResult(
            session=session,
            result=result_text,
            messages=[result_message],
            evidences=list(relevant_evidences),
            errors=errors,
            status=session.status,
        )
        trace.finalize(session=session, result=result, errors=errors)
        result.trace = trace.trace
        return result

    def synthesize_from_evidences(self, instruction: str, evidences: list[Evidence]) -> str:
        synthesis, _, _resp = self._synthesize_with_relevant_evidences(instruction, evidences)
        return synthesis

    def _synthesize_with_relevant_evidences(
        self, instruction: str, evidences: list[Evidence]
    ) -> tuple[str, list[Evidence], "Response | None"]:
        evidence_blocks = []
        evidence_id_to_item: dict[str, Evidence] = {}
        for idx, evidence in enumerate(evidences, start=1):
            summary = (
                str(evidence.takeaways).strip()
                if evidence.takeaways is not None
                else str(evidence.raw).strip()
            )
            if not summary:
                continue
            evidence_id = f"ev_{idx}"
            evidence_id_to_item[evidence_id] = evidence
            evidence_blocks.append(
                f"Evidence ID: {evidence_id}\nSource: {evidence.source}\nSummary: {summary}"
            )
        if not evidence_blocks:
            return "", [], None

        synthesis_prompt = (
            "You are a media verification evidence synthesizer.\n"
            "You will be given the full accepted evidence history for this media item.\n"
            "Use all of it as context, but select only the evidence items that are directly relevant to the current task.\n"
            "Be explicit about uncertainty, dates, locations, and whether findings come from reverse image search "
            "or geolocation.\n"
            "Return strict JSON only with schema:\n"
            '{"answer": "string", "relevant_evidence_ids": ["ev_1"]}\n'
            "The relevant_evidence_ids must contain only IDs from the accepted evidence list.\n\n"
            f"Task:\n{instruction}\n\n"
            f"Accepted evidence:\n{chr(10).join(evidence_blocks)}"
        )
        try:
            _resp = self.summarization_model.generate(
                [Message(role=MessageRole.USER, content=Prompt(text=synthesis_prompt))]
            )
            response_text = _resp.text.strip()
        except Exception:
            return "\n\n".join(evidence_blocks), list(evidences), None
        parsed = self._parse_synthesis_response(response_text, evidence_id_to_item)
        if parsed is None:
            return response_text, list(evidences), _resp
        answer, relevant_evidences = parsed
        return answer, relevant_evidences, _resp

    def _parse_synthesis_response(
        self, response_text: str, evidence_id_to_item: dict[str, Evidence]
    ) -> tuple[str, list[Evidence]] | None:
        text = response_text.strip()
        if text.startswith("```"):
            lines = [line for line in text.splitlines() if not line.startswith("```")]
            text = "\n".join(lines).strip()
        text = extract_json_object(text)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning(f"[{self.name}] Failed to parse synthesis response as JSON: {exc}")
            return None

        answer = payload.get("answer")
        relevant_ids = payload.get("relevant_evidence_ids")
        if not isinstance(answer, str):
            logger.warning(f"[{self.name}] Invalid synthesis response: answer must be a string.")
            return None
        if not isinstance(relevant_ids, list) or not all(isinstance(item, str) for item in relevant_ids):
            logger.warning(
                f"[{self.name}] Invalid synthesis response: relevant_evidence_ids must be a list of strings."
            )
            return None

        relevant_evidences: list[Evidence] = []
        for evidence_id in relevant_ids:
            evidence = evidence_id_to_item.get(evidence_id)
            if evidence is None:
                logger.warning(
                    f"[{self.name}] Ignoring unknown evidence ID from synthesis response: {evidence_id}"
                )
                continue
            if evidence not in relevant_evidences:
                relevant_evidences.append(evidence)
        if not relevant_evidences:
            relevant_evidences = list(evidence_id_to_item.values())
        return answer.strip(), relevant_evidences

    def _run_selected_tools(
        self,
        instruction: str,
        prior_context: str,
        media_item: Image | Video,
        errors: list[str],
        trace: MediaTraceRecorder | None = None,
    ) -> list[tuple[str, ToolResult]]:
        plan, planner_messages, planner_response = plan_media_tools(
            self, instruction, prior_context, errors, trace=trace
        )
        if trace is not None:
            trace.record_planner_messages(planner_messages)
            if planner_response is not None:
                trace.record_planner_response(planner_response)
        if plan is None:
            errors.append(
                "Media planner output could not be parsed. Falling back to running reverse image search and geolocation."
            )
            selected_tools = ["reverse_image_search", "geolocate"]
        else:
            selected_tools = plan.tools
        if trace is not None:
            trace.record_planned_tools(list(selected_tools))

        results: list[tuple[str, ToolResult]] = []
        for tool_name in selected_tools:
            if tool_name == "reverse_image_search":
                results.append((tool_name, self.ris_tool.perform(ReverseImageSearch(media_item.reference))))
            elif tool_name == "geolocate":
                results.append((tool_name, self.geolocator.perform(Geolocate(media_item.reference))))
        return results
