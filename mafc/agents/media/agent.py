from __future__ import annotations

import json
from typing import cast

from ezmm import MultimodalSequence
from ezmm.common.items import Image, Video

from mafc.agents.agent import Agent, AgentResult
from mafc.agents.common import AgentSession
from mafc.agents.media.planner import plan_media_tools
from mafc.agents.web_search.parsing import extract_json_object
from mafc.common.evidence import Evidence
from mafc.common.logger import logger
from mafc.common.modeling.message import Message, MessageRole
from mafc.common.modeling.model import Model
from mafc.common.modeling.prompt import Prompt
from mafc.tools.geolocate.geolocate import Geolocate, Geolocator
from mafc.tools.tool import Tool
from mafc.tools.tool_result import ToolResult
from mafc.tools.web_search.google_search import GoogleSearchPlatform
from mafc.tools.web_search.google_vision import GoogleRisResults
from mafc.tools.web_search.reverse_image_search import ReverseImageSearch, ReverseImageSearchTool


class MediaAgent(Agent):
    name = "MediaAgent"
    description = "Investigates image/video questions using reverse image search and geolocation."

    allowed_tools = cast(list[type[Tool]], [ReverseImageSearchTool, Geolocator, GoogleSearchPlatform])

    def __init__(
        self,
        model: Model,
        n_workers: int = 1,
        summarization_model: Model | None = None,
        ris_tool: ReverseImageSearchTool | None = None,
        geolocator: Geolocator | None = None,
        agent_id: str | None = None,
    ):
        super().__init__(model, n_workers=n_workers, agent_id=agent_id)
        self.summarization_model = summarization_model or model
        self.ris_tool = ris_tool or ReverseImageSearchTool()
        self.geolocator = geolocator or Geolocator()

    def run(self, session: AgentSession) -> AgentResult:
        self._mark_running(session)
        instruction = str(session.goal).strip()
        prior_context = self.build_prior_context(session)
        if self._should_stop:
            self._mark_failed(session)
            return AgentResult(
                session=session,
                result=None,
                errors=["Agent was stopped before execution started."],
                status=session.status,
            )
        if not instruction:
            self._mark_failed(session)
            return AgentResult(
                session=session,
                result=None,
                errors=["Task prompt is empty."],
                status=session.status,
            )

        errors: list[str] = []
        media_items = self._extract_media_items(session.goal)
        if not media_items:
            self._mark_failed(session)
            return AgentResult(
                session=session,
                result=None,
                errors=["Task does not contain any image or video item."],
                status=session.status,
            )
        if len(media_items) > 1:
            errors.append("Task contains multiple media items. Only the first item is processed for now.")

        media_item = media_items[0]
        for tool_result in self._run_selected_tools(instruction, prior_context, media_item, errors):
            for evidence in self._build_evidences_from_tool_result(tool_result, media_item.reference):
                if evidence not in session.evidences:
                    session.evidences.append(evidence)

        if not session.evidences:
            self._mark_failed(session)
            return AgentResult(
                session=session,
                result=None,
                evidences=[],
                errors=errors,
                status=session.status,
            )

        synthesis, relevant_evidences = self._synthesize_with_relevant_evidences(
            instruction, session.evidences
        )
        if not synthesis.strip():
            self._mark_failed(session)
            return AgentResult(
                session=session,
                result=None,
                evidences=list(relevant_evidences),
                errors=errors,
                status=session.status,
            )

        result_text = MultimodalSequence(synthesis)
        result_message = self.make_result_message(session, result_text, list(relevant_evidences))
        session.messages.append(result_message)
        self._mark_completed(session)
        return AgentResult(
            session=session,
            result=result_text,
            messages=[result_message],
            evidences=list(relevant_evidences),
            errors=errors,
            status=session.status,
        )

    def synthesize_from_evidences(self, instruction: str, evidences: list[Evidence]) -> str:
        synthesis, _ = self._synthesize_with_relevant_evidences(instruction, evidences)
        return synthesis

    def _synthesize_with_relevant_evidences(
        self, instruction: str, evidences: list[Evidence]
    ) -> tuple[str, list[Evidence]]:
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
            return "", []

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
            response_text = self.summarization_model.generate(
                [Message(role=MessageRole.USER, content=Prompt(text=synthesis_prompt))]
            ).text.strip()
        except Exception:
            return "\n\n".join(evidence_blocks), list(evidences)
        parsed = self._parse_synthesis_response(response_text, evidence_id_to_item)
        if parsed is None:
            return response_text, list(evidences)
        return parsed

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

    def _extract_media_items(self, goal: MultimodalSequence) -> list[Image | Video]:
        items: list[Image | Video] = []
        items.extend(goal.images)
        items.extend(goal.videos)
        return items

    def _run_selected_tools(
        self,
        instruction: str,
        prior_context: str,
        media_item: Image | Video,
        errors: list[str],
    ) -> list[ToolResult]:
        plan = plan_media_tools(self, instruction, prior_context, errors)
        if plan is None:
            errors.append(
                "Media planner output could not be parsed. Falling back to running reverse image search and geolocation."
            )
            selected_tools = ["reverse_image_search", "geolocate"]
        else:
            selected_tools = plan.tools

        results: list[ToolResult] = []
        for tool_name in selected_tools:
            if tool_name == "reverse_image_search":
                results.append(self.ris_tool.perform(ReverseImageSearch(media_item.reference)))
            elif tool_name == "geolocate":
                results.append(self.geolocator.perform(Geolocate(media_item.reference)))
        return results

    def _build_evidences_from_tool_result(
        self, tool_result: ToolResult, media_reference: str
    ) -> list[Evidence]:
        raw = tool_result.raw
        takeaways = tool_result.takeaways

        # RIS: promote each matched source into its own evidence item.
        if isinstance(raw, GoogleRisResults) and raw.sources:
            logger.info(
                "Raw tool output contains multiple sources. Promoting each source into its own evidence item."
            )
            evidences: list[Evidence] = []
            for source in raw.sources:
                summary_parts = []
                if takeaways is not None:
                    summary_parts.append(str(takeaways))
                summary = "\n".join(part for part in summary_parts if part.strip())
                evidences.append(
                    Evidence(
                        raw=MultimodalSequence(str(source)),
                        action=tool_result.action,
                        source=source.reference,
                        takeaways=MultimodalSequence(summary) if summary else None,
                    )
                )
            if evidences:
                return evidences

        # Geolocation or fallback: one evidence item per tool run.
        return [
            Evidence(
                raw=MultimodalSequence(str(raw)),
                action=tool_result.action,
                source=media_reference,
                takeaways=takeaways,
            )
        ]
