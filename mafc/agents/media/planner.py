from __future__ import annotations

import json
from typing import cast

from mafc.agents.media.models import MediaToolName, MediaToolPlan
from mafc.utils.parsing import extract_json_object, is_failed_model_text
from mafc.common.logger import logger
from mafc.common.modeling.message import Message, MessageRole
from mafc.common.modeling.prompt import Prompt

VALID_MEDIA_TOOLS = {"reverse_image_search", "geolocate"}


def plan_media_tools(
    agent,
    instruction: str,
    prior_context: str,
    errors: list[str],
) -> MediaToolPlan | None:
    """Generate and parse the media tool-selection plan from model output."""
    planner_prompt = (
        "You are a media investigation planner.\n"
        "Choose which media investigation tools should run for the task.\n"
        "Available tools:\n"
        '- "reverse_image_search": use when the task asks where the media appeared online, was published, or to find matching copies/context.\n'
        '- "geolocate": use when the task asks where the media was taken or what location is shown.\n\n'
        "Return strict JSON with schema:\n"
        '{"tools": ["reverse_image_search"]}\n\n'
        "Guidelines:\n"
        "- Return one or both tools.\n"
        "- If the task needs both publication/context and location, return both tools.\n"
        "- Use prior session context to answer follow-up questions efficiently.\n"
        "- Only return tool names from the available tools list.\n\n"
        f"Task:\n{instruction}\n\n"
        f"Prior session context:\n{prior_context if prior_context else 'None'}\n"
    )
    try:
        response = agent.model.generate(
            [Message(role=MessageRole.USER, content=Prompt(text=planner_prompt))]
        ).text
        logger.info(f"Media planner response:\n{response}")
        if is_failed_model_text(response):
            return None
        parsed = parse_media_tool_plan(agent, response)
        if parsed is not None:
            return parsed

        repair_prompt = (
            "Convert the following planner response to strict JSON with schema:\n"
            '{"tools": ["reverse_image_search"]}\n'
            "Only return JSON.\n\n"
            f"Response:\n{response}"
        )
        repaired = agent.model.generate(
            [Message(role=MessageRole.USER, content=Prompt(text=repair_prompt))]
        ).text
        if is_failed_model_text(repaired):
            return None
        return parse_media_tool_plan(agent, repaired)
    except Exception as exc:
        logger.error(f"Media planner call failed: {exc}")
        errors.append(f"Media planner call failed: {exc}")
        return None


def parse_media_tool_plan(agent, response_text: str) -> MediaToolPlan | None:
    """Parse planner output into a validated media tool plan."""
    text = response_text.strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.startswith("```")]
        text = "\n".join(lines).strip()
    text = extract_json_object(text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.error(f"[{agent.name}] Failed to parse media planner response as JSON: {exc}")
        return None

    tools = payload.get("tools")
    if not isinstance(tools, list) or not all(isinstance(tool, str) for tool in tools):
        logger.error(f"[{agent.name}] Invalid media planner response: tools must be a list of strings.")
        return None

    normalized_tools: list[MediaToolName] = []
    for tool in tools:
        if tool not in VALID_MEDIA_TOOLS:
            logger.error(f"[{agent.name}] Invalid media planner response: unknown tool '{tool}'.")
            return None
        typed_tool = cast(MediaToolName, tool)
        if typed_tool not in normalized_tools:
            normalized_tools.append(typed_tool)
    if not normalized_tools:
        logger.error(f"[{agent.name}] Invalid media planner response: tools must not be empty.")
        return None
    return MediaToolPlan(tools=normalized_tools)
