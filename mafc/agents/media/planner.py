from __future__ import annotations

import json
from typing import cast

from mafc.agents.media.models import MediaToolName, MediaToolPlan
from mafc.agents.media.tracing import MediaTraceRecorder
from mafc.utils.parsing import (
    extract_json_object,
    is_failed_model_text,
    strip_json_fences,
    try_parse_with_repair,
)
from mafc.common.logger import logger
from mafc.common.modeling.message import Message, MessageRole
from mafc.common.modeling.prompt import Prompt

VALID_MEDIA_TOOLS = {"reverse_image_search", "geolocate"}


def plan_media_tools(
    agent,
    instruction: str,
    prior_context: str,
    errors: list[str],
    trace: MediaTraceRecorder | None = None,
) -> tuple[MediaToolPlan | None, list[Message], str | None]:
    """Generate and parse the media tool-selection plan from model output.

    Returns a tuple of (plan, planner_messages, planner_response_text).
    """
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
    planner_messages = [Message(role=MessageRole.USER, content=Prompt(text=planner_prompt))]
    try:
        _resp = agent.model.generate(planner_messages)
        response = _resp.text
        if trace is not None:
            trace.add_usage(_resp, agent.model.name)
        logger.info(f"Media planner response:\n{response}")
        if is_failed_model_text(response):
            return None, planner_messages, response
        repair_prefix = (
            "Convert the following planner response to strict JSON with schema:\n"
            '{"tools": ["reverse_image_search"]}\n'
            "Only return JSON."
        )
        parsed, _ = try_parse_with_repair(
            response, lambda text: parse_media_tool_plan(agent, text), agent.model, repair_prefix, trace
        )
        return parsed, planner_messages, response
    except Exception as exc:
        logger.error(f"Media planner call failed: {exc}")
        errors.append(f"Media planner call failed: {exc}")
        return None, planner_messages, None


def parse_media_tool_plan(agent, response_text: str) -> MediaToolPlan | None:
    """Parse planner output into a validated media tool plan."""
    text = extract_json_object(strip_json_fences(response_text.strip()))
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
