from __future__ import annotations

import json

from mafc.common.logger import logger
from mafc.common.modeling.message import Message, MessageRole
from mafc.common.modeling.prompt import Prompt

from mafc.agents.web_search.models import SearchPlanStep
from mafc.agents.web_search.tracing import WebSearchTraceRecorder
from mafc.utils.parsing import (
    extract_json_object,
    is_failed_model_text,
    strip_json_fences,
    try_parse_with_repair,
)


def plan_step(
    agent,
    instruction: str,
    prior_context: str,
    errors: list[str],
    step: int | None = None,
    trace: WebSearchTraceRecorder | None = None,
) -> SearchPlanStep | None:
    """Generate and parse the next search plan step from model output."""
    planner_prompt = (
        "You are a web-search planner.\n"
        "Given the task and prior session context, propose next search queries.\n"
        "Return strict JSON with keys:\n"
        '- "queries": array of strings\n'
        '- "done": boolean\n\n'
        "Guidelines:\n"
        "- Keep queries specific and evidence-seeking.\n"
        "- If enough evidence is already gathered, set done=true and queries=[].\n"
        f"- If you want to gather more information, set done=false and propose up to {agent.max_queries_per_step} new queries.\n"
        "- You have multiple iterations to gather information, so you can search for facts building on previous findings.\n"
        "- Use prior session context to answer follow-up questions efficiently and avoid repeating work.\n\n"
        f"Task:\n{instruction}\n\n"
        f"Prior session context:\n{prior_context if prior_context else 'None'}\n\n"
    )
    planner_messages = [Message(role=MessageRole.USER, content=Prompt(text=planner_prompt))]
    if trace is not None and step is not None:
        trace.record_planner_messages(planner_messages, step=step)
    try:
        _resp = agent.model.generate(planner_messages)
        response = _resp.text
        if trace is not None:
            trace.add_usage(_resp, agent.model.name)
        if trace is not None and step is not None:
            trace.record_planner_response(response, step=step)
        logger.info(f"Planner response:\n{response}")
        if is_failed_model_text(response):
            return None
        repair_prefix = (
            "Convert the following planner response to strict JSON with schema:\n"
            '{"queries": ["..."], "done": false}\n'
            "Only return JSON."
        )
        parsed, repair_text = try_parse_with_repair(
            response, lambda text: parse_plan(agent, text), agent.model, repair_prefix, trace
        )
        if trace is not None and step is not None and repair_text is not None:
            trace.record_planner_repair(
                prompt=f"{repair_prefix}\n\nResponse:\n{response}",
                response_text=repair_text,
                step=step,
            )
        return parsed
    except Exception as exc:
        logger.error(f"Planner call failed: {exc}")
        errors.append(f"Planner call failed: {exc}")
        if trace is not None:
            trace.record_error(step=step, phase="planner_call", message=f"Planner call failed: {exc}")
        return None


def parse_plan(agent, response_text: str) -> SearchPlanStep | None:
    """Parse planner output into a validated `SearchPlanStep` object."""
    text = extract_json_object(strip_json_fences(response_text.strip()))
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.error(f"[{agent.name}] Failed to parse planner response as JSON: {exc}")
        return None
    queries = payload.get("queries")
    done = payload.get("done", False)
    if not isinstance(queries, list) or not all(isinstance(q, str) for q in queries):
        logger.error(f"[{agent.name}] Invalid planner response: queries must be a list of strings.")
        return None
    if not isinstance(done, bool):
        logger.error(f"[{agent.name}] Invalid planner response: done must be a boolean.")
        return None
    return SearchPlanStep(queries=queries, done=done)
