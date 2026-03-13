from __future__ import annotations

import json

from mafc.common.logger import logger
from mafc.common.modeling.message import Message, MessageRole
from mafc.common.modeling.prompt import Prompt

from mafc.agents.web_search.models import SearchPlanStep
from mafc.utils.parsing import extract_json_object, is_failed_model_text


def plan_step(
    agent,
    instruction: str,
    prior_context: str,
    errors: list[str],
    step: int | None = None,
    trace=None,
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
        response = agent.model.generate(planner_messages).text
        if trace is not None and step is not None:
            trace.record_planner_response(response, step=step)
        logger.info(f"Planner response:\n{response}")
        if is_failed_model_text(response):
            return None
        parsed = parse_plan(agent, response)
        if parsed is not None:
            return parsed

        repair_prompt = (
            "Convert the following planner response to strict JSON with schema:\n"
            '{"queries": ["..."], "done": false}\n'
            "Only return JSON.\n\n"
            f"Response:\n{response}"
        )
        repair_messages = [Message(role=MessageRole.USER, content=Prompt(text=repair_prompt))]
        repaired = agent.model.generate(repair_messages).text
        if trace is not None and step is not None:
            trace.record_planner_repair(prompt=repair_prompt, response_text=repaired, step=step)
        if is_failed_model_text(repaired):
            return None
        return parse_plan(agent, repaired)
    except Exception as exc:
        logger.error(f"Planner call failed: {exc}")
        errors.append(f"Planner call failed: {exc}")
        if trace is not None:
            trace.record_error(step=step, phase="planner_call", message=f"Planner call failed: {exc}")
        return None


def parse_plan(agent, response_text: str) -> SearchPlanStep | None:
    """Parse planner output into a validated `SearchPlanStep` object."""
    text = response_text.strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.startswith("```")]
        text = "\n".join(lines).strip()
    text = extract_json_object(text)
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
