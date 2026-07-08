from __future__ import annotations

import json
import re

from ezmm import MultimodalSequence
from ezmm.common.items import Image, Video

from mafc.common.modeling.model import Model, Response
from mafc.common.modeling.message import Message, MessageRole
from mafc.common.modeling.prompt import Prompt
from mafc.utils.parsing import extract_json_object, is_failed_model_text
from mafc.utils.media import build_media_json_instruction, parse_media_relevance

_MAX_MEDIA_PER_REQUEST = 50


def synthesize_step(model: Model, instruction: str, observations: list[str]) -> str:
    """Synthesize current observations across sources for the current step."""
    synthesis_prompt = (
        "You are a factual evidence synthesizer in a non-conversational setting.\n"
        "Synthesize the observations across sources, focusing only on information relevant to the task.\n"
        "State agreements and disagreements between sources when present.\n"
        "Include concrete facts and source references where possible.\n"
        "Call out important uncertainties or missing evidence.\n\n"
        f"Task:\n{instruction}\n\n"
        f"Observations:\n{chr(10).join(observations)}"
    )
    try:
        synthesis = model.generate(
            [Message(role=MessageRole.USER, content=Prompt(text=synthesis_prompt))]
        ).text.strip()
        if not synthesis or is_failed_model_text(synthesis):
            return "\n\n".join(observations)
        return synthesis
    except Exception:
        return "\n\n".join(observations)


def _neutralize_media_refs(text: str) -> str:
    """Turn ezmm media reference tags into plain text so they are not resolved
    into attached media when the text is embedded in another prompt."""
    return re.sub(r"<(image|video|audio):(\d+)>", r"[\1 \2]", text)


def summarize_observation(
    model: Model,
    instruction: str,
    observation: str,
    media_items: list[Image | Video] = [],
    claim_text: str | None = None,
) -> tuple[str, list[Image | Video], Response | None]:
    """Summarize a single observation block with the summarization model.

    ``claim_text`` anchors relevance to the claim under investigation (not just the
    search query), so pages that merely match the query's topic do not fill the fact
    budget with background irrelevant to verifying the claim.

    Returns (summary_text, relevant_media, response). summary_text is empty string if no relevant
    content. relevant_media contains only media items the model judged relevant. response is None
    if the call failed entirely.
    """
    media_instruction = build_media_json_instruction(media_items, context="The page")

    claim_block = f"Claim under investigation: {_neutralize_media_refs(claim_text)}\n" if claim_text else ""
    relevance_rule = (
        "- Relevance means: the fact helps confirm, refute, date, locate, or trace the provenance "
        "of the claim (or of its media). Skip generic background that matches the query's topic "
        "but does not bear on the claim's accuracy.\n"
        if claim_text
        else ""
    )
    summary_prompt = (
        "Extract factual statements from the web page below that are relevant to the search query"
        + (" and the claim under investigation" if claim_text else "")
        + ".\n"
        "Rules:\n"
        "- Only report facts explicitly stated in the page. Do not infer or conclude.\n"
        "- Preserve exact figures, dates, names, attributions, and any qualifying conditions "
        "(e.g. what exactly a statistic refers to, the time/place/scope it applies to). These "
        "specifics are often what decides a claim's accuracy, so do not generalize them away.\n"
        f"{relevance_rule}"
        "- Do not draw verdicts, make recommendations, or offer further help.\n"
        "- Do not use first-person. Do not address the reader.\n"
        "- Return strict JSON only with schema:\n"
        '  {"facts": ["...", ...], "no_relevant_content": true|false}\n'
        "- Set no_relevant_content to true if the page contains no relevant information.\n"
        "- Limit facts to 15 bullets maximum.\n"
        f"{media_instruction}\n"
        f"{claim_block}"
        f"Search query: {instruction}\n\n"
        f"Page content:\n{observation}"
    )
    content: MultimodalSequence = MultimodalSequence(summary_prompt, *media_items[:_MAX_MEDIA_PER_REQUEST])
    try:
        resp = model.generate([Message(role=MessageRole.USER, content=content)])
        summary_text, relevant_media = _parse_summarize_response(resp.text, media_items)
        return summary_text, relevant_media, resp
    except Exception:
        return "", [], None


def _parse_summarize_response(
    response_text: str, media_items: list[Image | Video]
) -> tuple[str, list[Image | Video]]:
    """Parse the JSON response from summarize_observation."""
    text = extract_json_object(response_text.strip())
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return response_text.strip(), []

    if payload.get("no_relevant_content"):
        return "", []

    facts = payload.get("facts", [])
    summary = "\n".join(f"- {f}" for f in facts if isinstance(f, str)).strip()

    relevant_media = parse_media_relevance(payload.get("media", []), media_items)
    return summary, relevant_media
