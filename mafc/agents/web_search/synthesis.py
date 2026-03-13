from __future__ import annotations

from mafc.common.modeling.model import Model
from mafc.common.modeling.message import Message, MessageRole
from mafc.common.modeling.prompt import Prompt
from mafc.utils.parsing import is_failed_model_text


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


def summarize_observation(model: Model, instruction: str, observation: str) -> str:
    """Summarize a single observation block with the summarization model."""
    summary_prompt = (
        "Extract factual statements from the web page below that are relevant to the search query.\n"
        "Rules:\n"
        "- Only report facts explicitly stated in the page. Do not infer or conclude.\n"
        "- Do not draw verdicts, make recommendations, or offer further help.\n"
        "- Do not use first-person. Do not address the reader.\n"
        "- Return a plain bulleted list of factual statements (max 7 bullets).\n"
        "- If the page contains no relevant information, return: NO_RELEVANT_CONTENT\n\n"
        f"Search query: {instruction}\n\n"
        f"Page content:\n{observation}"
    )
    try:
        summary = model.generate(
            [Message(role=MessageRole.USER, content=Prompt(text=summary_prompt))]
        ).text.strip()
        if not summary or is_failed_model_text(summary) or summary == "NO_RELEVANT_CONTENT":
            return ""
        return summary
    except Exception:
        return observation
