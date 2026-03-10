from __future__ import annotations

from mafc.common.modeling.prompt import Prompt

from .parsing import is_failed_model_text


def synthesize_step(agent, instruction: str, observations: list[str]) -> str:
    """Synthesize current observations across sources for the current step."""
    synthesis_prompt = (
        "You are a factual evidence synthesizer.\n"
        "Synthesize the observations across sources, focusing only on information relevant to the task.\n"
        "State agreements and disagreements between sources when present.\n"
        "Include concrete facts and source references where possible.\n"
        "Call out important uncertainties or missing evidence.\n\n"
        f"Task:\n{instruction}\n\n"
        f"Observations:\n{chr(10).join(observations)}"
    )
    try:
        synthesis = agent.summarization_model.generate(Prompt(text=synthesis_prompt)).text.strip()
        if not synthesis or is_failed_model_text(synthesis):
            return "\n\n".join(observations)
        return synthesis
    except Exception:
        return "\n\n".join(observations)


def summarize_observation(agent, instruction: str, observation: str) -> str:
    """Summarize a single observation block with the summarization model."""
    summary_prompt = (
        "You are a factual evidence summarizer.\n"
        "Summarize only information relevant to the task.\n"
        "Include concrete facts and source references where possible.\n\n"
        f"Task:\n{instruction}\n\n"
        f"Observation:\n{observation}"
    )
    try:
        summary = agent.summarization_model.generate(Prompt(text=summary_prompt)).text.strip()
        if not summary or is_failed_model_text(summary):
            return observation
        return summary
    except Exception:
        return observation
