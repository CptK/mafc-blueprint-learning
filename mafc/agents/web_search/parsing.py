from __future__ import annotations

import re


def extract_json_object(text: str) -> str:
    """Extract the first JSON object from potentially mixed model output."""
    if text.startswith("{") and text.endswith("}"):
        return text
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return match.group(0).strip() if match else text


def is_failed_model_text(text: str) -> bool:
    """Return True when the model output matches known failure placeholders."""
    normalized = text.strip().lower()
    return normalized in {
        "",
        "failed to generate a response.",
        "failed to generate a response",
    }
