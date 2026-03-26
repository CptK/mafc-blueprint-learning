from typing import Any, Callable, TypeVar
from urllib.parse import urlparse

T = TypeVar("T")


def strip_json_fences(text: str) -> str:
    """Strip markdown code fences from model output."""
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.startswith("```")]
        return "\n".join(lines).strip()
    return text


def try_parse_with_repair(
    response_text: str,
    parse_fn: Callable[[str], T | None],
    model: Any,
    repair_prompt_prefix: str,
    trace: Any = None,
) -> tuple[T | None, str | None]:
    """Parse response; on failure, repair with one model call.

    Returns (result, repair_response_text). repair_response_text is None if the
    first parse succeeded. result is None if both parses fail.
    trace, if provided, must have add_usage(response, model_name).
    """
    parsed = parse_fn(response_text)
    if parsed is not None:
        return parsed, None

    from mafc.common.modeling.message import Message, MessageRole
    from mafc.common.modeling.prompt import Prompt

    repair_prompt = f"{repair_prompt_prefix}\n\nResponse:\n{response_text}"
    repair_messages = [Message(role=MessageRole.USER, content=Prompt(text=repair_prompt))]
    _repair_resp = model.generate(repair_messages)
    repair_text = _repair_resp.text.strip()
    if trace is not None:
        trace.add_usage(_repair_resp, model.name)
    return parse_fn(repair_text), repair_text


def get_domain(url: str) -> str:
    parsed_url = urlparse(url)
    netloc = parsed_url.netloc.lower()  # get the network location (netloc)
    domain = ".".join(netloc.split(".")[-2:])  # remove subdomains
    return domain


def get_base_domain(url: str) -> str:
    """
    Extracts the base domain from a given URL, ignoring common subdomains like 'www' and 'm'.

    Args:
        url (str): The URL to extract the base domain from.

    Returns:
        str: The base domain (e.g., 'facebook.com').
    """
    netloc = urlparse(url).netloc

    # Remove common subdomains like 'www.' and 'm.'
    if netloc.startswith("www.") or netloc.startswith("m."):
        netloc = netloc.split(".", 1)[1]  # Remove the first part (e.g., 'www.', 'm.')

    return netloc


def extract_json_object(text: str) -> str:
    """Extract the first complete JSON object from potentially mixed model output."""
    text = text.strip()
    start = text.find("{")
    if start == -1:
        return text
    # Walk from the first '{' and find the matching closing '}' via bracket counting.
    # This correctly handles models that repeat their output (e.g. plain JSON followed
    # by the same JSON inside a ```json fence) by stopping at the first complete object.
    depth = 0
    in_string = False
    escape_next = False
    for i, ch in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1].strip()
    return text


def is_failed_model_text(text: str) -> bool:
    """Return True when the model output matches known failure placeholders."""
    normalized = text.strip().lower()
    return normalized in {
        "",
        "failed to generate a response.",
        "failed to generate a response",
    }
