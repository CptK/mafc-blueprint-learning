import os

import pytest

import config.globals  # noqa: F401  # Triggers .env loading like normal app startup.
from mafc.common.modeling.anthropic_model import AnthropicAPI
from mafc.common.modeling.gemini_model import GeminiAPI
from mafc.common.modeling.openai_model import OpenAIAPI
from mafc.common.modeling.prompt import Prompt


def _skip_if_missing_key(*key_names: str) -> None:
    if not any(os.environ.get(name) for name in key_names):
        pytest.skip(f"Missing API key. Provide one of: {', '.join(key_names)}")


def _call_or_skip(fn):
    try:
        return fn()
    except Exception as exc:  # pragma: no cover - depends on runtime network/provider status
        pytest.skip(f"Live API call skipped due to runtime/provider issue: {exc}")


@pytest.mark.integration
def test_openai_live_api_small_call() -> None:
    _skip_if_missing_key("OPENAI_API_KEY", "openai_api_key")
    api = OpenAIAPI(model="gpt-5-mini-2025-08-07", context_window=1024)
    out = _call_or_skip(
        lambda: api(
            prompt=Prompt(text="Reply with exactly one word: OK"),
            max_response_length=16,
            temperature=1.0,
            top_p=1.0,
        )
    )
    assert isinstance(out.text, str)
    assert out.text.strip()


@pytest.mark.integration
def test_anthropic_live_api_small_call() -> None:
    _skip_if_missing_key("ANTHROPIC_API_KEY", "anthropic_api_key")
    api = AnthropicAPI(model="claude-haiku-4-5-20251001", context_window=1024)
    out = _call_or_skip(
        lambda: api(
            prompt=Prompt(text="Reply with exactly one word: OK"),
            max_response_length=16,
            temperature=0.0,
            top_p=1.0,
        )
    )
    assert isinstance(out.text, str)
    assert out.text.strip()


@pytest.mark.integration
def test_gemini_live_api_small_call() -> None:
    _skip_if_missing_key("GEMINI_API_KEY", "gemini_api_key")
    api = GeminiAPI(model="gemini-3.1-flash-lite-preview", context_window=1024)
    out = _call_or_skip(
        lambda: api(
            prompt=Prompt(text="Reply with exactly one word: OK"),
            max_response_length=16,
            temperature=0.0,
            top_p=1.0,
        )
    )
    assert isinstance(out.text, str)
    assert out.text.strip()
