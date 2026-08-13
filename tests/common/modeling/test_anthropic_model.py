from types import SimpleNamespace
from typing import cast

import pytest

from mafc.common.modeling.anthropic_model import (
    AnthropicAPI,
    AnthropicModel,
    _resolve_anthropic_key,
    format_input,
)
from mafc.common.modeling.model import APIResponse
from mafc.common.modeling.message import Message, MessageRole
from mafc.common.modeling.prompt import Prompt
from tests.common.modeling.helpers import LONG_PROMPT, assert_abbreviated, capture_errors


class FakePromptBlocks:
    def __init__(self, blocks):
        self._blocks = blocks

    def to_list(self):
        return self._blocks


def test_resolve_anthropic_key(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("anthropic_api_key", "k")
    assert _resolve_anthropic_key() == "k"


def test_anthropic_format_input_text(monkeypatch) -> None:
    monkeypatch.setattr("mafc.common.modeling.anthropic_model.Image", type("FakeImage", (), {}))
    monkeypatch.setattr("mafc.common.modeling.anthropic_model.Video", type("FakeVideo", (), {}))
    out = format_input(cast(Prompt, FakePromptBlocks(["hello"])), context_window=100)
    assert out == [{"type": "text", "text": "hello"}]


def test_anthropic_format_input_image_and_video(monkeypatch) -> None:
    class FakeImage:
        width = 100
        height = 100

        def get_base64_encoded(self):
            return "abc"

    class FakeVideo:
        pass

    monkeypatch.setattr("mafc.common.modeling.anthropic_model.Image", FakeImage)
    monkeypatch.setattr("mafc.common.modeling.anthropic_model.Video", FakeVideo)
    monkeypatch.setattr("mafc.common.modeling.anthropic_model.count_image_tokens_estimate", lambda image: 1)

    out = format_input(cast(Prompt, FakePromptBlocks([FakeImage(), FakeVideo()])), context_window=5)
    assert out == [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": "abc"},
        }
    ]


def test_anthropic_format_input_image_too_large(monkeypatch) -> None:
    class FakeImage:
        width = 100
        height = 100

        def get_base64_encoded(self):
            return "abc"

    monkeypatch.setattr("mafc.common.modeling.anthropic_model.Image", FakeImage)
    monkeypatch.setattr("mafc.common.modeling.anthropic_model.Video", type("FakeVideo", (), {}))
    monkeypatch.setattr("mafc.common.modeling.anthropic_model.count_image_tokens_estimate", lambda image: 100)

    out = format_input(cast(Prompt, FakePromptBlocks([FakeImage()])), context_window=1)
    assert out == []


def test_anthropic_api_requires_key(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("anthropic_api_key", raising=False)
    with pytest.raises(ValueError):
        AnthropicAPI(model="x", context_window=10)


def test_anthropic_api_call_success_and_fallback(monkeypatch) -> None:
    class Block:
        def __init__(self, t, text):
            self.type = t
            self.text = text

    class FakeResponse:
        def __init__(self):
            self.content = [Block("text", "a"), Block("other", "x"), Block("text", "b")]
            self.usage = SimpleNamespace(input_tokens=7, output_tokens=3)

    class FakeClient:
        class messages:
            @staticmethod
            def create(**kwargs):
                return FakeResponse()

    warnings: list[str] = []
    monkeypatch.setattr("mafc.common.modeling.anthropic_model.logger.warning", warnings.append)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setattr(
        "mafc.common.modeling.anthropic_model.anthropic.Anthropic", lambda api_key, timeout: FakeClient()
    )
    monkeypatch.setattr(
        "mafc.common.modeling.anthropic_model.format_input", lambda content, context_window: ["x"]
    )

    api = AnthropicAPI(model="claude", context_window=100)
    out = api(
        messages=[
            Message(role=MessageRole.SYSTEM, content=Prompt(text="sys")),
            Message(role=MessageRole.USER, content=Prompt(text="u")),
        ],
        temperature=0.1,
        top_p=0.2,
        max_response_length=10,
    )
    assert out.text == "a\nb"
    assert out.total_token_count == 10
    assert warnings

    class BadIterable:
        def __iter__(self):
            raise RuntimeError("bad content")

    class BadContentResponse:
        def __init__(self):
            self.content = BadIterable()
            self.usage = None

    class BadClient:
        class messages:
            @staticmethod
            def create(**kwargs):
                return BadContentResponse()

    monkeypatch.setattr(
        "mafc.common.modeling.anthropic_model.anthropic.Anthropic", lambda api_key, timeout: BadClient()
    )
    api2 = AnthropicAPI(model="claude", context_window=100)
    out2 = api2(messages=[Message(role=MessageRole.USER, content=Prompt(text="u"))])
    assert out2.text


def test_anthropic_api_errors(monkeypatch) -> None:
    class DummyRateLimit(Exception):
        pass

    class DummyAuth(Exception):
        pass

    monkeypatch.setattr("mafc.common.modeling.anthropic_model.anthropic.RateLimitError", DummyRateLimit)
    monkeypatch.setattr("mafc.common.modeling.anthropic_model.anthropic.AuthenticationError", DummyAuth)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setattr(
        "mafc.common.modeling.anthropic_model.format_input", lambda content, context_window: ["x"]
    )

    class RateClient:
        class messages:
            @staticmethod
            def create(**kwargs):
                raise DummyRateLimit("rate")

    monkeypatch.setattr(
        "mafc.common.modeling.anthropic_model.anthropic.Anthropic", lambda api_key, timeout: RateClient()
    )
    api = AnthropicAPI(model="claude", context_window=100)
    with pytest.raises(DummyRateLimit):
        api(messages=[Message(role=MessageRole.USER, content=Prompt(text="u"))])

    class AuthClient:
        class messages:
            @staticmethod
            def create(**kwargs):
                raise DummyAuth("auth")

    monkeypatch.setattr(
        "mafc.common.modeling.anthropic_model.anthropic.Anthropic", lambda api_key, timeout: AuthClient()
    )
    api = AnthropicAPI(model="claude", context_window=100)
    with pytest.raises(DummyAuth):
        api(messages=[Message(role=MessageRole.USER, content=Prompt(text="u"))])

    class OtherClient:
        class messages:
            @staticmethod
            def create(**kwargs):
                raise RuntimeError("other")

    monkeypatch.setattr(
        "mafc.common.modeling.anthropic_model.anthropic.Anthropic", lambda api_key, timeout: OtherClient()
    )
    api = AnthropicAPI(model="claude", context_window=100)
    with pytest.raises(RuntimeError):
        api(messages=[Message(role=MessageRole.USER, content=Prompt(text="u"))])


def test_anthropic_model_generate(monkeypatch) -> None:
    monkeypatch.setattr(
        "mafc.common.modeling.model.model_specifier_to_shorthand", lambda s: ("claude_4.5_haiku", "m")
    )
    monkeypatch.setattr("mafc.common.modeling.model.get_model_context_window", lambda n: 1000)
    monkeypatch.setattr("mafc.common.modeling.model.get_model_api_pricing", lambda n: (1.0, 2.0))
    monkeypatch.setattr(
        "mafc.common.modeling.anthropic_model.messages_with_videos_as_frames", lambda messages, n: messages
    )
    monkeypatch.setattr(
        "mafc.common.modeling.anthropic_model.AnthropicAPI",
        lambda model, context_window: (
            lambda prompt, **kwargs: APIResponse(text="ok", input_token_count=1000, output_token_count=500)
        ),
    )

    model = AnthropicModel(specifier="ANTHROPIC:claude-haiku-4-5-20251001")
    prompt = [Message(role=MessageRole.USER, content=Prompt(text="hello"))]
    response = model.generate(prompt)
    assert response.text == "ok"
    assert response.total_cost == 0.002


def test_anthropic_model_generate_reraises(monkeypatch) -> None:
    monkeypatch.setattr(
        "mafc.common.modeling.model.model_specifier_to_shorthand", lambda s: ("claude_4.5_haiku", "m")
    )
    monkeypatch.setattr("mafc.common.modeling.model.get_model_context_window", lambda n: 1000)
    monkeypatch.setattr("mafc.common.modeling.model.get_model_api_pricing", lambda n: (1.0, 2.0))
    monkeypatch.setattr(
        "mafc.common.modeling.anthropic_model.messages_with_videos_as_frames", lambda messages, n: messages
    )
    monkeypatch.setattr(
        "mafc.common.modeling.anthropic_model.AnthropicAPI", lambda model, context_window: None
    )

    model = AnthropicModel(specifier="ANTHROPIC:claude-haiku-4-5-20251001")
    monkeypatch.setattr(model, "api", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    prompt = [Message(role=MessageRole.USER, content=Prompt(text="hello"))]
    with pytest.raises(RuntimeError):
        model.generate(prompt)


def test_anthropic_api_error_abbreviates_logged_input(monkeypatch) -> None:
    class BrokenClient:
        class messages:
            @staticmethod
            def create(**kwargs):
                raise RuntimeError("boom")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setattr(
        "mafc.common.modeling.anthropic_model.format_input", lambda content, context_window: ["x"]
    )
    monkeypatch.setattr(
        "mafc.common.modeling.anthropic_model.anthropic.Anthropic", lambda api_key, timeout: BrokenClient()
    )

    logged = capture_errors(monkeypatch)
    api = AnthropicAPI(model="claude", context_window=100)
    with pytest.raises(RuntimeError):
        api(messages=[Message(role=MessageRole.USER, content=Prompt(text=LONG_PROMPT))])

    assert_abbreviated(logged)


def _capture_create_kwargs(monkeypatch, model: str, **call_kwargs) -> dict:
    """Run one API call against a fake client and return the request it built."""
    from types import SimpleNamespace

    captured: dict = {}

    class FakeClient:
        class messages:
            @staticmethod
            def create(**kwargs):
                captured.update(kwargs)
                return SimpleNamespace(
                    content=[SimpleNamespace(type="text", text="ok")],
                    usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                )

    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setattr(
        "mafc.common.modeling.anthropic_model.anthropic.Anthropic", lambda api_key, timeout: FakeClient()
    )
    monkeypatch.setattr(
        "mafc.common.modeling.anthropic_model.format_input",
        lambda content, context_window: [{"type": "text", "text": "x"}],
    )
    api = AnthropicAPI(model=model, context_window=200000)
    system_text = call_kwargs.pop("system_text", None)
    msgs = [Message(role=MessageRole.USER, content=Prompt(text="u"))]
    if system_text:
        msgs.insert(0, Message(role=MessageRole.SYSTEM, content=Prompt(text=system_text)))
    api(
        messages=msgs,
        max_response_length=call_kwargs.pop("max_response_length", 64000),
        **call_kwargs,
    )
    return captured


def test_adaptive_thinking_keeps_sampling_params(monkeypatch) -> None:
    """Opus 4.6 takes adaptive thinking and still accepts temperature."""
    sent = _capture_create_kwargs(monkeypatch, "claude-opus-4-6", thinking=True, temperature=1.0)
    assert sent["thinking"] == {"type": "adaptive"}
    assert sent["temperature"] == 1.0
    assert "output_config" not in sent


def test_effort_rides_inside_output_config(monkeypatch) -> None:
    sent = _capture_create_kwargs(
        monkeypatch, "claude-opus-4-6", thinking=True, temperature=1.0, effort="medium"
    )
    assert sent["output_config"] == {"effort": "medium"}


def test_thinking_off_sends_no_thinking_block(monkeypatch) -> None:
    sent = _capture_create_kwargs(monkeypatch, "claude-opus-4-6", thinking=False, temperature=1.0)
    assert "thinking" not in sent


def test_legacy_models_get_a_budget_and_drop_sampling(monkeypatch) -> None:
    """Pre-4.6 models reject adaptive thinking, and a budget rules out sampling."""
    sent = _capture_create_kwargs(monkeypatch, "claude-haiku-4-5-20251001", thinking=True, temperature=0.7)
    assert sent["thinking"]["type"] == "enabled"
    assert sent["thinking"]["budget_tokens"] < 64000
    assert "temperature" not in sent


def test_thinking_skipped_when_no_budget_fits(monkeypatch) -> None:
    """The budget must stay under max_tokens, so a tight cap turns thinking off."""
    sent = _capture_create_kwargs(
        monkeypatch,
        "claude-haiku-4-5-20251001",
        thinking=True,
        temperature=0.7,
        max_response_length=1024,
    )
    assert "thinking" not in sent


def test_cache_system_marks_only_the_system_block(monkeypatch) -> None:
    """Planner shape: stable system prefix, volatile user turn."""
    sent = _capture_create_kwargs(monkeypatch, "claude-opus-4-6", cache="system", system_text="stable prefix")
    assert sent["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in sent["messages"][-1]["content"][-1]


def test_cache_prompt_marks_the_last_content_block(monkeypatch) -> None:
    """Judge shape: the whole request repeats, so the prefix spans system + messages."""
    sent = _capture_create_kwargs(monkeypatch, "claude-opus-4-6", cache="prompt", system_text="rules")
    assert sent["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in sent["system"][0]


def test_no_cache_placement_marks_nothing(monkeypatch) -> None:
    sent = _capture_create_kwargs(monkeypatch, "claude-opus-4-6", system_text="rules")
    assert "cache_control" not in sent["system"][0]
    assert "cache_control" not in sent["messages"][-1]["content"][-1]


def test_unknown_cache_placement_is_rejected(monkeypatch) -> None:
    """A typo must fail loudly; silently not caching is the failure mode to avoid."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setattr(
        "mafc.common.modeling.anthropic_model.anthropic.Anthropic", lambda api_key, timeout: object()
    )
    with pytest.raises(ValueError, match="Unknown cache placement"):
        AnthropicModel(specifier="ANTHROPIC:claude-opus-4-6", cache="sytsem")


def _model_with_fake_client(monkeypatch, captured: dict, **model_kwargs):
    from types import SimpleNamespace

    class FakeClient:
        class messages:
            @staticmethod
            def create(**kwargs):
                captured.update(kwargs)
                return SimpleNamespace(
                    content=[SimpleNamespace(type="text", text="ok")],
                    usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                )

    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setattr(
        "mafc.common.modeling.anthropic_model.anthropic.Anthropic", lambda api_key, timeout: FakeClient()
    )
    monkeypatch.setattr(
        "mafc.common.modeling.anthropic_model.format_input",
        lambda content, context_window: [{"type": "text", "text": "x"}],
    )
    return AnthropicModel(specifier="ANTHROPIC:claude-opus-4-6", **model_kwargs)


def test_unset_sampling_params_are_not_sent(monkeypatch) -> None:
    """Neither value was chosen, so neither is forwarded and nothing conflicts."""
    captured: dict = {}
    warnings: list[str] = []
    monkeypatch.setattr("mafc.common.modeling.anthropic_model.logger.warning", warnings.append)
    model = _model_with_fake_client(monkeypatch, captured)
    model.generate([Message(role=MessageRole.USER, content=Prompt(text="u"))])
    assert "temperature" not in captured
    assert "top_p" not in captured
    assert not warnings


def test_only_the_chosen_sampling_param_is_sent(monkeypatch) -> None:
    captured: dict = {}
    warnings: list[str] = []
    monkeypatch.setattr("mafc.common.modeling.anthropic_model.logger.warning", warnings.append)
    model = _model_with_fake_client(monkeypatch, captured, temperature=0.3)
    model.generate([Message(role=MessageRole.USER, content=Prompt(text="u"))])
    assert captured["temperature"] == 0.3
    assert "top_p" not in captured
    assert not warnings


def test_top_p_alone_survives(monkeypatch) -> None:
    """A caller who picked top_p instead of temperature keeps it."""
    captured: dict = {}
    model = _model_with_fake_client(monkeypatch, captured, top_p=0.8)
    model.generate([Message(role=MessageRole.USER, content=Prompt(text="u"))])
    assert captured["top_p"] == 0.8
    assert "temperature" not in captured


def test_real_conflict_warns_once_not_per_call(monkeypatch) -> None:
    """A genuine conflict is still reported, but cannot flood a 1000-claim run."""
    captured: dict = {}
    warnings: list[str] = []
    monkeypatch.setattr("mafc.common.modeling.anthropic_model.logger.warning", warnings.append)
    model = _model_with_fake_client(monkeypatch, captured, temperature=0.3, top_p=0.8)
    for _ in range(5):
        model.generate([Message(role=MessageRole.USER, content=Prompt(text="u"))])
    assert captured["temperature"] == 0.3
    assert "top_p" not in captured
    assert len(warnings) == 1
