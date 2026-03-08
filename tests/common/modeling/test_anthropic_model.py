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
from mafc.common.modeling.prompt import Prompt


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
        "mafc.common.modeling.anthropic_model.format_input", lambda prompt, context_window: ["x"]
    )

    api = AnthropicAPI(model="claude", context_window=100)
    out = api(
        prompt=cast(Prompt, SimpleNamespace()),
        system_prompt="sys",
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
    out2 = api2(prompt=cast(Prompt, SimpleNamespace()))
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
        "mafc.common.modeling.anthropic_model.format_input", lambda prompt, context_window: ["x"]
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
        api(prompt=cast(Prompt, SimpleNamespace()))

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
        api(prompt=cast(Prompt, SimpleNamespace()))

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
        api(prompt=cast(Prompt, SimpleNamespace()))


def test_anthropic_model_generate(monkeypatch) -> None:
    monkeypatch.setattr(
        "mafc.common.modeling.model.model_specifier_to_shorthand", lambda s: ("claude_4.5_haiku", "m")
    )
    monkeypatch.setattr("mafc.common.modeling.model.get_model_context_window", lambda n: 1000)
    monkeypatch.setattr("mafc.common.modeling.model.get_model_api_pricing", lambda n: (1.0, 2.0))
    monkeypatch.setattr(
        "mafc.common.modeling.anthropic_model.AnthropicAPI",
        lambda model, context_window: (
            lambda prompt, **kwargs: APIResponse(text="ok", input_token_count=1000, output_token_count=500)
        ),
    )

    model = AnthropicModel(specifier="ANTHROPIC:claude-haiku-4-5-20251001")
    prompt = cast(Prompt, SimpleNamespace(with_videos_as_frames=lambda n: "frames"))
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
        "mafc.common.modeling.anthropic_model.AnthropicAPI", lambda model, context_window: None
    )

    model = AnthropicModel(specifier="ANTHROPIC:claude-haiku-4-5-20251001")
    monkeypatch.setattr(model, "api", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    prompt = cast(Prompt, SimpleNamespace(with_videos_as_frames=lambda n: "frames"))
    with pytest.raises(RuntimeError):
        model.generate(prompt)
