from types import SimpleNamespace
from typing import cast

import pytest
from ezmm import Image

from mafc.common.modeling.model import APIResponse
from mafc.common.modeling.openai_model import (
    OpenAIAPI,
    OpenAIModel,
    count_image_tokens,
    count_tokens,
    format_input,
)
from mafc.common.modeling.prompt import Prompt


class FakePromptBlocks:
    def __init__(self, blocks):
        self._blocks = blocks

    def to_list(self):
        return self._blocks


def test_count_image_tokens() -> None:
    img = SimpleNamespace(width=600, height=512)
    assert count_image_tokens(cast(Image, img)) == 425


def test_format_input_text_truncation(monkeypatch) -> None:
    monkeypatch.setattr("mafc.common.modeling.openai_model.Image", type("FakeImage", (), {}))
    monkeypatch.setattr("mafc.common.modeling.openai_model.Video", type("FakeVideo", (), {}))
    prompt = FakePromptBlocks(["hello world"])
    out = format_input(cast(Prompt, prompt), context_window=1)
    assert len(out) == 1
    assert out[0]["type"] == "text"
    assert isinstance(out[0]["text"], str)


def test_format_input_with_image_and_video_branches(monkeypatch) -> None:
    class FakeImage:
        def get_base64_encoded(self):
            return "abc"

    class FakeVideo:
        pass

    monkeypatch.setattr("mafc.common.modeling.openai_model.Image", FakeImage)
    monkeypatch.setattr("mafc.common.modeling.openai_model.Video", FakeVideo)
    monkeypatch.setattr("mafc.common.modeling.openai_model.count_image_tokens", lambda image: 1)

    out = format_input(cast(Prompt, FakePromptBlocks([FakeImage(), FakeVideo()])), context_window=5)
    assert out == [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc"}}]


def test_format_input_skips_image_if_budget_too_small(monkeypatch) -> None:
    class FakeImage:
        def get_base64_encoded(self):
            return "abc"

    monkeypatch.setattr("mafc.common.modeling.openai_model.Image", FakeImage)
    monkeypatch.setattr("mafc.common.modeling.openai_model.Video", type("FakeVideo", (), {}))
    monkeypatch.setattr("mafc.common.modeling.openai_model.count_image_tokens", lambda image: 10)

    out = format_input(cast(Prompt, FakePromptBlocks([FakeImage()])), context_window=1)
    assert out == []


def test_count_tokens_for_prompt_with_images(monkeypatch) -> None:
    class FakePrompt:
        def __str__(self):
            return "abc"

        def has_images(self):
            return True

        @property
        def images(self):
            return [SimpleNamespace(width=512, height=512)]

    monkeypatch.setattr("mafc.common.modeling.openai_model.Prompt", FakePrompt)
    assert count_tokens(cast(Prompt, FakePrompt())) > 0


def test_count_tokens_for_plain_string() -> None:
    assert count_tokens("abc") > 0


def test_openai_api_requires_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("openai_api_key", raising=False)
    with pytest.raises(ValueError):
        OpenAIAPI(model="x", context_window=10)


def test_openai_api_call_success(monkeypatch) -> None:
    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    return SimpleNamespace(
                        choices=[SimpleNamespace(message=SimpleNamespace(content="hello"))],
                        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                    )

    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setattr("mafc.common.modeling.openai_model.OpenAI", lambda api_key, timeout: FakeClient())
    monkeypatch.setattr(
        "mafc.common.modeling.openai_model.format_input", lambda prompt, context_window: ["u"]
    )

    api = OpenAIAPI(model="gpt", context_window=100)
    out = api(
        prompt=cast(Prompt, SimpleNamespace()),
        system_prompt="sys",
        temperature=0.2,
        top_p=0.9,
        max_response_length=77,
    )

    assert out.text == "hello"
    assert out.input_token_count == 10
    assert out.output_token_count == 5
    assert out.total_token_count == 15


def test_openai_api_call_without_usage(monkeypatch) -> None:
    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    return SimpleNamespace(
                        choices=[SimpleNamespace(message=SimpleNamespace(content=None))],
                        usage=None,
                    )

    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setattr("mafc.common.modeling.openai_model.OpenAI", lambda api_key, timeout: FakeClient())
    monkeypatch.setattr(
        "mafc.common.modeling.openai_model.format_input", lambda prompt, context_window: ["u"]
    )

    api = OpenAIAPI(model="gpt", context_window=100)
    out = api(prompt=cast(Prompt, SimpleNamespace()))
    assert out.text == "Failed to generate a response."
    assert out.input_token_count is None


def test_openai_model_generate(monkeypatch) -> None:
    monkeypatch.setattr(
        "mafc.common.modeling.model.model_specifier_to_shorthand", lambda s: ("gpt_5_mini", "m")
    )
    monkeypatch.setattr("mafc.common.modeling.model.get_model_context_window", lambda n: 1000)
    monkeypatch.setattr("mafc.common.modeling.model.get_model_api_pricing", lambda n: (1.0, 2.0))
    monkeypatch.setattr(
        "mafc.common.modeling.openai_model.OpenAIAPI",
        lambda model, context_window: (
            lambda prompt, **kwargs: APIResponse(text="ok", input_token_count=1000, output_token_count=500)
        ),
    )

    model = OpenAIModel(specifier="OPENAI:gpt-5-mini-2025-08-07")
    prompt = cast(Prompt, SimpleNamespace(with_videos_as_frames=lambda n: "frames"))
    response = model.generate(prompt)
    assert response.text == "ok"
    assert response.total_cost == 0.002


def test_openai_model_generate_error_paths(monkeypatch) -> None:
    monkeypatch.setattr(
        "mafc.common.modeling.model.model_specifier_to_shorthand", lambda s: ("gpt_5_mini", "m")
    )
    monkeypatch.setattr("mafc.common.modeling.model.get_model_context_window", lambda n: 1000)
    monkeypatch.setattr("mafc.common.modeling.model.get_model_api_pricing", lambda n: (1.0, 2.0))

    class DummyRateLimitError(Exception):
        pass

    class DummyAuthError(Exception):
        pass

    monkeypatch.setattr("mafc.common.modeling.openai_model.openai.RateLimitError", DummyRateLimitError)
    monkeypatch.setattr("mafc.common.modeling.openai_model.openai.AuthenticationError", DummyAuthError)

    model = OpenAIModel(specifier="OPENAI:gpt-5-mini-2025-08-07")
    prompt = cast(Prompt, SimpleNamespace(with_videos_as_frames=lambda n: "frames"))

    monkeypatch.setattr(
        model, "api", lambda *_args, **_kwargs: (_ for _ in ()).throw(DummyRateLimitError("rate"))
    )
    with pytest.raises(DummyRateLimitError):
        model.generate(prompt)

    monkeypatch.setattr(model, "api", lambda *_args, **_kwargs: (_ for _ in ()).throw(DummyAuthError("auth")))
    with pytest.raises(DummyAuthError):
        model.generate(prompt)

    monkeypatch.setattr(model, "api", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("other")))
    with pytest.raises(RuntimeError):
        model.generate(prompt)
