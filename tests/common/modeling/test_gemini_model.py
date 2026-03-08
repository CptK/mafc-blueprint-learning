import base64
from types import SimpleNamespace
from typing import cast

import pytest
from ezmm import Image

from mafc.common.modeling.gemini_model import (
    GeminiAPI,
    GeminiModel,
    _image_bytes,
    _resolve_gemini_key,
    format_input,
)
from mafc.common.modeling.model import APIResponse
from mafc.common.modeling.prompt import Prompt


class FakePromptBlocks:
    def __init__(self, blocks):
        self._blocks = blocks

    def to_list(self):
        return self._blocks


def test_resolve_gemini_key(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("gemini_api_key", "k")
    assert _resolve_gemini_key() == "k"


def test_image_bytes() -> None:
    img = SimpleNamespace(get_base64_encoded=lambda: base64.b64encode(b"abc").decode("ascii"))
    assert _image_bytes(cast(Image, img)) == b"abc"


def test_image_bytes_failure() -> None:
    img = SimpleNamespace(get_base64_encoded=lambda: (_ for _ in ()).throw(RuntimeError("bad")))
    with pytest.raises(ValueError):
        _image_bytes(cast(Image, img))


def test_gemini_format_input_text(monkeypatch) -> None:
    monkeypatch.setattr("mafc.common.modeling.gemini_model.Image", type("FakeImage", (), {}))
    monkeypatch.setattr("mafc.common.modeling.gemini_model.Video", type("FakeVideo", (), {}))
    monkeypatch.setattr("mafc.common.modeling.gemini_model.Part", lambda **kwargs: kwargs)
    out = format_input(cast(Prompt, FakePromptBlocks(["hello"])), context_window=100)
    assert out == [{"text": "hello"}]


def test_gemini_format_input_image_video_and_budget(monkeypatch) -> None:
    class FakeImage:
        def get_base64_encoded(self):
            return base64.b64encode(b"abc").decode("ascii")

        width = 512
        height = 512

    class FakeVideo:
        pass

    monkeypatch.setattr("mafc.common.modeling.gemini_model.Image", FakeImage)
    monkeypatch.setattr("mafc.common.modeling.gemini_model.Video", FakeVideo)
    monkeypatch.setattr("mafc.common.modeling.gemini_model.Part", lambda **kwargs: kwargs)
    monkeypatch.setattr("mafc.common.modeling.gemini_model.Blob", lambda **kwargs: kwargs)
    monkeypatch.setattr("mafc.common.modeling.gemini_model.count_image_tokens_estimate", lambda image: 1)

    out = format_input(cast(Prompt, FakePromptBlocks([FakeImage(), FakeVideo()])), context_window=5)
    assert out == [{"inline_data": {"mime_type": "image/jpeg", "data": b"abc"}}]

    monkeypatch.setattr("mafc.common.modeling.gemini_model.count_image_tokens_estimate", lambda image: 99)
    out2 = format_input(cast(Prompt, FakePromptBlocks([FakeImage()])), context_window=1)
    assert out2 == []


def test_gemini_api_requires_key(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("gemini_api_key", raising=False)
    with pytest.raises(ValueError):
        GeminiAPI(model="x", context_window=10)


def test_gemini_api_call_success_and_fallback(monkeypatch) -> None:
    class FakeResponse:
        text = None
        candidates = [
            SimpleNamespace(
                content=SimpleNamespace(parts=[SimpleNamespace(text="a"), SimpleNamespace(text="b")])
            )
        ]
        usage_metadata = SimpleNamespace(prompt_token_count=9, candidates_token_count=4, total_token_count=13)

    class FakeClient:
        class models:
            @staticmethod
            def generate_content(**kwargs):
                return FakeResponse()

    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr("mafc.common.modeling.gemini_model.genai.Client", lambda api_key: FakeClient())
    monkeypatch.setattr(
        "mafc.common.modeling.gemini_model.format_input", lambda prompt, context_window: ["p"]
    )
    monkeypatch.setattr("mafc.common.modeling.gemini_model.Content", lambda **kwargs: kwargs)
    monkeypatch.setattr("mafc.common.modeling.gemini_model.GenerateContentConfig", lambda **kwargs: kwargs)

    api = GeminiAPI(model="gem", context_window=100)
    out = api(prompt=cast(Prompt, SimpleNamespace()), system_prompt="sys")
    assert out.text == "a\nb"
    assert out.total_token_count == 13


def test_gemini_api_error(monkeypatch) -> None:
    class BrokenClient:
        class models:
            @staticmethod
            def generate_content(**kwargs):
                raise RuntimeError("boom")

    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr("mafc.common.modeling.gemini_model.genai.Client", lambda api_key: BrokenClient())
    monkeypatch.setattr(
        "mafc.common.modeling.gemini_model.format_input", lambda prompt, context_window: ["p"]
    )
    monkeypatch.setattr("mafc.common.modeling.gemini_model.Content", lambda **kwargs: kwargs)
    monkeypatch.setattr("mafc.common.modeling.gemini_model.GenerateContentConfig", lambda **kwargs: kwargs)

    api = GeminiAPI(model="gem", context_window=100)
    with pytest.raises(RuntimeError):
        api(prompt=cast(Prompt, SimpleNamespace()))


def test_gemini_model_generate(monkeypatch) -> None:
    monkeypatch.setattr(
        "mafc.common.modeling.model.model_specifier_to_shorthand",
        lambda s: ("gemini_3.1_flash_lite", "m"),
    )
    monkeypatch.setattr("mafc.common.modeling.model.get_model_context_window", lambda n: 1000)
    monkeypatch.setattr("mafc.common.modeling.model.get_model_api_pricing", lambda n: (1.0, 2.0))
    monkeypatch.setattr(
        "mafc.common.modeling.gemini_model.GeminiAPI",
        lambda model, context_window: (
            lambda prompt, **kwargs: APIResponse(text="ok", input_token_count=1000, output_token_count=500)
        ),
    )

    model = GeminiModel(specifier="GOOGLE:gemini-3.1-flash-lite-preview")
    prompt = cast(Prompt, SimpleNamespace(with_videos_as_frames=lambda n: "frames"))
    response = model.generate(prompt)
    assert response.text == "ok"
    assert response.total_cost == 0.002
