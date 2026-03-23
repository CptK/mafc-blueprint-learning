from types import SimpleNamespace

import pytest

from mafc.common.modeling.model import APIResponse
from mafc.common.modeling.message import Message, MessageRole
from mafc.common.modeling.selfhosted_model import SelfhostedAPI, SelfhostedModel
from mafc.common.modeling.prompt import Prompt


def _fake_globals(url):
    return SimpleNamespace(selfhosted_url=url)


def test_selfhosted_api_requires_url(monkeypatch) -> None:
    monkeypatch.setattr("mafc.common.modeling.selfhosted_model.globals", _fake_globals(None))
    with pytest.raises(ValueError, match="selfhosted_url"):
        SelfhostedAPI(model="m", context_window=100)


def test_selfhosted_api_call_success(monkeypatch) -> None:
    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    return SimpleNamespace(
                        choices=[SimpleNamespace(message=SimpleNamespace(content="Paris"))],
                        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                    )

    monkeypatch.setattr("mafc.common.modeling.selfhosted_model.globals", _fake_globals("http://host/v1"))
    monkeypatch.setattr(
        "mafc.common.modeling.selfhosted_model.OpenAI",
        lambda base_url, api_key, timeout: FakeClient(),
    )
    monkeypatch.setattr(
        "mafc.common.modeling.selfhosted_model.format_input", lambda content, context_window: ["u"]
    )

    api = SelfhostedAPI(model="m", context_window=100)
    out = api(
        messages=[Message(role=MessageRole.USER, content=Prompt(text="capital?"))],
        temperature=0.5,
        top_p=0.9,
        max_response_length=256,
    )

    assert out.text == "Paris"
    assert out.input_token_count == 10
    assert out.output_token_count == 5
    assert out.total_token_count == 15


def test_selfhosted_api_call_empty_response(monkeypatch) -> None:
    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    return SimpleNamespace(
                        choices=[SimpleNamespace(
                            message=SimpleNamespace(content=None),
                            finish_reason="length",
                        )],
                        usage=None,
                    )

    monkeypatch.setattr("mafc.common.modeling.selfhosted_model.globals", _fake_globals("http://host/v1"))
    monkeypatch.setattr(
        "mafc.common.modeling.selfhosted_model.OpenAI",
        lambda base_url, api_key, timeout: FakeClient(),
    )
    monkeypatch.setattr(
        "mafc.common.modeling.selfhosted_model.format_input", lambda content, context_window: ["u"]
    )

    api = SelfhostedAPI(model="m", context_window=100)
    out = api(messages=[Message(role=MessageRole.USER, content=Prompt(text="?"))])

    assert out.text == "Failed to generate a response."
    assert out.input_token_count is None


def test_selfhosted_model_generate(monkeypatch) -> None:
    monkeypatch.setattr(
        "mafc.common.modeling.model.model_specifier_to_shorthand",
        lambda s: ("qwen_3_5_122b", "Qwen/Qwen3.5-122B-A10B-FP8"),
    )
    monkeypatch.setattr("mafc.common.modeling.model.get_model_context_window", lambda n: 1000)
    monkeypatch.setattr("mafc.common.modeling.model.get_model_api_pricing", lambda n: (0.0, 0.0))
    monkeypatch.setattr(
        "mafc.common.modeling.selfhosted_model.messages_with_videos_as_frames",
        lambda messages, n: messages,
    )
    monkeypatch.setattr("mafc.common.modeling.selfhosted_model.globals", _fake_globals("http://host/v1"))
    monkeypatch.setattr(
        "mafc.common.modeling.selfhosted_model.SelfhostedAPI",
        lambda model, context_window: (
            lambda prompt, **kwargs: APIResponse(text="Paris", input_token_count=10, output_token_count=5)
        ),
    )

    model = SelfhostedModel(specifier="SELFHOSTED:Qwen/Qwen3.5-122B-A10B-FP8")
    response = model.generate([Message(role=MessageRole.USER, content=Prompt(text="capital?"))])

    assert response.text == "Paris"
    assert response.total_cost == 0.0


def test_selfhosted_model_generate_error_paths(monkeypatch) -> None:
    monkeypatch.setattr(
        "mafc.common.modeling.model.model_specifier_to_shorthand",
        lambda s: ("qwen_3_5_122b", "Qwen/Qwen3.5-122B-A10B-FP8"),
    )
    monkeypatch.setattr("mafc.common.modeling.model.get_model_context_window", lambda n: 1000)
    monkeypatch.setattr("mafc.common.modeling.model.get_model_api_pricing", lambda n: (0.0, 0.0))
    monkeypatch.setattr(
        "mafc.common.modeling.selfhosted_model.messages_with_videos_as_frames",
        lambda messages, n: messages,
    )
    monkeypatch.setattr("mafc.common.modeling.selfhosted_model.globals", _fake_globals("http://host/v1"))
    monkeypatch.setattr(
        "mafc.common.modeling.selfhosted_model.SelfhostedAPI", lambda model, context_window: None
    )

    class DummyRateLimitError(Exception):
        pass

    class DummyAuthError(Exception):
        pass

    monkeypatch.setattr("mafc.common.modeling.selfhosted_model.openai.RateLimitError", DummyRateLimitError)
    monkeypatch.setattr("mafc.common.modeling.selfhosted_model.openai.AuthenticationError", DummyAuthError)

    model = SelfhostedModel(specifier="SELFHOSTED:Qwen/Qwen3.5-122B-A10B-FP8")
    prompt = [Message(role=MessageRole.USER, content=Prompt(text="?"))]

    monkeypatch.setattr(
        model, "api", lambda *_args, **_kwargs: (_ for _ in ()).throw(DummyRateLimitError("rate"))
    )
    with pytest.raises(DummyRateLimitError):
        model.generate(prompt)

    monkeypatch.setattr(
        model, "api", lambda *_args, **_kwargs: (_ for _ in ()).throw(DummyAuthError("auth"))
    )
    with pytest.raises(DummyAuthError):
        model.generate(prompt)

    monkeypatch.setattr(
        model, "api", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("other"))
    )
    with pytest.raises(RuntimeError):
        model.generate(prompt)
