from .anthropic_model import AnthropicAPI, AnthropicModel
from .gemini_model import GeminiAPI, GeminiModel
from .model import APIResponse, Model, Response
from .openai_model import OpenAIAPI, OpenAIModel
from .prompt import Prompt
from .utils import (
    AVAILABLE_MODELS,
    model_specifier_to_shorthand,
    model_shorthand_to_full_specifier,
    get_model_context_window,
    get_model_api_pricing,
)


def make_model(name: str, **kwargs) -> Model:
    """Factory function to load an (M)LLM. Use this instead of class instantiation."""
    if name in AVAILABLE_MODELS["Shorthand"].to_list():
        specifier = model_shorthand_to_full_specifier(name)
    else:
        specifier = name

    platform = specifier.split(":")[0].lower()
    model_name = specifier.split(":")[1].lower()

    match platform:
        case "openai":
            return OpenAIModel(model_name, **kwargs)
        case "anthropic":
            return AnthropicModel(model_name, **kwargs)
        case "gemini":
            return GeminiModel(model_name, **kwargs)
        case _:
            raise ValueError(
                f'Platform "{platform}" not supported. Check "config/available_models.csv" for available models.'
            )


__all__ = [
    "Model",
    "Response",
    "APIResponse",
    "Prompt",
    "make_model",
    "model_specifier_to_shorthand",
    "model_shorthand_to_full_specifier",
    "get_model_context_window",
    "get_model_api_pricing",
    "OpenAIModel",
    "OpenAIAPI",
    "AnthropicModel",
    "AnthropicAPI",
    "GeminiModel",
    "GeminiAPI",
]
