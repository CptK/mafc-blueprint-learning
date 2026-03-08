import pytest

from mafc.common.modeling.utils import (
    get_model_api_pricing,
    get_model_context_window,
    model_shorthand_to_full_specifier,
    model_specifier_to_shorthand,
)


def test_model_specifier_and_shorthand_mapping() -> None:
    shorthand, model_name = model_specifier_to_shorthand("OPENAI:gpt-5-mini-2025-08-07")
    assert shorthand == "gpt_5_mini"
    assert model_name == "gpt-5-mini-2025-08-07"
    assert model_shorthand_to_full_specifier("gpt_5_mini") == "OPENAI:gpt-5-mini-2025-08-07"


def test_model_specifier_invalid_raises() -> None:
    with pytest.raises(ValueError):
        model_specifier_to_shorthand("invalid")

    with pytest.raises(ValueError):
        model_specifier_to_shorthand("OPENAI:does-not-exist")


def test_context_window_and_pricing_lookup() -> None:
    assert get_model_context_window("gpt_5_mini") == 272000
    assert get_model_context_window("OPENAI:gpt-5-mini-2025-08-07") == 272000
    assert get_model_api_pricing("gpt_5_mini") == (0.25, 2.0)
