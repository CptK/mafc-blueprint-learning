import pytest

from mafc.common.modeling.utils import (
    abbreviate,
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


def test_abbreviate_keeps_short_text_verbatim() -> None:
    assert abbreviate("") == ""
    assert abbreviate("short") == "short"
    # The threshold is inclusive: exactly 2*edge_chars still fits without eliding.
    assert abbreviate("x" * 500) == "x" * 500


def test_abbreviate_elides_middle_and_keeps_both_edges() -> None:
    text = "HEAD" + "x" * 5000 + "TAIL"
    out = abbreviate(text)

    assert out.startswith("HEAD")
    assert out.endswith("TAIL")
    assert "..." in out
    assert f"[{len(text) - 500} chars omitted]" in out
    # Both edges survive at full width; only the middle is dropped.
    assert out[:250] == text[:250]
    assert out[-250:] == text[-250:]


def test_abbreviate_edge_chars_is_configurable() -> None:
    text = "y" * 10_000
    assert len(abbreviate(text, edge_chars=10)) < len(abbreviate(text, edge_chars=1000))
    assert abbreviate(text, edge_chars=10).startswith("y" * 10)
