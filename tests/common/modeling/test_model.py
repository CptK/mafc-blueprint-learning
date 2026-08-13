from typing import cast

from mafc.common.modeling.message import Message
from mafc.common.modeling.model import API, APIResponse, Model


class DummyModel(Model):
    def _do_generate(self, messages):
        raise NotImplementedError


def test_model_initialization_and_compute_cost() -> None:
    model = DummyModel(specifier="OPENAI:gpt-5-mini-2025-08-07")
    assert model.name == "gpt_5_mini"
    assert model.model == "gpt-5-mini-2025-08-07"
    assert model.context_window == 272000
    assert model.input_token_cost == 0.25
    assert model.output_token_cost == 2.0

    api_response = APIResponse(text="ok", input_token_count=1000, output_token_count=500)
    assert model.compute_cost(api_response) == (1000 / 1_000_000) * 0.25 + (500 / 1_000_000) * 2.0

    missing = APIResponse(text="ok", input_token_count=None, output_token_count=500)
    assert model.compute_cost(missing) == 0.0


def test_abstract_method_bodies_are_covered() -> None:
    # These execute the abstract base method bodies (which are `pass`) directly.
    assert API.__call__(cast(API, object()), messages=cast(list[Message], object())) is None
    assert Model._do_generate(cast(Model, object()), messages=cast(list[Message], object())) is None


def test_compute_cost_prices_cached_prompt_tokens() -> None:
    """Cache writes and reads are billed at their own multiples of the input rate.

    The provider reports them apart from input_token_count, so pricing only that
    field would under-report a cache-heavy run to near zero.
    """
    model = DummyModel(specifier="OPENAI:gpt-5-mini-2025-08-07")
    response = APIResponse(
        text="ok",
        input_token_count=100,
        output_token_count=500,
        cache_write_token_count=1000,
        cache_read_token_count=4000,
    )
    expected = (100 + 1000 * 1.25 + 4000 * 0.1) * 0.25 / 1_000_000 + 500 * 2.0 / 1_000_000
    assert model.compute_cost(response) == expected


def test_compute_cost_counts_a_full_cache_hit() -> None:
    """A fully cached prompt reports input_token_count as 0; that is not a free call."""
    model = DummyModel(specifier="OPENAI:gpt-5-mini-2025-08-07")
    response = APIResponse(text="ok", input_token_count=0, output_token_count=10, cache_read_token_count=8000)
    assert model.compute_cost(response) > 0.0
