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
