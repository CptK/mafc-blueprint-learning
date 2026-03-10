from dataclasses import dataclass
from typing import cast

import pytest
from ezmm import MultimodalSequence

from mafc.common.action import Action
from mafc.common.results import Results
from mafc.tools.tool_result import ToolResult
from mafc.tools.tool import Tool


class DummyAction(Action):
    name = "dummy"

    def __init__(self, value: int = 1):
        self._save_parameters(locals())
        self.value = value


class OtherAction(Action):
    name = "other"

    def __init__(self):
        self._save_parameters(locals())


@dataclass
class SizedDummyResults(Results):
    values: list[int]

    def __len__(self) -> int:
        return len(self.values)

    def __str__(self) -> str:
        return ",".join(str(v) for v in self.values)


@dataclass
class UnsizedDummyResults(Results):
    text: str

    def __str__(self) -> str:
        return self.text


class SizedDummyTool(Tool[DummyAction, SizedDummyResults]):
    name = "sized_dummy"
    actions = [DummyAction]

    def __init__(self):
        super().__init__()
        self.summarize_calls = 0
        self.last_kwargs: dict = {}

    def _perform(self, action: DummyAction) -> SizedDummyResults:
        return SizedDummyResults(values=[action.value, 2, 3])

    def _summarize(self, result: SizedDummyResults, **kwargs) -> MultimodalSequence | None:
        self.summarize_calls += 1
        self.last_kwargs = kwargs
        return MultimodalSequence(f"summary:{len(result)}")


class UnsizedDummyTool(Tool[DummyAction, UnsizedDummyResults]):
    name = "unsized_dummy"
    actions = [DummyAction]

    def _perform(self, action: DummyAction) -> UnsizedDummyResults:
        return UnsizedDummyResults(text="ok")

    def _summarize(self, result: UnsizedDummyResults, **kwargs) -> MultimodalSequence | None:
        return MultimodalSequence(result.text)


def test_perform_executes_and_summarizes() -> None:
    tool = SizedDummyTool()
    action = DummyAction(7)

    result = tool.perform(action, summarize=True, note="keep")

    assert isinstance(result, ToolResult)
    assert result.action == action
    raw = cast(SizedDummyResults, result.raw)
    assert raw.values == [7, 2, 3]
    assert result.takeaways is not None
    assert tool.summarize_calls == 1
    assert tool.last_kwargs == {"note": "keep"}


def test_perform_skips_summarization_when_disabled() -> None:
    tool = SizedDummyTool()

    result = tool.perform(DummyAction(), summarize=False)

    assert result.takeaways is None
    assert tool.summarize_calls == 0


def test_perform_rejects_forbidden_action() -> None:
    tool = SizedDummyTool()

    with pytest.raises(AssertionError, match="Forbidden action"):
        tool.perform(cast(DummyAction, OtherAction()))


def test_perform_handles_unsized_result() -> None:
    tool = UnsizedDummyTool()

    result = tool.perform(DummyAction(), summarize=False)

    raw = cast(UnsizedDummyResults, result.raw)
    assert raw.text == "ok"
