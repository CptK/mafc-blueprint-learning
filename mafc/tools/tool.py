from abc import ABC, abstractmethod
import torch
from typing import Generic, TypeVar

from ezmm import MultimodalSequence

from mafc.common.modeling.model import Model
from mafc.common.action import Action
from mafc.common.results import Results
from mafc.tools.tool_result import ToolResult

ActionType = TypeVar("ActionType", bound=Action)
ResultType = TypeVar("ResultType", bound=Results)


class Tool(ABC, Generic[ActionType, ResultType]):
    """Base class for all tools. Tools leverage integrations to retrieve evidence."""

    name: str
    actions: list[type[ActionType]]  # (classes of the) available actions this tool offers

    def __init__(self, llm: Model | None = None, device: str | torch.device | None = None):
        self.device = device
        self.llm = llm

        self.current_claim_id: str | None = None  # used by few tools to adjust claim-specific behavior

    def perform(self, action: ActionType, summarize: bool = True, **kwargs) -> ToolResult:
        assert type(action) in self.actions, f"Forbidden action: {action}"

        # Execute the action
        try:
            result = self._perform(action)
        except Exception as e:
            raise

        # Summarize the result
        if summarize:
            try:
                summary = self._summarize(result, **kwargs)
            except Exception as e:
                raise
        else:
            summary = None

        tool_result = ToolResult(raw=result, action=action, takeaways=summary)

        return tool_result

    @abstractmethod
    def _perform(self, action: ActionType) -> ResultType:
        """The actual function executing the action."""
        pass

    @abstractmethod
    def _summarize(self, result: ResultType, **kwargs) -> MultimodalSequence | None:
        """Turns the result into an LLM-friendly summary. May use additional
        context for summarization. Returns None iff the result does not contain any
        (potentially) helpful information."""
        pass
