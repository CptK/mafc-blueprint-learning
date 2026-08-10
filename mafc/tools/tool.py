from abc import ABC, abstractmethod
import torch
from typing import Generic, TypeVar

from ezmm import MultimodalSequence

from mafc.common.modeling.model import Model
from mafc.common.action import Action
from mafc.common.results import Results
from mafc.common.source_guard import filter_blocked_sources
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

    def perform(
        self,
        action: ActionType,
        summarize: bool = True,
        blocked_urls: set[str] | None = None,
        **kwargs,
    ) -> ToolResult:
        """Execute ``action`` and wrap the outcome as a ``ToolResult``.

        ``blocked_urls`` drops forbidden sources from the raw result. It is applied
        here, before summarization, because the summary is RENDERED FROM the result:
        filtering afterwards would leave the blocked URL in the takeaways text even
        though it was gone from ``sources``, and the takeaways are what becomes
        evidence. See ``mafc.common.source_guard``.
        """
        assert type(action) in self.actions, f"Forbidden action: {action}"

        # Execute the action
        try:
            result = self._perform(action)
        except Exception:
            raise

        if blocked_urls and getattr(result, "sources", None):
            result.sources = filter_blocked_sources(
                result.sources, blocked_urls, context=f"tool={action.name}"
            )

        # Summarize the result
        if summarize:
            try:
                summary = self._summarize(result, **kwargs)
            except Exception:
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
