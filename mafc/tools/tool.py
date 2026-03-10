from abc import ABC, abstractmethod
from collections.abc import Sized
import time
import torch
from typing import Generic, TypeVar

from ezmm import MultimodalSequence

from mafc.common.modeling.model import Model
from mafc.common.action import Action
from mafc.common.logger import logger
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
        logger.log(f"[Tool:{self.name}] Starting _perform for {type(action).__name__}")
        start_time = time.time()
        try:
            result = self._perform(action)
            execution_time = time.time() - start_time
            result_count = len(result) if isinstance(result, Sized) else "?"
            logger.log(
                f"[Tool:{self.name}] _perform completed in {execution_time:.2f}s, got {result_count} results"
            )
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"[Tool:{self.name}] _perform failed after {execution_time:.2f}s: {e}")
            raise

        # Summarize the result
        if summarize:
            logger.log(f"[Tool:{self.name}] Starting _summarize")
            start_time = time.time()
            try:
                summary = self._summarize(result, **kwargs)
                elapsed = time.time() - start_time
                logger.log(f"[Tool:{self.name}] _summarize completed in {elapsed:.2f}s")
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"[Tool:{self.name}] _summarize failed after {elapsed:.2f}s: {e}")
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
