from abc import ABC, abstractmethod
from dataclasses import dataclass
from ezmm import MultimodalSequence

from mafc.common.modeling.model import Model
from mafc.common.modeling.prompt import Prompt
from mafc.tools.tool import Tool


@dataclass
class AgentResult:
    result: MultimodalSequence | None
    errors: list[str]


class Agent(ABC):
    name: str
    description: str
    allowed_tools: list[type[Tool]]

    def __init__(self, model: Model, n_workers: int = 1):
        self.model = model
        self.n_workers = n_workers
        self._should_stop = False

    @abstractmethod
    def run(self, task: Prompt) -> AgentResult:
        """Run the agent on the given task. This is the main entry point for using the agent."""
        pass

    def stop(self):
        """Signal the agent to stop execution.

        The agent should check for this signal periodically and stop as soon as possible.
        """
        self._should_stop = True
