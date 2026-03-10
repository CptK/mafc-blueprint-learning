from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from ezmm import MultimodalSequence

from mafc.agents.common import AgentMessage, AgentSession, AgentStatus
from mafc.common.evidence import Evidence
from mafc.common.modeling.model import Model
from mafc.tools.tool import Tool


@dataclass
class AgentResult:
    session: AgentSession
    result: MultimodalSequence | None
    messages: list[AgentMessage] = field(default_factory=list)
    evidences: list[Evidence] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    status: AgentStatus | None = None


class Agent(ABC):
    name: str
    description: str
    allowed_tools: list[type[Tool]]

    def __init__(self, model: Model, n_workers: int = 1):
        self.model = model
        self.n_workers = n_workers
        self._should_stop = False

    @abstractmethod
    def run(self, session: AgentSession) -> AgentResult:
        """Run the agent on the given investigation session."""
        raise NotImplementedError

    def _mark_running(self, session: AgentSession) -> None:
        session.status = AgentStatus.RUNNING

    def _mark_completed(self, session: AgentSession) -> None:
        session.status = AgentStatus.COMPLETED

    def _mark_failed(self, session: AgentSession) -> None:
        session.status = AgentStatus.FAILED

    def stop(self):
        """Signal the agent to stop execution.

        The agent should check for this signal periodically and stop as soon as possible.
        """
        self._should_stop = True
