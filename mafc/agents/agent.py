from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from ezmm import MultimodalSequence

from mafc.agents.common import AgentMessage, AgentMessageType, AgentSession, AgentStatus
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
    trace: dict | None = None


class Agent(ABC):
    name: str
    description: str
    allowed_tools: list[type[Tool]]

    def __init__(self, model: Model, n_workers: int = 1, agent_id: str | None = None):
        self.model = model
        self.n_workers = n_workers
        self._should_stop = False
        self.agent_id = agent_id or f"{self.name}_{id(self)}"

    @abstractmethod
    def run(self, session: AgentSession, trace_scope=None) -> AgentResult:
        """Run the agent on the given investigation session."""
        raise NotImplementedError

    @abstractmethod
    def synthesize_from_evidences(self, instruction: str, evidences: list[Evidence]) -> str:
        """Generate an answer using only the provided evidence set."""
        raise NotImplementedError

    def _mark_running(self, session: AgentSession) -> None:
        session.status = AgentStatus.RUNNING

    def _mark_completed(self, session: AgentSession) -> None:
        session.status = AgentStatus.COMPLETED

    def _mark_failed(self, session: AgentSession) -> None:
        session.status = AgentStatus.FAILED

    def build_prior_context(self, session: AgentSession) -> str:
        """Build generic planner-visible context from prior messages and evidence."""
        sections: list[str] = []

        prior_messages: list[str] = []
        for message in session.messages:
            formatted = self.format_message_context(message)
            if formatted is not None:
                prior_messages.append(formatted)
        if prior_messages:
            sections.append("Previous messages:\n" + "\n".join(prior_messages))

        prior_evidences: list[str] = []
        for evidence in session.evidences:
            formatted = self.format_evidence_context(evidence)
            if formatted is not None:
                prior_evidences.append(formatted)
        if prior_evidences:
            sections.append("Accepted evidence:\n" + "\n".join(prior_evidences))

        return "\n\n".join(sections)

    def format_message_context(self, message: AgentMessage) -> str | None:
        """Format one prior message for planner-visible context."""
        content = str(message.content).strip()
        if not content:
            return None
        return f"[{message.message_type.value}] {message.sender} -> {message.receiver}: {content}"

    def format_evidence_context(self, evidence: Evidence) -> str | None:
        """Format one prior evidence item for planner-visible context."""
        summary = (
            str(evidence.takeaways).strip() if evidence.takeaways is not None else str(evidence.raw).strip()
        )
        if not summary:
            return None
        return f"- Source: {evidence.source}\n  Summary: {summary}"

    def make_result_message(
        self,
        session: AgentSession,
        content: MultimodalSequence,
        evidences: list[Evidence],
    ) -> AgentMessage:
        """Create a session-scoped result message with a unique identifier."""
        result_index = (
            sum(1 for message in session.messages if message.message_type == AgentMessageType.RESULT) + 1
        )
        return AgentMessage(
            id=f"{session.id}:result:{result_index}",
            session_id=session.id,
            sender=self.agent_id,
            receiver=session.parent_session_id or session.id,
            message_type=AgentMessageType.RESULT,
            content=content,
            evidences=list(evidences),
        )

    def stop(self):
        """Signal the agent to stop execution.

        The agent should check for this signal periodically and stop as soon as possible.
        """
        self._should_stop = True
