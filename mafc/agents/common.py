from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from ezmm import MultimodalSequence

from mafc.common.claim import Claim
from mafc.common.evidence import Evidence


class AgentStatus(Enum):
    """Represents the current lifecycle state of an investigation session."""

    NOT_STARTED = "not_started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentMessageType(Enum):
    """Categorizes workflow messages exchanged between agents or sessions."""

    TASK = "task"
    RESULT = "result"
    FOLLOW_UP = "follow_up"
    ERROR = "error"
    OTHER = "other"


@dataclass
class AgentMessage:
    """One communication event between agents.

    Messages are the transport mechanism between sessions. They may carry a
    natural-language summary in `content` and optionally attach evidence items
    that the sender wants to report to the receiver.
    """

    id: str
    session_id: str
    sender: str
    receiver: str
    message_type: AgentMessageType
    content: MultimodalSequence
    parent_id: str | None = None  # ID of the message this is replying to, if any.
    evidences: list[Evidence] = field(default_factory=list)  # Evidence payload reported with this message.


@dataclass
class AgentSession:
    """State owned by one investigation session.

    A session can represent either a top-level fact-check or a narrower
    sub-task delegated to a worker agent. The session owns its local working
    state, including messages seen in this session, evidence accepted into this
    session, and unresolved follow-up questions.

    Evidence ownership is session-local:
    - child sessions keep the evidence they discovered while investigating
    - messages can carry selected evidence from one session to another
    - parent sessions may decide to accept reported evidence into their own
      `evidences` list
    """

    id: str
    goal: MultimodalSequence
    claim: Claim | None = None
    cutoff_date: date | None = None  # Restrict web searches to sources published on or before this date.
    status: AgentStatus = AgentStatus.NOT_STARTED
    messages: list[AgentMessage] = field(default_factory=list)
    evidences: list[Evidence] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    parent_session_id: str | None = None  # Parent session for delegated sub-tasks, if any.
