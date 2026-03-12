from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict
from ezmm import MultimodalSequence


class MessageRole(str, Enum):
    """Supported chat-style roles for model inputs."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    """One role-scoped model input message with multimodal content."""

    role: MessageRole
    content: MultimodalSequence

    model_config = ConfigDict(arbitrary_types_allowed=True)
