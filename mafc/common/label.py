from enum import Enum
from typing import Protocol, runtime_checkable


class BaseLabel(str, Enum):
    """Base class for benchmark-specific label enums.

    Subclass this for each benchmark's labels, e.g., `class VeritasLabel(BaseLabel): ...`.
    Using `str` as a mixin ensures labels have stable string values.
    """
    pass


@runtime_checkable
class LabelLike(Protocol):
    """Structural type for label enums used across the codebase.

    Read-only name/value properties match Enum's interface.
    """
    @property
    def name(self) -> str: ...

    @property
    def value(self) -> str: ...
