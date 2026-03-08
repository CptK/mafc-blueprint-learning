from typing import TypedDict, NotRequired, Literal


class MediaScore(TypedDict, total=False):
    score: float


class MediaItem(TypedDict, total=False):
    id: str
    type: Literal["image", "video"]
    authenticity: NotRequired[float | MediaScore | None]
    contextualization: NotRequired[float | MediaScore | None]


class ClaimEntry(TypedDict, total=False):
    id: int | str
    text: str
    media: list[MediaItem]
    integrity: float | MediaScore | None
    veracity: NotRequired[float | MediaScore | None]
    context_coverage: NotRequired[float | MediaScore | None]
    date: NotRequired[str | None]
