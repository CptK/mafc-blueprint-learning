from dataclasses import dataclass, field
from typing import TypedDict, NotRequired, Literal

from mafc.eval.types import BenchmarkSample


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


@dataclass
class VeriTaSBenchmarkSample(BenchmarkSample):
    gt_score: float
    gt_veracity: float | None = None
    gt_context_coverage: float | None = None
    gt_media_verdicts: list[dict] = field(default_factory=list)
