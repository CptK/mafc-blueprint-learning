from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mafc.common.claim import Claim
from mafc.common.label import BaseLabel


@dataclass(slots=True)
class BenchmarkSample:
    id: str
    input: Claim
    label: BaseLabel
    justification: dict[str, Any] | None = None
