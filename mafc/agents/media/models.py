from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MediaToolName = Literal["reverse_image_search", "geolocate"]


@dataclass
class MediaToolPlan:
    tools: list[MediaToolName]
