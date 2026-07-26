from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MediaToolName = Literal[
    "reverse_image_search",
    "geolocate",
    # Grouped intent offered to the planner: fans out to every available
    # authenticity detector (see MediaAgent._run_authenticity_fanout).
    "assess_authenticity",
    # Individual authenticity detectors. Not offered to the planner directly
    # (it selects "assess_authenticity" instead), but still valid dispatch
    # targets when a MediaToolPlan is constructed explicitly.
    "check_c2pa_provenance",
    "detect_trufor_manipulation",
    "sightengine_detection",
]

# The individual detectors the "assess_authenticity" intent fans out to.
AUTHENTICITY_TOOLS: tuple[MediaToolName, ...] = (
    "check_c2pa_provenance",
    "detect_trufor_manipulation",
    "sightengine_detection",
)


@dataclass
class MediaToolPlan:
    tools: list[MediaToolName]
