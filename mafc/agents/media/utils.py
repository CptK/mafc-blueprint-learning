from __future__ import annotations

from ezmm import MultimodalSequence

from mafc.common.evidence import Evidence
from mafc.common.logger import logger
from mafc.tools.tool_result import ToolResult
from mafc.tools.web_search.google_vision import GoogleRisResults


def build_evidences_from_tool_result(tool_result: ToolResult, media_reference: str) -> list[Evidence]:
    raw = tool_result.raw
    takeaways = tool_result.takeaways

    # RIS: promote each matched source into its own evidence item.
    if isinstance(raw, GoogleRisResults) and raw.sources:
        logger.info(
            "Raw tool output contains multiple sources. Promoting each source into its own evidence item."
        )
        evidences: list[Evidence] = []
        for source in raw.sources:
            summary_parts = []
            if takeaways is not None:
                summary_parts.append(str(takeaways))
            summary = "\n".join(part for part in summary_parts if part.strip())
            evidences.append(
                Evidence(
                    raw=MultimodalSequence(str(source)),
                    action=tool_result.action,
                    source=source.reference,
                    takeaways=MultimodalSequence(summary) if summary else None,
                )
            )
        if evidences:
            return evidences

    # Geolocation or fallback: one evidence item per tool run.
    return [
        Evidence(
            raw=MultimodalSequence(str(raw)),
            action=tool_result.action,
            source=media_reference,
            takeaways=takeaways,
        )
    ]
