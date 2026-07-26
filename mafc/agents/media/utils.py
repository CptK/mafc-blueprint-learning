from __future__ import annotations

from ezmm import MultimodalSequence

from mafc.common.evidence import Evidence
from mafc.common.logger import logger
from mafc.tools.tool_result import ToolResult
from mafc.tools.web_search.common import WebSource
from mafc.tools.web_search.google_vision import GoogleRisResults, match_precision

# Must contain media_referent._NO_MATCH_MARKER verbatim so that
# extract_referent_digest() sets `no_match_reported` for this evidence.
_NO_MATCH_HEAD = (
    "No provenance could be established for this media: reverse image search returned no "
    "EXACT or PARTIAL match. Origin, earliest appearance, and link to any claimed event "
    "are unverified."
)
_NO_MATCH_LOOKALIKES = (
    " {n_similar} visually-similar page(s) were returned, but none is a confirmed match of "
    "this media, so they say nothing about its origin or content."
)
_NO_MATCH_NO_PAGES = " No pages containing this media were found at all."


def _no_match_note(n_similar: int) -> str:
    """Render the negative RIS finding. Deliberately carries no entity tags or
    best-guess labels: those describe Google's index neighbourhood, not this media,
    and downstream stages have been observed quoting them as content evidence."""
    tail = _NO_MATCH_LOOKALIKES.format(n_similar=n_similar) if n_similar else _NO_MATCH_NO_PAGES
    return _NO_MATCH_HEAD + tail


def build_evidences_from_tool_result(tool_result: ToolResult, media_reference: str) -> list[Evidence]:
    raw = tool_result.raw
    takeaways = tool_result.takeaways

    # RIS: promote each *confirmed* match into its own evidence item.
    #
    # Only pages Google labelled EXACT or PARTIAL are promoted. Unconfirmed
    # ("visually similar") pages used to be promoted too, each carrying a copy of the
    # aggregate takeaways block — so one API call returning N lookalikes produced N
    # evidence items all repeating the same entity tags, which downstream judging read
    # as N independent findings about this media. Confirmed matches now carry only
    # their own match line, and an all-unconfirmed result collapses into a single
    # explicit negative finding.
    if isinstance(raw, GoogleRisResults):
        confirmed = [
            source
            for source in raw.sources
            if isinstance(source, WebSource) and match_precision(source) is not None
        ]
        if not confirmed:
            logger.info(
                f"RIS returned {len(raw.sources)} source(s), none with a confirmed EXACT/PARTIAL "
                "match. Emitting a single no-provenance evidence item instead of per-page items."
            )
            note = _no_match_note(len(raw.sources))
            return [
                Evidence(
                    raw=MultimodalSequence(note),
                    action=tool_result.action,
                    source=media_reference,
                    # Populated on purpose: the fact-check planner drops evidence with no
                    # takeaways (see fact_check.prompts._planner_summary_for_evidence), and
                    # Evidence.is_useful() keys on it. "RIS found nothing" is a finding.
                    takeaways=MultimodalSequence(note),
                )
            ]

        logger.info(
            f"RIS returned {len(confirmed)} confirmed match(es) of {len(raw.sources)} source(s). "
            "Promoting each confirmed match into its own evidence item."
        )
        evidences: list[Evidence] = []
        for source in confirmed:
            # Per-source match line only — never the aggregate takeaways block, which
            # would duplicate whole-result context (entity tags, every other page) onto
            # each item and manufacture apparent corroboration.
            own_note = str(source).strip()
            evidences.append(
                Evidence(
                    raw=MultimodalSequence(own_note),
                    action=tool_result.action,
                    source=source.reference,
                    takeaways=MultimodalSequence(own_note),
                )
            )
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
