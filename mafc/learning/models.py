from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class ActionEvidenceLink:
    """Maps a single investigative action to the finding it produced."""

    action: str
    """Abstract action type, e.g. 'reverse_image_search', 'geolocation', 'source_lookup'."""

    finding: str
    """What the action revealed, e.g. 'video originates from Ghana school fire in 2019'."""

    query_or_input: str | None = None
    """The search query or input used, if mentioned in the article."""

    was_decisive: bool = False
    """Whether this finding directly contributed to the verdict."""


@dataclass
class ArticleAnalysis:
    """Structured extraction from a fact-check article.

    Fields that require an explicit process description in the article are
    None (not empty list) when absent, to distinguish 'not described' from
    'described but empty'.
    """

    # --- Always extractable ---

    claim_type: str
    """High-level category of the claim being checked.

    Examples: 'media_authenticity', 'quote_attribution', 'event_claim',
    'statistic', 'identity_claim', 'context_manipulation'.
    """

    verdict_summary: str
    """1-2 sentence plain explanation of why the claim is true/false/misleading."""

    key_evidence: list[str]
    """Concrete pieces of evidence cited in the article.

    Examples: ['video geotag matches Ghana, not Nigeria',
               'AFP photographer confirms image was taken in 2019'].
    """

    evidence_types: list[str]
    """Abstract types of evidence used, inferred from conclusions when process is not explicit.

    Examples: ['reverse_image_search', 'geolocation', 'source_lookup',
               'metadata_analysis', 'expert_consultation', 'official_records'].
    """

    # --- Present in process-rich articles only ---

    action_evidence_links: list[ActionEvidenceLink] | None = None
    """Ordered mapping from investigative actions to findings. None for result-only articles."""

    investigative_steps: list[str] | None = None
    """High-level ordered steps taken during investigation. None if not described."""

    search_queries: list[str] | None = None
    """Specific queries or searches mentioned in the article. None if not described."""

    # --- Meta ---

    process_richness: Literal["full", "partial", "result_only"] = "result_only"
    """How much of the investigative process is described.

    - 'full': Step-by-step process is described.
    - 'partial': Some steps mentioned, but incomplete.
    - 'result_only': Only conclusions and evidence are stated.
    """

    notes: str | None = None
    """Anything unusual, ambiguous, or hard to categorize that the analyzer flagged."""
