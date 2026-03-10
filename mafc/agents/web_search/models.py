from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from mafc.tools.web_search.common import Source, WebSource


@dataclass
class SearchPlanStep:
    queries: list[str]
    done: bool = False


@dataclass
class StepQueryPlan:
    """Resolved query plan used by the run loop for one iteration."""

    queries: list[str]
    done: bool
    should_terminate: bool = False


@dataclass
class QuerySearchResult:
    """Result container for one query search execution in the worker pool."""

    query_text: str
    sources: Sequence[Source] | None
    errors: list[str]
    mark_seen: bool = False
    stopped: bool = False


@dataclass
class GlobalSourceCandidate:
    """One URL candidate with query context for global relevance filtering."""

    query_text: str
    source: WebSource
