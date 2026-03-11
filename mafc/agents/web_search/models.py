from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from mafc.common.evidence import Evidence
from mafc.tools.web_search.common import Source, WebSource
from mafc.tools.web_search.common import Query, SearchResults


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


@dataclass
class QueryInvestigationResult:
    """Structured retrieval output for one handled query."""

    query_text: str
    observation_text: str
    evidences: list[Evidence]


@dataclass
class IterationOutcome:
    """Run-loop outcome for one iteration."""

    should_stop: bool


class SearchTool(Protocol):
    """Minimal search interface required by the web-search agent."""

    def search(self, query: Query) -> SearchResults | None:
        pass
