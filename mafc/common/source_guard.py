"""Harness-level guard against retrieving a benchmark's own answer key.

VeriTaS labels are derived from a specific fact-check article, and that article is
reachable on the open web. A system that retrieves it is not reasoning to the
verdict, it is reading it — but nothing in the retrieval stack can tell the
difference, so the exclusion has to be imposed from outside.

The cutoff date is the primary defence and this is the backstop, for two cases the
date cannot cover:

* Reverse image search returns pages with **no date at all** (the Vision API does
  not expose one), so a date bound cannot be applied on that path. RIS is also the
  single most likely tool to surface a fact-check, because fact-check articles
  embed the very media being checked.
* A fact-check published before the claim date -- for a recycled claim, say --
  passes any date bound legitimately.

Blocking is logged rather than silent: a guard that never reports is
indistinguishable from one that is not wired up, and the count is itself the
measurement of how much leakage the date bound is missing.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TypeVar

from mafc.common.logger import logger
from mafc.common.media_referent import normalize_url

_T = TypeVar("_T")


def normalize_blocked(urls: Iterable[str] | None) -> set[str]:
    """Normalize URLs into the comparable form used by ``is_blocked``."""
    if not urls:
        return set()
    return {key for url in urls if url and (key := normalize_url(url))}


def is_blocked(url: str | None, blocked: set[str]) -> bool:
    """True when ``url`` is one of the blocked references.

    Compares on normalized host+path, so a bare scheme or ``www.`` difference
    cannot slip the guard — 1 of the 274 leaks measured on the 2026 Q1 run
    differed from the recorded review URL by exactly that.
    """
    if not blocked or not url:
        return False
    return normalize_url(url) in blocked


def filter_blocked_sources(
    sources: Sequence[_T] | None,
    blocked: set[str],
    *,
    context: str = "",
) -> list[_T]:
    """Drop sources whose reference is blocked, logging each hit.

    Accepts anything exposing a ``reference`` attribute. Returns a list so callers
    can treat the result uniformly; ``None`` becomes ``[]``.
    """
    if not sources:
        return []
    if not blocked:
        return list(sources)

    kept: list[_T] = []
    dropped: list[str] = []
    for source in sources:
        reference = getattr(source, "reference", None)
        if is_blocked(reference, blocked):
            dropped.append(str(reference))
        else:
            kept.append(source)

    if dropped:
        where = f" [{context}]" if context else ""
        logger.warning(
            f"[SourceGuard]{where} Blocked {len(dropped)} answer-key source(s) that the "
            f"date cutoff did not exclude: {', '.join(dropped[:3])}"
            + (f" (+{len(dropped) - 3} more)" if len(dropped) > 3 else "")
        )
    return kept
