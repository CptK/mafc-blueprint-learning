"""Media referent status: link reverse-image-search matches to evidence sources.

The dominant residual error class (wrong-referent flips) happens because the
verdict stage sees "a fact-check debunks this footage" without knowing whether
the fact-checked footage IS the claim's footage. Reverse image search already
answers that question for many pages (Google's EXACT/PARTIAL page labels), but
the signal was buried inside one evidence item among many. This module extracts
it into a per-source referent status that the judge can enforce: measured on the
0707-112229 flip core, 44% of flipped media claims had an EXACT match in their
evidence and flipped anyway.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from mafc.common.evidence import Evidence

_RIS_ACTION_NAME = "reverse_image_search"

# Matches "Web Source <url>" followed by a match-type note, as emitted by
# GoogleRisResults / WebSource previews.
_MATCH_PAIR_RE = re.compile(
    r"(https?://\S+)[^\n]*\n[^\n]*Match type: (EXACT|PARTIAL|unknown)",
    re.IGNORECASE,
)
_NO_MATCH_MARKER = "No provenance could be established"


def normalize_url(url: str) -> str:
    """Normalize a URL for set membership: scheme, www., query, fragment and
    trailing slash are irrelevant to page identity here."""
    parts = urlsplit(url.strip())
    host = (parts.netloc or "").lower()
    host = host.removeprefix("www.")
    path = (parts.path or "").rstrip("/")
    return f"{host}{path}"


@dataclass
class ReferentDigest:
    """Per-claim referent knowledge extracted from reverse-image-search evidence."""

    exact: dict[str, str] = field(default_factory=dict)
    """normalized url -> original url of pages with a Google-verified EXACT copy."""

    partial: dict[str, str] = field(default_factory=dict)
    """normalized url -> original url of pages with only a PARTIAL/similar match."""

    local: dict[str, str] = field(default_factory=dict)
    """normalized url -> original url of pages confirmed by LOCAL frame comparison
    (mafc.common.referent_verifier) — same strength as ``exact``."""

    ris_ran: bool = False
    """True when any reverse-image-search evidence was present."""

    no_match_reported: bool = False
    """True when RIS explicitly reported that no provenance could be established."""

    @property
    def has_referent_info(self) -> bool:
        return self.ris_ran or bool(self.local)

    def classify(self, source_url: str) -> str | None:
        """Return 'exact', 'partial', or None for an evidence source URL."""
        key = normalize_url(source_url)
        if not key:
            return None
        if key in self.exact or key in self.local:
            return "exact"
        if key in self.partial:
            return "partial"
        return None


def extract_referent_digest(evidences: list[Evidence]) -> ReferentDigest:
    """Build the referent digest from all reverse-image-search evidence items."""
    digest = ReferentDigest()
    for evidence in evidences:
        if getattr(evidence.action, "name", None) != _RIS_ACTION_NAME:
            continue
        digest.ris_ran = True
        texts = [str(evidence.raw)]
        if evidence.takeaways is not None:
            texts.append(str(evidence.takeaways))
        combined = "\n".join(texts)

        if _NO_MATCH_MARKER.lower() in combined.lower():
            digest.no_match_reported = True

        pairs: list[tuple[str, str]] = _MATCH_PAIR_RE.findall(combined)
        # Per-page RIS evidence carries the page URL as its source; fall back to
        # it when the match note doesn't repeat the URL inline.
        if not pairs and evidence.source.startswith("http"):
            match = re.search(r"Match type: (EXACT|PARTIAL|unknown)", combined, re.IGNORECASE)
            if match:
                pairs = [(evidence.source, match.group(1))]

        for url, kind in pairs:
            url = url.rstrip(".,;)")
            key = normalize_url(url)
            if not key:
                continue
            if kind.upper() == "EXACT":
                digest.exact[key] = url
                digest.partial.pop(key, None)
            elif kind.upper() == "PARTIAL" and key not in digest.exact:
                digest.partial[key] = url
    return digest


def referent_tag(status: str | None) -> str | None:
    """One-line evidence annotation for a classified source, or None."""
    if status == "exact":
        return (
            "Referent status: SAME MEDIA CONFIRMED — an exact visual copy of the "
            "claim's media appears on this page."
        )
    if status == "partial":
        return (
            "Referent status: PARTIAL visual match only — a similar or edited "
            "version appears on this page; NOT confirmed to be the same media."
        )
    return None


def format_referent_block(digest: ReferentDigest, max_urls: int = 8) -> str | None:
    """Render the judge-facing referent section, or None without referent info."""
    if not digest.has_referent_info:
        return None

    def _fmt(urls: dict[str, str]) -> str:
        if not urls:
            return "none"
        shown = list(urls.values())[:max_urls]
        suffix = f" (+{len(urls) - max_urls} more)" if len(urls) > max_urls else ""
        return ", ".join(shown) + suffix

    lines = [
        "Media referent status (from reverse image search and direct frame comparison "
        "over the claim's media):",
        f"- Pages visually CONFIRMED to contain this exact media: {_fmt({**digest.exact, **digest.local})}",
        f"- Pages with only a PARTIAL/similar match (not confirmed same media): "
        f"{_fmt({k: v for k, v in digest.partial.items() if k not in digest.local})}",
    ]
    if digest.no_match_reported and not digest.exact and not digest.partial and not digest.local:
        lines.append("- Reverse image search found no matches for this media anywhere.")
    lines.append(
        "Binding rule: statements about this media's origin, date, location, context, "
        "or debunking apply to THIS media only if their source page is visually "
        "confirmed above, or the evidence itself demonstrates an exact match (identical "
        "frames or image). Evidence from unconfirmed pages may describe DIFFERENT "
        "footage or images of the same event — such evidence neither refutes nor "
        "confirms this media. This rule applies symmetrically to debunks and to "
        "origin/authenticity confirmations."
    )
    return "\n".join(lines)
