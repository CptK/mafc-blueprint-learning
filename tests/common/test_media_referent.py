"""Tests for referent-digest extraction from reverse-image-search evidence.

These build evidence without a stored ``referent``, so they exercise the legacy
text-parsing fallback that keeps archived traces rejudgeable. The structured path
that live runs use is covered in tests/agents/media/test_utils.py and
tests/agents/judge/test_referent_linkage.py.
"""

from __future__ import annotations

from mafc.common.action import Action
from mafc.common.evidence import Evidence
from mafc.common.media_referent import (
    extract_referent_digest,
    format_referent_block,
    normalize_url,
    referent_tag,
)
from mafc.common.modeling.prompt import Prompt


class RisAction(Action):
    name = "reverse_image_search"

    def __init__(self):
        self._save_parameters(locals())


class OtherAction(Action):
    name = "web_search"

    def __init__(self):
        self._save_parameters(locals())


def _ris_evidence(text: str, source: str = "ris://query") -> Evidence:
    return Evidence(raw=Prompt(text=text), action=RisAction(), source=source)


def test_extracts_exact_and_partial_pairs() -> None:
    text = (
        "Web Source https://www.example.com/article?utm_source=x\n"
        "Match type: EXACT copy of the media appears on this page.\n"
        "Web Source https://other.org/post/\n"
        "Match type: PARTIAL match — a cropped, edited, or overlapping version "
        "of the media appears on this page (not necessarily the same media)."
    )
    digest = extract_referent_digest([_ris_evidence(text)])
    assert digest.ris_ran
    assert digest.classify("https://example.com/article") == "exact"
    assert digest.classify("http://www.example.com/article?ref=abc") == "exact"
    assert digest.classify("https://other.org/post") == "partial"
    assert digest.classify("https://unrelated.com/page") is None


def test_matched_image_urls_are_not_captured_as_pages() -> None:
    text = (
        "Web Source https://page.com/a\n"
        "Match type: EXACT copy of the media appears on this page. "
        "Matched image file(s): https://cdn.com/img.jpg"
    )
    digest = extract_referent_digest([_ris_evidence(text)])
    assert digest.classify("https://page.com/a") == "exact"
    assert digest.classify("https://cdn.com/img.jpg") is None


def test_exact_wins_over_partial_for_same_page() -> None:
    partial_first = (
        "Web Source https://site.com/x\nMatch type: PARTIAL match — similar.\n"
        "Web Source https://site.com/x\nMatch type: EXACT copy of the media appears on this page."
    )
    digest = extract_referent_digest([_ris_evidence(partial_first)])
    assert digest.classify("https://site.com/x") == "exact"
    assert not digest.partial


def test_source_fallback_when_note_has_no_inline_url() -> None:
    evidence = _ris_evidence(
        "Match type: EXACT copy of the media appears on this page.",
        source="https://x.com/user/status/123",
    )
    digest = extract_referent_digest([evidence])
    assert digest.classify("https://x.com/user/status/123") == "exact"


def test_non_ris_evidence_is_ignored() -> None:
    evidence = Evidence(
        raw=Prompt(text="Match type: EXACT copy of the media appears on this page."),
        action=OtherAction(),
        source="https://example.com",
    )
    digest = extract_referent_digest([evidence])
    assert not digest.ris_ran
    assert format_referent_block(digest) is None


def test_no_match_reported() -> None:
    digest = extract_referent_digest([_ris_evidence("No provenance could be established for this media.")])
    assert digest.no_match_reported
    block = format_referent_block(digest)
    assert block is not None and "found no matches" in block


def test_format_block_contains_binding_rule_and_urls() -> None:
    text = "Web Source https://a.com/1\nMatch type: EXACT copy of the media appears on this page."
    block = format_referent_block(extract_referent_digest([_ris_evidence(text)]))
    assert "https://a.com/1" in block
    assert "Binding rule" in block
    assert "symmetrically" in block


def test_referent_tags() -> None:
    assert "SAME MEDIA CONFIRMED" in referent_tag("exact")
    assert "PARTIAL" in referent_tag("partial")
    assert referent_tag(None) is None


def test_normalize_url() -> None:
    assert normalize_url("https://www.Site.com/a/b/?q=1#f") == "site.com/a/b"
    assert normalize_url("http://site.com/a/b") == "site.com/a/b"
