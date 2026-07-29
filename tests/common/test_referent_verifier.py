"""Tests for local referent verification (network mocked)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

import mafc.common.referent_verifier as rv
from mafc.common.action import Action
from mafc.common.evidence import Evidence
from mafc.common.media_referent import ReferentDigest
from mafc.common.modeling.prompt import Prompt


class WebAction(Action):
    name = "web_search"

    def __init__(self):
        self._save_parameters(locals())


class RisAction(Action):
    name = "reverse_image_search"

    def __init__(self):
        self._save_parameters(locals())


def _frame(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (240, 240, 3), dtype=np.uint8)
    return cv2.GaussianBlur(img, (7, 7), 0)


def _jpeg(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


class FakeClaim:
    """Minimal stand-in exposing .videos/.images like a MultimodalSequence."""

    def __init__(self, frames: list[np.ndarray]):
        self._frames = frames

        class _Vid:
            reference = "<video:test>"

            def __init__(self, frames):
                self._frames = frames

            def sample_frames(self, n, format="rgb"):
                return self._frames[:n]

        self.videos = [_Vid(frames)]
        self.images = []


CLAIM_FRAME = _frame(1)
OTHER_FRAME = _frame(2)
PAGE_HTML = b'<html><meta property="og:image" content="https://cdn.site.com/still.jpg"></html>'


def _evidence(source: str) -> Evidence:
    return Evidence(raw=Prompt(text="finding"), action=WebAction(), source=source)


@pytest.fixture()
def patched_fetch(monkeypatch):
    """Fake network: article page -> og:image -> claim's own frame; other page -> unrelated."""

    def fake_fetch(url: str, byte_cap: int) -> bytes | None:
        return {
            "https://factcheck.org/debunk": PAGE_HTML,
            "https://cdn.site.com/still.jpg": _jpeg(CLAIM_FRAME),
            "https://other.org/article": b'<html><meta property="og:image" content="https://other.org/img.jpg"></html>',
            "https://other.org/img.jpg": _jpeg(OTHER_FRAME),
            "https://cdn.google.com/matched.jpg": _jpeg(CLAIM_FRAME),
        }.get(url)

    monkeypatch.setattr(rv, "_fetch_bytes", fake_fetch)


def test_confirms_page_whose_still_matches(patched_fetch) -> None:
    digest = ReferentDigest(ris_ran=True)
    claim = FakeClaim([CLAIM_FRAME])
    rv.verify_evidence_referents(
        claim, [_evidence("https://factcheck.org/debunk"), _evidence("https://other.org/article")], digest
    )
    assert digest.classify("https://factcheck.org/debunk") == "exact"
    assert digest.classify("https://other.org/article") is None  # non-match stays unverified


def test_matched_file_urls_are_compared_for_their_page(patched_fetch) -> None:
    ris_evidence = Evidence(
        raw=Prompt(
            text="Web Source https://walled.com/article\n"
            "Match type: PARTIAL match — similar. "
            "Matched image file(s): https://cdn.google.com/matched.jpg"
        ),
        action=RisAction(),
        source="ris://query",
    )
    digest = ReferentDigest(ris_ran=True, partial={"walled.com/article": "https://walled.com/article"})
    rv.verify_evidence_referents(FakeClaim([CLAIM_FRAME]), [ris_evidence], digest)
    # The walled page itself is unfetchable, but Google's matched image file
    # confirms it — the PARTIAL page upgrades to exact.
    assert digest.classify("https://walled.com/article") == "exact"


def test_no_claim_frames_is_a_noop(patched_fetch) -> None:
    digest = ReferentDigest(ris_ran=True)
    claim = FakeClaim([])
    claim.videos = []
    rv.verify_evidence_referents(claim, [_evidence("https://factcheck.org/debunk")], digest)
    assert not digest.local


def test_already_confirmed_pages_are_skipped(patched_fetch, monkeypatch) -> None:
    calls: list[str] = []
    original = rv._candidate_image_urls

    def spy(url: str):
        calls.append(url)
        return original(url)

    monkeypatch.setattr(rv, "_candidate_image_urls", spy)
    digest = ReferentDigest(ris_ran=True, exact={"factcheck.org/debunk": "https://factcheck.org/debunk"})
    rv.verify_evidence_referents(
        FakeClaim([CLAIM_FRAME]), [_evidence("https://factcheck.org/debunk")], digest
    )
    assert calls == []
