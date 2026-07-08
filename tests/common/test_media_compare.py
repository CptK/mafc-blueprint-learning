"""Tests for frame-level media comparison (hash + ORB tiers).

The invariants under test mirror the wrong-referent problem:
- re-encodes/resolution changes of the SAME frame must match (hash tier),
- crops of the SAME frame must match (ORB geometric tier),
- different imagery — even statistically similar — must NOT match,
- verdicts are only ever "same" or "no_match" (never "different").
"""

from __future__ import annotations

import cv2
import numpy as np

from mafc.common.media_compare import (
    MediaMatchResult,
    compare_frame_sets,
    dhash,
    hamming,
    phash,
    to_gray,
)


def _textured_frame(seed: int, size: int = 480) -> np.ndarray:
    """Deterministic textured RGB frame (blurred noise + seed-dependent shapes).

    Shape positions derive from the seed so that different seeds produce
    genuinely different imagery — not just different noise under an identical
    layout (which perceptual hashes would rightly consider similar).
    """
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    img = cv2.GaussianBlur(img, (7, 7), 0)
    for _ in range(6):
        x0, y0 = rng.integers(0, size - 80, 2)
        w, h = rng.integers(40, 160, 2)
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        if rng.random() < 0.5:
            cv2.rectangle(img, (int(x0), int(y0)), (int(min(x0 + w, size - 1)), int(min(y0 + h, size - 1))), color, -1)
        else:
            cv2.circle(img, (int(x0 + 40), int(y0 + 40)), int(w // 3), color, -1)
    cv2.putText(
        img, f"SAMPLE{seed}", (int(rng.integers(20, size // 3)), int(rng.integers(60, size - 20))),
        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 4,
    )
    return img


def _reencode_jpeg(img: np.ndarray, quality: int = 60) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def test_same_frame_different_resolution_and_encoding_matches() -> None:
    frame = _textured_frame(1)
    downscaled = cv2.resize(frame, (160, 160))
    candidate = _reencode_jpeg(downscaled, quality=40)
    result = compare_frame_sets([frame], [candidate])
    assert result.is_same and result.method == "hash"


def test_mirrored_copy_matches() -> None:
    frame = _textured_frame(2)
    mirrored = frame[:, ::-1].copy()
    assert compare_frame_sets([frame], [mirrored]).is_same


def test_cropped_frame_matches_via_orb() -> None:
    frame = _textured_frame(3, size=640)
    crop = frame[80:560, 160:640].copy()  # 75% crop, offset
    result = compare_frame_sets([frame], [crop])
    assert result.is_same
    assert result.method in ("orb", "hash")


def test_different_frames_do_not_match() -> None:
    result = compare_frame_sets([_textured_frame(4)], [_textured_frame(5)])
    assert not result.is_same
    assert result.verdict == "no_match"


def test_excerpt_detection_over_frame_sets() -> None:
    # Claim clip = frames 2..4 of a longer candidate video; unrelated frame ignored.
    candidate_frames = [_textured_frame(i) for i in range(6)]
    claim_frames = [candidate_frames[2], candidate_frames[3], candidate_frames[4]]
    result = compare_frame_sets(claim_frames, candidate_frames)
    assert result.is_same
    assert result.matched_claim_frames == 3


def test_orb_tier_can_be_disabled() -> None:
    frame = _textured_frame(3, size=640)
    crop = frame[80:560, 160:640].copy()
    result = compare_frame_sets([frame], [crop], allow_orb=False)
    # Without the geometric tier a crop cannot be confirmed — and must not be.
    if compare_frame_sets([frame], [crop]).method == "orb":
        assert result.verdict == "no_match"


def test_empty_inputs_return_no_match() -> None:
    assert compare_frame_sets([], [_textured_frame(1)]).verdict == "no_match"
    assert compare_frame_sets([_textured_frame(1)], []).verdict == "no_match"


def test_hashes_are_stable_and_discriminative() -> None:
    a = to_gray(_textured_frame(6))
    b = to_gray(_textured_frame(7))
    assert hamming(phash(a), phash(a)) == 0
    assert hamming(phash(a), phash(b)) > 12
    assert hamming(dhash(a), dhash(b)) > 12


def test_never_emits_different() -> None:
    for result in (
        compare_frame_sets([_textured_frame(8)], [_textured_frame(9)]),
        compare_frame_sets([], []),
    ):
        assert isinstance(result, MediaMatchResult)
        assert result.verdict in ("same", "no_match")
