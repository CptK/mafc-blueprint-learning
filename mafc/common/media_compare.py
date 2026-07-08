"""Frame-level media comparison: is candidate imagery the SAME footage as the claim's?

Separates "same scene, same camera, same take" from "same event, different
camera" — the distinction wrong-referent flips hinge on. Two tiers:

1. Perceptual hashes (pHash via DCT + dHash): catch re-encodes, resolution
   changes, and excerpts. Cheap enough to run frame×candidate exhaustively.
2. ORB keypoints + RANSAC homography: catch crops, vertical re-cuts, and
   caption overlays by verifying that fine image geometry aligns — which two
   different cameras filming the same event cannot produce.

Semantic embeddings are deliberately NOT used: two different videos of the same
event are semantically near-identical, which is precisely the confusion to kill.

Thresholds are conservative by design: a false SAME would bind a wrong-referent
debunk to the claim's media — the exact failure this exists to prevent. A missed
match merely leaves the referent unverified, which is the status quo.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Hash agreement: both hashes must be close. 64-bit hashes; unrelated images
# average ~32 differing bits, re-encoded copies 0-8.
PHASH_MAX_DISTANCE = 8
DHASH_MAX_DISTANCE = 10

# Geometric verification: inliers of a RANSAC homography between ORB keypoints.
# Strict on purpose: branded news cards, watermarks, and shared text overlays
# produce spurious geometric agreement between DIFFERENT frames (measured on the
# T2/T3 flip cohort: every og:image false-confirmation was ORB-tier). Callers
# should additionally gate ORB behind a prior (see compare_frame_sets(allow_orb)).
ORB_MIN_INLIERS = 45
ORB_MIN_INLIER_RATIO = 0.35
_ORB_FEATURES = 1000
_ORB_CANDIDATE_PAIRS = 6  # hash-closest pairs to try before giving up


def to_gray(image: np.ndarray) -> np.ndarray:
    """Convert an RGB/BGR/gray uint8 array to single-channel grayscale."""
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        image = image[:, :, :3]
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def phash(gray: np.ndarray) -> int:
    """64-bit DCT perceptual hash (low-frequency structure of the frame)."""
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)
    low = dct[:8, :8].flatten()
    median = np.median(low[1:])  # skip DC term
    bits = low > median
    return int(np.packbits(bits.astype(np.uint8)).view(">u8")[0])


def dhash(gray: np.ndarray) -> int:
    """64-bit gradient hash (per-row brightness direction)."""
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    return int(np.packbits(bits.astype(np.uint8)).view(">u8")[0])


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _mirror(gray: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(gray[:, ::-1])


@dataclass
class FrameHashes:
    gray: np.ndarray
    ph: int
    dh: int
    ph_flip: int
    dh_flip: int


def hash_frame(gray: np.ndarray) -> FrameHashes:
    flipped = _mirror(gray)
    return FrameHashes(
        gray=gray, ph=phash(gray), dh=dhash(gray), ph_flip=phash(flipped), dh_flip=dhash(flipped)
    )


def _hash_match(claim: FrameHashes, cand: FrameHashes) -> bool:
    """Hashes agree in either orientation (mirror flips are a common repost trick)."""
    direct = (
        hamming(claim.ph, cand.ph) <= PHASH_MAX_DISTANCE and hamming(claim.dh, cand.dh) <= DHASH_MAX_DISTANCE
    )
    flipped = (
        hamming(claim.ph_flip, cand.ph) <= PHASH_MAX_DISTANCE
        and hamming(claim.dh_flip, cand.dh) <= DHASH_MAX_DISTANCE
    )
    return direct or flipped


def orb_inliers(gray_a: np.ndarray, gray_b: np.ndarray) -> tuple[int, float]:
    """(inlier count, inlier ratio) of a RANSAC homography between ORB features."""
    orb = cv2.ORB_create(nfeatures=_ORB_FEATURES)
    kp_a, des_a = orb.detectAndCompute(gray_a, None)
    kp_b, des_b = orb.detectAndCompute(gray_b, None)
    if des_a is None or des_b is None or len(kp_a) < 8 or len(kp_b) < 8:
        return 0, 0.0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = matcher.knnMatch(des_a, des_b, k=2)
    good = [m for pair in knn if len(pair) == 2 for m, n in [pair] if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return 0, 0.0
    src = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if homography is None or mask is None:
        return 0, 0.0
    inliers = int(mask.sum())
    return inliers, inliers / len(good)


def _orb_match(gray_a: np.ndarray, gray_b: np.ndarray) -> bool:
    for candidate in (gray_b, _mirror(gray_b)):
        inliers, ratio = orb_inliers(gray_a, candidate)
        if inliers >= ORB_MIN_INLIERS and ratio >= ORB_MIN_INLIER_RATIO:
            return True
    return False


@dataclass
class MediaMatchResult:
    """Outcome of comparing claim frames against one candidate's imagery."""

    verdict: str  # "same" | "no_match"
    method: str | None = None  # "hash" | "orb" when verdict == "same"
    matched_claim_frames: int = 0
    n_claim_frames: int = 0
    n_candidate_images: int = 0

    @property
    def is_same(self) -> bool:
        return self.verdict == "same"


def compare_frame_sets(
    claim_frames: list[np.ndarray],
    candidate_images: list[np.ndarray],
    allow_orb: bool = True,
) -> MediaMatchResult:
    """Compare claim keyframes against a candidate's images.

    Returns "same" when any claim frame matches any candidate image by
    conservative dual-hash agreement, or — failing that — by ORB geometric
    verification on the hash-closest pairs. Never returns "different": a failed
    match is weak evidence (the candidate imagery may simply not include the
    relevant frame) and must stay "no_match" (i.e. unverified).

    Args:
        allow_orb: Enable the geometric tier. Callers without a prior that the
            candidate contains a variant of the claim frame (e.g. a Google
            PARTIAL label) should pass False — ORB alone can be fooled by
            shared branding/overlays between different frames.
    """
    claim_hashed = [hash_frame(to_gray(f)) for f in claim_frames]
    cand_hashed = [hash_frame(to_gray(f)) for f in candidate_images]
    if not claim_hashed or not cand_hashed:
        return MediaMatchResult(
            "no_match", n_claim_frames=len(claim_hashed), n_candidate_images=len(cand_hashed)
        )

    matched = {i for i, cf in enumerate(claim_hashed) for cand in cand_hashed if _hash_match(cf, cand)}
    if matched:
        return MediaMatchResult(
            "same",
            method="hash",
            matched_claim_frames=len(matched),
            n_claim_frames=len(claim_hashed),
            n_candidate_images=len(cand_hashed),
        )

    if not allow_orb:
        return MediaMatchResult(
            "no_match", n_claim_frames=len(claim_hashed), n_candidate_images=len(cand_hashed)
        )

    # ORB tier: try the hash-closest (claim frame, candidate) pairs — crops and
    # re-cuts defeat global hashes but keep local geometry.
    pairs = sorted(
        (
            (hamming(cf.ph, cand.ph), i, j)
            for i, cf in enumerate(claim_hashed)
            for j, cand in enumerate(cand_hashed)
        ),
    )[:_ORB_CANDIDATE_PAIRS]
    for _, i, j in pairs:
        if _orb_match(claim_hashed[i].gray, cand_hashed[j].gray):
            return MediaMatchResult(
                "same",
                method="orb",
                matched_claim_frames=1,
                n_claim_frames=len(claim_hashed),
                n_candidate_images=len(cand_hashed),
            )

    return MediaMatchResult("no_match", n_claim_frames=len(claim_hashed), n_candidate_images=len(cand_hashed))
