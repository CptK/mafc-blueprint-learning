"""Local referent verification: frame-match evidence pages against claim media.

Extends the RIS-based referent digest to sources Google didn't label — the
T2/T3 cohorts of the flip analysis (platform embeds and fact-check articles):
fetch each candidate page's imagery (og:image/twitter:image cover the video
thumbnail on platform pages; article <img> tags cover fact-check stills; RIS
"Matched image file(s)" URLs are fetched directly) and compare it against the
claim's keyframes with mafc.common.media_compare.

Only positive results are recorded: a page whose fetched imagery doesn't match
stays UNVERIFIED — its imagery may simply not include the relevant still, so a
non-match must never be read as "different media" (mirror of the RIS neutrality
rule). All failures are swallowed: verification is an enhancement, never a
reason for a judge call to fail.
"""

from __future__ import annotations

import re
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from ezmm import MultimodalSequence

from mafc.common.evidence import Evidence
from mafc.common.logger import logger
from mafc.common.media_compare import compare_frame_sets
from mafc.common.media_referent import ReferentDigest, normalize_url

_UA = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    )
}
_FETCH_TIMEOUT = 8
_PAGE_BYTE_CAP = 512 * 1024
_IMAGE_BYTE_CAP = 8 * 1024 * 1024
_MAX_SOURCES = 6
_MAX_IMAGES_PER_PAGE = 3
_CLAIM_VIDEO_FRAMES = 8
_WORKERS = 4

_IMAGE_EXT_RE = re.compile(r"\.(jpe?g|png|webp|gif|bmp)(\?|$)", re.IGNORECASE)
_META_IMAGE_RE = re.compile(
    r"<meta[^>]+(?:property|name)=[\"'](?:og:image|twitter:image(?::src)?)[\"'][^>]+content=[\"']([^\"']+)",
    re.IGNORECASE,
)
_META_IMAGE_RE_REVERSED = re.compile(
    r"<meta[^>]+content=[\"']([^\"']+)[\"'][^>]+(?:property|name)=[\"'](?:og:image|twitter:image(?::src)?)[\"']",
    re.IGNORECASE,
)
_IMG_TAG_RE = re.compile(r"<img[^>]+src=[\"'](https?://[^\"']+)", re.IGNORECASE)
_MATCHED_FILES_RE = re.compile(r"Matched image file\(s\): ([^\n]+)")


def _fetch_bytes(url: str, byte_cap: int) -> bytes | None:
    """GET up to byte_cap bytes; None on any failure. Module-level for test patching."""
    import requests

    try:
        response = requests.get(url, headers=_UA, timeout=_FETCH_TIMEOUT, stream=True)
        if response.status_code >= 400:
            return None
        body = b""
        for chunk in response.iter_content(65536):
            body += chunk
            if len(body) > byte_cap:
                break
        response.close()
        return body
    except Exception:
        return None


def _decode_image(data: bytes) -> np.ndarray | None:
    try:
        arr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if arr is None or arr.shape[0] < 32 or arr.shape[1] < 32:
            return None  # icons/trackers carry no referent signal
        return arr
    except Exception:
        return None


def _candidate_image_urls(source_url: str) -> list[str]:
    """Image URLs to compare for one evidence source."""
    if _IMAGE_EXT_RE.search(source_url):
        return [source_url]
    page = _fetch_bytes(source_url, _PAGE_BYTE_CAP)
    if page is None:
        return []
    html = page.decode("utf-8", errors="replace")
    urls: list[str] = _META_IMAGE_RE.findall(html) + _META_IMAGE_RE_REVERSED.findall(html)
    urls += [u for u in _IMG_TAG_RE.findall(html) if _IMAGE_EXT_RE.search(u)]
    seen: list[str] = []
    for url in urls:
        if url.startswith("http") and url not in seen:
            seen.append(url)
        if len(seen) >= _MAX_IMAGES_PER_PAGE:
            break
    return seen


def _fetch_candidate_images(source_url: str) -> list[np.ndarray]:
    images = []
    for url in _candidate_image_urls(source_url):
        data = _fetch_bytes(url, _IMAGE_BYTE_CAP)
        if data is None:
            continue
        decoded = _decode_image(data)
        if decoded is not None:
            images.append(decoded)
    return images


def extract_claim_frames(claim: MultimodalSequence) -> list[np.ndarray]:
    """Keyframes/images of the claim's media as uint8 arrays."""
    frames: list[np.ndarray] = []
    for video in claim.videos:
        try:
            frames.extend(video.sample_frames(_CLAIM_VIDEO_FRAMES, format="rgb"))
        except Exception as e:
            logger.warning(f"[ReferentVerifier] Frame sampling failed for {video.reference}: {e}")
    for image in claim.images:
        try:
            frames.append(np.array(image.image.convert("RGB")))
        except Exception as e:
            logger.warning(f"[ReferentVerifier] Image load failed for {image.reference}: {e}")
    return frames


def _matched_file_urls(evidences: list[Evidence]) -> dict[str, str]:
    """RIS 'Matched image file(s)' URLs, keyed by the page they were found for.

    The direct image files Google matched are fetchable even when the page
    itself is bot-walled, so they get compared under their page's URL.
    """
    out: dict[str, str] = {}
    for evidence in evidences:
        if getattr(evidence.action, "name", None) != "reverse_image_search":
            continue
        text = str(evidence.raw)
        for match in _MATCHED_FILES_RE.finditer(text):
            preceding = text[: match.start()]
            page_urls = re.findall(r"(https?://\S+)", preceding)
            if not page_urls:
                continue
            page = page_urls[-1].rstrip(".,;)")
            first_file = match.group(1).split(",")[0].strip().rstrip(".,;)")
            if first_file.startswith("http"):
                out.setdefault(page, first_file)
    return out


def verify_evidence_referents(
    claim: MultimodalSequence,
    evidences: list[Evidence],
    digest: ReferentDigest,
    max_sources: int = _MAX_SOURCES,
) -> None:
    """Frame-match unconfirmed evidence sources; record confirmations in the digest.

    Mutates ``digest.local`` (pages visually confirmed by local comparison).
    PARTIAL-labelled pages are verified first — Google already found similar
    imagery there, so local geometric verification has the best odds of turning
    them into confirmations (crops/edits are exactly what ORB covers).
    """
    claim_frames = extract_claim_frames(claim)
    if not claim_frames:
        return

    matched_files = _matched_file_urls(evidences)

    partial_first: list[str] = list(digest.partial.values())
    rest: list[str] = []
    seen: set[str] = set(map(normalize_url, partial_first))
    for evidence in evidences:
        url = evidence.source
        if not isinstance(url, str) or not url.startswith("http"):
            continue
        key = normalize_url(url)
        if not key or key in seen or key in digest.exact or key in digest.local:
            continue
        seen.add(key)
        rest.append(url)
    queue = (partial_first + rest)[:max_sources]
    if not queue:
        return

    lock = threading.Lock()

    def _verify(page_url: str) -> None:
        images: list[np.ndarray] = []
        direct = matched_files.get(page_url)
        if direct:
            data = _fetch_bytes(direct, _IMAGE_BYTE_CAP)
            decoded = _decode_image(data) if data else None
            if decoded is not None:
                images.append(decoded)
        images.extend(_fetch_candidate_images(page_url))
        if not images:
            return
        # ORB (crop-tolerant but foolable by shared branding between different
        # frames) is only trusted where Google's index provides a prior that a
        # variant of the claim frame exists on this page; elsewhere hash-only.
        has_prior = normalize_url(page_url) in digest.partial or direct is not None
        result = compare_frame_sets(claim_frames, images, allow_orb=has_prior)
        if result.is_same:
            with lock:
                digest.local[normalize_url(page_url)] = page_url
            logger.debug(
                f"[ReferentVerifier] Confirmed same media on {page_url} "
                f"(method={result.method}, frames={result.matched_claim_frames})."
            )

    try:
        with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
            list(pool.map(_verify, queue))
    except Exception as e:
        logger.warning(f"[ReferentVerifier] Verification pass failed: {e}")
