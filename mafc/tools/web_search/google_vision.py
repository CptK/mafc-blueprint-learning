import os
from dataclasses import dataclass
from typing import Any, Sequence, cast

from ezmm import Image, Video
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import vision

from config.globals import google_service_account_key_path
from mafc.common.logger import logger
from mafc.tools.web_search.common import Query, WebSource, SearchResults
from mafc.utils.parsing import get_base_domain

# Number of evenly-spaced keyframes to reverse-search for a video. A single frame
# (often a title/black intro frame) is a weak basis for finding a video's true
# origin; sampling several frames and merging the matches is far more reliable.
_VIDEO_RIS_FRAMES = 4

# Match-note prefixes. Kept as constants so the producer (`_parse_results`) and the
# consumer (`match_precision`, used to decide which pages may become evidence) can
# never drift apart. The "Match type: <EXACT|PARTIAL|unknown>" shape is also parsed
# by mafc.common.media_referent — do not reword it without updating that regex.
EXACT_MATCH_NOTE = "Match type: EXACT copy of the media appears on this page."
PARTIAL_MATCH_NOTE = (
    "Match type: PARTIAL match — a cropped, edited, or overlapping version "
    "of the media appears on this page (not necessarily the same media)."
)
UNKNOWN_MATCH_NOTE = "Match type: unknown precision (page listed as containing a matching image)."


def match_precision(source: WebSource) -> str | None:
    """Classify a RIS page as 'exact', 'partial', or None (no confirmed match).

    None means Google listed the page as visually similar but confirmed neither a
    full nor a partial image match. Such a page says nothing about THIS media's
    referent and must not be promoted into its own evidence item — that is how
    lookalike pages get read downstream as provenance findings.
    """
    note = (source.preview or "").strip()
    if note.startswith("Match type: EXACT"):
        return "exact"
    if note.startswith("Match type: PARTIAL"):
        return "partial"
    return None


@dataclass
class GoogleRisResults(SearchResults):
    """Reverse Image Search (RIS) results. Ship with additional object detection
    information next to the list of sources."""

    entities: dict[str, float]  # mapping between entity description and score
    best_guess_labels: list[str]

    def __str__(self):
        text = "**Reverse Image Search Results**"

        if self.entities:
            text += "\n\nPossible identified entities:\n"
            text += "\n".join(f"- {name}" for name, _ in self.entities.items())

        if self.best_guess_labels:
            text += f"\n\nBest guess about the topic of the image: {', '.join(self.best_guess_labels)}."

        if self.sources:
            text += (
                "\n\nPages where this media was found (match precision per page: an EXACT copy "
                "means the identical image appears there; a PARTIAL match means a cropped/edited/"
                "overlapping version):\n"
            )
            text += "\n".join(map(str, self.sources))
        else:
            text += (
                "\n\nNo pages containing this media were found by reverse image search. "
                "No provenance could be established: the media's origin, earliest appearance, "
                "and link to any claimed event remain unverified."
            )

        return text

    def __repr__(self):
        return (
            f"RisResults(n_sources={len(self.sources)}, "
            f"n_entities={len(self.entities)}, "
            f"n_best_guess_labels={len(self.best_guess_labels)})"
        )


class GoogleVisionAPI:
    """Wraps the Google Cloud Vision API for performing reverse image search (RIS)."""

    def __init__(self):
        self.client: vision.ImageAnnotatorClient | None = None
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_service_account_key_path.as_posix()
        try:
            self.client = vision.ImageAnnotatorClient()
        except DefaultCredentialsError:
            logger.error(
                f"[Google Vision API] ❌ No or invalid Google Cloud API credentials at "
                f"{google_service_account_key_path.as_posix()}."
            )
        else:
            logger.info("[Google Vision API] ✅ Successfully connected to Google Cloud Vision API.")

    def search(self, query: Query) -> GoogleRisResults:
        """Run image reverse search through Google Vision API and parse results."""
        if not query.has_media():
            logger.error("[Google Vision API] Query does not contain media for reverse search.")
            return GoogleRisResults(sources=[], query=query, entities={}, best_guess_labels=[])

        if self.client is None:
            logger.error("[Google Vision API] Cannot perform search because client is not initialized.")
            return GoogleRisResults(sources=[], query=query, entities={}, best_guess_labels=[])

        media = query.media
        if isinstance(media, Video):
            try:
                frame_contents = list(media.sample_frames(_VIDEO_RIS_FRAMES, format="jpeg"))
            except Exception as e:  # noqa: BLE001 — fall back to a single frame on any sampling error
                logger.warning(f"[Google Vision API] Video frame sampling failed ({e}); using one frame.")
                frame_contents = list(media.sample_frames(1, format="jpeg"))
        elif isinstance(media, Image):
            frame_contents = [media.get_base64_encoded()]
        else:
            logger.error(f"[Google Vision API] Unsupported media type for Google Vision API: {type(media)}.")
            return GoogleRisResults(sources=[], query=query, entities={}, best_guess_labels=[])

        per_frame: list[GoogleRisResults] = []
        for content in frame_contents:
            image = vision.Image(content=content)
            # `web_detection` exists at runtime, but some type stubs do not expose it.
            response = cast(Any, self.client).web_detection(image=image)
            if response.error.message:
                logger.warning(
                    f"{response.error.message}\nCheck Google Cloud Vision API documentation for more info."
                )
                continue
            per_frame.append(_parse_results(response.web_detection, query))

        if not per_frame:
            return GoogleRisResults(sources=[], query=query, entities={}, best_guess_labels=[])
        # Preserve identity for the single-frame (image) case; merge multiple keyframes.
        return per_frame[0] if len(per_frame) == 1 else _merge_ris_results(per_frame, query)


def _merge_ris_results(results: list[GoogleRisResults], query: Query) -> GoogleRisResults:
    """Merge reverse-image results from several video keyframes into one.

    Deduplicates sources by URL (preserving first-seen order), keeps the highest
    score per entity, and unions best-guess labels — then sorts entities by score.
    """
    sources: list[WebSource] = []
    seen_urls: set[str] = set()
    entities: dict[str, float] = {}
    labels: list[str] = []
    for result in results:
        for source in result.sources:
            url = getattr(source, "url", None) or source.reference
            if url not in seen_urls:
                seen_urls.add(url)
                sources.append(source)
        for name, score in result.entities.items():
            if score > entities.get(name, 0.0):
                entities[name] = score
        for label in result.best_guess_labels:
            if label not in labels:
                labels.append(label)
    entities = dict(sorted(entities.items(), key=lambda kv: kv[1], reverse=True))
    return GoogleRisResults(sources=sources, query=query, entities=entities, best_guess_labels=labels)


google_vision_api = GoogleVisionAPI()


def _parse_results(web_detection: vision.WebDetection, query: Query) -> GoogleRisResults:
    """Parse Google Vision API web detection results into SearchResult instances."""

    # Web Entities
    web_entities = {}
    for entity in web_detection.web_entities:
        if entity.description:
            web_entities[entity.description] = entity.score

    # Best Guess Labels
    best_guess_labels = []
    if web_detection.best_guess_labels:
        for label in web_detection.best_guess_labels:
            if label.label:
                best_guess_labels.append(label.label)

    # Pages with relevant images. Keep Vision's exact-vs-partial distinction: an exact
    # copy on a page is strong provenance evidence, a partial (cropped/edited) match is
    # weaker, and this difference is what downstream judging needs to weigh debunks and
    # authentications correctly.
    web_sources = []
    pages = sorted(
        web_detection.pages_with_matching_images,
        key=lambda p: 0 if getattr(p, "full_matching_images", None) else 1,
    )
    filtered_pages = _filter_unique_stem_pages(pages)
    for page in filtered_pages:
        url = page.url
        title = page.__dict__.get("page_title")
        if getattr(page, "full_matching_images", None):
            match_note = EXACT_MATCH_NOTE
            match_note += _format_matched_image_urls(page.full_matching_images)
        elif getattr(page, "partial_matching_images", None):
            match_note = PARTIAL_MATCH_NOTE
            match_note += _format_matched_image_urls(page.partial_matching_images)
        else:
            match_note = UNKNOWN_MATCH_NOTE
        web_source = WebSource(reference=url, title=title, preview=match_note)
        web_sources.append(web_source)

    return GoogleRisResults(
        sources=web_sources, query=query, entities=web_entities, best_guess_labels=best_guess_labels
    )


def _format_matched_image_urls(matching_images: Sequence, limit: int = 2) -> str:
    """Render Vision's direct URLs of the matched image files (previously dropped).

    These are the strongest referent pointers available: the exact image file
    Google compared against, fetchable for frame-level verification without
    scraping the (possibly bot-walled) page itself.
    """
    urls = [img.url for img in matching_images[:limit] if getattr(img, "url", None)]
    if not urls:
        return ""
    return " Matched image file(s): " + ", ".join(urls)


def _filter_unique_stem_pages(pages: Sequence):
    """
    Filters pages to ensure only one page per website base domain is included
    (e.g., 'facebook.com' regardless of subdomain),
    and limits the total number of pages to the specified limit.

    Args:
        pages (list): List of pages with matching images.

    Returns:
        list: Filtered list of pages.
    """
    unique_domains = set()
    filtered_pages = []

    for page in pages:
        base_domain = get_base_domain(page.url)

        # Check if we already have a page from this base domain
        if base_domain not in unique_domains:
            unique_domains.add(base_domain)
            filtered_pages.append(page)

    return filtered_pages
