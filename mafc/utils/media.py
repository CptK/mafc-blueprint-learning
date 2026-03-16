from __future__ import annotations

from ezmm.common.items import Image, Video


def parse_media_relevance(
    entries: list[dict],
    media_items: list[Image | Video],
) -> list[Image | Video]:
    """Filter media_items to those a model marked as relevant.

    Expects entries of the form [{"index": 1, "relevant": true}, ...] where
    index is 1-based and corresponds to the position in media_items.
    """
    relevant: list[Image | Video] = []
    for entry in entries:
        try:
            idx = int(entry["index"]) - 1
            if entry.get("relevant") and 0 <= idx < len(media_items):
                relevant.append(media_items[idx])
        except (KeyError, ValueError, TypeError):
            continue
    return relevant


def build_media_json_instruction(
    media_items: list[Image | Video],
    context: str = "The content",
) -> str:
    """Build the prompt fragment that instructs a model to assess media relevance.

    Returns an empty string when media_items is empty so callers can
    unconditionally concatenate it into a prompt.
    """
    if not media_items:
        return ""
    parts = []
    n_images = sum(1 for m in media_items if isinstance(m, Image))
    n_videos = sum(1 for m in media_items if isinstance(m, Video))
    if n_images:
        parts.append(f"{n_images} image(s)")
    if n_videos:
        parts.append(f"{n_videos} video(s)")
    return (
        f"\n{context} also contains {' and '.join(parts)} (shown below).\n"
        'Include a "media" key in the JSON: a list of objects with "index" (1-based) '
        'and "relevant" (true/false) for each media item.\n'
    )
