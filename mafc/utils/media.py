from __future__ import annotations

from ezmm import MultimodalSequence
from ezmm.common.items import Image, Item, Video


def deduplicate_media(seq: MultimodalSequence) -> MultimodalSequence:
    """Return a new MultimodalSequence with duplicate media items removed.

    Two items are considered duplicates if they share the same ezmm item id
    (same kind + registry id) or the same SHA-256 file hash. The first
    occurrence of each item is kept; subsequent duplicates are dropped.
    Non-media elements (strings) are always kept as-is.
    """
    seen_item_ids: set[tuple[str, int]] = set()
    seen_hashes: set[str] = set()
    new_data: list[str | Item] = []

    for element in seq.data:
        if not isinstance(element, Item):
            new_data.append(element)
            continue

        item_key = (element.kind, element.id)
        if item_key in seen_item_ids:
            continue

        item_hash = element.sha256
        if item_hash in seen_hashes:
            continue

        seen_item_ids.add(item_key)
        seen_hashes.add(item_hash)
        new_data.append(element)

    result = MultimodalSequence.__new__(MultimodalSequence)
    result.data = new_data
    return result


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
