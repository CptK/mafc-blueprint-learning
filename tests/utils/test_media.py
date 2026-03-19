from __future__ import annotations

import shutil
from pathlib import Path
from typing import cast

from ezmm import MultimodalSequence
from ezmm.common.items import Image, Video

from mafc.utils.media import deduplicate_media, extract_media_items

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"


def _image(path: Path) -> Image:
    return cast(Image, Image(file_path=path))


def test_deduplicate_media_empty_sequence() -> None:
    seq = MultimodalSequence("just text")
    result = deduplicate_media(seq)
    assert list(result.data) == list(seq.data)


def test_deduplicate_media_preserves_strings() -> None:
    seq = MultimodalSequence("hello world")
    result = deduplicate_media(seq)
    assert result.images == []
    assert result.videos == []
    assert "hello" in str(result)


def test_deduplicate_media_same_item_id_deduplicated(tmp_path: Path) -> None:
    img_path = tmp_path / "img.jpg"
    shutil.copy(ASSETS_DIR / "Greece.jpeg", img_path)
    img = _image(img_path)

    # Same item object referenced twice in one sequence
    seq = MultimodalSequence("before", img, "middle", img, "after")
    result = deduplicate_media(seq)

    assert result.images == [img]


def test_deduplicate_media_different_ids_same_hash_deduplicated(tmp_path: Path) -> None:
    # Two different file paths with identical content → different ids, same sha256
    path_a = tmp_path / "a.jpg"
    path_b = tmp_path / "b.jpg"
    shutil.copy(ASSETS_DIR / "Greece.jpeg", path_a)
    shutil.copy(ASSETS_DIR / "Greece.jpeg", path_b)

    img_a = _image(path_a)
    img_b = _image(path_b)
    assert img_a.id != img_b.id
    assert img_a.sha256 == img_b.sha256

    seq = MultimodalSequence("text", img_a, img_b)
    result = deduplicate_media(seq)

    assert len(result.images) == 1
    assert result.images[0] is img_a


def test_deduplicate_media_different_images_both_kept(tmp_path: Path) -> None:
    path_a = tmp_path / "a.jpg"
    path_b = tmp_path / "b.jpg"
    shutil.copy(ASSETS_DIR / "Greece.jpeg", path_a)
    shutil.copy(ASSETS_DIR / "Paris.avif", path_b)

    img_a = _image(path_a)
    img_b = _image(path_b)
    assert img_a.sha256 != img_b.sha256

    seq = MultimodalSequence("text", img_a, img_b)
    result = deduplicate_media(seq)

    assert len(result.images) == 2


def test_deduplicate_media_preserves_surrounding_text(tmp_path: Path) -> None:
    img_path = tmp_path / "img.jpg"
    shutil.copy(ASSETS_DIR / "Greece.jpeg", img_path)
    img = _image(img_path)

    seq = MultimodalSequence("before", img, img, "after")
    result = deduplicate_media(seq)

    assert "before" in str(result)
    assert "after" in str(result)
    assert len(result.images) == 1


def test_deduplicate_media_via_reference_resolution(tmp_path: Path) -> None:
    # Simulate the judge path: str(takeaways) embeds <<image:N>> references,
    # which Prompt(text=...) resolves back to Item objects — duplicates must be removed.
    img_path = tmp_path / "img.jpg"
    shutil.copy(ASSETS_DIR / "Greece.jpeg", img_path)
    img = _image(img_path)

    # Build a string with the same reference twice (as happens when two evidences
    # share the same image in their takeaways text)
    ref = img.reference  # e.g. "<<image:5>>"
    text_with_duplicate_refs = f"Evidence 1: {ref} Evidence 2: {ref}"
    seq = MultimodalSequence(text_with_duplicate_refs)

    # The sequence already resolved both refs to the same item object
    assert seq.images.count(img) == 2

    result = deduplicate_media(seq)
    assert len(result.images) == 1


# --- extract_media_items ---


def test_extract_media_items_empty_sequence() -> None:
    seq = MultimodalSequence("just text")
    assert extract_media_items(seq) == []


def test_extract_media_items_images_only(tmp_path: Path) -> None:
    img_path = tmp_path / "img.jpg"
    shutil.copy(ASSETS_DIR / "Greece.jpeg", img_path)
    img = Image(file_path=img_path)

    seq = MultimodalSequence("text", img)
    result = extract_media_items(seq)

    assert result == [img]


def test_extract_media_items_videos_only() -> None:
    video = Video(binary_data=b"fake-video")

    seq = MultimodalSequence("text", video)
    result = extract_media_items(seq)

    assert result == [video]


def test_extract_media_items_images_and_videos(tmp_path: Path) -> None:
    img_path = tmp_path / "img.jpg"
    shutil.copy(ASSETS_DIR / "Greece.jpeg", img_path)
    img = Image(file_path=img_path)
    video = Video(binary_data=b"fake-video")

    seq = MultimodalSequence("text", img, video)
    result = extract_media_items(seq)

    assert img in result
    assert video in result
    assert len(result) == 2
