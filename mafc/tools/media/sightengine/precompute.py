"""Scores a whole media directory once through Sightengine and writes the
results to a store.

    python -m mafc.tools.media.sightengine.precompute data/veritas_2026_q1/images
    python -m mafc.tools.media.sightengine.precompute data/veritas_2026_q1 --videos

Every check is a paid Sightengine API call, so this exists to pay once per
dataset. Re-running skips everything already in the store, so an interrupted
run just picks up where it left off. Requires SIGHTENGINE_API_USER /
SIGHTENGINE_API_SECRET in the environment.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys
import time

from ezmm.common.items import Image, Video

from mafc.common.logger import logger

from .store import SightengineRecord, SightengineStore, file_sha256
from .tool import SightengineChecker

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def collect_media(root: Path, include_videos: bool) -> list[Path]:
    suffixes = IMAGE_SUFFIXES | (VIDEO_SUFFIXES if include_videos else set())
    if root.is_file():
        return [root] if root.suffix.lower() in suffixes else []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)


def default_store_dir(media_dir: Path) -> Path:
    """A dataset's store lives beside its media, so it travels with the dataset."""
    base = media_dir if media_dir.is_dir() else media_dir.parent
    return (base.parent if base.name in {"images", "videos"} else base) / "sightengine"


def _as_item(path: Path) -> Image | Video:
    if path.suffix.lower() in VIDEO_SUFFIXES:
        return Video(file_path=str(path))
    return Image(file_path=str(path))


def precompute(
    media_dir: Path,
    store_dir: Path | None = None,
    include_videos: bool = False,
    limit: int | None = None,
    checker: SightengineChecker | None = None,
) -> SightengineStore:
    media_dir = Path(media_dir)
    store = SightengineStore(store_dir or default_store_dir(media_dir))
    files = collect_media(media_dir, include_videos)
    if limit:
        files = files[:limit]
    if not files:
        logger.warning(f"[Sightengine] No media found under {media_dir}")
        return store

    # No stores/cache of its own: it reads/writes the target store directly, and
    # every miss goes straight to the API.
    checker = checker or SightengineChecker(stores=[], use_cache=False)

    todo = [(f, file_sha256(f)) for f in files]
    todo = [(f, sha) for f, sha in todo if store.get(sha) is None]
    print(f"{len(files)} files found, {len(files) - len(todo)} already scored, {len(todo)} to do")

    started = time.time()
    failures = 0
    for i, (path, sha) in enumerate(todo, start=1):
        try:
            record: SightengineRecord = checker.compute_record(_as_item(path))
            store.put(sha, record)
        except Exception as e:  # keep going: one bad/unreachable file should not end the run
            failures += 1
            logger.error(f"[Sightengine] Failed on {path}: {e}")
            continue

        if i % 25 == 0 or i == len(todo):
            store.save()  # checkpoint, so an interrupted run keeps its progress
            elapsed = time.time() - started
            rate = elapsed / i
            eta = rate * (len(todo) - i)
            print(f"  {i}/{len(todo)}  {rate:.2f}s/file  eta {eta / 60:.1f} min", flush=True)

    store.save()
    print(f"Done: {len(todo) - failures} scored, {failures} failed -> {store.path}")
    return store


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "media_dir", type=Path, help="directory of images (searched recursively) or a single file"
    )
    parser.add_argument(
        "--store", type=Path, default=None, help="output store dir (default: <dataset>/sightengine)"
    )
    parser.add_argument("--videos", action="store_true", help="also score videos (paid, and slower)")
    parser.add_argument(
        "--no-ai-speech", action="store_true", help="skip the ai_speech check on videos"
    )
    parser.add_argument(
        "--video-aggregation", default="max", choices=["max", "mean", "median"],
        help="how per-frame video scores collapse into one (default: max)",
    )
    parser.add_argument("--limit", type=int, default=None, help="only process the first N files")
    args = parser.parse_args(argv)

    if not args.media_dir.exists():
        parser.error(f"{args.media_dir} does not exist")

    checker = SightengineChecker(
        stores=[],
        use_cache=False,
        check_ai_speech=not args.no_ai_speech,
        video_aggregation=args.video_aggregation,
    )
    api_user, api_secret = checker._resolve_key()
    if not api_user or not api_secret:
        parser.error(
            "Sightengine API credentials not configured "
            "(set SIGHTENGINE_API_USER / SIGHTENGINE_API_SECRET)"
        )

    store = precompute(
        args.media_dir,
        store_dir=args.store,
        include_videos=args.videos,
        limit=args.limit,
        checker=checker,
    )
    print(f"Store now holds {len(store)} records")
    return 0


if __name__ == "__main__":
    sys.exit(main())
