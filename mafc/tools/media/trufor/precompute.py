"""Scores a whole media directory once and writes the results to a store.

    python -m mafc.tools.media.trufor.precompute data/veritas_2026_q1/images
    python -m mafc.tools.media.trufor.precompute data/veritas_2026_q1 --videos

Re-running skips everything already in the store, so an interrupted run just
picks up where it left off.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys
import time

import numpy as np

from mafc.common.logger import logger

from .inference import TruForModel
from .store import ScoreRecord, ScoreStore, file_sha256

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def collect_media(root: Path, include_videos: bool) -> list[Path]:
    suffixes = IMAGE_SUFFIXES | (VIDEO_SUFFIXES if include_videos else set())
    if root.is_file():
        return [root] if root.suffix.lower() in suffixes else []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)


def default_store_dir(media_dir: Path) -> Path:
    """A dataset's store lives beside its media, so it travels with the dataset."""
    base = media_dir if media_dir.is_dir() else media_dir.parent
    return (base.parent if base.name in {"images", "videos"} else base) / "trufor"


def precompute(
    media_dir: Path,
    store_dir: Path | None = None,
    include_videos: bool = False,
    keep_maps: bool = False,
    n_video_frames: int = 5,
    device: str | None = None,
    limit: int | None = None,
) -> ScoreStore:
    media_dir = Path(media_dir)
    store = ScoreStore(store_dir or default_store_dir(media_dir))
    files = collect_media(media_dir, include_videos)
    if limit:
        files = files[:limit]
    if not files:
        logger.warning(f"[TruFor] No media found under {media_dir}")
        return store

    engine = TruForModel(device=device)
    todo = [(f, file_sha256(f)) for f in files]
    todo = [(f, sha) for f, sha in todo if store.get(sha) is None]
    print(f"{len(files)} files found, {len(files) - len(todo)} already scored, {len(todo)} to do")

    started = time.time()
    failures = 0
    for i, (path, sha) in enumerate(todo, start=1):
        try:
            if path.suffix.lower() in VIDEO_SUFFIXES:
                record = _score_video(engine, path, n_video_frames)
            else:
                prediction = engine.predict_image(path, return_maps=keep_maps)
                if keep_maps and prediction.localization_map is not None:
                    store.save_maps(sha, prediction.localization_map, prediction.confidence_map)
                record = ScoreRecord(
                    score=prediction.score,
                    source_name=path.name,
                    image_size=list(prediction.image_size) if prediction.image_size else None,
                    has_maps=keep_maps,
                )
            store.put(sha, record)
        except Exception as e:  # keep going: one unreadable file should not end the run
            failures += 1
            logger.error(f"[TruFor] Failed on {path}: {e}")
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


def _score_video(engine: TruForModel, path: Path, n_frames: int) -> ScoreRecord:
    from ezmm.common.items import Video

    frames = Video(file_path=str(path)).sample_frames(n_frames, format="rgb")
    if not frames:
        raise RuntimeError("no frames could be read")
    scores = [engine.predict_array(np.asarray(f)).score for f in frames]
    return ScoreRecord(
        score=float(max(scores)),
        source_name=path.name,
        n_frames=len(scores),
        frame_scores=scores,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "media_dir", type=Path, help="directory of images (searched recursively) or a single file"
    )
    parser.add_argument(
        "--store", type=Path, default=None, help="output store dir (default: <dataset>/trufor)"
    )
    parser.add_argument("--videos", action="store_true", help="also score videos by sampling frames")
    parser.add_argument("--frames", type=int, default=5, help="frames sampled per video (default: 5)")
    parser.add_argument("--maps", action="store_true", help="also store localization/confidence maps (large)")
    parser.add_argument("--device", default=None, help="torch device (default: mps > cuda > cpu)")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N files")
    args = parser.parse_args(argv)

    if not args.media_dir.exists():
        parser.error(f"{args.media_dir} does not exist")

    store = precompute(
        args.media_dir,
        store_dir=args.store,
        include_videos=args.videos,
        keep_maps=args.maps,
        n_video_frames=args.frames,
        device=args.device,
        limit=args.limit,
    )
    print(f"Store now holds {len(store)} records")
    return 0


if __name__ == "__main__":
    sys.exit(main())
