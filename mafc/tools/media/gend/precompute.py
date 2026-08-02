"""Scores a whole media directory once through GenD and writes the results to a store.

    python -m mafc.tools.media.gend.precompute data/veritas_2026_q1/images
    python -m mafc.tools.media.gend.precompute data/veritas_2026_q1 --videos

GenD runs locally, so this exists to avoid recomputing a slow model rather than
to avoid paying an API. Re-running skips everything already in the store, so an
interrupted run just picks up where it left off.

Files with no detectable face are stored with p_fake=None. That is deliberate:
it records that the file *was* examined and no face was found, which stops the
next run from retrying it, and keeps "no face" distinguishable from "a face
that scored low".
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

from tqdm import tqdm

from mafc.common.logger import logger

from .inference import AVAILABLE_MODELS, DEFAULT_MAX_FACES, DEFAULT_MIN_FACE_PX, DEFAULT_MODEL, GenDDetector
from .store import GenDRecord, GenDStore, file_sha256

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
    return (base.parent if base.name in {"images", "videos"} else base) / "gend"


def precompute(
    media_dir: Path,
    store_dir: Path,
    include_videos: bool = False,
    model_name: str = DEFAULT_MODEL,
    max_faces: int | None = DEFAULT_MAX_FACES,
    min_face_px: int = DEFAULT_MIN_FACE_PX,
    video_stride: int = 10,
    video_max_frames: int = 32,
    video_aggregation: str = "median",
    limit: int | None = None,
) -> int:
    paths = collect_media(media_dir, include_videos)
    if limit:
        paths = paths[:limit]
    if not paths:
        logger.warning(f"[GenD] no media found in {media_dir}")
        return 0

    store = GenDStore(store_dir)
    logger.info(f"[GenD] {len(paths)} files in {media_dir}; store has {len(store)} records")

    detector = GenDDetector(model_name=model_name, max_faces=max_faces, min_face_px=min_face_px)

    # Load both models up front. They are lazy, so otherwise the first iteration
    # absorbs the whole load (and, on a cold cache, the checkpoint download) —
    # the bar would sit at 0 and the ETA would start out badly wrong.
    logger.info("[GenD] loading models…")
    _ = detector.model, detector.detector

    scored = skipped = failed = no_face = 0

    # Only the first pass over a dataset is slow; on a resumed run most files are
    # store hits, so the bar counts every file and reports the breakdown live.
    progress = tqdm(paths, unit="file", desc="GenD", dynamic_ncols=True)
    for path in progress:
        sha = file_sha256(path)
        if store.get(sha) is not None:
            skipped += 1
            continue

        try:
            if path.suffix.lower() in VIDEO_SUFFIXES:
                prediction = detector.score_video(
                    path, stride=video_stride, max_frames=video_max_frames, aggregation=video_aggregation
                )
            else:
                prediction = detector.score_image(path)
        except Exception as e:
            # tqdm.write keeps the bar intact instead of it being redrawn mid-line.
            progress.write(f"[GenD] failed on {path.name}: {e}")
            failed += 1
            continue

        if prediction.p_fake is None:
            no_face += 1

        store.put(
            sha,
            GenDRecord(
                p_fake=prediction.p_fake,
                n_faces=prediction.n_faces,
                n_faces_skipped=prediction.n_faces_skipped,
                face_scores=[f.p_fake for f in prediction.faces],
                model_name=model_name,
                source_name=path.name,
                n_frames=prediction.n_frames,
                aggregation=prediction.aggregation,
            ),
        )
        scored += 1

        postfix = {"scored": scored, "no_face": no_face}
        if skipped:
            postfix["cached"] = skipped
        if failed:
            postfix["failed"] = failed
        progress.set_postfix(postfix, refresh=False)

        # Save periodically so a long run survives an interrupt.
        if scored % 25 == 0:
            store.save()

    progress.close()
    store.save()
    logger.info(
        f"[GenD] done: {scored} scored ({no_face} had no face), {skipped} already present, {failed} failed. "
        f"Store: {store_dir} ({len(store)} records)"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("media_dir", type=Path, help="directory (or single file) of media to score")
    parser.add_argument(
        "--store", type=Path, default=None, help="store directory; defaults to <dataset>/gend"
    )
    parser.add_argument("--videos", action="store_true", help="also score videos (slow)")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=AVAILABLE_MODELS)
    parser.add_argument(
        "--max-faces", type=int, default=DEFAULT_MAX_FACES, help="faces per frame, largest first; 0 = all"
    )
    parser.add_argument(
        "--min-face-px", type=int, default=DEFAULT_MIN_FACE_PX, help="skip aligned crops smaller than this"
    )
    parser.add_argument("--video-stride", type=int, default=10)
    parser.add_argument("--video-max-frames", type=int, default=32)
    parser.add_argument("--video-aggregation", default="median", choices=["median", "mean", "max"])
    parser.add_argument("--limit", type=int, default=None, help="score at most N files (for a trial run)")
    args = parser.parse_args()

    store_dir = args.store or default_store_dir(args.media_dir)
    return precompute(
        media_dir=args.media_dir,
        store_dir=store_dir,
        include_videos=args.videos,
        model_name=args.model,
        max_faces=None if args.max_faces == 0 else args.max_faces,
        min_face_px=args.min_face_px,
        video_stride=args.video_stride,
        video_max_frames=args.video_max_frames,
        video_aggregation=args.video_aggregation,
        limit=args.limit,
    )


if __name__ == "__main__":
    sys.exit(main())
