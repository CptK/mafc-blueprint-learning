"""TruFor image manipulation detection, wired into the mafc tool interface.

Works two ways, transparently:
  * on the fly for any image or video, running the model in-process, and
  * from precomputed stores, so scoring a whole dataset is a one-time cost
    (see precompute.py).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
import statistics
from typing import cast

import numpy as np
from ezmm import MultimodalSequence
from ezmm.common.items import Image, Video
from ezmm.common.registry import item_registry

from config.globals import trufor_cache_dir, trufor_stores
from mafc.common.action import Action, MediaRequirement
from mafc.common.logger import logger
from mafc.common.results import Results
from mafc.tools.tool import Tool

from .inference import TruForModel, TruForPrediction
from .store import ScoreRecord, ScoreStore, file_sha256

# The paper thresholds the detection score at 0.5.
DEFAULT_THRESHOLD = 0.5

# How per-frame video scores collapse into one score for the whole video.
_AGGREGATIONS: dict[str, Callable[[list[float]], float]] = {
    "max": max,
    "mean": lambda s: sum(s) / len(s),
    "median": lambda s: statistics.median(s),
}


class DetectTruForManipulation(Action):
    """Detects whether an image (or video frame) has been digitally manipulated,
    e.g. by splicing, copy-move or inpainting. Returns a manipulation score
    between 0 and 1, where higher means more likely manipulated. Does not detect
    fully AI-generated images, and says nothing about whether the content is
    shown in a misleading context."""

    name = "detect_trufor_manipulation"
    media_requirement = MediaRequirement.IMAGE_OR_VIDEO

    def __init__(self, media: str):
        """Args:
        media: reference to the image or video to analyze (must be in the item registry)
        """
        self._save_parameters(locals())
        item = item_registry.get(reference=media)
        if item is None:
            logger.error(f"[Action:{self.name}] Media not found in registry for reference: {media}")
            self.media = None
        elif not isinstance(item, Image | Video):
            logger.error(
                f"[Action:{self.name}] Item found for reference {media} is not an Image/Video: "
                f"{type(item).__name__}"
            )
            self.media = None
        else:
            self.media = cast(Image | Video, item)

    def __eq__(self, other):
        return isinstance(other, DetectTruForManipulation) and self.media == other.media

    def __hash__(self):
        return hash((self.name, self.media))


@dataclass
class ManipulationDetectionResults(Results):
    """TruFor output for one media item."""

    score: float  # in [0, 1]; higher = more likely manipulated
    is_manipulated: bool
    threshold: float = DEFAULT_THRESHOLD
    from_cache: bool = False
    n_frames: int | None = None  # videos: number of frames scored
    frame_scores: list[float] = field(default_factory=list)
    aggregation: str | None = None  # videos: how frame scores were combined
    localization_map: np.ndarray | None = None
    confidence_map: np.ndarray | None = None
    error: str | None = None

    def __str__(self) -> str:
        if self.error:
            return f"Manipulation detection failed: {self.error}"
        verdict = "signs of manipulation" if self.is_manipulated else "no signs of manipulation"
        text = f"TruFor manipulation score: {self.score:.3f} (threshold {self.threshold:.2f}) — {verdict}."
        if self.n_frames:
            text += (
                f" Scored {self.n_frames} sampled video frames; "
                f"the score is the {self.aggregation or 'max'} over frames."
            )
        return text

    def is_useful(self) -> bool | None:
        return self.error is None


class TruFor(Tool[DetectTruForManipulation, ManipulationDetectionResults]):
    """Detects image manipulation with TruFor (https://github.com/grip-unina/TruFor).

    Scores are looked up in precomputed stores first and only computed when
    missing. Newly computed scores are cached, so repeated runs over the same
    dataset cost nothing.
    """

    name = "trufor"
    description = "TruFor detects digitally manipulated (spliced, copy-moved, inpainted) images."
    actions = [DetectTruForManipulation]

    def __init__(
        self,
        stores: list[str | Path] | None = None,
        cache_dir: str | Path | None = None,
        use_cache: bool = True,
        threshold: float = DEFAULT_THRESHOLD,
        n_video_frames: int = 5,
        video_aggregation: str = "max",
        keep_maps: bool = False,
        weights_path: str | Path | None = None,
        **kwargs,
    ):
        """Args:
        stores: read-only score stores to consult before computing (in order).
            Defaults to the `trufor_stores` env var.
        cache_dir: writable store for newly computed scores. Defaults to temp/trufor.
        use_cache: set False to score without writing anything to disk.
        threshold: score above which an image counts as manipulated.
        n_video_frames: frames sampled per video.
        video_aggregation: how frame scores become one video score — "max"
            (most sensitive, but biased upward the more frames you sample),
            "median" or "mean". Per-frame scores are always stored, so this can
            be changed later without rescoring.
        keep_maps: also return (and cache) the localization/confidence maps.
        """
        if video_aggregation not in _AGGREGATIONS:
            raise ValueError(f"video_aggregation must be one of {sorted(_AGGREGATIONS)}")
        super().__init__(**kwargs)
        store_paths = [Path(p) for p in stores] if stores is not None else list(trufor_stores)
        self.stores = [ScoreStore(p, writable=False) for p in store_paths]
        self.cache: ScoreStore | None = (
            ScoreStore(Path(cache_dir) if cache_dir else Path(trufor_cache_dir)) if use_cache else None
        )
        self.threshold = threshold
        self.n_video_frames = n_video_frames
        self.video_aggregation = video_aggregation
        self.keep_maps = keep_maps
        self._engine = TruForModel(
            device=self.device, weights_path=Path(weights_path) if weights_path else None
        )

    # --- lookup ---------------------------------------------------------------

    def _lookup(self, sha256: str) -> tuple[ScoreRecord, ScoreStore] | None:
        for store in [*self.stores, *(s for s in [self.cache] if s is not None)]:
            record = store.get(sha256)
            if record is not None:
                return record, store
        return None

    def _result_from_record(
        self, record: ScoreRecord, store: ScoreStore, sha: str
    ) -> ManipulationDetectionResults:
        maps = store.load_maps(sha) if (self.keep_maps and record.has_maps) else None
        # Re-aggregate stored frame scores, so changing video_aggregation does not
        # require rescoring the videos.
        score = self._aggregate(record.frame_scores) if record.frame_scores else record.score
        return ManipulationDetectionResults(
            score=score,
            is_manipulated=score >= self.threshold,
            threshold=self.threshold,
            from_cache=True,
            n_frames=record.n_frames,
            frame_scores=list(record.frame_scores),
            aggregation=self.video_aggregation if record.frame_scores else None,
            localization_map=maps.get("map") if maps else None,
            confidence_map=maps.get("conf") if maps else None,
        )

    # --- scoring --------------------------------------------------------------

    def score_image(self, path: str | Path) -> ManipulationDetectionResults:
        """Scores a single image file, using the stores when possible."""
        sha = file_sha256(path)
        hit = self._lookup(sha)
        if hit is not None:
            record, store = hit
            return self._result_from_record(record, store, sha)

        prediction = self._engine.predict_image(path, return_maps=self.keep_maps)
        self._store(sha, Path(path).name, prediction)
        return ManipulationDetectionResults(
            score=prediction.score,
            is_manipulated=prediction.score >= self.threshold,
            threshold=self.threshold,
            localization_map=prediction.localization_map,
            confidence_map=prediction.confidence_map,
        )

    def score_video(self, video: Video) -> ManipulationDetectionResults:
        """Samples frames and returns the maximum frame score. Frames are taken as
        raw RGB arrays, so they are not re-compressed before scoring."""
        frames = video.sample_frames(self.n_video_frames, format="rgb")
        if not frames:
            return ManipulationDetectionResults(
                score=float("nan"), is_manipulated=False, error="no frames could be read from the video"
            )
        frame_scores = [self._engine.predict_array(np.asarray(f)).score for f in frames]
        score = self._aggregate(frame_scores)
        return ManipulationDetectionResults(
            score=score,
            is_manipulated=score >= self.threshold,
            threshold=self.threshold,
            n_frames=len(frame_scores),
            frame_scores=frame_scores,
            aggregation=self.video_aggregation,
        )

    def _aggregate(self, frame_scores: list[float]) -> float:
        return float(_AGGREGATIONS[self.video_aggregation](frame_scores))

    def _store(self, sha: str, name: str, prediction: TruForPrediction) -> None:
        if self.cache is None:
            return
        loc_map = prediction.localization_map
        has_maps = self.keep_maps and loc_map is not None
        if has_maps and loc_map is not None:
            self.cache.save_maps(sha, loc_map, prediction.confidence_map)
        self.cache.put(
            sha,
            ScoreRecord(
                score=prediction.score,
                source_name=name,
                image_size=list(prediction.image_size) if prediction.image_size else None,
                has_maps=has_maps,
            ),
        )
        self.cache.save()

    # --- Tool interface -------------------------------------------------------

    def _perform(self, action: DetectTruForManipulation) -> ManipulationDetectionResults:
        if action.media is None:
            return ManipulationDetectionResults(
                score=float("nan"), is_manipulated=False, error="media not found in the item registry"
            )
        try:
            if isinstance(action.media, Video):
                sha = file_sha256(action.media.file_path)
                hit = self._lookup(sha)
                if hit is not None:
                    record, store = hit
                    return self._result_from_record(record, store, sha)
                result = self.score_video(cast(Video, action.media))
                if self.cache is not None and not np.isnan(result.score):
                    self.cache.put(
                        sha,
                        ScoreRecord(
                            score=result.score,
                            source_name=Path(action.media.file_path).name,
                            n_frames=result.n_frames,
                            frame_scores=result.frame_scores,
                        ),
                    )
                    self.cache.save()
                return result
            return self.score_image(action.media.file_path)
        except Exception as e:
            logger.error(f"[Tool:{self.name}] Failed to score {action.media.reference}: {e}")
            return ManipulationDetectionResults(score=float("nan"), is_manipulated=False, error=str(e))

    def _summarize(self, result: ManipulationDetectionResults, **kwargs) -> MultimodalSequence | None:
        if result.error is not None:
            return None
        if result.is_manipulated:
            reading = (
                f"TruFor rates this media as **likely manipulated** (score {result.score:.2f} of 1.00, "
                f"threshold {result.threshold:.2f}). This points to local edits such as splicing, "
                f"copy-move or inpainting."
            )
        else:
            reading = (
                f"TruFor finds **no evidence of manipulation** (score {result.score:.2f} of 1.00, "
                f"threshold {result.threshold:.2f})."
            )
        caveat = (
            " Note that TruFor detects local edits only: it does not flag fully AI-generated images, "
            "and heavy re-compression or resizing (e.g. by social platforms) can both hide real edits "
            "and cause false alarms. Treat the score as one signal, not proof."
        )
        return MultimodalSequence(reading + caveat)
