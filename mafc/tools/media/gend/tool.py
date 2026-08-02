"""GenD face-deepfake detection, wired into the mafc tool interface.

Works two ways, transparently:
  * on the fly for any image or video, running the model in-process, and
  * from precomputed stores, so scoring a whole dataset is a one-time cost
    (see precompute.py).

GenD answers a narrower question than TruFor or Sightengine: "is this *face*
synthetic or swapped?" It has nothing to say about spliced backgrounds, forged
screenshots, or generated images without people, and it returns no score at all
where no face is present.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from ezmm import MultimodalSequence
from ezmm.common.items import Image, Video
from ezmm.common.registry import item_registry

from config.globals import gend_cache_dir, gend_stores
from mafc.common.action import Action, MediaRequirement
from mafc.common.logger import logger
from mafc.common.results import Results
from mafc.tools.tool import Tool

from .inference import (
    DEFAULT_IMAGE_AGGREGATION,
    DEFAULT_MAX_FACES,
    DEFAULT_MIN_FACE_PX,
    DEFAULT_MODEL,
    GenDDetector,
    GenDPrediction,
)
from .store import GenDRecord, GenDStore, file_sha256

# The model is trained with a balanced 2-class head, so 0.5 is its natural
# decision point. It has not been calibrated on in-the-wild fact-check imagery.
DEFAULT_THRESHOLD = 0.5


class DetectGenDDeepfake(Action):
    """Detects whether a face in an image or video is a deepfake — swapped,
    reenacted, or synthetically generated. Returns a probability between 0 and 1,
    where higher means more likely synthetic. Only works where a face is
    present, and says nothing about manipulation elsewhere in the image, about
    fully synthetic images without people, or about misleading context."""

    name = "detect_gend_deepfake"
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
        return isinstance(other, DetectGenDDeepfake) and self.media == other.media

    def __hash__(self):
        return hash((self.name, self.media))


@dataclass
class DeepfakeDetectionResults(Results):
    """GenD output for one media item."""

    p_fake: float | None = None  # None when no face was scored
    n_faces: int = 0
    n_faces_skipped: int = 0  # found, but too small to judge
    is_deepfake: bool | None = None  # None when there was nothing to judge
    threshold: float = DEFAULT_THRESHOLD
    face_scores: list[float] = field(default_factory=list)
    from_cache: bool = False
    n_frames: int | None = None
    aggregation: str | None = None
    error: str | None = None

    def __str__(self) -> str:
        if self.error:
            return f"Deepfake detection failed: {self.error}"
        if self.p_fake is None:
            if self.n_faces_skipped:
                return (
                    f"GenD found {self.n_faces_skipped} face(s), but all were too small to judge "
                    "reliably, so no score is reported. This is not evidence either way."
                )
            return (
                "GenD found no face in this media. It detects face deepfakes only, so this is "
                "not evidence either way — the media may still be manipulated in other ways."
            )
        verdict = "likely a deepfake" if self.is_deepfake else "no signs of a face deepfake"
        text = f"GenD deepfake probability: {self.p_fake:.3f} (threshold {self.threshold:.2f}) — {verdict}."
        if self.n_faces > 1:
            spread = (
                f" (range {min(self.face_scores):.3f}–{max(self.face_scores):.3f})"
                if self.face_scores
                else ""
            )
            text += f" Scored {self.n_faces} faces, {self.aggregation} of their scores{spread}."
        else:
            text += " Scored 1 face."
        if self.n_faces_skipped:
            text += f" {self.n_faces_skipped} further face(s) were too small to judge."
        if self.n_frames:
            text += f" Sampled {self.n_frames} video frames; combined across frames by {self.aggregation}."
        return text

    def is_useful(self) -> bool | None:
        # No face means the tool never got to weigh in; that is not a result.
        return self.error is None and self.p_fake is not None


class GenDChecker(Tool[DetectGenDDeepfake, DeepfakeDetectionResults]):
    """Detects face deepfakes with GenD (https://github.com/yermandy/deepfake-detection).

    Scores are looked up in precomputed stores first and only computed when
    missing. Newly computed scores are cached, so repeated runs over the same
    dataset cost nothing.
    """

    name = "gend"
    description = "GenD detects deepfaked faces (swapped, reenacted, or synthetic) in images and videos."
    actions = [DetectGenDDeepfake]

    def __init__(
        self,
        stores: list[str | Path] | None = None,
        cache_dir: str | Path | None = None,
        use_cache: bool = True,
        threshold: float = DEFAULT_THRESHOLD,
        model_name: str = DEFAULT_MODEL,
        n_video_frames: int = 32,
        video_stride: int = 10,
        video_aggregation: str = "median",
        max_faces: int | None = DEFAULT_MAX_FACES,
        min_face_px: int = DEFAULT_MIN_FACE_PX,
        image_aggregation: str = DEFAULT_IMAGE_AGGREGATION,
        **kwargs,
    ):
        """Args:
        stores: read-only score stores to consult before computing (in order).
            Defaults to the `gend_stores` env var.
        cache_dir: writable store for newly computed scores. Defaults to temp/gend.
        use_cache: set False to score without writing anything to disk.
        threshold: probability above which a face counts as a deepfake.
        model_name: which released GenD checkpoint to use.
        n_video_frames: maximum frames scored per video.
        video_stride: sample every Nth frame.
        video_aggregation: "median" (default; robust to a few bad frames),
            "mean", or "max".
        max_faces: how many faces per frame to score, largest first. None scores all.
        min_face_px: aligned crops smaller than this are skipped rather than
            scored; below it there is not enough real detail to judge.
        image_aggregation: how several faces in one image become one score.
        """
        super().__init__(**kwargs)
        store_paths = [Path(p) for p in stores] if stores is not None else list(gend_stores)
        self.stores = [GenDStore(p, writable=False) for p in store_paths]
        self.cache: GenDStore | None = (
            GenDStore(Path(cache_dir) if cache_dir else Path(gend_cache_dir)) if use_cache else None
        )
        self.threshold = threshold
        self.model_name = model_name
        self.n_video_frames = n_video_frames
        self.video_stride = video_stride
        self.video_aggregation = video_aggregation
        self._engine = GenDDetector(
            model_name=model_name,
            device=self.device,
            max_faces=max_faces,
            min_face_px=min_face_px,
            image_aggregation=image_aggregation,
        )

    # --- lookup ---------------------------------------------------------------

    def _lookup(self, sha256: str) -> GenDRecord | None:
        for store in [*self.stores, *(s for s in [self.cache] if s is not None)]:
            record = store.get(sha256)
            if record is not None:
                return record
        return None

    def _result_from_record(self, record: GenDRecord) -> DeepfakeDetectionResults:
        return DeepfakeDetectionResults(
            p_fake=record.p_fake,
            n_faces=record.n_faces,
            n_faces_skipped=record.n_faces_skipped,
            is_deepfake=None if record.p_fake is None else record.p_fake >= self.threshold,
            threshold=self.threshold,
            face_scores=list(record.face_scores),
            from_cache=True,
            n_frames=record.n_frames,
            aggregation=record.aggregation,
        )

    def _result_from_prediction(self, prediction: GenDPrediction) -> DeepfakeDetectionResults:
        return DeepfakeDetectionResults(
            p_fake=prediction.p_fake,
            n_faces=prediction.n_faces,
            n_faces_skipped=prediction.n_faces_skipped,
            is_deepfake=None if prediction.p_fake is None else prediction.p_fake >= self.threshold,
            threshold=self.threshold,
            face_scores=[f.p_fake for f in prediction.faces],
            n_frames=prediction.n_frames,
            aggregation=prediction.aggregation,
        )

    def _store(self, sha: str, name: str, prediction: GenDPrediction) -> None:
        if self.cache is None:
            return
        self.cache.put(
            sha,
            GenDRecord(
                p_fake=prediction.p_fake,
                n_faces=prediction.n_faces,
                n_faces_skipped=prediction.n_faces_skipped,
                face_scores=[f.p_fake for f in prediction.faces],
                model_name=self.model_name,
                source_name=name,
                n_frames=prediction.n_frames,
                aggregation=prediction.aggregation,
            ),
        )
        self.cache.save()

    # --- scoring --------------------------------------------------------------

    def score_image(self, path: str | Path) -> DeepfakeDetectionResults:
        sha = file_sha256(path)
        record = self._lookup(sha)
        if record is not None:
            return self._result_from_record(record)

        prediction = self._engine.score_image(path)
        self._store(sha, Path(path).name, prediction)
        return self._result_from_prediction(prediction)

    def score_video(self, path: str | Path) -> DeepfakeDetectionResults:
        sha = file_sha256(path)
        record = self._lookup(sha)
        if record is not None:
            return self._result_from_record(record)

        prediction = self._engine.score_video(
            path,
            stride=self.video_stride,
            max_frames=self.n_video_frames,
            aggregation=self.video_aggregation,
        )
        self._store(sha, Path(path).name, prediction)
        return self._result_from_prediction(prediction)

    def _perform(self, action: DetectGenDDeepfake) -> DeepfakeDetectionResults:
        if action.media is None:
            return DeepfakeDetectionResults(error="media not found in the item registry")
        try:
            if isinstance(action.media, Video):
                return self.score_video(action.media.file_path)
            return self.score_image(action.media.file_path)
        except Exception as e:
            logger.error(f"[Tool:{self.name}] Failed on {action.media.reference}: {e}")
            return DeepfakeDetectionResults(error=str(e))

    def _summarize(self, result: DeepfakeDetectionResults, **kwargs) -> MultimodalSequence | None:
        if result.error is not None:
            return None
        return MultimodalSequence(str(result))
