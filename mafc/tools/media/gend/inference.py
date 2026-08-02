"""Runs GenD over an image or video: detect faces, align them, score each one.

GenD (WACV 2026, https://arxiv.org/abs/2508.06248) is a *face* deepfake
detector. It answers one question — "is this face synthetic or swapped?" — and
it can only answer it where a face exists. An image with no detectable face
gets no score, which is a different outcome from a low score and must not be
collapsed into one: absence of a face is absence of evidence.

Weights are pulled from Hugging Face on first use and cached there; nothing is
vendored except the model definition and the face pipeline.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
import statistics

import cv2
import numpy as np
import torch
from PIL import Image

from mafc.common.logger import logger

from .align import DEFAULT_SCALE, align_face
from .modeling_gend import GenD
from .retinaface import RetinaFace, prepare_model

# https://huggingface.co/collections/yermandy/gend
DEFAULT_MODEL = "yermandy/GenD_CLIP_L_14"
AVAILABLE_MODELS = ("yermandy/GenD_CLIP_L_14", "yermandy/GenD_PE_L", "yermandy/GenD_DINOv3_L")

# Softmax index 1 is the synthetic class (matches upstream app/run.py).
FAKE_INDEX = 1

DEFAULT_FACE_THRESHOLD = 0.5  # RetinaFace detection confidence

# Upstream scores only the largest face. That makes the verdict hinge on a
# near-tie whenever several faces are similar in size: on a crowd scene the
# largest face (47px) scored 0.976 while the runner-up (45px) scored 0.193 —
# same image, opposite verdicts. Score several and aggregate instead.
DEFAULT_MAX_FACES = 5
DEFAULT_IMAGE_AGGREGATION = "median"

# Faces smaller than this (aligned crop edge, in pixels) are dropped rather than
# scored. Below roughly this size the crop is mostly upsampling artefacts, and
# GenD trains on far larger crops — a confident number off 14 real pixels is
# noise dressed up as evidence.
DEFAULT_MIN_FACE_PX = 50

_AGGREGATIONS: dict[str, Callable[[list[float]], float]] = {
    "median": statistics.median,
    "mean": statistics.fmean,
    "max": max,
}


@dataclass
class FaceScore:
    """One detected face and its probability of being synthetic."""

    bbox: tuple[int, int, int, int]
    p_fake: float
    area: int
    crop_px: int = 0  # edge length of the aligned crop actually fed to the model


@dataclass
class GenDPrediction:
    """GenD's read on one media file.

    `p_fake` is None when no face was scored — the detector had nothing to look
    at, which is not the same as finding a genuine face. That includes the case
    where faces were present but all too small to judge; `n_faces_skipped`
    records how many, so "too small to tell" stays distinguishable from
    "nobody in frame".
    """

    p_fake: float | None
    n_faces: int
    faces: list[FaceScore]
    n_faces_skipped: int = 0
    n_frames: int | None = None  # videos only
    aggregation: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def has_face(self) -> bool:
        return self.n_faces > 0


class GenDDetector:
    """Loads GenD plus the face detector once and scores media with them."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        face_threshold: float = DEFAULT_FACE_THRESHOLD,
        scale: float = DEFAULT_SCALE,
        max_faces: int | None = DEFAULT_MAX_FACES,
        min_face_px: int = DEFAULT_MIN_FACE_PX,
        image_aggregation: str = DEFAULT_IMAGE_AGGREGATION,
    ):
        if image_aggregation not in _AGGREGATIONS:
            raise ValueError(f"image_aggregation must be one of {sorted(_AGGREGATIONS)}")
        self.model_name = model_name
        self.face_threshold = face_threshold
        self.scale = scale
        self.max_faces = max_faces
        self.min_face_px = min_face_px
        self.image_aggregation = image_aggregation
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._model: GenD | None = None
        self._detector: RetinaFace | None = None

    @property
    def model(self) -> GenD:
        if self._model is None:
            logger.info(f"[GenD] loading {self.model_name} on {self.device}")
            model = GenD.from_pretrained(self.model_name)
            model.eval()
            model.to(self.device)
            self._model = model
        return self._model

    @property
    def detector(self) -> RetinaFace:
        if self._detector is None:
            self._detector = prepare_model(det_thres=self.face_threshold)
        return self._detector

    def score_faces(self, img_bgr: np.ndarray) -> tuple[list[FaceScore], int]:
        """Detect, align, and score the faces in one BGR frame.

        Returns the scored faces and how many were skipped for being too small.
        """
        try:
            xyxy, landmarks = self.detector.detect(img_bgr)
        except Exception as e:
            logger.warning(f"[GenD] face detection failed: {e}")
            return [], 0

        if xyxy is None or len(xyxy) == 0:
            return [], 0

        # Largest face first, so capping at max_faces keeps the most prominent.
        order = sorted(
            range(len(xyxy)),
            key=lambda i: (xyxy[i][2] - xyxy[i][0]) * (xyxy[i][3] - xyxy[i][1]),
            reverse=True,
        )
        if self.max_faces is not None:
            order = order[: max(1, self.max_faces)]

        results: list[FaceScore] = []
        skipped = 0
        for i in order:
            try:
                aligned = align_face(img_bgr, landmarks[i], target_size=None, scale=self.scale)
            except Exception as e:
                logger.warning(f"[GenD] alignment failed for a face: {e}")
                continue

            # Judged on the aligned crop, not the raw box: the crop is what the
            # model actually sees, and alignment can shrink an oblique face.
            crop_px = int(min(aligned.shape[:2]))
            if crop_px < self.min_face_px:
                skipped += 1
                continue

            pil = Image.fromarray(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))
            with torch.no_grad():
                batch = self.model.feature_extractor.preprocess(pil).unsqueeze(0).to(self.device)
                probs = self.model(batch).softmax(dim=-1).cpu().numpy()[0]

            x1, y1, x2, y2 = (int(v) for v in xyxy[i][:4])
            results.append(
                FaceScore(
                    bbox=(x1, y1, x2, y2),
                    p_fake=float(probs[FAKE_INDEX]),
                    area=(x2 - x1) * (y2 - y1),
                    crop_px=crop_px,
                )
            )
        return results, skipped

    def score_image(self, path: str | Path) -> GenDPrediction:
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            raise ValueError(f"could not read image: {path}")

        faces, skipped = self.score_faces(img_bgr)
        if not faces:
            notes = (
                [f"{skipped} face(s) found but all below {self.min_face_px}px; too small to judge"]
                if skipped
                else []
            )
            return GenDPrediction(p_fake=None, n_faces=0, faces=[], n_faces_skipped=skipped, notes=notes)

        # Aggregate across faces rather than trusting the largest one: on crowd
        # scenes the top two are often within a couple of pixels of each other
        # yet score at opposite ends, so "largest" is close to arbitrary.
        combine = _AGGREGATIONS[self.image_aggregation]
        return GenDPrediction(
            p_fake=float(combine([f.p_fake for f in faces])),
            n_faces=len(faces),
            faces=faces,
            n_faces_skipped=skipped,
            aggregation=self.image_aggregation if len(faces) > 1 else None,
        )

    def score_video(
        self, path: str | Path, stride: int = 10, max_frames: int = 32, aggregation: str = "median"
    ) -> GenDPrediction:
        """Score sampled frames and aggregate.

        Median by default: a handful of badly-lit or motion-blurred frames
        should not swing the verdict, which a max would let them do.
        """
        import imageio.v3 as iio

        if aggregation not in _AGGREGATIONS:
            raise ValueError(f"aggregation must be one of {sorted(_AGGREGATIONS)}")

        per_frame: list[float] = []
        all_faces: list[FaceScore] = []
        n_frames = 0
        skipped = 0

        combine_faces = _AGGREGATIONS[self.image_aggregation]
        for idx, frame_rgb in enumerate(iio.imiter(str(path), plugin="pyav")):
            if idx % stride:
                continue
            faces, frame_skipped = self.score_faces(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            n_frames += 1
            skipped += frame_skipped
            if faces:
                # Collapse each frame the same way a still image is collapsed,
                # then aggregate across frames.
                per_frame.append(float(combine_faces([f.p_fake for f in faces])))
                all_faces.extend(faces)
            if len(per_frame) >= max_frames:
                break

        if not per_frame:
            notes = (
                [
                    f"{skipped} face(s) found across frames but all below {self.min_face_px}px; too small to judge"
                ]
                if skipped
                else []
            )
            return GenDPrediction(
                p_fake=None,
                n_faces=0,
                faces=[],
                n_faces_skipped=skipped,
                n_frames=n_frames,
                aggregation=aggregation,
                notes=notes,
            )

        return GenDPrediction(
            p_fake=float(_AGGREGATIONS[aggregation](per_frame)),
            n_faces=len(all_faces),
            faces=all_faces,
            n_faces_skipped=skipped,
            n_frames=n_frames,
            aggregation=aggregation,
        )
