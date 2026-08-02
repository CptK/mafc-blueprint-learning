"""Similarity-transform face alignment onto GenD's reference landmark layout.

Vendored from the GenD repository (https://github.com/yermandy/deepfake-detection,
MIT, see LICENSE.txt) at detector.py::align_face, unchanged apart from dropping
the mask branch, which only mattered for building training data.

This step is not optional. GenD is trained exclusively on faces aligned this
way, so feeding it a raw image — or even an unaligned face crop — puts it
outside its training distribution and its output stops meaning anything.
"""

from __future__ import annotations

import cv2
import numpy as np

# The five-point layout (eyes, nose, mouth corners) every face is warped onto,
# in fractions of the output square. Changing these silently invalidates the
# model's calibration.
REFERENCE_LANDMARKS = np.array(
    [
        [0.34, 0.46],
        [0.66, 0.46],
        [0.50, 0.64],
        [0.37, 0.82],
        [0.63, 0.82],
    ],
    dtype=np.float32,
)

DEFAULT_SCALE = 1.3  # how much context around the face to keep


def align_face(
    img: np.ndarray,
    landmarks: np.ndarray,
    target_size: tuple[int, int] | None = None,
    scale: float = DEFAULT_SCALE,
) -> np.ndarray:
    """Align a face from 5-point landmarks.

    Args:
        img: BGR image containing the face.
        landmarks: (5, 2) array of facial landmarks.
        target_size: output (width, height); None derives it from the face's
            own size, which is what the upstream app does by default.
        scale: context margin around the face; 1.3 is what GenD trained with.

    Returns:
        The aligned BGR face crop.
    """
    dst = REFERENCE_LANDMARKS.copy()

    if target_size is None:
        # Size the output from the face itself, so small faces are not upsampled
        # into detail the sensor never captured.
        desired_dists = np.linalg.norm(landmarks[:, None, :] - landmarks[None, :, :], axis=-1)
        dst_dists = np.linalg.norm(dst[:, None, :] - dst[None, :, :], axis=-1)
        upper = np.triu_indices(len(dst), k=1)
        approx = np.round(np.mean(desired_dists[upper] / dst_dists[upper]) * scale).astype(int)
        target_size = (int(approx), int(approx))

    dst[:, 0] *= target_size[0]
    dst[:, 1] *= target_size[1]

    margin_rate = scale - 1
    x_margin = target_size[0] * margin_rate / 2.0
    y_margin = target_size[1] * margin_rate / 2.0

    dst[:, 0] += x_margin
    dst[:, 1] += y_margin

    dst[:, 0] *= target_size[0] / (target_size[0] + 2 * x_margin)
    dst[:, 1] *= target_size[1] / (target_size[1] + 2 * y_margin)

    matrix = cv2.estimateAffinePartial2D(landmarks.astype(np.float32), dst, method=cv2.LMEDS)[0]
    if matrix is None:
        raise ValueError("could not estimate an alignment transform for these landmarks")

    return cv2.warpAffine(img, matrix, target_size, flags=cv2.INTER_LINEAR)
