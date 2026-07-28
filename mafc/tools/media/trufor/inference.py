"""Runs the vendored TruFor network. Replaces upstream's test_docker/src/trufor_test.py.

No Docker involved: the model runs in-process on the Apple GPU (MPS) when
available, else CUDA, else CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import urllib.request
import zipfile

import numpy as np
import torch
from PIL import Image as PILImage
from torch.nn import functional as F

from mafc.common.logger import logger

from .model.config import default_config

WEIGHTS_URL = "https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip"
WEIGHTS_FILENAME = "trufor.pth.tar"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

# Stage-1 self-attention in the SegFormer backbone scales with H*W on the query
# side (sr_ratio only shrinks the key/value side), so an unresized image well
# beyond typical benchmark resolutions can blow the attention matrix up to a
# multi-hundred-GiB allocation. 2048 matches the largest images already flowing
# through the real eval pipeline without issue.
MAX_LONG_SIDE = 2048


def resolve_device(device: str | torch.device | None = None) -> str:
    """MPS on Apple Silicon, else CUDA, else CPU."""
    if device is not None:
        return str(device)
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def ensure_weights(weights_path: Path | None = None) -> Path:
    """Returns the checkpoint path, downloading it (~280 MB) on first use."""
    if weights_path is None:
        env_path = os.environ.get("trufor_weights_path")
        weights_path = Path(env_path) if env_path else DEFAULT_WEIGHTS_DIR / WEIGHTS_FILENAME
    weights_path = Path(weights_path)
    if weights_path.is_file():
        return weights_path

    weights_path.parent.mkdir(parents=True, exist_ok=True)
    archive = weights_path.parent / "TruFor_weights.zip"
    logger.info(f"[TruFor] Downloading weights to {weights_path} (~280 MB, one time)")
    urllib.request.urlretrieve(WEIGHTS_URL, archive)
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            if member.endswith(WEIGHTS_FILENAME):
                # flatten: the archive stores it under weights/
                with zf.open(member) as src, open(weights_path, "wb") as dst:
                    dst.write(src.read())
                break
        else:
            raise RuntimeError(f"{WEIGHTS_FILENAME} not found in {WEIGHTS_URL}")
    archive.unlink()
    return weights_path


@dataclass
class TruForPrediction:
    """Raw network output for one image."""

    score: float  # detection score in [0, 1]; higher = more likely manipulated
    localization_map: np.ndarray | None = None  # per-pixel manipulation probability
    confidence_map: np.ndarray | None = None
    image_size: tuple[int, int] | None = None  # (height, width)


class TruForModel:
    """Lazily-loaded TruFor network. Keep one instance and reuse it: loading the
    checkpoint takes a few seconds, inference is ~1 s/MPix on MPS."""

    def __init__(self, device: str | torch.device | None = None, weights_path: Path | None = None):
        self.device = resolve_device(device)
        self.weights_path = weights_path
        self._model: torch.nn.Module | None = None

    @property
    def model(self):
        if self._model is None:
            self._load()
        return self._model

    def _load(self) -> None:
        from .model.cmx.builder_np_conf import myEncoderDecoder

        weights = ensure_weights(self.weights_path)
        logger.info(f"[TruFor] Loading model on {self.device}")
        # weights_only=False: the checkpoint stores more than plain tensors, and
        # torch >= 2.6 defaults this to True. Older torch has no such argument.
        try:
            checkpoint = torch.load(weights, map_location=torch.device(self.device), weights_only=False)
        except TypeError:
            checkpoint = torch.load(weights, map_location=torch.device(self.device))
        model = myEncoderDecoder(cfg=default_config())
        model.load_state_dict(checkpoint["state_dict"])
        self._model = model.to(self.device).eval()

    def _empty_cache(self) -> None:
        """Release the allocator pool between images: after a large image the cached
        block is big enough to push the following ones into swap."""
        if self.device.startswith("mps") and hasattr(torch, "mps"):
            torch.mps.empty_cache()
        elif self.device.startswith("cuda"):
            torch.cuda.empty_cache()

    def predict_array(self, rgb: np.ndarray, return_maps: bool = False) -> TruForPrediction:
        """`rgb`: uint8 HxWx3 array."""
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"expected an HxWx3 RGB array, got shape {rgb.shape}")

        orig_h, orig_w = rgb.shape[:2]
        tensor = torch.tensor(rgb.transpose(2, 0, 1), dtype=torch.float)[None] / 256.0
        scale = min(1.0, MAX_LONG_SIDE / max(orig_h, orig_w))
        if scale < 1.0:
            new_h, new_w = round(orig_h * scale), round(orig_w * scale)
            tensor = F.interpolate(tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)

        with torch.no_grad():
            pred, conf, det, _ = self.model(tensor.to(self.device))

            score = float(torch.sigmoid(det).item()) if det is not None else float("nan")
            loc_map = conf_map = None
            if return_maps:
                loc_map = F.softmax(torch.squeeze(pred, 0), dim=0)[1]
                conf_map = torch.sigmoid(torch.squeeze(conf, 0))[0] if conf is not None else None
                if scale < 1.0:
                    loc_map = F.interpolate(
                        loc_map[None, None], size=(orig_h, orig_w), mode="bilinear", align_corners=False
                    )[0, 0]
                    if conf_map is not None:
                        conf_map = F.interpolate(
                            conf_map[None, None], size=(orig_h, orig_w), mode="bilinear", align_corners=False
                        )[0, 0]
                loc_map = loc_map.cpu().numpy()
                conf_map = conf_map.cpu().numpy() if conf_map is not None else None

        self._empty_cache()
        return TruForPrediction(
            score=score,
            localization_map=loc_map,
            confidence_map=conf_map,
            image_size=(rgb.shape[0], rgb.shape[1]),
        )

    def predict_image(self, path: str | Path, return_maps: bool = False) -> TruForPrediction:
        rgb = np.array(PILImage.open(path).convert("RGB"))
        return self.predict_array(rgb, return_maps=return_maps)
