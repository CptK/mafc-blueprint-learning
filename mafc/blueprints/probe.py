from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mafc.common.logger import logger

PROBE_FILENAME = "selector_probe.json"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"


@dataclass(frozen=True)
class ProbePrediction:
    """One probe routing decision."""

    blueprint_name: str
    confidence: float
    scores: dict[str, float]


class BlueprintProbe:
    """Multinomial logistic regression routing a claim embedding to a blueprint.

    Blueprints are synthesized from clusters of ground-truth fact-check articles, but
    routing happens on the claim alone. The probe learns that mapping directly from
    claims whose cluster is known — it never compares a claim vector to an article
    vector, only to other claims.

    Coefficients are stored as plain JSON so the artifact travels with the blueprint
    pool and needs no pickle or scikit-learn at inference time.
    """

    def __init__(
        self,
        classes: list[str],
        coefficients: np.ndarray,
        intercepts: np.ndarray,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ):
        """Initialize the probe from fitted parameters."""
        self.classes = classes
        self.coefficients = np.asarray(coefficients, dtype=np.float32)
        self.intercepts = np.asarray(intercepts, dtype=np.float32)
        self.embedding_model = embedding_model

    # -- persistence ----------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> "BlueprintProbe":
        """Load a probe artifact from disk."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            classes=payload["classes"],
            coefficients=np.array(payload["coefficients"], dtype=np.float32),
            intercepts=np.array(payload["intercepts"], dtype=np.float32),
            embedding_model=payload.get("embedding_model", DEFAULT_EMBEDDING_MODEL),
        )

    @classmethod
    def load_from_blueprint_dir(cls, blueprint_dir: str | Path) -> "BlueprintProbe | None":
        """Load the probe stored alongside a blueprint pool, or None when absent."""
        path = Path(blueprint_dir) / PROBE_FILENAME
        if not path.is_file():
            return None
        return cls.load(path)

    def save(self, path: str | Path) -> None:
        """Write the probe artifact to disk."""
        Path(path).write_text(
            json.dumps(
                {
                    "classes": self.classes,
                    "coefficients": self.coefficients.tolist(),
                    "intercepts": self.intercepts.tolist(),
                    "embedding_model": self.embedding_model,
                }
            ),
            encoding="utf-8",
        )

    # -- inference ------------------------------------------------------

    def predict(self, embedding: np.ndarray) -> ProbePrediction:
        """Route one L2-normalized claim embedding to a blueprint."""
        vector = np.asarray(embedding, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector = vector / norm

        logits = self.coefficients @ vector + self.intercepts
        # Binary logistic regression stores a single row; expand to two-class scores.
        if logits.shape[0] == 1:
            logits = np.array([-logits[0], logits[0]], dtype=np.float32)
        shifted = logits - float(np.max(logits))
        exp = np.exp(shifted)
        probabilities = exp / float(np.sum(exp))

        best = int(np.argmax(probabilities))
        return ProbePrediction(
            blueprint_name=self.classes[best],
            confidence=float(probabilities[best]),
            scores={name: float(p) for name, p in zip(self.classes, probabilities)},
        )


def embed_claim(text: str, model: str = DEFAULT_EMBEDDING_MODEL) -> np.ndarray | None:
    """Embed one claim for probe routing, or return None when embedding fails.

    Returning None lets the caller fall back to the LLM tie-break rather than
    failing the run: a routing optimization must never be able to abort a claim.
    """
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("openai_api_key")
    if not api_key or not text.strip():
        return None
    try:
        from openai import OpenAI

        response = OpenAI(api_key=api_key, timeout=60).embeddings.create(
            model=model, input=[text]
        )
        return np.array(response.data[0].embedding, dtype=np.float32)
    except Exception as exc:  # noqa: BLE001 - any embedding failure degrades to tie-break
        logger.warning(f"[BlueprintProbe] Claim embedding failed, falling back: {exc}")
        return None
