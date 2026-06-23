"""Reduce a high-dim justification embedding to compact scalar features.

The full justification embedding (3072-dim for ``text-embedding-3-large``) is too
wide to drop straight into a tree model on small data. Rather than truncate to a
few *raw* leading dimensions — which carry no special ordering — we expose the
**full** vector as ``just_emb_0..just_emb_{D-1}`` plus its L2 norm, and let the
trainer's pipeline PCA-reduce them *in-fold* (so the projection is never fit on the
evaluation rows). This module stays pure (numpy only).
"""

from __future__ import annotations

import numpy as np


def just_embedding_features(vec) -> dict[str, float]:
    """Expose a justification embedding as ``just_emb_norm`` + all raw components.

    The raw ``just_emb_*`` columns are intended to be PCA-reduced inside the model
    pipeline (refit per fold); they should not be fed to the model un-reduced.
    """
    vec = np.asarray(vec, dtype=np.float32)
    out: dict[str, float] = {"just_emb_norm": float(np.linalg.norm(vec))}
    for i in range(len(vec)):
        out[f"just_emb_{i}"] = float(vec[i])
    return out
