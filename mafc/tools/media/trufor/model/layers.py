"""Drop-in replacements for the three `timm.models.layers` helpers the vendored
TruFor network uses, so the tool does not need `timm` as a dependency.

`DropPath` is inactive at eval time and `trunc_normal_` only affects
initialization (every weight is overwritten by the checkpoint), so these only
need to match timm structurally, not bit-exactly.
"""

from itertools import repeat
import collections.abc

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_  # noqa: F401  (re-exported)


def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        return tuple(x)
    return tuple(repeat(x, 2))


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Stochastic depth, per sample (when applied in the main path of residual blocks)."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # broadcast over all but the batch dim
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float | None = None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob or 0.0, self.training)

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob}"
