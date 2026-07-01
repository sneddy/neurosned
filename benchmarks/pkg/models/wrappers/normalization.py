"""Normalization wrappers for external models."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from typing import Any

import torch
from torch import nn

from benchmarks.pkg.models.layers import StdPerSample


def build_model(config: Mapping[str, Any]) -> nn.Module:
    """Build a model from a module/class/params mapping."""
    module = import_module(config["module_name"])
    model_cls = getattr(module, config["class_name"])
    return model_cls(**config.get("params", {}))


class WithStdPerSample(nn.Module):
    """Apply StdPerSample before an inner model."""

    def __init__(self, inner: nn.Module | Mapping[str, Any], eps: float = 1e-5):
        super().__init__()
        self.norm = StdPerSample(eps=eps)
        self.inner = inner if isinstance(inner, nn.Module) else build_model(inner)

        n_chans = getattr(self.inner, "n_chans", None)
        if n_chans is None and isinstance(inner, Mapping):
            n_chans = inner.get("params", {}).get("n_chans")
        if n_chans is not None:
            self.n_chans = n_chans

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input and call the inner model."""
        return self.inner(self.norm(x))
