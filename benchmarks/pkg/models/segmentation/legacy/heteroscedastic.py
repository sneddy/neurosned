"""Heteroscedastic wrappers for event-time segmentation models."""

from __future__ import annotations

import math
from importlib import import_module
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def _build_backbone(config: dict[str, Any]) -> nn.Module:
    """Build a backbone from the same importable object schema used in YAML."""
    if not isinstance(config, dict):
        raise TypeError("backbone must be a mapping with module_name, class_name and params.")
    module_name = config["module_name"]
    class_name = config["class_name"]
    params = dict(config.get("params", {}))
    cls = getattr(import_module(module_name), class_name)
    return cls(**params)


def _inverse_sigmoid(value: float) -> float:
    """Return logit(value) for a value strictly inside (0, 1)."""
    value = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


class HeteroscedasticEventTimeUNet(nn.Module):
    """Wrap a temporal-logit backbone with a trial-wise EventNLL sigma head.

    The wrapper preserves the standard segmentation forward contract: calling
    the model returns only temporal logits. The most recent per-window
    observation scale is exposed through ``event_observation_sigma()`` for
    likelihood losses that support heteroscedastic kernels.
    """

    def __init__(
        self,
        *,
        backbone: dict[str, Any],
        sigma_min: float = 0.03,
        sigma_max: float = 0.50,
        sigma_init: float = 0.15,
        sigma_hidden: int = 16,
        summary_temperature: float = 0.65,
    ):
        super().__init__()
        if not sigma_min < sigma_init < sigma_max:
            raise ValueError("Expected sigma_min < sigma_init < sigma_max.")
        if sigma_hidden <= 0:
            raise ValueError("sigma_hidden must be positive.")

        self.backbone = _build_backbone(backbone)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.sigma_init = float(sigma_init)
        self.summary_temperature = float(summary_temperature)
        self._last_event_sigma: torch.Tensor | None = None

        self.sigma_head = nn.Sequential(
            nn.Linear(7, int(sigma_hidden)),
            nn.GELU(),
            nn.Linear(int(sigma_hidden), 1),
        )
        self._init_sigma_head()

        for name in ("n_chans", "n_times", "sfreq", "out_channels", "use_norm"):
            if hasattr(self.backbone, name):
                setattr(self, name, getattr(self.backbone, name))

    def _init_sigma_head(self) -> None:
        """Initialize the sigma head to predict sigma_init before training."""
        final = self.sigma_head[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("The final sigma head layer must be nn.Linear.")
        fraction = (self.sigma_init - self.sigma_min) / (self.sigma_max - self.sigma_min)
        nn.init.zeros_(final.weight)
        nn.init.constant_(final.bias, _inverse_sigmoid(fraction))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return temporal logits and cache the per-window observation scale."""
        logits = self.backbone(x)
        self._last_event_sigma = self._predict_sigma(logits)
        return logits

    def _predict_sigma(self, logits: torch.Tensor) -> torch.Tensor:
        """Predict one bounded observation scale per sample from logit shape."""
        if logits.ndim != 3 or logits.shape[1] != 1:
            raise ValueError("HeteroscedasticEventTimeUNet expects logits with shape (B, 1, T).")
        z = logits.squeeze(1)
        p = F.softmax(z / self.summary_temperature, dim=-1)
        entropy = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=-1, keepdim=True)
        entropy = entropy / math.log(float(z.shape[-1]))
        peak = p.amax(dim=-1, keepdim=True)
        z_max = z.amax(dim=-1, keepdim=True)
        z_min = z.amin(dim=-1, keepdim=True)
        summary = torch.cat(
            [
                z.mean(dim=-1, keepdim=True),
                z.std(dim=-1, unbiased=False, keepdim=True),
                z_max,
                z_min,
                z_max - z_min,
                entropy,
                peak,
            ],
            dim=-1,
        )
        raw = self.sigma_head(summary).squeeze(-1)
        fraction = torch.sigmoid(raw)
        return self.sigma_min + (self.sigma_max - self.sigma_min) * fraction

    def event_observation_sigma(self) -> torch.Tensor:
        """Return the per-sample observation scale from the latest forward pass."""
        if self._last_event_sigma is None:
            raise RuntimeError("event_observation_sigma() was called before forward().")
        return self._last_event_sigma

    @torch.no_grad()
    def predict(self, *args, **kwargs):
        """Delegate prediction helpers to the wrapped backbone."""
        return self.backbone.predict(*args, **kwargs)

    @torch.no_grad()
    def predict_mask(self, *args, **kwargs):
        """Delegate probability-mask helpers to the wrapped backbone."""
        return self.backbone.predict_mask(*args, **kwargs)
