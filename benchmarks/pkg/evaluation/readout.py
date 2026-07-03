"""Readout helpers for temporal logits."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def logits_to_probabilities(logits, *, temperature: float):
    """Return temperature-scaled softmax probabilities."""
    z = torch.as_tensor(logits, dtype=torch.float32)
    return F.softmax(z / float(temperature), dim=-1)


def soft_argmax_predictions(logits, *, temperature: float, sfreq: float = 100.0, win_offset: float = 0.5):
    """Read scalar event times from temporal logits via soft-argmax."""
    z = torch.as_tensor(logits, dtype=torch.float32)
    probabilities = logits_to_probabilities(z, temperature=temperature)
    dt = 1.0 / float(sfreq)
    time_grid = torch.arange(z.shape[-1], dtype=z.dtype, device=z.device)[None, :] * dt
    predictions = (probabilities * time_grid).sum(dim=-1) + float(win_offset)
    return predictions.detach().cpu().numpy().astype(np.float32, copy=False)
