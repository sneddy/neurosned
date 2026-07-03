"""Scalar evaluation metrics."""

from __future__ import annotations

import numpy as np


def rmse(predictions, targets) -> float:
    """Return root mean squared error."""
    prediction = np.asarray(predictions, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    return float(np.sqrt(np.mean((prediction - target) ** 2)))


def nrmse(predictions, targets) -> float:
    """Return RMSE normalized by target standard deviation."""
    target = np.asarray(targets, dtype=np.float64)
    denominator = float(np.std(target, ddof=1)) if len(target) > 1 else 0.0
    value = rmse(predictions, target)
    return value / denominator if denominator else value
