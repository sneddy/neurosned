"""Regression metrics used by stacking experiments."""

from __future__ import annotations

import numpy as np


def rmse(y_true, y_pred) -> float:
    """Return root mean squared error."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_true, y_pred) -> float:
    """Return mean absolute error."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_pred - y_true)))


def nrmse(y_true, y_pred) -> float:
    """Return RMSE divided by target standard deviation."""
    y_true = np.asarray(y_true, dtype=np.float64)
    denom = float(np.std(y_true))
    if denom <= 0:
        raise ValueError("Cannot compute nRMSE when target standard deviation is zero.")
    return rmse(y_true, y_pred) / denom


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Return the scalar regression metrics used in stacking tables."""
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "nrmse": nrmse(y_true, y_pred),
    }
