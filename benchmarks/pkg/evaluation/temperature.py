"""Post-hoc temperature calibration for temporal logits."""

from __future__ import annotations

import numpy as np

from benchmarks.pkg.evaluation.metrics import nrmse, rmse
from benchmarks.pkg.evaluation.readout import soft_argmax_predictions


def make_temperature_grid(min_value: float = 0.4, max_value: float = 1.8, step: float = 0.05) -> np.ndarray:
    """Return an inclusive temperature grid."""
    if step <= 0:
        raise ValueError("Temperature grid step must be positive.")
    if max_value < min_value:
        raise ValueError("Temperature grid max must be >= min.")
    n_steps = int(np.floor((max_value - min_value) / step + 0.5))
    values = min_value + np.arange(n_steps + 1, dtype=np.float64) * step
    if values[-1] < max_value - step * 0.5:
        values = np.append(values, max_value)
    values = values[values <= max_value + step * 0.5]
    return np.round(values, 10)


def apply_temperature(logits, temperature: float, *, sfreq: float = 100.0, win_offset: float = 0.5) -> np.ndarray:
    """Return soft-argmax predictions at one temperature."""
    return soft_argmax_predictions(logits, temperature=temperature, sfreq=sfreq, win_offset=win_offset)


def fit_temperature(
    logits,
    targets,
    *,
    min_value: float = 0.4,
    max_value: float = 1.8,
    step: float = 0.05,
    sfreq: float = 100.0,
    win_offset: float = 0.5,
) -> dict:
    """Select the temperature with the lowest validation NRMSE."""
    target = np.asarray(targets, dtype=np.float32)
    grid = make_temperature_grid(min_value, max_value, step)
    rows = []
    best = None
    for temperature in grid:
        predictions = apply_temperature(logits, float(temperature), sfreq=sfreq, win_offset=win_offset)
        row = {
            "temperature": float(temperature),
            "rmse": rmse(predictions, target),
            "nrmse": nrmse(predictions, target),
        }
        rows.append(row)
        if best is None or row["nrmse"] < best["nrmse"]:
            best = row

    return {
        "best_temperature": float(best["temperature"]),
        "best_rmse": float(best["rmse"]),
        "best_nrmse": float(best["nrmse"]),
        "grid": {
            "min": float(min_value),
            "max": float(max_value),
            "step": float(step),
            "values": [float(value) for value in grid],
        },
        "results": rows,
    }
