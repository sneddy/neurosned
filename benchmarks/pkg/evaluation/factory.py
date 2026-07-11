"""Builders for evaluation datasets, readouts, and summary statistics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from benchmarks.pkg.evaluation.bootstrap import subject_bootstrap_nrmse
from benchmarks.pkg.evaluation.readout import soft_argmax_predictions, softmax_probabilities
from benchmarks.pkg.losses.hazard import hazard_expected_time, hazard_logits_to_pmf


PredictionFn = Callable[[Any, float], np.ndarray]


@dataclass(frozen=True)
class TemperatureReadout:
    """A temperature-scaled scalar readout for temporal logits."""

    name: str
    prediction_fn: PredictionFn
    probability_fn: PredictionFn
    metadata: dict[str, Any]


def build_dataset_wrapper(dataset_config, base_dataset, channels_list, params: dict | None = None):
    """Build a dataset wrapper and pass configured channel selection."""
    wrapper_params = dict(dataset_config.params)
    if params is not None:
        wrapper_params.update(params)
    if channels_list is not None and "use_channels" not in wrapper_params:
        wrapper_params["use_channels"] = channels_list
    dataset_cls = dataset_config.load_class()
    return dataset_cls(base_dataset, **wrapper_params)


def build_eval_dataset(config, base_dataset, channels_list):
    """Build a non-training dataset wrapper."""
    if config.data.valid_dataset is None:
        return base_dataset
    return build_dataset_wrapper(config.data.valid_dataset, base_dataset, channels_list)


def build_confidence_interval(metrics: dict, metadata, evaluation) -> dict | None:
    """Return configured confidence interval metrics for saved predictions."""
    ci_config = evaluation.confidence_interval
    prediction_key = _prediction_key(metrics)
    if not ci_config.enabled or prediction_key is None:
        return None

    if ci_config.method != "subject_bootstrap":
        raise ValueError(f"Unsupported confidence interval method: {ci_config.method!r}")
    return subject_bootstrap_nrmse(
        metrics[prediction_key],
        metadata,
        n_samples=ci_config.n_samples,
        resampling_seed=ci_config.resampling_seed,
    )


def build_temperature_readout(config, *, sfreq: float, win_offset: float) -> TemperatureReadout:
    """Build the configured temperature-calibration readout."""
    params = getattr(config.trainer, "params", {}) or {}
    readout = _normalize_readout(params.get("temperature_readout", params.get("readout_distribution", "softmax")))

    if readout == "softmax":
        return TemperatureReadout(
            name="softmax",
            prediction_fn=lambda logits, temperature: soft_argmax_predictions(
                logits,
                temperature=temperature,
                sfreq=sfreq,
                win_offset=win_offset,
            ),
            probability_fn=lambda logits, temperature: softmax_probabilities(
                logits,
                temperature=temperature,
            ),
            metadata={"readout": "softmax"},
        )

    if readout == "hazard":
        condition_inside = bool(params.get("hazard_condition_inside", True))
        return TemperatureReadout(
            name="hazard",
            prediction_fn=lambda logits, temperature: _hazard_temperature_predictions(
                logits,
                temperature,
                sfreq=sfreq,
                win_offset=win_offset,
                condition_inside=condition_inside,
            ),
            probability_fn=lambda logits, temperature: _hazard_temperature_probabilities(
                logits,
                temperature,
                condition_inside=condition_inside,
            ),
            metadata={
                "readout": "hazard",
                "hazard_condition_inside": condition_inside,
            },
        )

    raise RuntimeError(f"Unsupported temperature readout: {readout!r}")


def _hazard_temperature_predictions(
    logits,
    temperature: float,
    *,
    sfreq: float = 100.0,
    win_offset: float = 0.5,
    condition_inside: bool = True,
) -> np.ndarray:
    """Return expected event times from temperature-scaled hazard logits."""
    z = torch.as_tensor(logits, dtype=torch.float32)
    dt = 1.0 / float(sfreq)
    time_grid = torch.arange(z.shape[-1], dtype=z.dtype, device=z.device)[None, :] * dt
    predictions = hazard_expected_time(
        z,
        time_grid,
        temperature=temperature,
        condition_inside=condition_inside,
    )
    return (predictions + float(win_offset)).detach().cpu().numpy().astype(np.float32, copy=False)


def _hazard_temperature_probabilities(
    logits,
    temperature: float,
    *,
    condition_inside: bool = True,
) -> np.ndarray:
    """Return event-bin probabilities from temperature-scaled hazard logits."""
    z = torch.as_tensor(logits, dtype=torch.float32)
    probabilities = hazard_logits_to_pmf(
        z,
        temperature=temperature,
        condition_inside=condition_inside,
    )
    return probabilities.detach().cpu().numpy().astype(np.float32, copy=False)


def _prediction_key(metrics: dict) -> str | None:
    for key in ("preds_abs", "preds"):
        if key in metrics:
            return key
    return None


def _normalize_readout(readout: str) -> str:
    name = str(readout).lower().replace("-", "_")
    if name in {"softmax", "soft_argmax", "event_softmax"}:
        return "softmax"
    if name in {"hazard", "survival", "event_hazard"}:
        return "hazard"
    raise ValueError(f"Unknown temperature readout: {readout!r}")
