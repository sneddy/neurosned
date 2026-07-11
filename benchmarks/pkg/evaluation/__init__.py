"""Evaluation helpers for benchmark post-processing."""

from benchmarks.pkg.evaluation.bootstrap import subject_bootstrap_nrmse
from benchmarks.pkg.evaluation.calibration import apply_temperature, fit_temperature, make_temperature_grid
from benchmarks.pkg.evaluation.factory import (
    TemperatureReadout,
    build_confidence_interval,
    build_dataset_wrapper,
    build_eval_dataset,
    build_temperature_readout,
)
from benchmarks.pkg.evaluation.metrics import (
    crps_discrete,
    fixed_kernel_event_nll,
    nrmse,
    posterior_distributional_metrics,
    rmse,
)
from benchmarks.pkg.evaluation.readout import logits_to_probabilities, soft_argmax_predictions, softmax_probabilities
from benchmarks.pkg.evaluation.runner import run_holdout_evaluation, run_temperature_calibration
from benchmarks.pkg.evaluation.shifted import run_shifted_eval_from_run_dir, run_shifted_evaluation

__all__ = [
    "TemperatureReadout",
    "apply_temperature",
    "build_confidence_interval",
    "build_dataset_wrapper",
    "build_eval_dataset",
    "build_temperature_readout",
    "fit_temperature",
    "crps_discrete",
    "fixed_kernel_event_nll",
    "logits_to_probabilities",
    "make_temperature_grid",
    "nrmse",
    "posterior_distributional_metrics",
    "rmse",
    "run_holdout_evaluation",
    "run_shifted_eval_from_run_dir",
    "run_shifted_evaluation",
    "run_temperature_calibration",
    "soft_argmax_predictions",
    "softmax_probabilities",
    "subject_bootstrap_nrmse",
]
