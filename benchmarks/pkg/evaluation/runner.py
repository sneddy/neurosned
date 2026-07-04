"""Evaluation orchestration shared by benchmark CLIs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from benchmarks.pkg.artefacts_manager import ArtefactsManager
from benchmarks.pkg.evaluation.calibration import apply_temperature, fit_temperature
from benchmarks.pkg.evaluation.factory import build_confidence_interval, build_eval_dataset, build_temperature_readout
from benchmarks.pkg.evaluation.metrics import nrmse, posterior_distributional_metrics, rmse
from benchmarks.pkg.runtime import PROJECT_ROOT, path_text


DEFAULT_DISTRIBUTIONAL_EVENT_NLL_SIGMA = 0.15


def run_holdout_evaluation(
    *,
    config,
    trainer,
    model,
    channels_list,
    valid_metadata,
    output_checkpoint_path: Path,
    default_rmse_by_split: dict[str, float],
    artefacts: ArtefactsManager,
    device,
    project_root: Path = PROJECT_ROOT,
    verbose: bool = True,
):
    """Run optional post-training holdout evaluation."""
    evaluation = config.evaluation
    if not evaluation.holdout_eval:
        return None

    split = evaluation.holdout_split
    if verbose:
        print(f"Holdout evaluation: split={split}")
    holdout_dataset = config.build_dataset(split, project_root)
    holdout_metadata = holdout_dataset.get_metadata()
    holdout_std = float(holdout_metadata["target"].std())
    default_rmse_by_split[split] = holdout_std
    if hasattr(trainer, "default_rmse_by_split"):
        trainer.default_rmse_by_split[split] = holdout_std

    holdout_dataset_for_loader = build_eval_dataset(config, holdout_dataset, channels_list)
    holdout_loader = DataLoader(holdout_dataset_for_loader, **config.loaders.valid.to_kwargs())
    if verbose:
        print(
            f"Holdout loader: rows={len(holdout_dataset_for_loader):,}, "
            f"batches={len(holdout_loader):,}, denominator={holdout_std:.4f}"
        )

    checkpoint_loaded = False
    if output_checkpoint_path.exists():
        model.load_state_dict(torch.load(output_checkpoint_path, map_location=device))
        checkpoint_loaded = True
        if verbose:
            print(f"Loaded best checkpoint for holdout: {path_text(output_checkpoint_path)}")
    else:
        if verbose:
            print("Best checkpoint is missing; evaluating current model state.")

    metrics = trainer.run_eval_epoch(holdout_loader, split=split, epoch=0)
    add_posterior_distributional_metrics(
        config=config,
        trainer=trainer,
        metrics=metrics,
        targets=holdout_metadata["target"].to_numpy(),
    )
    artefacts.save_holdout_evaluation(
        split=split,
        metrics=metrics,
        metadata=holdout_metadata,
        evaluation=evaluation,
        checkpoint_loaded=checkpoint_loaded,
        confidence_interval=build_confidence_interval(metrics, holdout_metadata, evaluation),
    )
    run_temperature_calibration(
        config=config,
        trainer=trainer,
        holdout_metrics=metrics,
        valid_metadata=valid_metadata,
        holdout_metadata=holdout_metadata,
        split=split,
        artefacts=artefacts,
        checkpoint_loaded=checkpoint_loaded,
        verbose=verbose,
    )
    monitor_value = metrics.get(config.trainer.monitor)
    if verbose and monitor_value is not None:
        print(f"Holdout {config.trainer.monitor}: {monitor_value:.6f}")
        print_distributional_metric_line("Holdout posterior scores", metrics)
    return metrics


def run_temperature_calibration(
    *,
    config,
    trainer,
    holdout_metrics: dict,
    valid_metadata,
    holdout_metadata,
    split: str,
    artefacts: ArtefactsManager,
    checkpoint_loaded: bool,
    verbose: bool = True,
) -> dict | None:
    """Fit validation temperature and apply it to holdout logits."""
    temperature_config = config.calibration.temperature
    if not temperature_config.enabled:
        return None
    if config.task != "segmentation":
        raise RuntimeError("Temperature calibration requires segmentation logits.")

    best_valid_metrics = trainer.best_valid_metrics or {}
    valid_logits = best_valid_metrics.get("logits")
    holdout_logits = holdout_metrics.get("logits")
    if valid_logits is None:
        raise RuntimeError("Temperature calibration is enabled, but best validation logits are missing.")
    if holdout_logits is None:
        raise RuntimeError("Temperature calibration is enabled, but holdout logits are missing.")

    sfreq = float(getattr(trainer.model, "sfreq", 100.0))
    win_offset = float(getattr(trainer, "win_offset", 0.5))
    valid_targets = valid_metadata["target"].to_numpy()
    holdout_targets = holdout_metadata["target"].to_numpy()
    readout = build_temperature_readout(config, sfreq=sfreq, win_offset=win_offset)

    calibration = fit_temperature(
        valid_logits,
        valid_targets,
        min_value=temperature_config.min,
        max_value=temperature_config.max,
        step=temperature_config.step,
        prediction_fn=readout.prediction_fn,
    )
    calibration.update(
        {
            "selection_split": "valid",
            "holdout_split": split,
            "metric": "nrmse",
            "sfreq": sfreq,
            "win_offset": win_offset,
            **readout.metadata,
        }
    )
    calibration_path = artefacts.save_temperature_calibration(calibration)

    temperature = calibration["best_temperature"]
    predictions = apply_temperature(
        holdout_logits,
        temperature,
        prediction_fn=readout.prediction_fn,
    )
    calibrated_metrics = {
        "rmse": rmse(predictions, holdout_targets),
        "nrmse": nrmse(predictions, holdout_targets),
        "temperature": temperature,
        "preds_abs": predictions,
    }
    add_posterior_distributional_metrics(
        config=config,
        trainer=trainer,
        metrics=calibrated_metrics,
        targets=holdout_targets,
        logits=holdout_logits,
        temperature=temperature,
        sfreq=sfreq,
        win_offset=win_offset,
        readout=readout,
    )
    artefacts.save_holdout_evaluation(
        split=f"{split}_tau",
        metrics=calibrated_metrics,
        metadata=holdout_metadata,
        evaluation=config.evaluation,
        checkpoint_loaded=checkpoint_loaded,
        confidence_interval=build_confidence_interval(calibrated_metrics, holdout_metadata, config.evaluation),
    )
    if verbose:
        print(
            f"Temperature calibration: tau={temperature:.4f}, "
            f"readout={readout.name}, valid_nrmse={calibration['best_nrmse']:.6f}, "
            f"{split}_tau_nrmse={calibrated_metrics['nrmse']:.6f}"
        )
        print_distributional_metric_line(f"{split}_tau posterior scores", calibrated_metrics)
        print(f"Saved temperature calibration: {path_text(calibration_path)}")
    return calibrated_metrics


def print_distributional_metric_line(label: str, metrics: dict) -> None:
    """Print posterior distributional scores when present."""
    crps = metrics.get("posterior_crps")
    event_nll = metrics.get("posterior_fixed_kernel_event_nll")
    if crps is None or event_nll is None:
        return
    print(f"{label}: CRPS={float(crps) * 1000.0:.1f} ms, fixed-kernel EventNLL={float(event_nll):.4f}")


def add_posterior_distributional_metrics(
    *,
    config,
    trainer,
    metrics: dict,
    targets,
    logits=None,
    temperature: float | None = None,
    sfreq: float | None = None,
    win_offset: float | None = None,
    readout=None,
) -> None:
    """Attach CRPS and fixed-kernel EventNLL when temporal logits are available."""
    if config.task != "segmentation":
        return
    logits = metrics.get("logits") if logits is None else logits
    if logits is None:
        return

    sfreq = float(sfreq if sfreq is not None else getattr(trainer.model, "sfreq", 100.0))
    win_offset = float(win_offset if win_offset is not None else getattr(trainer, "win_offset", 0.5))
    temperature = float(temperature if temperature is not None else getattr(trainer, "eval_temperature", 1.0))
    readout = readout or build_temperature_readout(config, sfreq=sfreq, win_offset=win_offset)
    probabilities = readout.probability_fn(logits, temperature)
    grid = np.arange(np.asarray(logits).shape[-1], dtype=np.float64) / sfreq + win_offset
    sigma = float(getattr(trainer, "posterior_score_sigma", DEFAULT_DISTRIBUTIONAL_EVENT_NLL_SIGMA))

    metrics.update(
        posterior_distributional_metrics(
            probabilities,
            grid,
            targets,
            event_nll_sigma=sigma,
        )
    )
