"""Evaluate trained RT models on temporally shifted crops."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

os.environ["MNE_DONTWRITE_HOME"] = "true"
os.environ["NUMBA_DISABLE_JIT"] = "1"
os.environ["MPLCONFIGDIR"] = "/tmp/neurosned-matplotlib"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from benchmarks.data.regression import ShiftedFixedWindowDataset
from benchmarks.pkg.config import ExperimentConfig, resolve_path
from benchmarks.pkg.evaluation.metrics import nrmse, rmse
from benchmarks.pkg.runtime import choose_device, path_text


DEFAULT_STARTS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
DEFAULT_DATASET = Path("data/new_validation/r11_test_5sec.pkl")


def build_parser() -> argparse.ArgumentParser:
    """Build the shifted-crop evaluation CLI."""
    parser = argparse.ArgumentParser(description="Evaluate a trained regression or segmentation run on shifted 2 s crops.")
    parser.add_argument("run_dir", type=Path, help="Existing run directory with config.yaml and best_model.pth.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Long-window dataset pickle used for shifted crops. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--starts",
        type=float,
        nargs="+",
        default=list(DEFAULT_STARTS),
        help="Crop starts in seconds relative to stimulus onset. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--reference-start",
        type=float,
        default=0.5,
        help="Canonical crop start used by the trained regression protocol. Defaults to %(default)s.",
    )
    parser.add_argument("--crop-sec", type=float, default=None, help="Crop length in seconds. Defaults to model n_times/sfreq.")
    parser.add_argument("--sfreq", type=float, default=None, help="Sampling rate. Defaults to model/config sfreq, then 100.")
    parser.add_argument("--batch-size", type=int, default=None, help="Evaluation batch size. Defaults to config valid batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers. Defaults to config valid workers.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto", help="Torch device selection.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Defaults to <run_dir>/shifted_eval.")
    parser.add_argument(
        "--segmentation-temperature",
        type=float,
        default=None,
        help="Softmax temperature for segmentation soft-argmax. Defaults to eval_temperature/temperature from config.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=1000, help="Subject-bootstrap samples for CI tables.")
    parser.add_argument("--bootstrap-seed", type=int, default=2025, help="Subject-bootstrap random seed.")
    return parser


def load_run_snapshot(run_dir: Path) -> tuple[dict, ExperimentConfig]:
    """Load the run snapshot config from an existing run directory."""
    snapshot_path = run_dir / "config.yaml"
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Missing run snapshot: {snapshot_path}")
    with snapshot_path.open("r", encoding="utf-8") as f:
        snapshot = yaml.safe_load(f)
    if "config" not in snapshot:
        raise ValueError(f"Run snapshot does not contain a 'config' section: {snapshot_path}")
    return snapshot, ExperimentConfig.model_validate(snapshot["config"])


def load_pickle_dataset(path: Path):
    """Load a prepared benchmark pickle dataset."""
    with path.open("rb") as f:
        return pickle.load(f)


def model_sfreq(config: ExperimentConfig, model, override: float | None) -> float:
    """Return the sampling rate used for shifted crop construction."""
    if override is not None:
        return float(override)
    if hasattr(model, "sfreq"):
        return float(model.sfreq)
    value = config.model.params.get("sfreq", 100.0)
    return float(value)


def model_crop_sec(config: ExperimentConfig, model, sfreq: float, override: float | None) -> float:
    """Return the crop length expected by the model."""
    if override is not None:
        return float(override)
    if hasattr(model, "n_times"):
        return float(model.n_times) / float(sfreq)
    n_times = config.model.params.get("n_times")
    if n_times is not None:
        return float(n_times) / float(sfreq)
    return 2.0


def selected_channels(model):
    """Return the channel selection used by benchmark wrappers."""
    return np.arange(model.n_chans) if hasattr(model, "n_chans") else None


def segmentation_temperature(config: ExperimentConfig, override: float | None) -> float:
    """Return the soft-argmax temperature for segmentation shifted eval."""
    if override is not None:
        return float(override)
    params = config.trainer.params
    return float(params.get("eval_temperature", params.get("temperature", 1.0)))


def predict_start(
    *,
    task: str,
    model,
    base_dataset,
    crop_start: float,
    reference_start: float,
    sfreq: float,
    crop_sec: float,
    segmentation_tau: float,
    channels_list,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return raw, corrected, relative predictions and absolute targets for one crop start."""
    dataset = ShiftedFixedWindowDataset(
        base_dataset,
        sfreq=sfreq,
        crop_sec=crop_sec,
        crop_start=crop_start,
        target_mode="absolute",
        use_channels=channels_list,
        use_augmentation=False,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    raw_predictions = []
    corrected_predictions = []
    relative_predictions = []
    targets = []
    dt = 1.0 / float(sfreq)
    model.eval()
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device).float().contiguous()
            if task == "regression":
                pred_abs = model(X).detach().view(-1)
                pred_raw = pred_abs
                pred_corrected = pred_abs + (float(crop_start) - float(reference_start))
                pred_rel = pred_abs - float(crop_start)
            elif task == "segmentation":
                logits = model(X).detach().squeeze(1)
                probabilities = torch.softmax(logits / float(segmentation_tau), dim=-1)
                time_grid = torch.arange(logits.shape[-1], device=logits.device, dtype=logits.dtype)[None, :] * dt
                pred_rel = (probabilities * time_grid).sum(dim=-1)
                pred_raw = pred_rel + float(reference_start)
                pred_corrected = pred_rel + float(crop_start)
            else:
                raise RuntimeError(f"Unsupported task for shifted eval: {task!r}")
            raw_predictions.append(pred_raw.cpu().numpy())
            corrected_predictions.append(pred_corrected.cpu().numpy())
            relative_predictions.append(pred_rel.cpu().numpy())
            targets.append(y.detach().view(-1).cpu().numpy())
    return (
        np.concatenate(raw_predictions),
        np.concatenate(corrected_predictions),
        np.concatenate(relative_predictions),
        np.concatenate(targets),
    )


def metadata_frame(base_dataset, targets: np.ndarray) -> pd.DataFrame:
    """Return row metadata aligned with base dataset order."""
    if hasattr(base_dataset, "get_metadata"):
        frame = base_dataset.get_metadata().reset_index(drop=True).copy()
    else:
        frame = pd.DataFrame(index=np.arange(len(targets)))
    frame.insert(0, "row_id", np.arange(len(frame)))
    if "target" not in frame:
        frame["target"] = targets
    return frame


def mask_for(frame: pd.DataFrame, name: str) -> np.ndarray:
    """Return the named metric mask."""
    if name == "all":
        return np.ones(len(frame), dtype=bool)
    if name == "inside_crop":
        return frame["inside_crop"].to_numpy(dtype=bool)
    if name == "common_inside":
        return frame["common_inside"].to_numpy(dtype=bool)
    raise ValueError(f"Unknown mask: {name}")


def mean_or_nan(values) -> float:
    """Return a finite mean or NaN for empty/all-NaN arrays."""
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    return float(np.mean(array))


def add_shift_tracking_columns(predictions: pd.DataFrame, *, reference_start: float) -> pd.DataFrame:
    """Attach reference-crop deltas and localizer tracking metrics."""
    frame = predictions.copy()
    reference_mask = np.isclose(frame["crop_start"].to_numpy(dtype=np.float64), float(reference_start))
    if not reference_mask.any():
        raise ValueError(f"reference_start={reference_start} is not present in shifted predictions.")

    reference = frame.loc[reference_mask, ["row_id", "prediction_raw"]].rename(
        columns={"prediction_raw": "prediction_reference"}
    )
    frame = frame.merge(reference, on="row_id", how="left", validate="many_to_one")
    frame["shift_delta"] = frame["crop_start"] - float(reference_start)
    frame["raw_delta_vs_reference"] = frame["prediction_raw"] - frame["prediction_reference"]
    frame["expected_localizer_delta"] = -frame["shift_delta"]
    frame["shift_tracking_error"] = frame["raw_delta_vs_reference"] - frame["expected_localizer_delta"]

    non_reference = ~np.isclose(frame["shift_delta"].to_numpy(dtype=np.float64), 0.0)
    frame["is_reference_start"] = ~non_reference
    frame["shift_sensitivity_ratio"] = np.nan
    frame.loc[non_reference, "shift_sensitivity_ratio"] = (
        frame.loc[non_reference, "raw_delta_vs_reference"].abs() / frame.loc[non_reference, "shift_delta"].abs()
    )
    frame["correct_shift_direction"] = np.nan
    frame.loc[non_reference, "correct_shift_direction"] = (
        frame.loc[non_reference, "raw_delta_vs_reference"] * frame.loc[non_reference, "shift_delta"] < 0
    ).astype(float)
    return frame


def metric_row(frame: pd.DataFrame, *, crop_start: float, mask_name: str) -> dict:
    """Summarize raw and shift-corrected prediction errors for one crop start."""
    mask = mask_for(frame, mask_name)
    target = frame.loc[mask, "target_abs"].to_numpy(dtype=np.float64)
    raw = frame.loc[mask, "prediction_raw"].to_numpy(dtype=np.float64)
    corrected = frame.loc[mask, "prediction_corrected"].to_numpy(dtype=np.float64)
    tracked = frame.loc[mask & ~frame["is_reference_start"], :]
    return {
        "crop_start": float(crop_start),
        "mask": mask_name,
        "n_rows": int(mask.sum()),
        "rmse_raw": rmse(raw, target),
        "nrmse_raw": nrmse(raw, target),
        "mae_raw": float(np.mean(np.abs(raw - target))),
        "bias_raw": float(np.mean(raw - target)),
        "rmse_corrected": rmse(corrected, target),
        "nrmse_corrected": nrmse(corrected, target),
        "mae_corrected": float(np.mean(np.abs(corrected - target))),
        "bias_corrected": float(np.mean(corrected - target)),
        "mean_abs_shift_tracking_error": mean_or_nan(tracked["shift_tracking_error"].abs()),
        "mean_shift_sensitivity_ratio": mean_or_nan(tracked["shift_sensitivity_ratio"]),
        "median_shift_sensitivity_ratio": float(np.nanmedian(tracked["shift_sensitivity_ratio"]))
        if tracked["shift_sensitivity_ratio"].notna().any()
        else float("nan"),
        "correct_shift_direction_rate": mean_or_nan(tracked["correct_shift_direction"]),
    }


def summarize_per_start(predictions: pd.DataFrame, *, reference_start: float) -> pd.DataFrame:
    """Return per-start scalar error summaries."""
    rows = []
    for crop_start, frame in predictions.groupby("crop_start", sort=True):
        for mask_name in ("all", "inside_crop", "common_inside"):
            rows.append(metric_row(frame, crop_start=float(crop_start), mask_name=mask_name))
    return add_reference_deltas(pd.DataFrame(rows), reference_start=reference_start)


def add_reference_deltas(summary: pd.DataFrame, *, reference_start: float) -> pd.DataFrame:
    """Attach degradation metrics relative to the canonical crop."""
    frame = summary.copy()
    metric_names = (
        "rmse_raw",
        "nrmse_raw",
        "mae_raw",
        "rmse_corrected",
        "nrmse_corrected",
        "mae_corrected",
    )
    for metric in metric_names:
        frame[f"delta_{metric}_vs_reference"] = np.nan

    for mask_name, group in frame.groupby("mask", sort=False):
        reference_rows = group[np.isclose(group["crop_start"].to_numpy(dtype=np.float64), float(reference_start))]
        if reference_rows.empty:
            continue
        reference = reference_rows.iloc[0]
        idx = frame["mask"] == mask_name
        for metric in metric_names:
            frame.loc[idx, f"delta_{metric}_vs_reference"] = frame.loc[idx, metric] - float(reference[metric])
    return frame


def fit_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Fit a simple y ~ a + b*x slope."""
    if len(np.unique(x)) < 2:
        return float("nan")
    return float(np.polyfit(x, y, deg=1)[0])


def per_trial_equivariance(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return per-trial shift-equivalence diagnostics."""
    rows = []
    for row_id, frame in predictions.groupby("row_id", sort=False):
        starts = frame["crop_start"].to_numpy(dtype=np.float64)
        raw = frame["prediction_raw"].to_numpy(dtype=np.float64)
        corrected = frame["prediction_corrected"].to_numpy(dtype=np.float64)
        target = float(frame["target_abs"].iloc[0])
        raw_slope = fit_slope(starts, raw)
        corrected_slope = fit_slope(starts, corrected)
        rows.append(
            {
                "row_id": int(row_id),
                "subject": frame["subject"].iloc[0] if "subject" in frame else int(row_id),
                "target_abs": target,
                "common_inside": bool(frame["common_inside"].iloc[0]),
                "raw_slope_vs_start": raw_slope,
                "corrected_slope_vs_start": corrected_slope,
                "raw_slope_abs_error_to_localizer": abs(raw_slope + 1.0),
                "corrected_slope_abs_error_to_stable": abs(corrected_slope),
                "is_localizer_like": -1.25 <= raw_slope <= -0.75,
                "is_invariant_like": -0.25 <= raw_slope <= 0.25,
                "is_wrong_direction": raw_slope > 0.0,
                "raw_std": float(np.std(raw, ddof=1)),
                "corrected_std": float(np.std(corrected, ddof=1)),
                "raw_range": float(np.max(raw) - np.min(raw)),
                "corrected_range": float(np.max(corrected) - np.min(corrected)),
                "mean_abs_error_raw": float(np.mean(np.abs(raw - target))),
                "mean_abs_error_corrected": float(np.mean(np.abs(corrected - target))),
            }
        )
    return pd.DataFrame(rows)


def summarize_equivariance(per_trial: pd.DataFrame) -> dict:
    """Summarize per-trial equivariance diagnostics."""
    summaries = {}
    for mask_name, frame in {
        "all": per_trial,
        "common_inside": per_trial[per_trial["common_inside"]],
    }.items():
        summaries[mask_name] = {
            "n_rows": int(len(frame)),
            "raw_slope_mean": float(frame["raw_slope_vs_start"].mean()),
            "raw_slope_median": float(frame["raw_slope_vs_start"].median()),
            "raw_slope_mae_to_localizer": float(np.mean(np.abs(frame["raw_slope_vs_start"] + 1.0))),
            "corrected_slope_mean": float(frame["corrected_slope_vs_start"].mean()),
            "corrected_slope_median": float(frame["corrected_slope_vs_start"].median()),
            "corrected_slope_mae_to_stable": float(np.mean(np.abs(frame["corrected_slope_vs_start"]))),
            "localizer_like_fraction": float(frame["is_localizer_like"].mean()),
            "invariant_like_fraction": float(frame["is_invariant_like"].mean()),
            "wrong_direction_fraction": float(frame["is_wrong_direction"].mean()),
            "raw_std_mean": float(frame["raw_std"].mean()),
            "corrected_std_mean": float(frame["corrected_std"].mean()),
            "raw_range_mean": float(frame["raw_range"].mean()),
            "corrected_range_mean": float(frame["corrected_range"].mean()),
            "mean_abs_error_raw": float(frame["mean_abs_error_raw"].mean()),
            "mean_abs_error_corrected": float(frame["mean_abs_error_corrected"].mean()),
        }
    return summaries


def subject_column(frame: pd.DataFrame) -> str:
    """Return the subject identifier column used for bootstrap grouping."""
    return "subject" if "subject" in frame else "row_id"


def bootstrap_subject_means(
    frame: pd.DataFrame,
    *,
    metric: str,
    subject_col: str,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, int]:
    """Return observed mean and 95% subject-bootstrap CI for one metric."""
    values = frame[[subject_col, metric]].replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return float("nan"), float("nan"), float("nan"), 0
    subject_means = values.groupby(subject_col, sort=False)[metric].mean().to_numpy(dtype=np.float64)
    n_subjects = len(subject_means)
    if n_subjects == 0:
        return float("nan"), float("nan"), float("nan"), 0
    samples = np.empty(int(n_samples), dtype=np.float64)
    for sample_idx in range(int(n_samples)):
        sampled = rng.choice(subject_means, size=n_subjects, replace=True)
        samples[sample_idx] = np.mean(sampled)
    observed = float(np.mean(subject_means))
    return observed, float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975)), n_subjects


def bootstrap_per_trial_ci(per_trial: pd.DataFrame, *, n_samples: int, seed: int) -> pd.DataFrame:
    """Bootstrap per-trial equivariance metrics over subjects."""
    metrics = (
        "raw_slope_vs_start",
        "raw_slope_abs_error_to_localizer",
        "corrected_slope_vs_start",
        "corrected_slope_abs_error_to_stable",
        "is_localizer_like",
        "is_invariant_like",
        "is_wrong_direction",
        "raw_std",
        "corrected_std",
        "raw_range",
        "corrected_range",
        "mean_abs_error_raw",
        "mean_abs_error_corrected",
    )
    subject_col = subject_column(per_trial)
    rows = []
    for mask_name, frame in {
        "all": per_trial,
        "common_inside": per_trial[per_trial["common_inside"]],
    }.items():
        for metric in metrics:
            rng = np.random.default_rng(seed + len(rows))
            observed, ci_low, ci_high, n_subjects = bootstrap_subject_means(
                frame,
                metric=metric,
                subject_col=subject_col,
                n_samples=n_samples,
                rng=rng,
            )
            rows.append(
                {
                    "section": "per_trial",
                    "mask": mask_name,
                    "crop_start": np.nan,
                    "metric": metric,
                    "observed": observed,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "n_subjects": n_subjects,
                    "n_rows": int(len(frame)),
                    "n_bootstrap": int(n_samples),
                    "bootstrap_unit": "subject_mean",
                }
            )
    return pd.DataFrame(rows)


def per_subject_start_metrics(frame: pd.DataFrame, *, subject_col: str) -> pd.DataFrame:
    """Compute subject-level per-start metrics used for bootstrap CIs."""
    rows = []
    for (subject, crop_start), group in frame.groupby([subject_col, "crop_start"], sort=False):
        rows.append(
            {
                subject_col: subject,
                "crop_start": float(crop_start),
                "mae_raw": float(np.mean(np.abs(group["error_raw"]))),
                "mae_corrected": float(np.mean(np.abs(group["error_corrected"]))),
                "mean_abs_shift_tracking_error": mean_or_nan(group["shift_tracking_error"].abs()),
                "mean_shift_sensitivity_ratio": mean_or_nan(group["shift_sensitivity_ratio"]),
                "correct_shift_direction_rate": mean_or_nan(group["correct_shift_direction"]),
            }
        )
    return pd.DataFrame(rows)


def attach_subject_reference_deltas(subject_metrics: pd.DataFrame, *, subject_col: str, reference_start: float) -> pd.DataFrame:
    """Attach subject-level degradation relative to the reference crop."""
    reference = subject_metrics[np.isclose(subject_metrics["crop_start"].to_numpy(dtype=np.float64), float(reference_start))]
    reference = reference[[subject_col, "mae_raw", "mae_corrected"]].rename(
        columns={
            "mae_raw": "reference_mae_raw",
            "mae_corrected": "reference_mae_corrected",
        }
    )
    frame = subject_metrics.merge(reference, on=subject_col, how="left", validate="many_to_one")
    frame["delta_mae_raw_vs_reference"] = frame["mae_raw"] - frame["reference_mae_raw"]
    frame["delta_mae_corrected_vs_reference"] = frame["mae_corrected"] - frame["reference_mae_corrected"]
    return frame


def bootstrap_per_start_ci(predictions: pd.DataFrame, *, reference_start: float, n_samples: int, seed: int) -> pd.DataFrame:
    """Bootstrap per-start MAE, degradation, and shift-tracking metrics over subjects."""
    metrics = (
        "mae_raw",
        "mae_corrected",
        "delta_mae_raw_vs_reference",
        "delta_mae_corrected_vs_reference",
        "mean_abs_shift_tracking_error",
        "mean_shift_sensitivity_ratio",
        "correct_shift_direction_rate",
    )
    subject_col = subject_column(predictions)
    rows = []
    for mask_name in ("all", "inside_crop", "common_inside"):
        masked = predictions[mask_for(predictions, mask_name)]
        subject_metrics = per_subject_start_metrics(masked, subject_col=subject_col)
        subject_metrics = attach_subject_reference_deltas(
            subject_metrics,
            subject_col=subject_col,
            reference_start=reference_start,
        )
        for crop_start, start_frame in subject_metrics.groupby("crop_start", sort=True):
            for metric in metrics:
                rng = np.random.default_rng(seed + 10000 + len(rows))
                observed, ci_low, ci_high, n_subjects = bootstrap_subject_means(
                    start_frame,
                    metric=metric,
                    subject_col=subject_col,
                    n_samples=n_samples,
                    rng=rng,
                )
                rows.append(
                    {
                        "section": "per_start",
                        "mask": mask_name,
                        "crop_start": float(crop_start),
                        "metric": metric,
                        "observed": observed,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "n_subjects": n_subjects,
                        "n_rows": int(len(masked[masked["crop_start"] == crop_start])),
                        "n_bootstrap": int(n_samples),
                        "bootstrap_unit": "subject_mean",
                    }
                )
    return pd.DataFrame(rows)


def bootstrap_ci_tables(
    predictions: pd.DataFrame,
    per_trial: pd.DataFrame,
    *,
    reference_start: float,
    n_samples: int,
    seed: int,
) -> pd.DataFrame:
    """Return combined bootstrap CI tables for shifted diagnostics."""
    if n_samples <= 0:
        return pd.DataFrame()
    return pd.concat(
        [
            bootstrap_per_start_ci(
                predictions,
                reference_start=reference_start,
                n_samples=n_samples,
                seed=seed,
            ),
            bootstrap_per_trial_ci(per_trial, n_samples=n_samples, seed=seed),
        ],
        ignore_index=True,
    )


def run_shifted_eval(args: argparse.Namespace) -> None:
    """Run shifted-crop evaluation for one regression run."""
    run_dir = resolve_path(args.run_dir, PROJECT_ROOT)
    dataset_path = resolve_path(args.dataset, PROJECT_ROOT)
    if run_dir is None:
        raise ValueError("run_dir cannot be None.")
    if dataset_path is None:
        raise ValueError("dataset path cannot be None.")
    run_dir = run_dir.resolve()
    dataset_path = dataset_path.resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Missing shifted-eval dataset: {dataset_path}. "
            "Build it with benchmarks/preparation/scripts/reload_test_5sec_dataset.py."
        )

    _, config = load_run_snapshot(run_dir)
    if config.task not in {"regression", "segmentation"}:
        raise RuntimeError(f"eval_shifted expects a regression or segmentation run, got task={config.task!r}.")

    device = choose_device(args.device)
    model = config.model.build().to(device)
    checkpoint_path = run_dir / "best_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing best checkpoint: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)

    sfreq = model_sfreq(config, model, args.sfreq)
    crop_sec = model_crop_sec(config, model, sfreq, args.crop_sec)
    seg_tau = segmentation_temperature(config, args.segmentation_temperature)
    batch_size = int(args.batch_size or config.loaders.valid.batch_size)
    num_workers = int(args.num_workers if args.num_workers is not None else config.loaders.valid.num_workers)
    starts = tuple(float(value) for value in args.starts)
    output_dir = (resolve_path(args.output_dir, PROJECT_ROOT) if args.output_dir else run_dir / "shifted_eval")
    if output_dir is None:
        raise ValueError("output_dir cannot be None.")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_dataset = load_pickle_dataset(dataset_path)
    channels_list = selected_channels(model)
    all_frames = []
    reference_start = float(args.reference_start)
    common_low = max(starts)
    common_high = min(starts) + crop_sec

    print(f"Run directory: {path_text(run_dir)}")
    print(f"Dataset: {path_text(dataset_path)}")
    print(f"Device: {device}")
    print(f"Task: {config.task}")
    print(f"Starts: {', '.join(f'{value:.3f}' for value in starts)}")
    print(f"Crop length: {crop_sec:.3f}s | sfreq={sfreq:g} | batch_size={batch_size} | workers={num_workers}")
    if config.task == "segmentation":
        print(f"Segmentation soft-argmax temperature: {seg_tau:.4f}")
    print(f"Common-inside target range: [{common_low:.3f}, {common_high:.3f}]s")
    print(f"Loaded checkpoint: {path_text(checkpoint_path)}")

    for crop_start in starts:
        predictions_raw, predictions_corrected, predictions_rel, targets = predict_start(
            task=config.task,
            model=model,
            base_dataset=base_dataset,
            crop_start=crop_start,
            reference_start=reference_start,
            sfreq=sfreq,
            crop_sec=crop_sec,
            segmentation_tau=seg_tau,
            channels_list=channels_list,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
        )
        frame = metadata_frame(base_dataset, targets)
        frame["crop_start"] = float(crop_start)
        frame["crop_end"] = float(crop_start + crop_sec)
        frame["target_abs"] = targets.astype(np.float64)
        frame["target_rel"] = frame["target_abs"] - float(crop_start)
        frame["inside_crop"] = (frame["target_abs"] >= crop_start) & (frame["target_abs"] <= crop_start + crop_sec)
        frame["common_inside"] = (frame["target_abs"] >= common_low) & (frame["target_abs"] <= common_high)
        frame["prediction_raw"] = predictions_raw.astype(np.float64)
        frame["prediction_corrected"] = predictions_corrected.astype(np.float64)
        frame["prediction_rel"] = predictions_rel.astype(np.float64)
        frame["prediction_abs"] = frame["prediction_corrected"]
        frame["error_raw"] = frame["prediction_raw"] - frame["target_abs"]
        frame["error_corrected"] = frame["prediction_corrected"] - frame["target_abs"]
        all_frames.append(frame)
        print(f"Evaluated crop_start={crop_start:.3f}: rows={len(frame):,}")

    prediction_frame = add_shift_tracking_columns(pd.concat(all_frames, ignore_index=True), reference_start=reference_start)
    summary_frame = summarize_per_start(prediction_frame, reference_start=reference_start)
    per_trial_frame = per_trial_equivariance(prediction_frame)
    equivariance_summary = summarize_equivariance(per_trial_frame)
    bootstrap_frame = bootstrap_ci_tables(
        prediction_frame,
        per_trial_frame,
        reference_start=reference_start,
        n_samples=int(args.bootstrap_samples),
        seed=int(args.bootstrap_seed),
    )
    predictions_path = output_dir / "shifted_predictions.csv"
    summary_path = output_dir / "shifted_summary.csv"
    per_trial_path = output_dir / "shifted_per_trial_equivariance.csv"
    bootstrap_path = output_dir / "shifted_bootstrap_ci.csv"
    json_path = output_dir / "shifted_equivariance.json"
    metadata = {
        "run_dir": str(run_dir),
        "dataset": str(dataset_path),
        "checkpoint": str(checkpoint_path),
        "task": config.task,
        "starts": list(starts),
        "reference_start": reference_start,
        "segmentation_temperature": seg_tau if config.task == "segmentation" else None,
        "crop_sec": crop_sec,
        "sfreq": sfreq,
        "common_inside_low": common_low,
        "common_inside_high": common_high,
        "n_base_rows": int(len(base_dataset)),
        "n_prediction_rows": int(len(prediction_frame)),
        "bootstrap_samples": int(args.bootstrap_samples),
        "bootstrap_seed": int(args.bootstrap_seed),
        "artifacts": {
            "predictions": str(predictions_path),
            "summary": str(summary_path),
            "per_trial_equivariance": str(per_trial_path),
            "bootstrap_ci": str(bootstrap_path) if not bootstrap_frame.empty else None,
        },
        "equivariance": equivariance_summary,
    }
    prediction_frame.to_csv(predictions_path, index=False)
    summary_frame.to_csv(summary_path, index=False)
    per_trial_frame.to_csv(per_trial_path, index=False)
    if not bootstrap_frame.empty:
        bootstrap_frame.to_csv(bootstrap_path, index=False)
    json_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    print("Saved shifted evaluation:")
    print(f"- predictions: {path_text(predictions_path)}")
    print(f"- summary: {path_text(summary_path)}")
    print(f"- per-trial equivariance: {path_text(per_trial_path)}")
    if not bootstrap_frame.empty:
        print(f"- bootstrap CI: {path_text(bootstrap_path)}")
    print(f"- metadata: {path_text(json_path)}")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    run_shifted_eval(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
