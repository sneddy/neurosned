"""Refresh segmentation temperature calibration from saved logits."""

from __future__ import annotations

import argparse
import os
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
import yaml

from benchmarks.pkg.artefacts_manager import ArtefactsManager, now_utc_iso
from benchmarks.pkg.config import ExperimentConfig, resolve_path
from benchmarks.pkg.evaluation.calibration import apply_temperature, fit_temperature
from benchmarks.pkg.evaluation.factory import build_confidence_interval, build_temperature_readout
from benchmarks.pkg.evaluation.metrics import nrmse, rmse
from benchmarks.pkg.runtime import path_text


def build_parser() -> argparse.ArgumentParser:
    """Build the temperature-refresh CLI parser."""
    parser = argparse.ArgumentParser(
        description="Refit segmentation temperature calibration from saved validation/test logits."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Run directories, or experiment directories containing run subdirectories.",
    )
    parser.add_argument("--split", default="test", help="Holdout split prefix, usually 'test'.")
    parser.add_argument("--temperature-min", type=float, default=0.2, help="Temperature grid minimum.")
    parser.add_argument("--temperature-max", type=float, default=3.5, help="Temperature grid maximum.")
    parser.add_argument("--temperature-step", type=float, default=0.05, help="Temperature grid step.")
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Only include run directories whose name contains this text. Can be passed multiple times.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Exclude run directories whose name contains this text. Can be passed multiple times.",
    )
    return parser


def load_snapshot(run_dir: Path) -> tuple[dict, ExperimentConfig]:
    """Load a run config snapshot."""
    path = run_dir / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Missing run snapshot: {path}")
    with path.open("r", encoding="utf-8") as f:
        snapshot = yaml.safe_load(f)
    if "config" not in snapshot:
        raise ValueError(f"Run snapshot does not contain a 'config' section: {path}")
    return snapshot, ExperimentConfig.model_validate(snapshot["config"])


def snapshot_path(snapshot: dict, key: str) -> Path | None:
    """Resolve an optional path from the snapshot run section."""
    value = snapshot.get("run", {}).get(key)
    return resolve_path(value, PROJECT_ROOT) if value else None


def apply_temperature_config(
    config: ExperimentConfig,
    *,
    min_value: float,
    max_value: float,
    step: float,
) -> ExperimentConfig:
    """Enable and set the temperature calibration grid."""
    temperature = config.calibration.temperature.model_copy(
        update={
            "enabled": True,
            "min": float(min_value),
            "max": float(max_value),
            "step": float(step),
        }
    )
    calibration = config.calibration.model_copy(update={"temperature": temperature})
    return config.model_copy(update={"calibration": calibration})


def discover_run_dirs(paths: list[Path], *, include: list[str], exclude: list[str], split: str) -> list[Path]:
    """Resolve run directories from run paths or experiment paths."""
    runs = []
    for raw_path in paths:
        path = resolve_path(raw_path, PROJECT_ROOT)
        if path is None:
            continue
        path = path.resolve()
        if (path / "config.yaml").exists():
            candidates = [path]
        else:
            candidates = sorted(child for child in path.iterdir() if child.is_dir() and (child / "config.yaml").exists())
        for candidate in candidates:
            text = candidate.name
            if include and not any(token in text for token in include):
                continue
            if exclude and any(token in text for token in exclude):
                continue
            required = [
                candidate / "predictions" / "best_logits.npy",
                candidate / "predictions" / "best_val_predictions.csv",
                candidate / "predictions" / f"{split}_logits.npy",
                candidate / "predictions" / f"{split}_predictions.csv",
            ]
            if all(path.exists() for path in required):
                runs.append(candidate)
    return sorted(dict.fromkeys(runs))


def read_time_grid(config: ExperimentConfig, logits: np.ndarray) -> tuple[float, float]:
    """Return sfreq and win_offset for segmentation readout."""
    sfreq = float(config.model.params.get("sfreq", 100.0))
    win_offset = float(config.trainer.params.get("win_offset", 0.5))
    if logits.shape[-1] <= 0:
        raise ValueError("Logits must have a non-empty time dimension.")
    return sfreq, win_offset


def refresh_run(
    run_dir: Path,
    *,
    split: str,
    min_value: float,
    max_value: float,
    step: float,
) -> dict:
    """Refresh calibration and calibrated holdout metrics for one run."""
    snapshot, config = load_snapshot(run_dir)
    if config.task != "segmentation":
        raise RuntimeError(f"Temperature refresh only supports segmentation runs: {run_dir}")
    config = apply_temperature_config(config, min_value=min_value, max_value=max_value, step=step)

    data_paths = config.data_paths(PROJECT_ROOT)
    artefacts = ArtefactsManager.open_existing(
        run_dir=run_dir,
        config=config,
        project_root=PROJECT_ROOT,
        config_path=snapshot_path(snapshot, "config_path"),
        input_checkpoint_path=snapshot_path(snapshot, "input_checkpoint_path"),
        data_paths=data_paths,
    )
    artefacts.save_config_snapshot()

    valid_logits = np.load(run_dir / "predictions" / "best_logits.npy")
    holdout_logits = np.load(run_dir / "predictions" / f"{split}_logits.npy")
    valid_metadata = pd.read_csv(run_dir / "predictions" / "best_val_predictions.csv")
    holdout_metadata = pd.read_csv(run_dir / "predictions" / f"{split}_predictions.csv")
    valid_targets = valid_metadata["target"].to_numpy()
    holdout_targets = holdout_metadata["target"].to_numpy()
    sfreq, win_offset = read_time_grid(config, holdout_logits)
    readout = build_temperature_readout(config, sfreq=sfreq, win_offset=win_offset)

    calibration = fit_temperature(
        valid_logits,
        valid_targets,
        min_value=min_value,
        max_value=max_value,
        step=step,
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

    temperature = float(calibration["best_temperature"])
    predictions = apply_temperature(
        holdout_logits,
        temperature,
        prediction_fn=readout.prediction_fn,
    )
    metrics = {
        "rmse": rmse(predictions, holdout_targets),
        "nrmse": nrmse(predictions, holdout_targets),
        "temperature": temperature,
        "preds_abs": predictions,
    }
    artefacts.save_holdout_evaluation(
        split=f"{split}_tau",
        metrics=metrics,
        metadata=holdout_metadata,
        evaluation=config.evaluation,
        checkpoint_loaded=True,
        confidence_interval=build_confidence_interval(metrics, holdout_metadata, config.evaluation),
    )
    artefacts.save_summary(status="temperature_refreshed", temperature_refreshed_at=now_utc_iso())
    return {
        "run_dir": run_dir,
        "calibration_path": calibration_path,
        "temperature": temperature,
        "valid_nrmse": float(calibration["best_nrmse"]),
        "holdout_nrmse": float(metrics["nrmse"]),
    }


def main(argv: list[str] | None = None) -> int:
    """Run the temperature refresh CLI."""
    args = build_parser().parse_args(argv)
    run_dirs = discover_run_dirs(args.paths, include=args.include, exclude=args.exclude, split=args.split)
    if not run_dirs:
        raise RuntimeError("No run directories with saved validation and holdout logits were found.")

    print("\n=== Temperature refresh from saved logits ===")
    print(f"Runs: {len(run_dirs)}")
    print(f"Split: {args.split}")
    print(f"Grid: {args.temperature_min:g}..{args.temperature_max:g} step {args.temperature_step:g}")
    for run_dir in run_dirs:
        result = refresh_run(
            run_dir,
            split=args.split,
            min_value=args.temperature_min,
            max_value=args.temperature_max,
            step=args.temperature_step,
        )
        print(
            f"- {path_text(result['run_dir'])}: "
            f"tau={result['temperature']:.4f}, "
            f"valid_nrmse={result['valid_nrmse']:.6f}, "
            f"{args.split}_tau_nrmse={result['holdout_nrmse']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
