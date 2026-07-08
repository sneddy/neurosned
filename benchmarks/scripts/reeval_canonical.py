"""Re-evaluate an existing run into a separate canonical-evaluation tree.

This is intentionally separate from ``reeval.py``. The normal reeval command
writes back into the source run directory; this command keeps the source run
immutable and writes evaluation-only artefacts under a new experiment root.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["MNE_DONTWRITE_HOME"] = "true"
os.environ["NUMBA_DISABLE_JIT"] = "1"
os.environ["MPLCONFIGDIR"] = "/tmp/neurosned-matplotlib"
os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from benchmarks.pkg.artefacts_manager import ArtefactsManager, now_utc_iso
from benchmarks.pkg.config import ExperimentConfig, resolve_path
from benchmarks.pkg.evaluation.factory import build_eval_dataset
from benchmarks.pkg.evaluation.runner import run_holdout_evaluation
from benchmarks.pkg.runtime import choose_device, path_text, tee_output


def build_parser() -> argparse.ArgumentParser:
    """Build the canonical re-evaluation CLI parser."""
    parser = argparse.ArgumentParser(
        description="Re-evaluate an existing benchmark run into a separate canonical eval folder."
    )
    parser.add_argument("run_dir", type=Path, help="Source run directory with config.yaml and best_model.pth.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto", help="Torch device selection.")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Canonical eval experiment root. Defaults to benchmarks/experiments/<source_experiment>_canonical_eval.",
    )
    parser.add_argument("--target-min", type=float, default=0.5, help="Canonical eval target lower bound.")
    parser.add_argument("--target-max", type=float, default=2.5, help="Canonical eval target upper bound.")
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Experiment name written to the canonical config. Defaults to <source_experiment>_canonical_eval.",
    )
    parser.add_argument("--enable-temperature", action="store_true", help="Force-enable temperature calibration.")
    parser.add_argument("--disable-temperature", action="store_true", help="Force-disable temperature calibration.")
    parser.add_argument("--temperature-min", type=float, default=None, help="Override temperature grid minimum.")
    parser.add_argument("--temperature-max", type=float, default=None, help="Override temperature grid maximum.")
    parser.add_argument("--temperature-step", type=float, default=None, help="Override temperature grid step.")
    parser.add_argument("--disable-ci", action="store_true", help="Disable holdout confidence intervals for this reeval.")
    return parser


def load_run_snapshot(run_dir: Path) -> tuple[dict, ExperimentConfig]:
    """Load a source run snapshot and typed config."""
    snapshot_path = run_dir / "config.yaml"
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Missing source run snapshot: {snapshot_path}")
    with snapshot_path.open("r", encoding="utf-8") as f:
        snapshot = yaml.safe_load(f)
    if "config" not in snapshot:
        raise ValueError(f"Run snapshot does not contain a 'config' section: {snapshot_path}")
    return snapshot, ExperimentConfig.model_validate(snapshot["config"])


def load_run_summary(run_dir: Path) -> dict:
    """Load source summary.json when available."""
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return {}
    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def apply_canonical_overrides(config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    """Return an evaluation-only config with canonical target support."""
    experiment_name = args.experiment_name or f"{config.experiment}_canonical_eval"
    data = config.data.model_copy(
        update={
            "target_min": args.target_min,
            "target_max": args.target_max,
        }
    )

    confidence_interval = config.evaluation.confidence_interval
    if args.disable_ci:
        confidence_interval = confidence_interval.model_copy(update={"enabled": False})
    evaluation = config.evaluation.model_copy(
        update={
            "holdout_eval": True,
            "save_predictions": True,
            "save_logits": True,
            "confidence_interval": confidence_interval,
        }
    )

    temperature_updates = {}
    if args.enable_temperature:
        if config.task != "segmentation":
            raise RuntimeError("--enable-temperature is only supported for segmentation runs.")
        temperature_updates["enabled"] = True
    if args.disable_temperature:
        temperature_updates["enabled"] = False
    if args.temperature_min is not None:
        temperature_updates["min"] = args.temperature_min
    if args.temperature_max is not None:
        temperature_updates["max"] = args.temperature_max
    if args.temperature_step is not None:
        temperature_updates["step"] = args.temperature_step

    calibration = config.calibration
    if temperature_updates:
        temperature = calibration.temperature.model_copy(update=temperature_updates)
        calibration = calibration.model_copy(update={"temperature": temperature})

    return config.model_copy(
        update={
            "experiment": experiment_name,
            "data": data,
            "evaluation": evaluation,
            "calibration": calibration,
        }
    )


def canonical_output_root(source_config: ExperimentConfig, args: argparse.Namespace) -> Path:
    """Return the canonical eval root directory."""
    if args.out_root is not None:
        root = resolve_path(args.out_root, PROJECT_ROOT)
        if root is None:
            raise ValueError("--out-root cannot resolve to None.")
        return root
    return PROJECT_ROOT / "benchmarks" / "experiments" / f"{source_config.experiment}_canonical_eval"


def canonical_run_dir(source_run_dir: Path, out_root: Path) -> Path:
    """Mirror repeated-run grouping when the source leaf is seedNNNN."""
    if source_run_dir.name.startswith("seed"):
        return out_root / source_run_dir.parent.name / source_run_dir.name
    return out_root / source_run_dir.name


def canonical_summary_run_name(source_run_dir: Path) -> str:
    """Build a unique run name for global summary rows."""
    if source_run_dir.name.startswith("seed"):
        return f"{source_run_dir.parent.name}__{source_run_dir.name}"
    return source_run_dir.name


def relative(path: Path | None) -> str | None:
    """Return a project-relative path when possible."""
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def build_eval_trainer(
    *,
    config: ExperimentConfig,
    model,
    valid_loader,
    default_rmse_by_split: dict[str, float],
    channels_list,
    artefacts: ArtefactsManager,
    device: torch.device,
):
    """Build the configured trainer for eval-only forward passes."""
    optimizer_cls = config.optimizer.load_class()
    optimizer = optimizer_cls(model.parameters(), **config.optimizer.params)

    trainer_params = dict(config.trainer.params)
    if config.task == "segmentation":
        trainer_params["channels_list"] = channels_list
        trainer_params["plot_last_batch"] = False
        trainer_params.setdefault("plot_save_dir", artefacts.paths.figures_dir)

    trainer_cls = config.trainer.load_class()
    return trainer_cls(
        model=model,
        train_loader=valid_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        device=device,
        n_epochs=0,
        checkpoint_path=artefacts.checkpoint_path,
        monitor=config.trainer.monitor,
        minimize=config.trainer.minimize,
        early_stopping_patience=None,
        print_batch_stats=config.trainer.print_batch_stats,
        default_rmse_by_split=default_rmse_by_split,
        **trainer_params,
    )


def source_metadata(
    *,
    source_run_dir: Path,
    source_snapshot: dict,
    source_config: ExperimentConfig,
    eval_config: ExperimentConfig,
    source_checkpoint: Path,
    dest_run_dir: Path,
) -> dict:
    """Build provenance metadata for a canonical eval run."""
    return {
        "created_at": now_utc_iso(),
        "source_run_dir": relative(source_run_dir),
        "source_config_snapshot": relative(source_run_dir / "config.yaml"),
        "source_checkpoint_path": relative(source_checkpoint),
        "source_run_config_path": source_snapshot.get("run", {}).get("config_path"),
        "source_experiment": source_config.experiment,
        "source_name": source_config.name,
        "source_seed": source_config.seed,
        "source_data_target_min": source_config.data.target_min,
        "source_data_target_max": source_config.data.target_max,
        "canonical_run_dir": relative(dest_run_dir),
        "canonical_experiment": eval_config.experiment,
        "canonical_name": eval_config.name,
        "canonical_seed": eval_config.seed,
        "canonical_data_target_min": eval_config.data.target_min,
        "canonical_data_target_max": eval_config.data.target_max,
        "note": (
            "The checkpoint is loaded from source_checkpoint_path. This directory contains "
            "evaluation-only artefacts produced with canonical data.target_min/max."
        ),
    }


def write_source_metadata(run_dir: Path, metadata: dict) -> Path:
    """Write source_run.json."""
    path = run_dir / "source_run.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return path


def write_canonical_config_snapshot(
    *,
    artefacts: ArtefactsManager,
    config: ExperimentConfig,
    source_checkpoint: Path,
    source_run_dir: Path,
    data_paths: dict[str, Path | None],
) -> None:
    """Save an eval-oriented config snapshot without pretending to own training weights."""
    snapshot = {
        "run": {
            "run_name": artefacts.run_name,
            "created_at": artefacts.created_at,
            "config_path": None,
            "input_checkpoint_path": relative(source_checkpoint),
            "output_checkpoint_path": None,
            "source_run_dir": relative(source_run_dir),
            "run_log": relative(artefacts.paths.run_log),
        },
        "data_paths": {key: relative(path) for key, path in data_paths.items()},
        "config": config.model_dump(mode="json"),
    }
    with artefacts.paths.config_snapshot.open("w", encoding="utf-8") as f:
        yaml.safe_dump(snapshot, f, sort_keys=False)


def reeval_canonical_run(run_dir: Path, *, device: torch.device, args: argparse.Namespace) -> ArtefactsManager:
    """Run canonical re-evaluation for one source run."""
    source_run_dir = resolve_path(run_dir, PROJECT_ROOT)
    if source_run_dir is None:
        raise ValueError("run_dir cannot be None.")
    source_run_dir = source_run_dir.resolve()
    source_snapshot, source_config = load_run_snapshot(source_run_dir)
    source_summary = load_run_summary(source_run_dir)
    eval_config = apply_canonical_overrides(source_config, args)
    if eval_config.data.test is None:
        raise RuntimeError("Canonical reeval requires data.test in the source config.")

    source_checkpoint = source_run_dir / "best_model.pth"
    if not source_checkpoint.exists():
        raise FileNotFoundError(f"Missing source checkpoint: {source_checkpoint}")

    out_root = canonical_output_root(source_config, args).resolve()
    dest_run_dir = canonical_run_dir(source_run_dir, out_root)
    dest_run_dir.mkdir(parents=True, exist_ok=True)

    model = eval_config.model.build().to(device)
    model.load_state_dict(torch.load(source_checkpoint, map_location=device))
    model.to(device)

    data_paths = eval_config.data_paths(PROJECT_ROOT)
    artefacts = ArtefactsManager.open_existing(
        run_dir=dest_run_dir,
        config=eval_config,
        project_root=PROJECT_ROOT,
        input_checkpoint_path=source_checkpoint,
        data_paths=data_paths,
        root_dir=out_root,
    )
    artefacts.run_name = canonical_summary_run_name(source_run_dir)
    artefacts.summary = artefacts._base_summary(status="canonical_reeval_started")
    artefacts.summary["checkpoint"] = relative(source_checkpoint)
    artefacts.save_model_summary(model)
    write_canonical_config_snapshot(
        artefacts=artefacts,
        config=eval_config,
        source_checkpoint=source_checkpoint,
        source_run_dir=source_run_dir,
        data_paths=data_paths,
    )
    source_info = source_metadata(
        source_run_dir=source_run_dir,
        source_snapshot=source_snapshot,
        source_config=source_config,
        eval_config=eval_config,
        source_checkpoint=source_checkpoint,
        dest_run_dir=dest_run_dir,
    )
    source_info_path = write_source_metadata(dest_run_dir, source_info)
    artefacts.save_summary(
        status="canonical_reeval_started",
        source_run_dir=relative(source_run_dir),
        source_checkpoint=relative(source_checkpoint),
        source_run_metadata=relative(source_info_path),
        canonical_target_min=eval_config.data.target_min,
        canonical_target_max=eval_config.data.target_max,
    )

    with tee_output(artefacts.paths.run_log):
        print("\n=== Canonical re-evaluation ===")
        print(f"Source run: {path_text(source_run_dir)}")
        print(f"Canonical run: {path_text(dest_run_dir)}")
        print(f"Device: {device}")
        print(f"Task: {eval_config.task}")
        print(f"Loaded checkpoint: {path_text(source_checkpoint)}")
        print(
            "Source data target range: "
            f"[{source_config.data.target_min}, {source_config.data.target_max}]"
        )
        print(
            "Canonical data target range: "
            f"[{eval_config.data.target_min}, {eval_config.data.target_max}]"
        )
        print(f"Temperature calibration: {eval_config.calibration.temperature.enabled}")

        valid_dataset = eval_config.build_dataset("valid", PROJECT_ROOT)
        valid_metadata = valid_dataset.get_metadata()
        valid_std = float(valid_metadata["target"].std())
        default_rmse_by_split = {"valid": valid_std}

        channels_list = np.arange(model.n_chans) if hasattr(model, "n_chans") else None
        valid_dataset_for_loader = build_eval_dataset(eval_config, valid_dataset, channels_list)
        valid_loader = DataLoader(valid_dataset_for_loader, **eval_config.loaders.valid.to_kwargs())
        print(f"\nRecomputing valid: rows={len(valid_dataset_for_loader):,}, batches={len(valid_loader):,}")

        trainer = build_eval_trainer(
            config=eval_config,
            model=model,
            valid_loader=valid_loader,
            default_rmse_by_split=default_rmse_by_split,
            channels_list=channels_list,
            artefacts=artefacts,
            device=device,
        )
        valid_metrics = trainer.run_valid_epoch(0)
        trainer.best_epoch = source_summary.get("best_epoch")
        trainer.best_metric = float(valid_metrics[eval_config.trainer.monitor])
        trainer.best_valid_metrics = valid_metrics
        artefacts.save_best_validation_predictions(trainer, valid_metadata)

        print(f"\nRecomputing {eval_config.evaluation.holdout_split} holdout")
        holdout_metrics = run_holdout_evaluation(
            config=eval_config,
            trainer=trainer,
            model=model,
            channels_list=channels_list,
            valid_metadata=valid_metadata,
            output_checkpoint_path=source_checkpoint,
            default_rmse_by_split=default_rmse_by_split,
            artefacts=artefacts,
            device=device,
            verbose=False,
        )
        updates = {
            "status": "canonical_reevaluated",
            "canonical_reevaluated_at": now_utc_iso(),
            "reeval_valid_nrmse": valid_metrics.get("nrmse"),
            "source_run_dir": relative(source_run_dir),
            "source_checkpoint": relative(source_checkpoint),
            "checkpoint": relative(source_checkpoint),
            "canonical_target_min": eval_config.data.target_min,
            "canonical_target_max": eval_config.data.target_max,
        }
        if holdout_metrics is not None:
            updates["reeval_holdout_nrmse"] = holdout_metrics.get("nrmse")
        artefacts.save_summary(**updates)

        print("\nCanonical metrics")
        print(f"valid_nrmse: {float(valid_metrics.get('nrmse')):.6f}")
        if holdout_metrics is not None:
            print(f"{eval_config.evaluation.holdout_split}_nrmse: {float(holdout_metrics.get('nrmse')):.6f}")
        if eval_config.calibration.temperature.enabled:
            tau = artefacts.summary.get("calibration_temperature")
            tau_nrmse = artefacts.summary.get(f"{eval_config.evaluation.holdout_split}_tau_nrmse")
            print(f"tau: {tau}")
            print(f"{eval_config.evaluation.holdout_split}_tau_nrmse: {tau_nrmse}")
        print(f"Saved canonical config: {path_text(artefacts.paths.config_snapshot)}")
        print(f"Saved source metadata: {path_text(source_info_path)}")
        print(f"Saved canonical summary: {path_text(artefacts.paths.run_summary)}")

    return artefacts


def main(argv: list[str] | None = None) -> int:
    """Run the canonical re-evaluation CLI."""
    args = build_parser().parse_args(argv)
    if args.enable_temperature and args.disable_temperature:
        raise SystemExit("--enable-temperature and --disable-temperature cannot be used together.")
    device = choose_device(args.device)
    reeval_canonical_run(args.run_dir, device=device, args=args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
