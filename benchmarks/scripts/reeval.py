"""Re-evaluate an existing benchmark run from its best checkpoint."""

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
import torch
import yaml
from torch.utils.data import DataLoader

from benchmarks.pkg.artefacts_manager import ArtefactsManager, now_utc_iso
from benchmarks.pkg.config import ExperimentConfig, resolve_path
from benchmarks.scripts.run import (
    build_eval_dataset,
    choose_device,
    path_text,
    run_holdout_evaluation,
    tee_output,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the re-evaluation CLI parser."""
    parser = argparse.ArgumentParser(description="Refresh holdout evaluation for an existing benchmark run.")
    parser.add_argument("run_dir", type=Path, help="Existing run directory with config.yaml and best_model.pth.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto", help="Torch device selection.")
    parser.add_argument("--enable-temperature", action="store_true", help="Enable post-hoc temperature calibration.")
    parser.add_argument("--temperature-min", type=float, default=None, help="Override temperature grid minimum.")
    parser.add_argument("--temperature-max", type=float, default=None, help="Override temperature grid maximum.")
    parser.add_argument("--temperature-step", type=float, default=None, help="Override temperature grid step.")
    parser.add_argument("--show-recompute-check", action="store_true", help="Print previous vs recomputed base metrics.")
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


def snapshot_path(snapshot: dict, key: str) -> Path | None:
    """Resolve an optional path from the snapshot run section."""
    value = snapshot.get("run", {}).get(key)
    return resolve_path(value, PROJECT_ROOT) if value else None


def apply_cli_overrides(config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    """Apply re-evaluation CLI overrides to the snapshot config."""
    temperature_updates = {}
    if args.enable_temperature:
        if config.task != "segmentation":
            raise RuntimeError("--enable-temperature is only supported for segmentation runs.")
        temperature_updates["enabled"] = True
    if args.temperature_min is not None:
        temperature_updates["min"] = args.temperature_min
    if args.temperature_max is not None:
        temperature_updates["max"] = args.temperature_max
    if args.temperature_step is not None:
        temperature_updates["step"] = args.temperature_step

    if not temperature_updates:
        return config

    temperature = config.calibration.temperature.model_copy(update=temperature_updates)
    calibration = config.calibration.model_copy(update={"temperature": temperature})
    return config.model_copy(update={"calibration": calibration})


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


def format_value(value, *, digits: int = 6) -> str:
    """Format an optional scalar for compact comparison output."""
    if value is None:
        return "none"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def metric_change(label: str, before, after, *, digits: int = 6) -> str:
    """Format one previous-to-current metric line."""
    text = f"{label}: {format_value(before, digits=digits)} -> {format_value(after, digits=digits)}"
    if before is None and after is not None:
        return f"{text} (new)"
    if before is not None and after is not None:
        try:
            return f"{text} ({float(after) - float(before):+.{digits}f})"
        except (TypeError, ValueError):
            return text
    return text


def configured_eval_temperature(config: ExperimentConfig) -> float | None:
    """Return the configured eval softmax temperature when present."""
    params = config.trainer.params
    return params.get("eval_temperature", params.get("temperature"))


def reeval_run(run_dir: Path, *, device: torch.device, args: argparse.Namespace) -> ArtefactsManager:
    """Refresh validation-derived outputs and holdout metrics for one run."""
    run_dir = resolve_path(run_dir, PROJECT_ROOT)
    if run_dir is None:
        raise ValueError("run_dir cannot be None.")
    run_dir = run_dir.resolve()
    snapshot, config = load_run_snapshot(run_dir)
    config = apply_cli_overrides(config, args)
    if not config.evaluation.holdout_eval:
        raise RuntimeError("Snapshot config has evaluation.holdout_eval=false; reeval expects a holdout split.")
    if config.data.test is None:
        raise RuntimeError("Snapshot config has no data.test path.")

    model = config.model.build().to(device)
    checkpoint_path = run_dir / "best_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing best checkpoint: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)

    data_paths = config.data_paths(PROJECT_ROOT)
    artefacts = ArtefactsManager.open_existing(
        run_dir=run_dir,
        config=config,
        project_root=PROJECT_ROOT,
        config_path=snapshot_path(snapshot, "config_path"),
        input_checkpoint_path=snapshot_path(snapshot, "input_checkpoint_path"),
        data_paths=data_paths,
    )
    previous_summary = dict(artefacts.summary)
    artefacts.save_config_snapshot()
    artefacts.clear_evaluation_summary(
        splits=(config.evaluation.holdout_split, f"{config.evaluation.holdout_split}_tau")
    )

    with tee_output(artefacts.paths.run_log):
        print("\n=== Re-evaluation ===")
        print(f"Run directory: {path_text(run_dir)}")
        print(f"Device: {device}")
        print(f"Task: {config.task}")
        print(f"Temperature calibration: {config.calibration.temperature.enabled}")
        print(f"Loaded checkpoint: {path_text(checkpoint_path)}")
        split = config.evaluation.holdout_split
        if args.show_recompute_check:
            print("\nPrevious saved base metrics")
            print(f"valid_nrmse: {format_value(previous_summary.get('best_metric'))}")
            print(f"{split}_nrmse: {format_value(previous_summary.get(f'{split}_nrmse'))}")
        if previous_summary.get("calibration_temperature") is not None:
            print(f"previous_fitted_tau: {format_value(previous_summary.get('calibration_temperature'), digits=4)}")
            print(f"previous_{split}_tau_nrmse: {format_value(previous_summary.get(f'{split}_tau_nrmse'))}")

        valid_dataset = config.build_dataset("valid", PROJECT_ROOT)
        valid_metadata = valid_dataset.get_metadata()
        valid_std = float(valid_metadata["target"].std())
        default_rmse_by_split = {"valid": valid_std}

        channels_list = np.arange(model.n_chans) if hasattr(model, "n_chans") else None
        valid_dataset_for_loader = build_eval_dataset(config, valid_dataset, channels_list)
        valid_loader = DataLoader(valid_dataset_for_loader, **config.loaders.valid.to_kwargs())
        print(f"\nRecomputing valid: rows={len(valid_dataset_for_loader):,}, batches={len(valid_loader):,}")

        trainer = build_eval_trainer(
            config=config,
            model=model,
            valid_loader=valid_loader,
            default_rmse_by_split=default_rmse_by_split,
            channels_list=channels_list,
            artefacts=artefacts,
            device=device,
        )
        valid_metrics = trainer.run_valid_epoch(0)
        trainer.best_epoch = artefacts.summary.get("best_epoch")
        trainer.best_metric = float(valid_metrics[config.trainer.monitor])
        trainer.best_valid_metrics = valid_metrics
        artefacts.save_best_validation_predictions(trainer, valid_metadata)

        artefacts.save_summary(
            status="reevaluating",
            reevaluated_at=now_utc_iso(),
            reeval_valid_nrmse=valid_metrics.get("nrmse"),
        )
        print(f"\nRecomputing {config.evaluation.holdout_split} holdout")
        holdout_metrics = run_holdout_evaluation(
            config=config,
            trainer=trainer,
            model=model,
            channels_list=channels_list,
            valid_metadata=valid_metadata,
            output_checkpoint_path=checkpoint_path,
            default_rmse_by_split=default_rmse_by_split,
            artefacts=artefacts,
            device=device,
            verbose=False,
        )
        updates = {
            "status": "reevaluated",
            "reevaluated_at": now_utc_iso(),
            "reeval_valid_nrmse": valid_metrics.get("nrmse"),
        }
        if holdout_metrics is not None:
            updates["reeval_holdout_nrmse"] = holdout_metrics.get("nrmse")
        artefacts.save_summary(**updates)

        if args.show_recompute_check:
            print("\nRecompute check")
            print(metric_change("valid_nrmse", previous_summary.get("best_metric"), valid_metrics.get("nrmse")))
            if holdout_metrics is not None:
                print(metric_change(f"{split}_nrmse", previous_summary.get(f"{split}_nrmse"), holdout_metrics.get("nrmse")))
        if config.calibration.temperature.enabled:
            print("\nPost-hoc temperature calibration")
            print(metric_change("tau", configured_eval_temperature(config), artefacts.summary.get("calibration_temperature"), digits=4))
            print(metric_change("valid_nrmse", valid_metrics.get("nrmse"), artefacts.summary.get("calibration_temperature_valid_nrmse")))
            print(metric_change(f"{split}_nrmse", holdout_metrics.get("nrmse") if holdout_metrics is not None else None, artefacts.summary.get(f"{split}_tau_nrmse")))
        else:
            print("\nBase metrics")
            print(f"valid_nrmse: {format_value(valid_metrics.get('nrmse'))}")
            if holdout_metrics is not None:
                print(f"{split}_nrmse: {format_value(holdout_metrics.get('nrmse'))}")
        print(f"Updated config snapshot: {path_text(artefacts.paths.config_snapshot)}")
        print(f"Updated run summary: {path_text(artefacts.paths.run_summary)}")
        print(f"Global summary: {path_text(artefacts.paths.summary_md)}")

    return artefacts


def main(argv: list[str] | None = None) -> int:
    """Run the re-evaluation CLI."""
    args = build_parser().parse_args(argv)
    device = choose_device(args.device)
    reeval_run(args.run_dir, device=device, args=args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
