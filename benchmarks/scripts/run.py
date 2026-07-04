"""Command-line runner matching benchmarks/prod.ipynb."""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

os.environ["MNE_DONTWRITE_HOME"] = "true"
os.environ["NUMBA_DISABLE_JIT"] = "1"
os.environ["MPLCONFIGDIR"] = "/tmp/neurosned-matplotlib"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_OUTPUT_DIR = Path("benchmarks/experiments")
GPU_MONITOR_INTERVAL_SEC = 10.0

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from benchmarks.pkg.artefacts_manager import ArtefactsManager
from benchmarks.pkg.config import load_experiment_config, resolve_path
from benchmarks.pkg.evaluation.factory import build_dataset_wrapper
from benchmarks.pkg.evaluation.runner import run_holdout_evaluation
from benchmarks.pkg.gpu import GpuMonitor
from benchmarks.pkg.runtime import choose_device, path_text, tee_output
from benchmarks.pkg.training import ReloadBestOnPlateau
from benchmarks.pkg.utils import set_seed


def build_parser() -> argparse.ArgumentParser:
    """Build the benchmark runner CLI parser."""
    parser = argparse.ArgumentParser(description="Run one benchmark YAML config.")
    parser.add_argument("config", type=Path, help="Path to experiment YAML config.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Artefact root directory.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto", help="Torch device selection.")
    parser.add_argument("--show-plots", action="store_true", help="Show diagnostic plots in addition to saving them.")
    parser.add_argument("--skip-initial-validation", action="store_true", help="Skip validation before training.")
    return parser


def target_summary(name: str, metadata) -> dict[str, float | int | str]:
    """Summarize target metadata for one split."""
    target = metadata["target"]
    return {
        "split": name,
        "rows": len(metadata),
        "target_mean": float(target.mean()),
        "target_std": float(target.std()),
        "target_min": float(target.min()),
        "target_max": float(target.max()),
    }


def stage_name(stage) -> str:
    """Return a readable stage name."""
    return stage.name if stage is not None else "default"


def stage_train_dataset(config, base_dataset, channels_list, stage):
    """Build the train dataset wrapper for one stage."""
    if config.data.train_dataset is None:
        return base_dataset
    params = dict(config.data.train_dataset.params)
    if stage is not None:
        params.update(stage.train_dataset_params)
    return build_dataset_wrapper(config.data.train_dataset, base_dataset, channels_list, params=params)


def stage_train_loader_config(config, stage):
    """Return the train DataLoader config for one stage."""
    if stage is not None and stage.train_loader is not None:
        return stage.train_loader
    return config.loaders.train


def stage_optimizer_config(config, stage):
    """Return the optimizer config for one stage."""
    if stage is not None and stage.optimizer is not None:
        return stage.optimizer
    return config.optimizer


def stage_plateau_config(config, stage):
    """Return the plateau config for one stage."""
    if stage is not None and stage.plateau is not None:
        return stage.plateau
    return config.trainer.plateau


def stage_trainer_params(config, stage) -> dict:
    """Merge global trainer params with stage overrides."""
    params = dict(config.trainer.params)
    if stage is not None:
        params.update(stage.params)
    return params


def stage_patience(config, stage):
    """Return early-stopping patience for one stage."""
    if stage is not None and stage.early_stopping_patience is not None:
        return stage.early_stopping_patience
    return config.trainer.early_stopping_patience


def stage_print_batch_stats(config, stage) -> bool:
    """Return progress printing setting for one stage."""
    if stage is not None and stage.print_batch_stats is not None:
        return stage.print_batch_stats
    return config.trainer.print_batch_stats


def reload_checkpoint_path(output_checkpoint_path: Path, input_checkpoint_path: Path | None) -> Path:
    """Return the best available checkpoint for stage reload."""
    if output_checkpoint_path.exists():
        return output_checkpoint_path
    if input_checkpoint_path is not None and input_checkpoint_path.exists():
        return input_checkpoint_path
    raise FileNotFoundError("Stage requested reload='best', but no checkpoint is available.")


def build_stage_trainer(
    *,
    config,
    stage,
    model,
    train_dataset,
    valid_loader,
    default_rmse_by_split: dict[str, float],
    channels_list,
    output_checkpoint_path: Path,
    input_checkpoint_path: Path | None,
    device,
    show_plots: bool,
    artefacts: ArtefactsManager,
    epoch_offset: int,
    completed_history: list[dict],
):
    """Build trainer, optimizer and train loader for one stage."""
    name = stage_name(stage)
    if stage is not None and stage.reload == "best":
        reload_path = reload_checkpoint_path(output_checkpoint_path, input_checkpoint_path)
        print(f"Stage {name}: reload best checkpoint {path_text(reload_path)}")
        model.load_state_dict(torch.load(reload_path, map_location=device))

    train_dataset_for_loader = stage_train_dataset(config, train_dataset, channels_list, stage)
    train_loader_config = stage_train_loader_config(config, stage)
    train_loader = DataLoader(train_dataset_for_loader, **train_loader_config.to_kwargs())

    optimizer_config = stage_optimizer_config(config, stage)
    optimizer_cls = optimizer_config.load_class()
    optimizer = optimizer_cls(model.parameters(), **optimizer_config.params)
    optimizer_kwargs = {key: value for key, value in optimizer_config.params.items() if key != "lr"}

    plateau_config = stage_plateau_config(config, stage)
    plateau_scheduler = None
    if plateau_config.enabled:
        plateau_scheduler = ReloadBestOnPlateau(
            optimizer_factory=optimizer_cls,
            lr=optimizer_config.params["lr"],
            factor=plateau_config.factor,
            optimizer_kwargs=optimizer_kwargs,
            fallback_checkpoint_path=input_checkpoint_path,
            max_restarts=plateau_config.max_restarts,
        )

    trainer_cls = config.trainer.load_class()
    trainer_params = stage_trainer_params(config, stage)
    trainer_kwargs = {
        "model": model,
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "optimizer": optimizer,
        "device": device,
        "n_epochs": stage.n_epochs if stage is not None else config.trainer.n_epochs,
        "checkpoint_path": output_checkpoint_path,
        "monitor": config.trainer.monitor,
        "minimize": config.trainer.minimize,
        "early_stopping_patience": stage_patience(config, stage),
        "plateau_scheduler": plateau_scheduler,
        "print_batch_stats": stage_print_batch_stats(config, stage),
        "on_epoch_end": lambda trainer, stage_history: artefacts.save_epoch(trainer, completed_history + stage_history),
        "stage_name": name if stage is not None else None,
        "epoch_offset": epoch_offset,
        "seconds_offset": completed_history[-1]["cumulative_training_seconds"] if completed_history else 0.0,
        "default_rmse_by_split": default_rmse_by_split,
    }
    if config.task == "segmentation":
        trainer_kwargs["channels_list"] = channels_list
        trainer_params.setdefault("plot_save_dir", artefacts.paths.figures_dir)
        trainer_params.setdefault("plot_show", show_plots)

    trainer = trainer_cls(**trainer_kwargs, **trainer_params)
    return trainer, train_loader, train_dataset_for_loader, optimizer_config, plateau_config


def run_config(
    config_path: Path,
    *,
    output_dir: Path,
    device: torch.device,
    show_plots: bool,
    skip_initial_validation: bool,
    seed_override: int | None = None,
    name_suffix: str | None = None,
) -> ArtefactsManager:
    """Run one benchmark config and return its artefacts manager."""
    config_path = config_path.resolve()
    config = load_experiment_config(config_path)
    updates = {}
    if seed_override is not None:
        updates["seed"] = int(seed_override)
    if name_suffix is not None:
        updates["name"] = f"{config.name}_{name_suffix}"
    if updates:
        config = config.model_copy(update=updates)
    set_seed(config.seed)

    train_dataset, valid_dataset = config.build_datasets(PROJECT_ROOT)
    data_paths = config.data_paths(PROJECT_ROOT)

    meta_information = train_dataset.get_metadata()
    meta_information_valid = valid_dataset.get_metadata()
    summary = pd.DataFrame(
        [
            target_summary("train", meta_information),
            target_summary("valid", meta_information_valid),
        ]
    )
    default_rmse_train = meta_information["target"].std()
    default_rmse_valid = meta_information_valid["target"].std()
    default_rmse_by_split = {
        "train": float(default_rmse_train),
        "valid": float(default_rmse_valid),
    }

    model = config.model.build().to(device)
    input_checkpoint_path = resolve_path(config.trainer.checkpoint.input, PROJECT_ROOT)

    checkpoint_loaded = False
    if input_checkpoint_path is not None and input_checkpoint_path.exists():
        model.load_state_dict(torch.load(input_checkpoint_path, map_location=device))
        checkpoint_loaded = True
    model.to(device)

    artefacts = ArtefactsManager.create(
        config=config,
        project_root=PROJECT_ROOT,
        model=model,
        config_path=config_path,
        input_checkpoint_path=input_checkpoint_path,
        data_paths=data_paths,
        root_dir=output_dir,
    )
    output_checkpoint_path = artefacts.checkpoint_path

    total_params = sum(p.numel() for p in model.parameters())
    mb_total = total_params * 4 / 1024**2

    with tee_output(artefacts.paths.run_log):
        try:
            print(f"Project root: {PROJECT_ROOT}")
            print(f"Config path: {path_text(config_path)}")
            print(f"Device: {device}")
            print(f"Torch: {torch.__version__}")
            print(f"Experiment: {config.experiment}/{config.name}")
            print(f"Task: {config.task}")
            print(f"Seed: {config.seed}")
            print(f"Model: {config.model.module_name}.{config.model.class_name}")
            print(f"Trainer: {config.trainer.module_name}.{config.trainer.class_name}")
            print(f"Optimizer: {config.optimizer.module_name}.{config.optimizer.class_name}")
            print(f"Train path: {path_text(data_paths['train'])}")
            print(f"Valid path: {path_text(data_paths['valid'])}")
            if data_paths["test"] is not None:
                print(f"Test path: {path_text(data_paths['test'])}")
            print(f"Train windows: {len(train_dataset):,}")
            print(f"Valid windows: {len(valid_dataset):,}")
            print(summary.to_string(index=False))
            print(f"Default RMSE denominators: train={default_rmse_train:.4f}, valid={default_rmse_valid:.4f}")
            print(f"Model class: {config.model.class_name}")
            print(f"Parameters: {total_params:,} ({mb_total:.2f} MB float32)")
            print(f"Input checkpoint: {path_text(input_checkpoint_path)}")
            print(f"Checkpoint loaded: {checkpoint_loaded}")
            print(f"Run directory: {path_text(artefacts.run_dir)}")
            print(f"Output checkpoint: {path_text(output_checkpoint_path)}")
            print(f"Run log: {path_text(artefacts.paths.run_log)}")

            channels_list = np.arange(model.n_chans) if hasattr(model, "n_chans") else None

            valid_dataset_for_loader = valid_dataset
            if config.data.valid_dataset is not None:
                valid_dataset_for_loader = build_dataset_wrapper(config.data.valid_dataset, valid_dataset, channels_list)

            valid_wrapper = config.data.valid_dataset.class_name if config.data.valid_dataset is not None else "None"

            print(f"Channels: {len(channels_list) if channels_list is not None else 'all'}")
            print(f"Valid wrapper: {valid_wrapper} | rows={len(valid_dataset_for_loader):,}")

            valid_loader = DataLoader(valid_dataset_for_loader, **config.loaders.valid.to_kwargs())

            print(
                f"Valid loader: batches={len(valid_loader):,}, "
                f"batch_size={config.loaders.valid.batch_size}, "
                f"workers={config.loaders.valid.num_workers}, "
                f"shuffle={config.loaders.valid.shuffle}"
            )

            stages = config.trainer.stages or [None]
            completed_history: list[dict] = []
            first_stage = stages[0]
            trainer, train_loader, train_dataset_for_loader, optimizer_config, plateau_config = build_stage_trainer(
                config=config,
                stage=first_stage,
                model=model,
                train_dataset=train_dataset,
                valid_loader=valid_loader,
                default_rmse_by_split=default_rmse_by_split,
                channels_list=channels_list,
                output_checkpoint_path=output_checkpoint_path,
                input_checkpoint_path=input_checkpoint_path,
                device=device,
                show_plots=show_plots,
                artefacts=artefacts,
                epoch_offset=0,
                completed_history=completed_history,
            )

            train_wrapper = config.data.train_dataset.class_name if config.data.train_dataset is not None else "None"
            train_loader_config = stage_train_loader_config(config, first_stage)
            print(f"Train wrapper: {train_wrapper} | rows={len(train_dataset_for_loader):,}")
            print(
                f"Train loader: batches={len(train_loader):,}, "
                f"batch_size={train_loader_config.batch_size}, "
                f"workers={train_loader_config.num_workers}, "
                f"shuffle={train_loader_config.shuffle}"
            )
            print(f"Trainer: {type(trainer).__name__}")
            print(f"Stages: {', '.join(stage_name(stage) for stage in stages)}")
            print(f"Monitor: {trainer.monitor} | minimize={trainer.minimize}")
            print(f"Optimizer: {optimizer_config.class_name} | params={optimizer_config.params}")
            print(
                f"Plateau: enabled={plateau_config.enabled}, "
                f"factor={plateau_config.factor}, max_restarts={plateau_config.max_restarts}"
            )

            initial_valid_metrics = None
            if not skip_initial_validation and input_checkpoint_path is not None and input_checkpoint_path.exists():
                initial_valid_metrics = trainer.run_valid_epoch(0)
                trainer.best_metric = initial_valid_metrics[config.trainer.monitor]
                trainer.best_epoch = 0
                trainer.best_valid_metrics = initial_valid_metrics
                trainer.save_checkpoint()
                print(f"Initial {config.trainer.monitor}: {trainer.best_metric:.6f}")
            else:
                print("Skipped initial validation.")
            artefacts.save_initial_validation(initial_valid_metrics)

            artefacts.start_training()
            gpu_monitor = None
            if device.type == "cuda":
                gpu_index = device.index if device.index is not None else torch.cuda.current_device()
                gpu_monitor = GpuMonitor(
                    gpu_index=gpu_index,
                    csv_path=artefacts.paths.gpu_metrics,
                    plot_path=artefacts.paths.gpu_plot,
                    interval_sec=GPU_MONITOR_INTERVAL_SEC,
                )
                print(f"GPU monitoring: every {GPU_MONITOR_INTERVAL_SEC:.0f}s -> {path_text(artefacts.paths.gpu_metrics)}")
                gpu_monitor.start()

            try:
                best_metric = trainer.best_metric
                best_epoch = trainer.best_epoch
                best_valid_metrics = trainer.best_valid_metrics
                for stage_index, stage in enumerate(stages):
                    if stage_index > 0:
                        trainer, train_loader, train_dataset_for_loader, optimizer_config, plateau_config = build_stage_trainer(
                            config=config,
                            stage=stage,
                            model=model,
                            train_dataset=train_dataset,
                            valid_loader=valid_loader,
                            default_rmse_by_split=default_rmse_by_split,
                            channels_list=channels_list,
                            output_checkpoint_path=output_checkpoint_path,
                            input_checkpoint_path=input_checkpoint_path,
                            device=device,
                            show_plots=show_plots,
                            artefacts=artefacts,
                            epoch_offset=len(completed_history),
                            completed_history=completed_history,
                        )
                        trainer.best_metric = best_metric
                        trainer.best_epoch = best_epoch
                        trainer.best_valid_metrics = best_valid_metrics
                    print(f"Stage {stage_name(stage)}: {trainer.n_epochs} epoch(s), patience={trainer.early_stopping_patience}")
                    print(f"Stage {stage_name(stage)} optimizer: {optimizer_config.class_name} {optimizer_config.params}")
                    stage_loader_config = stage_train_loader_config(config, stage)
                    print(
                        f"Stage {stage_name(stage)} train loader: batches={len(train_loader):,}, "
                        f"batch_size={stage_loader_config.batch_size}, workers={stage_loader_config.num_workers}, "
                        f"shuffle={stage_loader_config.shuffle}"
                    )
                    stage_history = trainer.run()
                    completed_history.extend(stage_history)
                    best_metric = trainer.best_metric
                    best_epoch = trainer.best_epoch
                    best_valid_metrics = trainer.best_valid_metrics
                history = completed_history
            finally:
                if gpu_monitor is not None:
                    gpu_summary = gpu_monitor.stop()
                    artefacts.save_gpu_monitoring(gpu_summary)
                    print(f"Saved GPU monitoring: {path_text(artefacts.paths.gpu_metrics)}")
                    if artefacts.paths.gpu_plot.exists():
                        print(f"Saved GPU plot: {path_text(artefacts.paths.gpu_plot)}")
            artefacts.finish_training(trainer, history)
            best_val_predictions_path = artefacts.save_best_validation_predictions(trainer, meta_information_valid)
            holdout_metrics = run_holdout_evaluation(
                config=config,
                trainer=trainer,
                model=model,
                channels_list=channels_list,
                valid_metadata=meta_information_valid,
                output_checkpoint_path=output_checkpoint_path,
                default_rmse_by_split=default_rmse_by_split,
                artefacts=artefacts,
                device=device,
            )

            print(f"Best {config.trainer.monitor}: {best_metric:.6f}")
            print(f"Best epoch: {best_epoch}")
            print(f"Saved checkpoint: {path_text(output_checkpoint_path)}")
            print(f"Saved metrics: {path_text(artefacts.paths.metrics)}")
            if best_val_predictions_path is not None:
                print(f"Saved best validation predictions: {path_text(best_val_predictions_path)}")
            if holdout_metrics is not None:
                print(f"Saved holdout evaluation: {path_text(artefacts.paths.run_dir / f'{config.evaluation.holdout_split}_metrics.json')}")
            print(f"Run summary: {path_text(artefacts.paths.run_summary)}")

            history_df = pd.DataFrame(history)
            if history_df.empty:
                print("History is empty.")
            else:
                print(history_df.tail().to_string(index=False))
                valid_col = f"valid_{config.trainer.monitor}"
                if valid_col in history_df:
                    best_row = history_df.loc[
                        history_df[valid_col].idxmin() if config.trainer.minimize else history_df[valid_col].idxmax()
                    ]
                    print(f"Best row by {valid_col}: epoch={int(best_row['epoch'])}, value={best_row[valid_col]:.6f}")
            print(f"Global summary: {path_text(artefacts.paths.summary_md)}")
        except KeyboardInterrupt:
            artefacts.save_summary(status="interrupted")
            print("Interrupted by user.")
            raise
        except Exception as exc:
            artefacts.save_summary(status="failed", error=repr(exc))
            traceback.print_exc()
            raise

    return artefacts


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark CLI."""
    args = build_parser().parse_args(argv)
    device = choose_device(args.device)
    run_config(
        args.config,
        output_dir=args.output_dir,
        device=device,
        show_plots=args.show_plots,
        skip_initial_validation=args.skip_initial_validation,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
