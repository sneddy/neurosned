from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sync_if_cuda(device) -> None:
    if getattr(device, "type", str(device)) == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _slug_value(value: Any) -> str:
    return (
        str(value)
        .replace("-", "m")
        .replace(".", "p")
        .replace("+", "p")
        .replace("/", "_")
        .replace(" ", "")
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def model_config_slug(model_config: dict[str, Any]) -> str:
    if "bottleneck_type" in model_config or "drop_path" in model_config:
        depth = model_config.get("depth_per_stage")
        if isinstance(depth, (list, tuple)):
            depth = "-".join(_slug_value(value) for value in depth)
        else:
            depth = _slug_value(depth)
        parts = [
            f"ch{model_config.get('n_chans')}",
            f"c{model_config.get('c0')}",
            f"w{_slug_value(model_config.get('widen'))}",
            f"st{model_config.get('num_stages')}",
            f"d{depth}",
            f"k{model_config.get('k')}",
            f"bn{model_config.get('bottleneck_type')}",
            f"bd{model_config.get('bottleneck_depth')}",
            f"drop{_slug_value(model_config.get('dropout'))}",
            f"dp{_slug_value(model_config.get('drop_path'))}",
            f"sg{int(bool(model_config.get('skip_gating')))}",
        ]
        if "rnn_type" in model_config:
            parts.extend(
                [
                    f"rnn{model_config.get('rnn_type')}",
                    f"bi{int(bool(model_config.get('bidirectional')))}",
                    f"hm{_slug_value(model_config.get('rnn_hidden_mult'))}",
                    f"rpb{model_config.get('rnn_layers_per_block')}",
                    f"brl{model_config.get('bottleneck_rnn_layers')}",
                    f"rdrop{_slug_value(model_config.get('rnn_dropout'))}",
                    f"dwpos{int(bool(model_config.get('use_dwpos')))}",
                ]
            )
        return "_".join(parts)

    if "fm_factors_front" in model_config or "use_stage_fm" in model_config:
        return "_".join(
            [
                f"c{model_config.get('c0')}",
                f"w{model_config.get('widen')}",
                f"st{model_config.get('n_stages')}",
                f"d{model_config.get('depth_per_stage')}",
                f"k{model_config.get('k')}",
                f"drop{_slug_value(model_config.get('dropout'))}",
                f"fmfront{model_config.get('fm_factors_front')}",
                f"stagefm{int(bool(model_config.get('use_stage_fm')))}",
                f"fmstage{model_config.get('fm_factors_stage')}",
            ]
        )

    if {"c0", "widen", "depth_per_stage"}.issubset(model_config):
        return (
            f"c{model_config.get('c0')}_"
            f"w{model_config.get('widen')}_"
            f"d{model_config.get('depth_per_stage')}"
        )

    if "branch_out" in model_config:
        scales = "-".join(_slug_value(value) for value in model_config.get("scales_samples_s", ()))
        pools = "-".join(_slug_value(value) for value in model_config.get("pooling_sizes", ()))
        return "_".join(
            [
                f"br{model_config.get('branch_out')}",
                f"scales{scales}",
                f"pool{pools}",
                f"drop{_slug_value(model_config.get('dropout'))}",
            ]
        )

    keys = ["n_chans", "n_times", "sfreq", "dropout", "out_channels"]
    return "_".join(f"{key}{_slug_value(model_config[key])}" for key in keys if key in model_config)


def build_run_name(
    experiment_name: str,
    model_name: str,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    timestamp: str | None = None,
) -> str:
    timestamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return "__".join(
        [
            experiment_name,
            model_name,
            model_config_slug(model_config),
            f"bs{training_config['batch_size']}",
            f"lr{_slug_value(training_config['initial_lr'])}",
            f"sig{_slug_value(training_config['sigma'])}",
            f"tau{_slug_value(training_config['temperature'])}",
            f"lt{_slug_value(training_config['lambda_time'])}_ce{_slug_value(training_config['lambda_ce'])}",
            timestamp,
        ]
    )


@dataclass
class RunPaths:
    root: Path
    run_dir: Path
    train_img_dir: Path
    val_img_dir: Path
    checkpoint: Path
    config: Path
    metrics: Path
    model_summary: Path
    best_val_predictions: Path
    holdout_predictions: Path
    summary_md: Path
    summary_jsonl: Path
    summary_csv: Path


def _format_md_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if np.isnan(value):
            return ""
        return f"{value:.6f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def update_summary_md(summary: dict[str, Any], summary_jsonl_path: Path, summary_md_path: Path) -> None:
    records = []
    if summary_jsonl_path.exists():
        with summary_jsonl_path.open("r") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    records = [record for record in records if record.get("run_name") != summary.get("run_name")]
    records.append(summary)
    records = sorted(records, key=lambda record: record.get("created_at", ""), reverse=True)

    with summary_jsonl_path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, default=_json_default) + "\n")
    pd.DataFrame(records).to_csv(summary_jsonl_path.with_suffix(".csv"), index=False)

    columns = [
        "created_at",
        "experiment_group",
        "experiment",
        "run_name",
        "model",
        "seed",
        "batch_size",
        "lr",
        "sigma",
        "temperature",
        "train_temperature",
        "eval_temperature",
        "lambda_time",
        "lambda_ce",
        "train_lambda_ce",
        "best_epoch",
        "best_val_nrmse",
        "holdout_nrmse",
        "training_wall_seconds",
        "avg_epoch_seconds",
        "time_to_best_seconds",
        "holdout_seconds",
        "status",
        "run_dir",
    ]
    lines = [
        "# Segmentation Runs Summary",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for record in records:
        lines.append("| " + " | ".join(_format_md_value(record.get(column)) for column in columns) + " |")
    summary_md_path.write_text("\n".join(lines) + "\n")


def write_config(config_path: Path, run_config: dict[str, Any]) -> None:
    with config_path.open("w") as f:
        json.dump(run_config, f, indent=2, default=_json_default)


def create_run(
    *,
    artifacts_dir: Path,
    experiment_name: str,
    experiment_group: str | None = None,
    model,
    model_name: str,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    data_config: dict[str, Any],
    device,
    notebook: str,
    load_initial_checkpoint: bool = False,
    input_checkpoint_path: str | Path | None = None,
) -> tuple[RunPaths, dict[str, Any], dict[str, Any]]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    experiment_group = experiment_group or experiment_name
    created_at = now_utc_iso()
    run_name = build_run_name(experiment_name, model_name, model_config, training_config)
    run_dir = artifacts_dir / run_name
    if run_dir.exists():
        raise FileExistsError(f"Run artifact directory already exists: {run_dir}")
    run_dir.mkdir(parents=False, exist_ok=False)

    paths = RunPaths(
        root=artifacts_dir,
        run_dir=run_dir,
        train_img_dir=run_dir / "img" / "train",
        val_img_dir=run_dir / "img" / "val",
        checkpoint=run_dir / "best_model.pth",
        config=run_dir / "config.json",
        metrics=run_dir / "metrics.csv",
        model_summary=run_dir / "model.txt",
        best_val_predictions=run_dir / "best_val_predictions.csv",
        holdout_predictions=run_dir / "holdout_predictions.csv",
        summary_md=artifacts_dir / "benchmark_summary.md",
        summary_jsonl=artifacts_dir / "benchmark_summary.jsonl",
        summary_csv=artifacts_dir / "benchmark_summary.csv",
    )
    paths.train_img_dir.mkdir(parents=True, exist_ok=True)
    paths.val_img_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "run_name": run_name,
        "experiment_name": experiment_name,
        "experiment_group": experiment_group,
        "created_at": created_at,
        "device": str(device),
        "model_name": model_name,
        "model_config": model_config,
        "training_config": training_config,
        "data_config": data_config,
        "load_initial_checkpoint": load_initial_checkpoint,
        "input_checkpoint_path": input_checkpoint_path,
        "output_checkpoint_path": paths.checkpoint,
        "image_dirs": {"train": paths.train_img_dir, "val": paths.val_img_dir},
        "timing": {"run_created_at": created_at},
        "notebook": notebook,
    }
    write_config(paths.config, run_config)

    with paths.model_summary.open("w") as f:
        f.write(str(model))
        f.write("\n")
        f.write(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    run_summary = {
        "created_at": created_at,
        "run_name": run_name,
        "experiment_group": experiment_group,
        "experiment": experiment_name,
        "model": model_name,
        "seed": training_config.get("seed"),
        "batch_size": training_config["batch_size"],
        "lr": training_config["initial_lr"],
        "sigma": training_config["sigma"],
        "temperature": training_config["temperature"],
        "train_temperature": training_config.get("train_temperature", training_config["temperature"]),
        "eval_temperature": training_config.get("eval_temperature", training_config["temperature"]),
        "lambda_time": training_config["lambda_time"],
        "lambda_ce": training_config["lambda_ce"],
        "train_lambda_ce": training_config.get("train_lambda_ce", training_config["lambda_ce"]),
        "best_epoch": None,
        "best_val_nrmse": None,
        "holdout_nrmse": None,
        "training_started_at": None,
        "training_finished_at": None,
        "training_wall_seconds": None,
        "avg_epoch_seconds": None,
        "time_to_best_seconds": None,
        "holdout_seconds": None,
        "status": "started",
        "run_dir": str(run_dir.relative_to(artifacts_dir.parent)),
    }
    update_summary_md(run_summary, paths.summary_jsonl, paths.summary_md)
    return paths, run_config, run_summary


def append_metrics(metrics_path: Path, history: list[dict[str, Any]]) -> None:
    pd.DataFrame(history).to_csv(metrics_path, index=False)


def save_predictions(path: Path, preds, metadata) -> None:
    if len(preds) != len(metadata):
        np.save(path.with_suffix(".npy"), preds)
        return

    frame = {
        "prediction": preds,
        "target": metadata["target"].to_numpy(),
    }
    if "subject" in metadata:
        frame["subject"] = metadata["subject"].to_numpy()
    pd.DataFrame(frame).to_csv(path, index=False)


def save_summary(summary: dict[str, Any], paths: RunPaths) -> None:
    update_summary_md(summary, paths.summary_jsonl, paths.summary_md)


def finish_run(
    *,
    paths: RunPaths,
    run_config: dict[str, Any],
    run_summary: dict[str, Any],
    epoch_history: list[dict[str, Any]],
    best_epoch: int | None,
    best_rmse: float,
    training_wall_seconds: float,
) -> None:
    avg_epoch_seconds = training_wall_seconds / len(epoch_history) if epoch_history else None
    finished_at = now_utc_iso()

    run_config["timing"].update(
        {
            "training_finished_at": finished_at,
            "training_wall_seconds": training_wall_seconds,
            "avg_epoch_seconds": avg_epoch_seconds,
            "best_epoch_finished_at": run_summary.get("best_epoch_finished_at"),
            "time_to_best_seconds": run_summary.get("time_to_best_seconds"),
        }
    )
    write_config(paths.config, run_config)

    run_summary.update(
        {
            "best_epoch": best_epoch,
            "best_val_nrmse": float(best_rmse) if best_rmse < float("inf") else None,
            "training_finished_at": finished_at,
            "training_wall_seconds": training_wall_seconds,
            "avg_epoch_seconds": avg_epoch_seconds,
            "status": "training_finished",
        }
    )
    save_summary(run_summary, paths)
