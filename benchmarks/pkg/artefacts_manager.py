"""Artefact output management for benchmark experiments."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from benchmarks.pkg.config import ExperimentConfig, resolve_path


def now_utc_iso() -> str:
    """Return a compact UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _safe_slug(value: Any) -> str:
    text = str(value).strip().replace(".", "p")
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", text)
    return text.strip("_") or "run"


def _scalar_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if key not in {"preds_abs", "preds", "diffs", "logits"}
    }


def _relative(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _prediction_key(metrics: dict[str, Any]) -> str | None:
    for key in ("preds_abs", "preds"):
        if key in metrics:
            return key
    return None


def _subject_bootstrap_nrmse(predictions, metadata, *, n_samples: int, resampling_seed: int) -> dict[str, Any]:
    """Return a subject-level bootstrap interval for NRMSE."""
    frame = pd.DataFrame(
        {
            "subject": metadata["subject"].to_numpy() if "subject" in metadata else np.arange(len(metadata)),
            "target": metadata["target"].to_numpy(),
            "prediction": np.asarray(predictions),
        }
    )
    subjects = frame["subject"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(resampling_seed)
    values = []
    grouped = {subject: group for subject, group in frame.groupby("subject", sort=False)}

    for _ in range(n_samples):
        sampled_subjects = rng.choice(subjects, size=len(subjects), replace=True)
        sample = pd.concat([grouped[subject] for subject in sampled_subjects], ignore_index=True)
        target = sample["target"].to_numpy()
        prediction = sample["prediction"].to_numpy()
        rmse = float(np.sqrt(np.mean((prediction - target) ** 2)))
        denominator = float(np.std(target, ddof=1)) if len(target) > 1 else 0.0
        values.append(rmse / denominator if denominator else rmse)

    values = np.asarray(values)
    return {
        "method": "subject_bootstrap",
        "n_samples": int(n_samples),
        "resampling_seed": int(resampling_seed),
        "n_subjects": int(len(subjects)),
        "n_rows": int(len(frame)),
        "nrmse_mean": float(np.mean(values)),
        "nrmse_ci_low": float(np.quantile(values, 0.025)),
        "nrmse_ci_high": float(np.quantile(values, 0.975)),
    }


@dataclass(frozen=True)
class ArtefactPaths:
    """Filesystem paths owned by one experiment run."""

    root: Path
    run_dir: Path
    checkpoint: Path
    config_snapshot: Path
    run_summary: Path
    metrics: Path
    model_summary: Path
    monitoring_dir: Path
    gpu_metrics: Path
    gpu_plot: Path
    logs_dir: Path
    run_log: Path
    predictions_dir: Path
    figures_dir: Path
    summary_jsonl: Path
    summary_csv: Path
    summary_md: Path


class ArtefactsManager:
    """Own all output files for one benchmark experiment run."""

    def __init__(
        self,
        *,
        config: ExperimentConfig,
        project_root: Path,
        paths: ArtefactPaths,
        run_name: str,
        created_at: str,
        config_path: Path | None = None,
        input_checkpoint_path: Path | None = None,
        data_paths: dict[str, Path | None] | None = None,
    ):
        self.config = config
        self.project_root = project_root
        self.paths = paths
        self.run_name = run_name
        self.created_at = created_at
        self.config_path = config_path
        self.input_checkpoint_path = input_checkpoint_path
        self.data_paths = data_paths or {}
        self.summary: dict[str, Any] = self._base_summary(status="created")
        self._training_started_perf: float | None = None

    @classmethod
    def create(
        cls,
        *,
        config: ExperimentConfig,
        project_root: str | Path,
        model,
        config_path: str | Path | None = None,
        input_checkpoint_path: str | Path | None = None,
        data_paths: dict[str, Path | None] | None = None,
        root_dir: str | Path = "benchmarks/experiments",
    ) -> "ArtefactsManager":
        """Create a run directory and write initial output files."""
        project_root = Path(project_root).resolve()
        root = resolve_path(root_dir, project_root)
        root.mkdir(parents=True, exist_ok=True)
        experiment_root = root / _safe_slug(config.experiment)
        experiment_root.mkdir(parents=True, exist_ok=True)

        created_at = now_utc_iso()
        run_name = cls.build_run_name(config, created_at)
        run_dir = cls._unique_run_dir(experiment_root, run_name)
        paths = ArtefactPaths(
            root=root,
            run_dir=run_dir,
            checkpoint=run_dir / "best_model.pth",
            config_snapshot=run_dir / "config.yaml",
            run_summary=run_dir / "summary.json",
            metrics=run_dir / "metrics.csv",
            model_summary=run_dir / "model.txt",
            monitoring_dir=run_dir / "monitoring",
            gpu_metrics=run_dir / "monitoring" / "gpu.csv",
            gpu_plot=run_dir / "figures" / "gpu_usage.png",
            logs_dir=run_dir / "logs",
            run_log=run_dir / "logs" / "run.log",
            predictions_dir=run_dir / "predictions",
            figures_dir=run_dir / "figures",
            summary_jsonl=root / "summary.jsonl",
            summary_csv=root / "summary.csv",
            summary_md=root / "summary.md",
        )
        run_dir.mkdir(parents=False, exist_ok=False)
        paths.monitoring_dir.mkdir(parents=True, exist_ok=True)
        paths.logs_dir.mkdir(parents=True, exist_ok=True)
        paths.predictions_dir.mkdir(parents=True, exist_ok=True)
        paths.figures_dir.mkdir(parents=True, exist_ok=True)

        manager = cls(
            config=config,
            project_root=project_root,
            paths=paths,
            run_name=run_dir.name,
            created_at=created_at,
            config_path=Path(config_path).resolve() if config_path is not None else None,
            input_checkpoint_path=Path(input_checkpoint_path).resolve() if input_checkpoint_path is not None else None,
            data_paths=data_paths,
        )
        manager.save_config_snapshot()
        manager.save_model_summary(model)
        manager.save_summary(status="created")
        return manager

    @classmethod
    def open_existing(
        cls,
        *,
        run_dir: str | Path,
        config: ExperimentConfig,
        project_root: str | Path,
        config_path: str | Path | None = None,
        input_checkpoint_path: str | Path | None = None,
        data_paths: dict[str, Path | None] | None = None,
        root_dir: str | Path | None = None,
    ) -> "ArtefactsManager":
        """Open an existing run directory for post-training updates."""
        project_root = Path(project_root).resolve()
        run_dir = resolve_path(run_dir, project_root)
        if run_dir is None:
            raise ValueError("run_dir cannot be None.")
        run_dir = run_dir.resolve()
        if root_dir is None:
            root = run_dir.parents[1]
        else:
            root = resolve_path(root_dir, project_root)
            if root is None:
                raise ValueError("root_dir cannot be None.")
            root = root.resolve()

        paths = ArtefactPaths(
            root=root,
            run_dir=run_dir,
            checkpoint=run_dir / "best_model.pth",
            config_snapshot=run_dir / "config.yaml",
            run_summary=run_dir / "summary.json",
            metrics=run_dir / "metrics.csv",
            model_summary=run_dir / "model.txt",
            monitoring_dir=run_dir / "monitoring",
            gpu_metrics=run_dir / "monitoring" / "gpu.csv",
            gpu_plot=run_dir / "figures" / "gpu_usage.png",
            logs_dir=run_dir / "logs",
            run_log=run_dir / "logs" / "run.log",
            predictions_dir=run_dir / "predictions",
            figures_dir=run_dir / "figures",
            summary_jsonl=root / "summary.jsonl",
            summary_csv=root / "summary.csv",
            summary_md=root / "summary.md",
        )
        paths.monitoring_dir.mkdir(parents=True, exist_ok=True)
        paths.logs_dir.mkdir(parents=True, exist_ok=True)
        paths.predictions_dir.mkdir(parents=True, exist_ok=True)
        paths.figures_dir.mkdir(parents=True, exist_ok=True)

        summary = None
        created_at = now_utc_iso()
        if paths.run_summary.exists():
            with paths.run_summary.open("r", encoding="utf-8") as f:
                summary = json.load(f)
            created_at = summary.get("created_at", created_at)

        manager = cls(
            config=config,
            project_root=project_root,
            paths=paths,
            run_name=run_dir.name,
            created_at=created_at,
            config_path=Path(config_path).resolve() if config_path is not None else None,
            input_checkpoint_path=Path(input_checkpoint_path).resolve() if input_checkpoint_path is not None else None,
            data_paths=data_paths,
        )
        if summary is not None:
            manager.summary = summary
        return manager

    @staticmethod
    def build_run_name(config: ExperimentConfig, created_at: str | None = None) -> str:
        """Build a readable run directory name."""
        created_at = created_at or now_utc_iso()
        timestamp = created_at.replace("-", "").replace(":", "").replace("T", "_").replace("Z", "")
        return "__".join(
            [
                _safe_slug(config.name),
                timestamp,
            ]
        )

    @staticmethod
    def _unique_run_dir(root: Path, run_name: str) -> Path:
        run_dir = root / run_name
        if not run_dir.exists():
            return run_dir
        index = 2
        while True:
            candidate = root / f"{run_name}__v{index}"
            if not candidate.exists():
                return candidate
            index += 1

    @property
    def run_dir(self) -> Path:
        """Return the run directory."""
        return self.paths.run_dir

    @property
    def checkpoint_path(self) -> Path:
        """Return the best-model checkpoint path."""
        return self.paths.checkpoint

    def figure_path(self, name: str) -> Path:
        """Return a path inside the figures directory."""
        return self.paths.figures_dir / f"{_safe_slug(name)}.png"

    def save_config_snapshot(self) -> None:
        """Save the resolved experiment config used for this run."""
        snapshot = {
            "run": {
                "run_name": self.run_name,
                "created_at": self.created_at,
                "config_path": _relative(self.config_path, self.project_root),
                "input_checkpoint_path": _relative(self.input_checkpoint_path, self.project_root),
                "output_checkpoint_path": _relative(self.paths.checkpoint, self.project_root),
                "run_log": _relative(self.paths.run_log, self.project_root),
            },
            "data_paths": {key: _relative(path, self.project_root) for key, path in self.data_paths.items()},
            "config": self.config.model_dump(mode="json"),
        }
        with self.paths.config_snapshot.open("w", encoding="utf-8") as f:
            yaml.safe_dump(snapshot, f, sort_keys=False)

    def save_model_summary(self, model) -> None:
        """Save model text representation and parameter count."""
        total_params = sum(param.numel() for param in model.parameters())
        with self.paths.model_summary.open("w", encoding="utf-8") as f:
            f.write(str(model))
            f.write("\n\n")
            f.write(f"Total parameters: {total_params:,}\n")

    def save_initial_validation(self, metrics: dict[str, Any] | None) -> None:
        """Save optional initial validation metrics."""
        if metrics is None:
            self.save_summary(status="initial_validation_skipped")
            return
        metrics = _scalar_metrics(metrics)
        metrics_path = self.paths.run_dir / "initial_validation.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=_json_default)
        monitor_value = metrics.get(self.config.trainer.monitor)
        self.save_summary(status="initial_validation_done", initial_metric=monitor_value)

    def save_metrics(self, history: list[dict[str, Any]]) -> None:
        """Write epoch history to metrics.csv."""
        pd.DataFrame(history).to_csv(self.paths.metrics, index=False)

    def save_epoch(self, trainer, history: list[dict[str, Any]]) -> None:
        """Flush metrics and summary after one completed epoch."""
        self.save_metrics(history)
        metric = trainer.best_metric
        if metric in (float("inf"), -float("inf")):
            metric = None

        last_row = history[-1] if history else {}
        training_wall_seconds = last_row.get("cumulative_training_seconds")
        avg_epoch_seconds = training_wall_seconds / len(history) if training_wall_seconds and history else None

        self.save_summary(
            status="running",
            last_epoch=last_row.get("epoch"),
            best_epoch=trainer.best_epoch,
            best_metric=metric,
            checkpoint_exists=self.paths.checkpoint.exists(),
            history_rows=len(history),
            training_started_at=getattr(trainer, "training_started_at", self.summary.get("training_started_at")),
            training_finished_at=None,
            training_wall_seconds=training_wall_seconds,
            avg_epoch_seconds=avg_epoch_seconds,
            best_epoch_finished_at=getattr(trainer, "best_epoch_finished_at", None),
            time_to_best_seconds=getattr(trainer, "time_to_best_seconds", None),
        )

    def save_predictions(self, name: str, predictions, metadata) -> Path:
        """Save predictions with target metadata when lengths match."""
        path = self.paths.predictions_dir / f"{_safe_slug(name)}.csv"
        predictions = np.asarray(predictions)
        if len(predictions) != len(metadata):
            npy_path = path.with_suffix(".npy")
            np.save(npy_path, predictions)
            return npy_path

        frame: dict[str, Any] = {"row_id": np.arange(len(predictions))}
        for column in ("subject", "session", "run", "target"):
            if column in metadata:
                frame[column] = metadata[column].to_numpy()
        frame["prediction"] = predictions
        pd.DataFrame(frame).to_csv(path, index=False)
        return path

    def save_logits(self, name: str, logits) -> Path:
        """Save dense model logits in NumPy format."""
        path = self.paths.predictions_dir / f"{_safe_slug(name)}.npy"
        np.save(path, np.asarray(logits, dtype=np.float32))
        return path

    def save_best_validation_predictions(self, trainer, metadata) -> Path | None:
        """Save best validation predictions captured by the trainer."""
        metrics = getattr(trainer, "best_valid_metrics", None)
        if not metrics:
            return None
        logits_path = None
        if "logits" in metrics:
            logits_path = self.save_logits("best_logits", metrics["logits"])
        for key in ("preds_abs", "preds"):
            if key in metrics:
                path = self.save_predictions("best_val_predictions", metrics[key], metadata)
                self.save_summary(
                    best_val_predictions=_relative(path, self.project_root),
                    best_logits=_relative(logits_path, self.project_root),
                )
                return path
        if logits_path is not None:
            self.save_summary(best_logits=_relative(logits_path, self.project_root))
        return None

    def save_holdout_evaluation(
        self,
        *,
        split: str,
        metrics: dict[str, Any],
        metadata,
        evaluation,
        checkpoint_loaded: bool,
    ) -> Path:
        """Save holdout metrics, predictions and optional confidence interval."""
        scalar_metrics = _scalar_metrics(metrics)
        metrics_path = self.paths.run_dir / f"{_safe_slug(split)}_metrics.json"
        prediction_path = None
        logits_path = None
        ci_path = None
        ci_metrics = None

        prediction_key = _prediction_key(metrics)
        if evaluation.save_predictions and prediction_key is not None:
            prediction_path = self.save_predictions(f"{split}_predictions", metrics[prediction_key], metadata)

        if evaluation.save_logits and "logits" in metrics:
            logits_path = self.save_logits(f"{split}_logits", metrics["logits"])

        ci_config = evaluation.confidence_interval
        if ci_config.enabled and prediction_key is not None:
            ci_metrics = _subject_bootstrap_nrmse(
                metrics[prediction_key],
                metadata,
                n_samples=ci_config.n_samples,
                resampling_seed=ci_config.resampling_seed,
            )
            ci_path = self.paths.run_dir / f"{_safe_slug(split)}_ci.json"
            with ci_path.open("w", encoding="utf-8") as f:
                json.dump(ci_metrics, f, indent=2, default=_json_default)

        output = {
            "split": split,
            "checkpoint_loaded": checkpoint_loaded,
            "metrics": scalar_metrics,
            "predictions": _relative(prediction_path, self.project_root),
            "logits": _relative(logits_path, self.project_root),
            "confidence_interval": ci_metrics,
        }
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, default=_json_default)

        summary_updates = {
            f"{split}_{key}": value for key, value in scalar_metrics.items()
        }
        summary_updates.update(
            {
                f"{split}_metrics": _relative(metrics_path, self.project_root),
                f"{split}_predictions": _relative(prediction_path, self.project_root),
                f"{split}_logits": _relative(logits_path, self.project_root),
                f"{split}_ci": _relative(ci_path, self.project_root),
                f"{split}_checkpoint_loaded": checkpoint_loaded,
            }
        )
        if ci_metrics is not None:
            summary_updates.update(
                {
                    f"{split}_nrmse_ci_low": ci_metrics["nrmse_ci_low"],
                    f"{split}_nrmse_ci_high": ci_metrics["nrmse_ci_high"],
                }
            )
        self.save_summary(**summary_updates)
        return metrics_path

    def save_temperature_calibration(self, calibration: dict[str, Any]) -> Path:
        """Save post-hoc temperature calibration details."""
        calibration_dir = self.paths.run_dir / "calibration"
        calibration_dir.mkdir(parents=True, exist_ok=True)
        path = calibration_dir / "temperature.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(calibration, f, indent=2, default=_json_default)
        self.save_summary(
            calibration_temperature=float(calibration["best_temperature"]),
            calibration_temperature_valid_nrmse=float(calibration["best_nrmse"]),
            calibration_temperature_metrics=_relative(path, self.project_root),
        )
        return path

    def clear_evaluation_summary(self, *, splits: tuple[str, ...] = ("test", "test_tau")) -> None:
        """Remove stale evaluation and calibration fields from the run summary."""
        prefixes = tuple(f"{split}_" for split in splits) + ("calibration_",)
        for key in list(self.summary):
            if key.startswith(prefixes) or key in {"error", "reevaluated_at"}:
                self.summary.pop(key, None)

    def save_gpu_monitoring(self, summary: dict[str, Any] | None) -> None:
        """Save GPU monitoring aggregate fields in summary.json."""
        if not summary:
            return
        updates = dict(summary)
        for key in ("gpu_csv", "gpu_plot"):
            if key in updates:
                updates[key] = _relative(Path(updates[key]), self.project_root)
        self.save_summary(**updates)

    def start_training(self) -> None:
        """Mark training start in the run summary."""
        self._training_started_perf = time.perf_counter()
        self.save_summary(status="training_started", training_started_at=now_utc_iso())

    def finish_training(self, trainer, history: list[dict[str, Any]]) -> None:
        """Save final training outputs from a completed trainer run."""
        self.save_metrics(history)
        metric = trainer.best_metric
        if metric in (float("inf"), -float("inf")):
            metric = None

        training_wall_seconds = None
        if self._training_started_perf is not None:
            training_wall_seconds = time.perf_counter() - self._training_started_perf
        if training_wall_seconds is None:
            training_wall_seconds = getattr(trainer, "training_wall_seconds", None)
        avg_epoch_seconds = training_wall_seconds / len(history) if training_wall_seconds and history else None

        self.save_summary(
            status="training_finished",
            best_epoch=trainer.best_epoch,
            best_metric=metric,
            checkpoint_exists=self.paths.checkpoint.exists(),
            history_rows=len(history),
            training_started_at=getattr(trainer, "training_started_at", self.summary.get("training_started_at")),
            training_finished_at=getattr(trainer, "training_finished_at", now_utc_iso()),
            training_wall_seconds=training_wall_seconds,
            avg_epoch_seconds=getattr(trainer, "avg_epoch_seconds", avg_epoch_seconds),
            best_epoch_finished_at=getattr(trainer, "best_epoch_finished_at", None),
            time_to_best_seconds=getattr(trainer, "time_to_best_seconds", None),
        )

    def save_summary(self, **updates: Any) -> None:
        """Update run summary and global experiment summary files."""
        self.summary.update(updates)
        with self.paths.run_summary.open("w", encoding="utf-8") as f:
            json.dump(self.summary, f, indent=2, default=_json_default)
        self._write_global_summary()

    def _base_summary(self, *, status: str) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "run_name": self.run_name,
            "experiment": self.config.experiment,
            "config": self.config.name,
            "task": self.config.task,
            "seed": self.config.seed,
            "model": self.config.model.class_name,
            "trainer": self.config.trainer.class_name,
            "optimizer": self.config.optimizer.class_name,
            "monitor": self.config.trainer.monitor,
            "best_epoch": None,
            "best_metric": None,
            "last_epoch": None,
            "history_rows": 0,
            "training_started_at": None,
            "training_finished_at": None,
            "training_wall_seconds": None,
            "avg_epoch_seconds": None,
            "best_epoch_finished_at": None,
            "time_to_best_seconds": None,
            "gpu_name": None,
            "gpu_memory_total_mb": None,
            "gpu_memory_peak_mb": None,
            "gpu_memory_mean_mb": None,
            "gpu_util_mean_pct": None,
            "gpu_util_max_pct": None,
            "gpu_samples": 0,
            "gpu_monitoring_error": None,
            "gpu_csv": None,
            "gpu_plot": None,
            "status": status,
            "run_dir": _relative(self.paths.run_dir, self.project_root),
            "checkpoint": _relative(self.paths.checkpoint, self.project_root),
            "run_log": _relative(self.paths.run_log, self.project_root),
        }

    def _read_global_records(self) -> list[dict[str, Any]]:
        if not self.paths.summary_jsonl.exists():
            return []
        records = []
        with self.paths.summary_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def _write_global_summary(self) -> None:
        records = [record for record in self._read_global_records() if record.get("run_name") != self.run_name]
        records.append(self.summary)
        records = sorted(records, key=lambda record: record.get("created_at", ""), reverse=True)

        with self.paths.summary_jsonl.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, default=_json_default) + "\n")
        pd.DataFrame(records).to_csv(self.paths.summary_csv, index=False)
        self._write_summary_md(records)

    def _write_summary_md(self, records: list[dict[str, Any]]) -> None:
        columns = [
            "created_at",
            "experiment",
            "config",
            "task",
            "model",
            "seed",
            "best_epoch",
            "best_metric",
            "test_nrmse",
            "test_nrmse_ci_low",
            "test_nrmse_ci_high",
            "calibration_temperature",
            "calibration_temperature_valid_nrmse",
            "test_tau_nrmse",
            "test_tau_nrmse_ci_low",
            "test_tau_nrmse_ci_high",
            "last_epoch",
            "training_wall_seconds",
            "time_to_best_seconds",
            "gpu_util_mean_pct",
            "gpu_memory_peak_mb",
            "gpu_samples",
            "status",
            "run_dir",
        ]
        lines = [
            "# Benchmark Runs",
            "",
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
        ]
        for record in records:
            values = [str(record.get(column, "") if record.get(column, "") is not None else "") for column in columns]
            values = [value.replace("|", "\\|").replace("\n", " ") for value in values]
            lines.append("| " + " | ".join(values) + " |")
        self.paths.summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
