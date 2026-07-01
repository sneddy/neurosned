"""Base trainer interface for benchmark experiments."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm


def now_utc_iso() -> str:
    """Return a compact UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class BaseTrainer(ABC):
    """Orchestrate epochs while task trainers own batch math."""

    def __init__(
        self,
        *,
        model,
        train_loader,
        valid_loader,
        optimizer,
        device,
        n_epochs: int,
        checkpoint_path: str | Path | None = None,
        monitor: str = "nrmse",
        minimize: bool = True,
        early_stopping_patience: int | None = None,
        plateau_scheduler=None,
        print_batch_stats: bool = True,
        on_epoch_end=None,
        stage_name: str | None = None,
        epoch_offset: int = 0,
        seconds_offset: float = 0.0,
        history_exclude_keys: tuple[str, ...] = ("preds_abs", "preds", "diffs", "logits"),
    ):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.device = device
        self.n_epochs = int(n_epochs)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self.monitor = monitor
        self.minimize = minimize
        self.early_stopping_patience = early_stopping_patience
        self.plateau_scheduler = plateau_scheduler
        self.print_batch_stats = print_batch_stats
        self.on_epoch_end = on_epoch_end
        self.stage_name = stage_name
        self.epoch_offset = int(epoch_offset)
        self.seconds_offset = float(seconds_offset)
        self.history_exclude_keys = set(history_exclude_keys)

        self.best_metric = self._initial_best_metric()
        self.best_epoch: int | None = None
        self.best_valid_metrics: dict[str, Any] | None = None
        self.epochs_no_improve = 0
        self.history: list[dict[str, Any]] = []
        self.training_started_at: str | None = None
        self.training_finished_at: str | None = None
        self.training_wall_seconds: float | None = None
        self.avg_epoch_seconds: float | None = None
        self.best_epoch_finished_at: str | None = None
        self.time_to_best_seconds: float | None = None

    def _initial_best_metric(self) -> float:
        """Return the hardcoded initial best metric."""
        if self.minimize and self.monitor == "nrmse":
            return 1.0
        return float("inf") if self.minimize else -float("inf")

    def run(self) -> list[dict[str, Any]]:
        """Run train/validation epochs and return metric history."""
        self.training_started_at = now_utc_iso()
        started = time.perf_counter()
        for stage_epoch in range(1, self.n_epochs + 1):
            epoch = self.epoch_offset + stage_epoch
            epoch_started = time.perf_counter()
            train_started = time.perf_counter()
            train_metrics = self.run_train_epoch(epoch)
            train_seconds = time.perf_counter() - train_started
            valid_started = time.perf_counter()
            valid_metrics = self.run_valid_epoch(epoch)
            valid_seconds = time.perf_counter() - valid_started
            epoch_seconds = time.perf_counter() - epoch_started
            cumulative_seconds = self.seconds_offset + time.perf_counter() - started
            row = {
                "epoch": epoch,
                "stage_epoch": stage_epoch,
                "train_seconds": train_seconds,
                "valid_seconds": valid_seconds,
                "epoch_seconds": epoch_seconds,
                "cumulative_training_seconds": cumulative_seconds,
                **self._prefixed_metrics("train", train_metrics),
                **self._prefixed_metrics("valid", valid_metrics),
            }
            if self.stage_name is not None:
                row["stage"] = self.stage_name
            self.history.append(row)

            should_break = False
            if self.is_best(valid_metrics):
                self.best_epoch = epoch
                self.best_metric = float(valid_metrics[self.monitor])
                self.best_valid_metrics = valid_metrics
                self.best_epoch_finished_at = now_utc_iso()
                self.time_to_best_seconds = cumulative_seconds
                self.epochs_no_improve = 0
                print(f"New best validation {self.monitor}: {self.best_metric:.6f} at epoch {epoch}")
                self.save_checkpoint()
            else:
                self.epochs_no_improve += 1
                if self.should_stop():
                    if self.plateau_scheduler is None:
                        should_break = True
                    else:
                        should_break = not self.plateau_scheduler.step(self)
                        if not should_break:
                            self.epochs_no_improve = 0
            if self.on_epoch_end is not None:
                self.on_epoch_end(self, self.history)
            if should_break:
                break
        self.training_finished_at = now_utc_iso()
        self.training_wall_seconds = time.perf_counter() - started
        self.avg_epoch_seconds = self.training_wall_seconds / len(self.history) if self.history else None
        return self.history

    def run_train_epoch(self, epoch: int) -> dict[str, Any]:
        """Run one training epoch."""
        return self.run_epoch(self.train_loader, epoch=epoch, train=True, split="train")

    def run_valid_epoch(self, epoch: int) -> dict[str, Any]:
        """Run one validation epoch."""
        return self.run_eval_epoch(self.valid_loader, split="valid", epoch=epoch)

    def run_eval_epoch(self, dataloader, *, split: str, epoch: int = 0) -> dict[str, Any]:
        """Run one non-training epoch on an arbitrary evaluation split."""
        return self.run_epoch(dataloader, epoch=epoch, train=False, split=split)

    def run_epoch(self, dataloader, *, epoch: int, train: bool, split: str) -> dict[str, Any]:
        """Run one epoch in train or validation mode."""
        self.model.train(train)
        if train:
            self.optimizer.zero_grad()

        state = self.create_epoch_state(train=train, split=split)
        state["epoch"] = epoch
        state["split"] = split
        n_batches = len(dataloader)
        progress = tqdm(enumerate(dataloader), total=n_batches, disable=not self.print_batch_stats)

        for batch_idx, batch in progress:
            metrics = self.run_batch(
                batch,
                epoch=epoch,
                batch_idx=batch_idx,
                n_batches=n_batches,
                train=train,
                state=state,
            )
            if self.print_batch_stats:
                progress.set_description(self.format_progress(epoch, batch_idx, n_batches, metrics, train=train))

        return self.finalize_epoch(state, train=train)

    def run_batch(
        self,
        batch,
        *,
        epoch: int,
        batch_idx: int,
        n_batches: int,
        train: bool,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """Route a batch to the task-specific train or eval implementation."""
        if train:
            return self.train_batch(batch, epoch=epoch, batch_idx=batch_idx, n_batches=n_batches, state=state)
        with torch.no_grad():
            return self.eval_batch(batch, epoch=epoch, batch_idx=batch_idx, n_batches=n_batches, state=state)

    def is_best(self, valid_metrics: dict[str, Any]) -> bool:
        """Check whether validation metrics improve the monitored metric."""
        value = valid_metrics.get(self.monitor)
        if value is None:
            return False
        return value < self.best_metric if self.minimize else value > self.best_metric

    def should_stop(self) -> bool:
        """Check early stopping state."""
        return self.early_stopping_patience is not None and self.epochs_no_improve >= self.early_stopping_patience

    def save_checkpoint(self) -> None:
        """Save model weights when checkpointing is configured."""
        if self.checkpoint_path is None:
            return
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.checkpoint_path)

    def _prefixed_metrics(self, prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
        """Prefix scalar-like metrics for history rows."""
        return {f"{prefix}_{key}": value for key, value in metrics.items() if key not in self.history_exclude_keys}

    def create_epoch_state(self, *, train: bool, split: str) -> dict[str, Any]:
        """Create task-specific accumulator state."""
        return {}

    def format_progress(
        self,
        epoch: int,
        batch_idx: int,
        n_batches: int,
        metrics: dict[str, Any],
        *,
        train: bool,
    ) -> str:
        """Format progress-bar text."""
        phase = "Train" if train else "Valid"
        return f"{phase} {epoch} [{batch_idx + 1}/{n_batches}]"

    @abstractmethod
    def train_batch(
        self,
        batch,
        *,
        epoch: int,
        batch_idx: int,
        n_batches: int,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """Run one train batch and update epoch state."""

    @abstractmethod
    def eval_batch(
        self,
        batch,
        *,
        epoch: int,
        batch_idx: int,
        n_batches: int,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """Run one eval batch and update epoch state."""

    @abstractmethod
    def finalize_epoch(self, state: dict[str, Any], *, train: bool) -> dict[str, Any]:
        """Return final epoch metrics from accumulated state."""
