"""Base trainer interface for benchmark experiments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm


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
        history_exclude_keys: tuple[str, ...] = ("preds_abs", "preds", "diffs"),
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
        self.history_exclude_keys = set(history_exclude_keys)

        self.best_metric = float("inf") if minimize else -float("inf")
        self.best_epoch: int | None = None
        self.epochs_no_improve = 0
        self.history: list[dict[str, Any]] = []

    def run(self) -> list[dict[str, Any]]:
        """Run train/validation epochs and return metric history."""
        for epoch in range(1, self.n_epochs + 1):
            train_metrics = self.run_train_epoch(epoch)
            valid_metrics = self.run_valid_epoch(epoch)
            row = {"epoch": epoch, **self._prefixed_metrics("train", train_metrics), **self._prefixed_metrics("valid", valid_metrics)}
            self.history.append(row)

            if self.is_best(valid_metrics):
                self.best_epoch = epoch
                self.best_metric = float(valid_metrics[self.monitor])
                self.epochs_no_improve = 0
                print(f"New best validation {self.monitor}: {self.best_metric:.6f} at epoch {epoch}")
                self.save_checkpoint()
            else:
                self.epochs_no_improve += 1
                if self.should_stop():
                    if self.plateau_scheduler is None:
                        break
                    self.plateau_scheduler.step(self)
                    self.epochs_no_improve = 0
        return self.history

    def run_train_epoch(self, epoch: int) -> dict[str, Any]:
        """Run one training epoch."""
        return self.run_epoch(self.train_loader, epoch=epoch, train=True)

    def run_valid_epoch(self, epoch: int) -> dict[str, Any]:
        """Run one validation epoch."""
        return self.run_epoch(self.valid_loader, epoch=epoch, train=False)

    def run_epoch(self, dataloader, *, epoch: int, train: bool) -> dict[str, Any]:
        """Run one epoch in train or validation mode."""
        self.model.train(train)
        if train:
            self.optimizer.zero_grad()

        state = self.create_epoch_state(train=train)
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

    def create_epoch_state(self, *, train: bool) -> dict[str, Any]:
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
