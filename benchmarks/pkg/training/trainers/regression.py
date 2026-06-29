"""Regression trainer implementation."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from benchmarks.pkg.training.trainers.base import BaseTrainer


class RegrTrainer(BaseTrainer):
    """Trainer for scalar reaction-time regression.

    Regression batches are `(X, y)` from fixed-window datasets. The model
    predicts a scalar reaction time and validation reports normalized RMSE plus
    threshold accuracies at 100 ms and 250 ms.

    Main tuning knobs:

    - `mixup_p`, `mixup_alpha`: batch mixup for EEG and scalar targets.
    - `default_rmse`: denominator for normalized RMSE.
    """

    def __init__(
        self,
        *,
        model,
        train_loader,
        valid_loader,
        optimizer,
        device,
        n_epochs: int,
        loss_fn=None,
        scheduler=None,
        default_rmse: float | None = None,
        mixup_p: float = 0.5,
        mixup_alpha: float = 0.4,
        **base_kwargs,
    ):
        base_kwargs.setdefault("monitor", "rmse")
        super().__init__(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            optimizer=optimizer,
            device=device,
            n_epochs=n_epochs,
            **base_kwargs,
        )
        self.loss_fn = loss_fn or nn.MSELoss()
        self.scheduler = scheduler
        self.default_rmse = default_rmse
        self.mixup_p = mixup_p
        self.mixup_alpha = mixup_alpha

    def create_epoch_state(self, *, train: bool) -> dict[str, Any]:
        """Create regression metric accumulators."""
        state = {
            "total_loss": 0.0,
            "sse": 0.0,
            "n_samples": 0,
            "n_batches": 0,
        }
        if not train:
            state["preds"] = []
            state["diffs"] = []
        return state

    def train_batch(
        self,
        batch,
        *,
        epoch: int,
        batch_idx: int,
        n_batches: int,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """Run one regression train batch."""
        X, y = self._prepare_batch(batch)
        X, y = self._mixup_batch(X, y)
        X = X.contiguous()

        self.optimizer.zero_grad()
        preds = self.model(X)
        loss = self.loss_fn(preds, y)

        loss.backward()
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return self._update_regression_state(state, preds, y, float(loss.detach()), collect_predictions=False)

    def eval_batch(
        self,
        batch,
        *,
        epoch: int,
        batch_idx: int,
        n_batches: int,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """Run one regression eval batch."""
        X, y = self._prepare_batch(batch)
        preds = self.model(X)
        loss = self.loss_fn(preds, y)
        return self._update_regression_state(state, preds, y, float(loss.detach()), collect_predictions=True)

    def finalize_epoch(self, state: dict[str, Any], *, train: bool) -> dict[str, Any]:
        """Return regression epoch metrics."""
        n_batches = max(state["n_batches"], 1)
        metrics = {
            "loss": state["total_loss"] / n_batches,
            "rmse": self._normalized_rmse(state),
        }
        if train:
            return metrics

        diffs = np.array(state["diffs"])
        metrics.update(
            {
                "preds": np.array(state["preds"]),
                "acc_100": float(np.mean(np.abs(diffs) < 0.1)) if diffs.size else float("nan"),
                "acc_250": float(np.mean(np.abs(diffs) < 0.25)) if diffs.size else float("nan"),
            }
        )
        print(
            f"Val RMSE: {metrics['rmse']:.6f}, "
            f"Val Loss: {metrics['loss']:.6f}, "
            f"acc_100: {metrics['acc_100']:.6f}, "
            f"acc_250: {metrics['acc_250']:.6f}"
        )
        return metrics

    def format_progress(
        self,
        epoch: int,
        batch_idx: int,
        n_batches: int,
        metrics: dict[str, Any],
        *,
        train: bool,
    ) -> str:
        """Format regression progress text."""
        if train:
            return (
                f"Epoch {epoch}, Batch {batch_idx + 1}/{n_batches}, "
                f"Loss: {metrics['loss']:.6f}, RMSE: {metrics['rmse']:.6f}"
            )
        return (
            f"Val Batch {batch_idx + 1}/{n_batches}, "
            f"Loss: {metrics['loss']:.6f}, RMSE: {metrics['rmse']:.6f}"
        )

    def _prepare_batch(self, batch):
        X, y = batch[0], batch[1]
        return X.to(self.device).float(), y.to(self.device).float()

    def _mixup_batch(self, X, y):
        if self.mixup_p > 0 and torch.rand(1).item() < self.mixup_p:
            batch_size = X.shape[0]
            perm = torch.randperm(batch_size, device=X.device)
            lam = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample((batch_size,)).to(X.device)
            X = lam.view(batch_size, 1, 1) * X + (1 - lam).view(batch_size, 1, 1) * X[perm]
            lam_y = lam.view(batch_size, *([1] * (y.ndim - 1)))
            y = lam_y * y + (1 - lam_y) * y[perm]

        return X, y

    def _update_regression_state(
        self,
        state: dict[str, Any],
        preds: torch.Tensor,
        y: torch.Tensor,
        loss: float,
        *,
        collect_predictions: bool,
    ) -> dict[str, float]:
        state["total_loss"] += loss
        state["n_batches"] += 1

        preds_flat = preds.detach().view(-1)
        y_flat = y.detach().view(-1)
        diffs = preds_flat - y_flat
        state["sse"] += torch.sum(diffs * diffs).item()
        state["n_samples"] += y_flat.numel()

        if collect_predictions:
            state["preds"].extend(preds_flat.cpu().numpy())
            state["diffs"].extend(diffs.cpu().numpy())

        return {"loss": loss, "rmse": self._normalized_rmse(state)}

    def _normalized_rmse(self, state: dict[str, Any]) -> float:
        rmse = (state["sse"] / max(state["n_samples"], 1)) ** 0.5
        return rmse / self.default_rmse if self.default_rmse else rmse
