"""Regression trainer implementation."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from benchmarks.training.base_trainer import BaseTrainer


class RegrTrainer(BaseTrainer):
    """Trainer for scalar reaction-time regression.

    Regression batches are `(X, y)` from the prepared pickle dataset, without
    the segmentation crop wrapper. The model predicts a scalar reaction time and
    validation reports normalized RMSE plus threshold accuracies at 100 ms and
    250 ms.

    Main tuning knobs:

    - `channel_dropout_max_ratio`: random per-sample channel masking strength.
    - `cutout_proba`, `cutout_min_len`, `cutout_max_len`: temporal cutout.
    - `noise_proba`, `noise_base_std`, `noise_random_std`: Gaussian noise.
    - `mixup_p`, `mixup_alpha`: batch mixup for EEG and scalar targets.
    - `channels_list`: optional fixed channel subset for train and validation.
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
        channels_list: list | None = None,
        default_rmse: float | None = None,
        channel_dropout_max_ratio: float = 0.5,
        cutout_proba: float = 0.5,
        cutout_min_len: int = 10,
        cutout_max_len: int = 100,
        noise_proba: float = 0.2,
        noise_base_std: float = 0.01,
        noise_random_std: float = 0.03,
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
        self.channels_list = channels_list
        self.default_rmse = default_rmse
        self.channel_dropout_max_ratio = channel_dropout_max_ratio
        self.cutout_proba = cutout_proba
        self.cutout_min_len = cutout_min_len
        self.cutout_max_len = cutout_max_len
        self.noise_proba = noise_proba
        self.noise_base_std = noise_base_std
        self.noise_random_std = noise_random_std
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
        X, y = self._augment_train_batch(X, y)
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
        if self.channels_list is not None:
            X = X[:, self.channels_list, :]
        return X.to(self.device).float(), y.to(self.device).float()

    def _augment_train_batch(self, X, y):
        if self.channel_dropout_max_ratio > 0:
            channel_dropout_ratio = torch.rand(1).item() * self.channel_dropout_max_ratio
            if channel_dropout_ratio > 0:
                batch_size, channels, _ = X.shape
                mask = (torch.rand(batch_size, channels, device=X.device) > channel_dropout_ratio).float().unsqueeze(-1)
                X = X * mask

        if self.cutout_proba > 0 and torch.rand(1).item() < self.cutout_proba:
            batch_size, _, times = X.shape
            max_len = min(self.cutout_max_len, times)
            for batch_idx in range(batch_size):
                seg_len = int(torch.randint(self.cutout_min_len, max_len + 1, (1,)).item())
                start = int(torch.randint(0, times - seg_len + 1, (1,)).item())
                X[batch_idx, :, start:start + seg_len] = 0

        if self.noise_proba > 0 and torch.rand(1).item() < self.noise_proba:
            noise_std = self.noise_random_std * torch.randn(1).abs().item() + self.noise_base_std
            X = X + torch.randn_like(X) * noise_std

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
