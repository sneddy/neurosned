"""Segmentation trainer implementation."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from benchmarks.training.base_trainer import BaseTrainer
from benchmarks.training.labels import soft_label_1d


class SegmTrainer(BaseTrainer):
    """Trainer for segmentation-style reaction-time prediction.

    Training batches come from `TrainCroppingDataset` as `(X, q, y_rel)`.
    Validation batches use the prepared pickle dataset directly as `(X, y)`.
    The model outputs temporal logits and the trainer reads reaction time with
    soft-argmax by default.

    Main tuning knobs:

    - `temperature` / `eval_temperature`: softmax sharpness for train/eval.
    - `lambda_time`: weight for RMSE of expected time on train crops.
    - `eval_lambda_time`: validation loss time term; metrics still report RMSE.
    - `lambda_ce`: soft-label cross entropy on the Gaussian target `q`.
    - `lambda_kl`, `lambda_wass`, `lambda_entropy`, `lambda_focal`: optional
      distribution-shape terms retained from the notebook experiments.
    - `mixup_p`, `mixup_alpha`: time-aware mixup probability and Beta alpha.
    - `grad_accum`: gradient accumulation for large effective batches.
    - `sigma`: soft-label width in seconds, shared with validation labels.
    - `plot_last_batch`: show the final validation batch diagnostic plot.
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
        temperature: float = 1.0,
        eval_temperature: float | None = None,
        lambda_time: float = 1.0,
        eval_lambda_time: float | None = None,
        lambda_ce: float = 1.0,
        lambda_kl: float = 0.0,
        lambda_wass: float = 0.0,
        lambda_entropy: float = 0.0,
        lambda_focal: float = 0.0,
        mixup_p: float = 0.5,
        mixup_alpha: float = 0.4,
        grad_accum: int = 1,
        sigma: float = 0.10,
        win_offset: float = 0.5,
        channels_list: list | None = None,
        default_rmse: float | None = None,
        use_soft_argmax: bool = True,
        plot_last_batch: bool = True,
        **base_kwargs,
    ):
        super().__init__(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            optimizer=optimizer,
            device=device,
            n_epochs=n_epochs,
            **base_kwargs,
        )
        self.temperature = temperature
        self.eval_temperature = temperature if eval_temperature is None else eval_temperature
        self.lambda_time = lambda_time
        self.eval_lambda_time = lambda_time if eval_lambda_time is None else eval_lambda_time
        self.lambda_ce = lambda_ce
        self.lambda_kl = lambda_kl
        self.lambda_wass = lambda_wass
        self.lambda_entropy = lambda_entropy
        self.lambda_focal = lambda_focal
        self.mixup_p = mixup_p
        self.mixup_alpha = mixup_alpha
        self.grad_accum = grad_accum
        self.sigma = sigma
        self.win_offset = win_offset
        self.channels_list = channels_list
        self.default_rmse = default_rmse
        self.use_soft_argmax = use_soft_argmax
        self.plot_last_batch = plot_last_batch

    def create_epoch_state(self, *, train: bool) -> dict[str, Any]:
        """Create segmentation metric accumulators."""
        state = {
            "total_loss": 0.0,
            "total_ce": 0.0,
            "total_rmse": 0.0,
            "n_batches": 0,
            "sse": 0.0,
            "n_samples": 0,
        }
        if train:
            state.update({"total_kl": 0.0, "total_wass": 0.0, "sum_y": 0.0, "sum_y2": 0.0})
        else:
            state["preds_abs"] = []
            state["last_batch"] = None
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
        """Run one segmentation train batch."""
        X, q, y_rel = batch
        X = X.to(self.device).float()
        q = q.to(self.device).float()
        y_rel = y_rel.to(self.device).float().view(-1)

        batch_size, _, T = X.shape
        dt = 1.0 / float(self.model.sfreq)

        if self.mixup_p > 0 and torch.rand((), device=X.device) < self.mixup_p:
            X, q, y_rel = self._mixup(X, q, y_rel, T, dt, batch_size)

        z = self.model(X).squeeze(1)
        losses = self._train_losses(z, q, y_rel, T, dt)
        loss = losses["total"]
        loss_for_backward = loss / self.grad_accum
        loss_for_backward.backward()

        if (batch_idx + 1) % self.grad_accum == 0 or (batch_idx + 1) == n_batches:
            self.optimizer.step()
            self.optimizer.zero_grad()

        t_hat = losses["t_hat"]
        diff = t_hat.detach() - y_rel.detach()
        state["sse"] += torch.sum(diff * diff).item()
        state["n_samples"] += y_rel.numel()
        state["sum_y"] += y_rel.detach().sum().item()
        state["sum_y2"] += torch.sum(y_rel.detach() * y_rel.detach()).item()
        state["n_batches"] += 1

        state["total_loss"] += float(loss_for_backward.detach())
        state["total_ce"] += float(losses["ce"].detach())
        state["total_rmse"] += float(losses["rmse"].detach())
        state["total_kl"] += float(losses["kl"].detach())
        state["total_wass"] += float(losses["wass"].detach())

        metrics = self._running_train_metrics(state)
        metrics.update(
            {
                "loss": float(loss_for_backward.detach()),
                "ce": float(losses["ce"].detach()),
                "rmse": float(losses["rmse"].detach()),
                "kl": float(losses["kl"].detach()),
                "wass": float(losses["wass"].detach()),
                "entropy": float(losses["entropy"].detach()),
            }
        )
        return metrics

    def eval_batch(
        self,
        batch,
        *,
        epoch: int,
        batch_idx: int,
        n_batches: int,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """Run one segmentation eval batch."""
        X = batch[0].to(self.device).float()
        y = batch[1].to(self.device).float().view(-1)
        if self.channels_list is not None:
            X = X[:, self.channels_list, :]

        _, _, T = X.shape
        dt = 1.0 / float(self.model.sfreq)
        win_sec = T * dt

        y_rel = (y - self.win_offset).clamp(0.0, win_sec)
        q = soft_label_1d(y_rel, T=T, dt=dt, sigma=self.sigma, density=True)

        z = self.model(X).squeeze(1)
        log_p = F.log_softmax(z / self.eval_temperature, dim=-1)
        ce = -(q * log_p).sum(dim=-1).mean()

        if self.use_soft_argmax:
            t_grid = (torch.arange(T, device=self.device, dtype=z.dtype) * dt)[None, :]
            p = torch.softmax(z / self.eval_temperature, dim=-1)
            t_hat_abs = (p * t_grid).sum(dim=-1) + self.win_offset
        else:
            t_hat_abs = z.argmax(dim=-1) * dt + self.win_offset

        rmse_time = F.mse_loss(t_hat_abs, y).sqrt()
        loss = ce + self.eval_lambda_time * rmse_time

        state["total_loss"] += float(loss.detach())
        state["total_ce"] += float(ce.detach())
        state["total_rmse"] += float(rmse_time.detach())
        state["n_batches"] += 1

        diff_abs = (t_hat_abs - y).detach()
        state["sse"] += torch.sum(diff_abs * diff_abs).item()
        state["n_samples"] += y.numel()
        state["preds_abs"].extend(t_hat_abs.cpu().numpy())
        state["last_batch"] = {
            "X": X.cpu(),
            "y": y.cpu(),
            "z": z.cpu(),
            "t_hat_abs": t_hat_abs.cpu(),
            "ce": float(ce.detach()),
            "rmse_time": float(rmse_time.detach()),
            "T": T,
            "dt": dt,
            "win_offset": self.win_offset,
            "q": q.cpu(),
            "temperature": self.eval_temperature,
        }

        metrics = self._running_eval_metrics(state, final=False)
        metrics.update({"loss": float(loss.detach()), "ce": float(ce.detach()), "rmse": float(rmse_time.detach())})
        return metrics

    def finalize_epoch(self, state: dict[str, Any], *, train: bool) -> dict[str, Any]:
        """Return segmentation epoch metrics."""
        n_batches = max(state["n_batches"], 1)
        if train:
            metrics = self._running_train_metrics(state)
            metrics.update(
                {
                    "loss": state["total_loss"] / n_batches,
                    "ce": state["total_ce"] / n_batches,
                    "rmse": state["total_rmse"] / n_batches,
                    "kl": state["total_kl"] / n_batches,
                    "wass": state["total_wass"] / n_batches,
                }
            )
            return metrics

        metrics = self._running_eval_metrics(state, final=True)
        metrics.update(
            {
                "loss": state["total_loss"] / n_batches,
                "ce": state["total_ce"] / n_batches,
                "rmse": state["total_rmse"] / n_batches,
                "preds_abs": np.array(state["preds_abs"]),
            }
        )
        if self.plot_last_batch and state["last_batch"] is not None:
            from benchmarks.training.visualization import draw_batch

            draw_batch(**state["last_batch"])
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
        """Format segmentation progress text."""
        if train:
            return (
                f"Epoch {epoch} [{batch_idx + 1}/{n_batches}] "
                f"Loss {metrics['loss']:.4f} CE {metrics['ce']:.4f} "
                f"RMSE {metrics['rmse']:.4f} KL {metrics['kl']:.4f} WASS {metrics['wass']:.4f} "
                f"NRMSE {metrics['nrmse']:.4f} Entropy {metrics['entropy']:.3f}"
            )
        return (
            f"Val [{batch_idx + 1}/{n_batches}] Loss {metrics['loss']:.4f} "
            f"CE {metrics['ce']:.4f} RMSE {metrics['rmse']:.4f} NRMSE {metrics['nrmse']:.4f}"
        )

    def _mixup(self, X, q, y_rel, T: int, dt: float, batch_size: int):
        bins = 50
        b = torch.clamp((y_rel / (T * dt) * bins).long(), 0, bins - 1)
        partner = torch.arange(batch_size, device=X.device)
        valid = torch.zeros(batch_size, dtype=torch.bool, device=X.device)
        for k in range(bins):
            group = torch.where(b == k)[0]
            if group.numel() >= 2:
                permuted = group[torch.randperm(group.numel(), device=X.device)]
                partner[permuted] = permuted.roll(1)
                valid[permuted] = True
        if valid.any():
            lam = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample((batch_size,)).to(X.device)
            lam = torch.where(valid, lam, torch.ones_like(lam))
            X = lam.view(-1, 1, 1) * X + (1 - lam).view(-1, 1, 1) * X[partner]
            q = lam.view(-1, 1) * q + (1 - lam).view(-1, 1) * q[partner]
            y_rel = lam * y_rel + (1 - lam) * y_rel[partner]
        return X, q, y_rel

    def _train_losses(self, z, q, y_rel, T: int, dt: float) -> dict[str, torch.Tensor]:
        log_p = F.log_softmax(z / self.temperature, dim=-1)
        p = torch.softmax(z / self.temperature, dim=-1)
        entropy = -(p * log_p).sum(dim=-1).mean()

        if self.lambda_ce:
            if self.lambda_focal > 0:
                focal_weight = (1.0 - p.detach()).pow(self.lambda_focal)
                ce = -((focal_weight * q) * log_p).sum(dim=-1).mean()
            else:
                ce = -(q * log_p).sum(dim=-1).mean()
        else:
            ce = torch.tensor(0.0, device=self.device)

        t_grid = (torch.arange(T, device=self.device, dtype=z.dtype) * dt)[None, :]
        t_hat = (p * t_grid).sum(dim=-1)
        rmse = F.mse_loss(t_hat, y_rel).sqrt()
        kl = (q * (torch.log(q + 1e-8) - log_p)).sum(dim=-1).mean() if self.lambda_kl else torch.tensor(0.0, device=self.device)
        if self.lambda_wass > 0:
            wass = (torch.abs(torch.cumsum(p, -1) - torch.cumsum(q, -1)).sum(-1).mean()) * dt
        else:
            wass = torch.tensor(0.0, device=self.device)

        total = self.lambda_ce * ce + self.lambda_time * rmse + self.lambda_kl * kl + self.lambda_wass * wass + self.lambda_entropy * entropy
        return {"total": total, "ce": ce, "rmse": rmse, "kl": kl, "wass": wass, "entropy": entropy, "t_hat": t_hat}

    def _running_train_metrics(self, state: dict[str, Any]) -> dict[str, float]:
        rmse_ds = (state["sse"] / max(state["n_samples"], 1)) ** 0.5
        mean_y = state["sum_y"] / max(state["n_samples"], 1)
        var_y = max((state["sum_y2"] / max(state["n_samples"], 1)) - mean_y**2, 1e-12)
        return {"nrmse": rmse_ds / (var_y**0.5)}

    def _running_eval_metrics(self, state: dict[str, Any], *, final: bool) -> dict[str, float]:
        rmse = (state["sse"] / max(state["n_samples"], 1)) ** 0.5
        if final:
            rmse = rmse * 1000
        nrmse = rmse / self.default_rmse if self.default_rmse else rmse
        return {"nrmse": nrmse}
