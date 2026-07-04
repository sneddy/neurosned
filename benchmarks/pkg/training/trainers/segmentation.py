"""Segmentation trainer implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from benchmarks.pkg.losses.event_time import (
    cdf_distance,
    event_mixture_nll,
    expected_time,
    posterior_entropy,
    posterior_kl,
    soft_label_cross_entropy,
)
from benchmarks.pkg.losses.hazard import hazard_discrete_nll, hazard_logits_to_log_pmf
from benchmarks.pkg.training.trainers.base import BaseTrainer
from benchmarks.pkg.training.labels import soft_label_1d


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
    - `eval_lambda_ce`: validation loss CE weight, kept at 1.0 by default.
    - `lambda_event_nll`: continuous event-time mixture likelihood term.
    - `event_nll_kernel`: observation model for the event-time likelihood:
      `"gaussian"`, `"student_t"`, `"laplace"`, or `"gaussian_mixture"`.
    - `readout_distribution`: logits parameterization, `"softmax"` or `"hazard"`.
    - `lambda_hazard_discrete_nll`: exact-bin discrete survival likelihood term.
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
        eval_lambda_ce: float = 1.0,
        lambda_event_nll: float = 0.0,
        eval_lambda_event_nll: float | None = None,
        lambda_hazard_discrete_nll: float = 0.0,
        eval_lambda_hazard_discrete_nll: float | None = None,
        event_nll_kernel: str = "gaussian",
        event_nll_df: float = 3.0,
        event_nll_mixture_weight: float = 0.1,
        event_nll_mixture_sigma_narrow: float | None = None,
        event_nll_mixture_sigma_wide: float | None = None,
        readout_distribution: str = "softmax",
        hazard_condition_inside: bool = True,
        lambda_kl: float = 0.0,
        lambda_wass: float = 0.0,
        eval_lambda_wass: float = 0.0,
        lambda_entropy: float = 0.0,
        lambda_focal: float = 0.0,
        mixup_p: float = 0.5,
        mixup_alpha: float = 0.4,
        grad_accum: int = 1,
        sigma: float = 0.10,
        win_offset: float = 0.5,
        channels_list: list | None = None,
        default_rmse: float | None = None,
        default_rmse_by_split: dict[str, float] | None = None,
        use_soft_argmax: bool = True,
        plot_last_batch: bool = True,
        plot_save_dir: str | Path | None = None,
        plot_show: bool = True,
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
        self.eval_lambda_ce = eval_lambda_ce
        self.lambda_event_nll = lambda_event_nll
        self.eval_lambda_event_nll = lambda_event_nll if eval_lambda_event_nll is None else eval_lambda_event_nll
        self.lambda_hazard_discrete_nll = lambda_hazard_discrete_nll
        self.eval_lambda_hazard_discrete_nll = (
            lambda_hazard_discrete_nll
            if eval_lambda_hazard_discrete_nll is None
            else eval_lambda_hazard_discrete_nll
        )
        self.event_nll_kernel = event_nll_kernel
        self.event_nll_df = event_nll_df
        self.event_nll_mixture_weight = event_nll_mixture_weight
        self.event_nll_mixture_sigma_narrow = event_nll_mixture_sigma_narrow
        self.event_nll_mixture_sigma_wide = event_nll_mixture_sigma_wide
        self.readout_distribution = self._normalize_readout_distribution(readout_distribution)
        self.hazard_condition_inside = bool(hazard_condition_inside)
        self.lambda_kl = lambda_kl
        self.lambda_wass = lambda_wass
        self.eval_lambda_wass = eval_lambda_wass
        self.lambda_entropy = lambda_entropy
        self.lambda_focal = lambda_focal
        self.mixup_p = mixup_p
        self.mixup_alpha = mixup_alpha
        self.grad_accum = grad_accum
        self.sigma = sigma
        self.win_offset = win_offset
        self.channels_list = channels_list
        self.default_rmse_by_split = dict(default_rmse_by_split or {})
        if default_rmse is not None:
            self.default_rmse_by_split.setdefault("valid", default_rmse)
        self.use_soft_argmax = use_soft_argmax
        self.plot_last_batch = plot_last_batch
        self.plot_save_dir = Path(plot_save_dir) if plot_save_dir is not None else None
        self.plot_show = plot_show

    def create_epoch_state(self, *, train: bool, split: str) -> dict[str, Any]:
        """Create segmentation metric accumulators."""
        state = {
            "split": split,
            "default_rmse": self.default_rmse_for(split),
            "total_loss": 0.0,
            "total_ce": 0.0,
            "total_event_nll": 0.0,
            "total_hazard_discrete_nll": 0.0,
            "total_event_sigma": 0.0,
            "total_rmse": 0.0,
            "total_wass": 0.0,
            "n_batches": 0,
            "sse": 0.0,
            "n_samples": 0,
            "n_event_sigma": 0,
        }
        if train:
            state.update({"total_kl": 0.0, "sum_y": 0.0, "sum_y2": 0.0})
        else:
            state["preds_abs"] = []
            state["logits"] = []
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
        state["total_event_nll"] += float(losses["event_nll"].detach())
        state["total_hazard_discrete_nll"] += float(losses["hazard_discrete_nll"].detach())
        state["total_rmse"] += float(losses["rmse"].detach())
        state["total_kl"] += float(losses["kl"].detach())
        state["total_wass"] += float(losses["wass"].detach())
        self._record_event_sigma(state, losses["event_sigma"], batch_size)

        metrics = self._running_train_metrics(state)
        metrics.update(
            {
                "loss": float(loss_for_backward.detach()),
                "ce": float(losses["ce"].detach()),
                "event_nll": float(losses["event_nll"].detach()),
                "hazard_discrete_nll": float(losses["hazard_discrete_nll"].detach()),
                "rmse": float(losses["rmse"].detach()),
                "kl": float(losses["kl"].detach()),
                "wass": float(losses["wass"].detach()),
                "entropy": float(losses["entropy"].detach()),
            }
        )
        event_sigma = self._running_event_sigma(state)
        if event_sigma is not None:
            metrics["event_sigma"] = event_sigma
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
        log_p, p = self._event_distribution(z, self.eval_temperature)
        ce = soft_label_cross_entropy(log_p, q)
        t_grid = (torch.arange(T, device=self.device, dtype=z.dtype) * dt)[None, :]
        event_sigma = self._model_event_sigma()
        event_nll = (
            event_mixture_nll(
                log_p,
                y_rel,
                t_grid,
                sigma=self.sigma,
                event_sigma=event_sigma,
                kernel=self.event_nll_kernel,
                df=self.event_nll_df,
                mixture_weight=self.event_nll_mixture_weight,
                mixture_sigma_narrow=self.event_nll_mixture_sigma_narrow,
                mixture_sigma_wide=self.event_nll_mixture_sigma_wide,
            )
            if self.eval_lambda_event_nll
            else torch.tensor(0.0, device=self.device)
        )
        hazard_nll = (
            hazard_discrete_nll(z, y_rel, dt=dt, temperature=self.eval_temperature)
            if self.eval_lambda_hazard_discrete_nll
            else torch.tensor(0.0, device=self.device)
        )
        if self.eval_lambda_wass > 0:
            wass = cdf_distance(p, q, dt)
        else:
            wass = torch.tensor(0.0, device=self.device)

        if self.use_soft_argmax:
            t_hat_abs = expected_time(p, t_grid) + self.win_offset
        else:
            t_hat_abs = p.argmax(dim=-1) * dt + self.win_offset

        rmse_time = F.mse_loss(t_hat_abs, y).sqrt()
        loss = (
            self.eval_lambda_ce * ce
            + self.eval_lambda_event_nll * event_nll
            + self.eval_lambda_hazard_discrete_nll * hazard_nll
            + self.eval_lambda_time * rmse_time
            + self.eval_lambda_wass * wass
        )

        state["total_loss"] += float(loss.detach())
        state["total_ce"] += float(ce.detach())
        state["total_event_nll"] += float(event_nll.detach())
        state["total_hazard_discrete_nll"] += float(hazard_nll.detach())
        state["total_rmse"] += float(rmse_time.detach())
        state["total_wass"] += float(wass.detach())
        state["n_batches"] += 1
        self._record_event_sigma(state, event_sigma, y.numel())

        diff_abs = (t_hat_abs - y).detach()
        state["sse"] += torch.sum(diff_abs * diff_abs).item()
        state["n_samples"] += y.numel()
        state["preds_abs"].extend(t_hat_abs.detach().cpu().numpy())
        state["logits"].append(z.detach().cpu().numpy().astype(np.float32, copy=False))
        state["last_batch"] = {
            "X": X.detach().cpu(),
            "y": y.detach().cpu(),
            "z": z.detach().cpu(),
            "t_hat_abs": t_hat_abs.detach().cpu(),
            "ce": float(ce.detach()),
            "rmse_time": float(rmse_time.detach()),
            "T": T,
            "dt": dt,
            "win_offset": self.win_offset,
            "q": q.detach().cpu(),
            "temperature": self.eval_temperature,
            "probabilities": p.detach().cpu(),
            "readout_label": f"Predicted {self.readout_distribution} p(t)",
        }

        metrics = self._running_eval_metrics(state)
        metrics.update({"loss": float(loss.detach()), "ce": float(ce.detach()), "rmse": float(rmse_time.detach())})
        if self.eval_lambda_event_nll:
            metrics["event_nll"] = float(event_nll.detach())
        if self.eval_lambda_hazard_discrete_nll:
            metrics["hazard_discrete_nll"] = float(hazard_nll.detach())
        if self.eval_lambda_wass:
            metrics["wass"] = float(wass.detach())
        event_sigma = self._running_event_sigma(state)
        if event_sigma is not None:
            metrics["event_sigma"] = event_sigma
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
            if self.lambda_event_nll:
                metrics["event_nll"] = state["total_event_nll"] / n_batches
            if self.lambda_hazard_discrete_nll:
                metrics["hazard_discrete_nll"] = state["total_hazard_discrete_nll"] / n_batches
            event_sigma = self._running_event_sigma(state)
            if event_sigma is not None:
                metrics["event_sigma"] = event_sigma
            return metrics

        metrics = self._running_eval_metrics(state)
        metrics.update(
            {
                "loss": state["total_loss"] / n_batches,
                "ce": state["total_ce"] / n_batches,
                "rmse": state["total_rmse"] / n_batches,
                "preds_abs": np.array(state["preds_abs"]),
                "logits": np.concatenate(state["logits"], axis=0) if state["logits"] else np.empty((0, 0), dtype=np.float32),
            }
        )
        if self.eval_lambda_event_nll:
            metrics["event_nll"] = state["total_event_nll"] / n_batches
        if self.eval_lambda_hazard_discrete_nll:
            metrics["hazard_discrete_nll"] = state["total_hazard_discrete_nll"] / n_batches
        if self.eval_lambda_wass:
            metrics["wass"] = state["total_wass"] / n_batches
        event_sigma = self._running_event_sigma(state)
        if event_sigma is not None:
            metrics["event_sigma"] = event_sigma
        if self.plot_last_batch and state["last_batch"] is not None:
            from benchmarks.pkg.training.visualization import draw_batch

            plot_path = self._plot_path(state)
            draw_batch(
                **state["last_batch"],
                save_path=plot_path,
                title_prefix=f"epoch {state.get('epoch')} validation",
                show=self.plot_show,
            )
            if plot_path is not None:
                metrics["diagnostic_plot"] = str(plot_path)
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
            parts = [f"Epoch {epoch} [{batch_idx + 1}/{n_batches}]", f"Loss {metrics['loss']:.4f}"]
            if self.lambda_ce:
                parts.append(f"CE {metrics['ce']:.4f}")
            if self.lambda_event_nll:
                parts.append(f"EventNLL {metrics['event_nll']:.4f}")
            if self.lambda_hazard_discrete_nll:
                parts.append(f"HazardNLL {metrics['hazard_discrete_nll']:.4f}")
            if self.lambda_time:
                parts.append(f"RMSE {metrics['rmse']:.4f}")
            if self.lambda_kl:
                parts.append(f"KL {metrics['kl']:.4f}")
            if self.lambda_wass:
                parts.append(f"WASS {metrics['wass']:.4f}")
            parts.append(f"NRMSE {metrics['nrmse']:.4f}")
            if self.lambda_entropy:
                parts.append(f"Entropy {metrics['entropy']:.3f}")
            if "event_sigma" in metrics:
                parts.append(f"Sigma {metrics['event_sigma']:.3f}")
            return " ".join(parts)
        return (
            f"Val [{batch_idx + 1}/{n_batches}] Loss {metrics['loss']:.4f} "
            f"CE {metrics['ce']:.4f} "
            f"{'EventNLL ' + format(metrics['event_nll'], '.4f') + ' ' if self.eval_lambda_event_nll else ''}"
            f"{'HazardNLL ' + format(metrics['hazard_discrete_nll'], '.4f') + ' ' if self.eval_lambda_hazard_discrete_nll else ''}"
            f"{'WASS ' + format(metrics['wass'], '.4f') + ' ' if self.eval_lambda_wass else ''}"
            f"{'Sigma ' + format(metrics['event_sigma'], '.3f') + ' ' if 'event_sigma' in metrics else ''}"
            f"RMSE {metrics['rmse']:.4f} NRMSE {metrics['nrmse']:.4f}"
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
        log_p, p = self._event_distribution(z, self.temperature)
        entropy = posterior_entropy(p, log_p)

        if self.lambda_ce:
            ce = soft_label_cross_entropy(log_p, q, probabilities=p, focal_gamma=self.lambda_focal)
        else:
            ce = torch.tensor(0.0, device=self.device)

        t_grid = (torch.arange(T, device=self.device, dtype=z.dtype) * dt)[None, :]
        t_hat = expected_time(p, t_grid)
        rmse = F.mse_loss(t_hat, y_rel).sqrt()
        event_sigma = self._model_event_sigma()
        event_nll = (
            event_mixture_nll(
                log_p,
                y_rel,
                t_grid,
                sigma=self.sigma,
                event_sigma=event_sigma,
                kernel=self.event_nll_kernel,
                df=self.event_nll_df,
                mixture_weight=self.event_nll_mixture_weight,
                mixture_sigma_narrow=self.event_nll_mixture_sigma_narrow,
                mixture_sigma_wide=self.event_nll_mixture_sigma_wide,
            )
            if self.lambda_event_nll
            else torch.tensor(0.0, device=self.device)
        )
        hazard_nll = (
            hazard_discrete_nll(z, y_rel, dt=dt, temperature=self.temperature)
            if self.lambda_hazard_discrete_nll
            else torch.tensor(0.0, device=self.device)
        )
        kl = posterior_kl(q, log_p) if self.lambda_kl else torch.tensor(0.0, device=self.device)
        if self.lambda_wass > 0:
            wass = cdf_distance(p, q, dt)
        else:
            wass = torch.tensor(0.0, device=self.device)

        total = (
            self.lambda_ce * ce
            + self.lambda_event_nll * event_nll
            + self.lambda_hazard_discrete_nll * hazard_nll
            + self.lambda_time * rmse
            + self.lambda_kl * kl
            + self.lambda_wass * wass
            + self.lambda_entropy * entropy
        )
        return {
            "total": total,
            "ce": ce,
            "event_nll": event_nll,
            "hazard_discrete_nll": hazard_nll,
            "rmse": rmse,
            "kl": kl,
            "wass": wass,
            "entropy": entropy,
            "t_hat": t_hat,
            "event_sigma": event_sigma,
        }

    def _event_distribution(self, logits: torch.Tensor, temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Return temporal event log-probabilities and probabilities."""
        if self.readout_distribution == "softmax":
            log_p = F.log_softmax(logits / float(temperature), dim=-1)
        elif self.readout_distribution == "hazard":
            log_p = hazard_logits_to_log_pmf(
                logits,
                temperature=float(temperature),
                condition_inside=self.hazard_condition_inside,
            )
        else:
            raise RuntimeError(f"Unsupported readout distribution: {self.readout_distribution!r}")
        return log_p, log_p.exp()

    @staticmethod
    def _normalize_readout_distribution(readout_distribution: str) -> str:
        """Normalize configured temporal readout names."""
        name = str(readout_distribution).lower().replace("-", "_")
        if name in {"softmax", "soft_argmax", "event_softmax"}:
            return "softmax"
        if name in {"hazard", "survival", "event_hazard"}:
            return "hazard"
        raise ValueError(f"Unknown readout_distribution: {readout_distribution!r}")

    def _model_event_sigma(self) -> torch.Tensor | None:
        """Return model-provided EventNLL observation scales when available."""
        getter = getattr(self.model, "event_observation_sigma", None)
        if getter is None:
            return None
        return getter()

    def _record_event_sigma(self, state: dict[str, Any], sigma: torch.Tensor | None, batch_size: int) -> None:
        """Accumulate model-provided EventNLL sigma for epoch-level logging."""
        if sigma is None:
            return
        values = sigma.detach().view(-1)
        if values.numel() == 1:
            state["total_event_sigma"] += float(values.item()) * int(batch_size)
            state["n_event_sigma"] += int(batch_size)
            return
        state["total_event_sigma"] += float(values.sum().item())
        state["n_event_sigma"] += int(values.numel())

    def _running_event_sigma(self, state: dict[str, Any]) -> float | None:
        """Return the running mean EventNLL sigma if a model provides it."""
        count = int(state.get("n_event_sigma", 0))
        if count <= 0:
            return None
        return float(state["total_event_sigma"] / count)

    def _running_train_metrics(self, state: dict[str, Any]) -> dict[str, float]:
        rmse_ds = (state["sse"] / max(state["n_samples"], 1)) ** 0.5
        default_rmse = state.get("default_rmse")
        if default_rmse:
            return {"nrmse": rmse_ds / default_rmse}

        mean_y = state["sum_y"] / max(state["n_samples"], 1)
        var_y = max((state["sum_y2"] / max(state["n_samples"], 1)) - mean_y**2, 1e-12)
        return {"nrmse": rmse_ds / (var_y**0.5)}

    def _running_eval_metrics(self, state: dict[str, Any]) -> dict[str, float]:
        rmse = (state["sse"] / max(state["n_samples"], 1)) ** 0.5
        default_rmse = state.get("default_rmse")
        nrmse = rmse / default_rmse if default_rmse else rmse
        return {"nrmse": nrmse}

    def default_rmse_for(self, split: str) -> float | None:
        """Return the configured NRMSE denominator for one split."""
        return self.default_rmse_by_split.get(split)

    def _plot_path(self, state: dict[str, Any]) -> Path | None:
        """Return the diagnostic validation plot path for this epoch."""
        if self.plot_save_dir is None:
            return None
        epoch = int(state.get("epoch", 0))
        return self.plot_save_dir / f"epoch_{epoch:04d}_validation_batch.png"
