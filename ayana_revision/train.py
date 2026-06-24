from __future__ import annotations

import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

from ayana_revision import logsave
from ayana_revision.data import TrainCroppingDataset, soft_label_1d
from ayana_revision.models import build_model, channels_for_model


OPTIMIZERS = {
    "Adam": torch.optim.Adam,
    "SGD": torch.optim.SGD,
}


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_checkpoint_path(path_like: str | Path | None, project_root: Path) -> Path | None:
    if path_like is None:
        return None
    path = Path(path_like)
    if path.is_absolute():
        return path

    candidates = [
        project_root / path,
        project_root / "notebooks" / "challenge_1" / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def draw_batch(
    X,
    y,
    z,
    t_hat_abs,
    ce,
    rmse_time,
    T,
    dt,
    win_offset,
    q,
    temperature,
    n_samples=4,
    rows=2,
    save_path=None,
    title_prefix=None,
    show=True,
    close=True,
):
    b = min(n_samples, X.shape[0])
    if b == 0:
        return None

    idxs = np.random.choice(X.shape[0], size=b, replace=False)
    rows = min(rows, b)
    cols = int(np.ceil(b / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 2), squeeze=False)
    axes = axes.ravel()
    t_grid = (np.arange(T) * dt) + win_offset

    for plot_idx, i in enumerate(idxs):
        ax = axes[plot_idx]
        p = torch.softmax(z[i] / temperature, dim=-1).detach().cpu().numpy()
        qi = q[i].detach().cpu().numpy()
        ax.plot(t_grid, p, label="Predicted p(t)", color="blue")
        ax.plot(t_grid, qi, label="Soft label q(t)", color="green", linestyle="--", alpha=0.8)
        ax.axvline(t_hat_abs[i].detach().cpu().item(), color="red", label="Predicted time", linestyle="--")
        ax.axvline(y[i].detach().cpu().item(), color="black", label="Actual time", linestyle="--")
        ax.set_title(f"Sample {i} | CE: {ce:.3f} | RMSE: {rmse_time:.3f}")
        ax.set_xlabel("Time (sec)")
        ax.set_ylabel("Probability / Weight")
        ax.legend()

    for ax in axes[b:]:
        ax.axis("off")

    if title_prefix:
        fig.suptitle(title_prefix, y=1.02)
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    if close:
        plt.close(fig)
    return fig


def train_one_epoch(
    dataloader: DataLoader,
    model: Module,
    optimizer,
    epoch: int,
    device,
    print_batch_stats: bool = True,
    temperature: float = 1.0,
    lambda_time: float = 1.0,
    lambda_ce: float = 1.0,
    lambda_kl: float = 0.0,
    lambda_wass: float = 0.0,
    lambda_entropy: float = 0.0,
    lambda_focal: float = 0.0,
    mixup_p: float = 0.5,
    mixup_alpha: float = 0.4,
    grad_accum: int = 1,
):
    model.train()
    optimizer.zero_grad()

    total_loss = total_ce = total_rmse = total_kl = total_wass = 0.0
    sse, n_samples = 0.0, 0
    sum_y, sum_y2 = 0.0, 0.0

    n_batches = len(dataloader)
    progress = tqdm(enumerate(dataloader), total=n_batches, disable=not print_batch_stats)

    for i, (X, q, y_rel) in progress:
        X = X.to(device).float()
        q = q.to(device).float()
        y_rel = y_rel.to(device).float().view(-1)

        batch_size, _, T = X.shape
        dt = 1.0 / float(model.sfreq)

        if mixup_p > 0 and torch.rand((), device=X.device) < mixup_p:
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
                lam = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample((batch_size,)).to(X.device)
                lam = torch.where(valid, lam, torch.ones_like(lam))
                X = lam.view(-1, 1, 1) * X + (1 - lam).view(-1, 1, 1) * X[partner]
                q = lam.view(-1, 1) * q + (1 - lam).view(-1, 1) * q[partner]
                y_rel = lam * y_rel + (1 - lam) * y_rel[partner]

        z = model(X).squeeze(1)
        log_p = F.log_softmax(z / temperature, dim=-1)
        p = torch.softmax(z / temperature, dim=-1)
        entropy = -(p * log_p).sum(dim=-1).mean()

        if lambda_ce:
            if lambda_focal > 0:
                focal_weight = (1.0 - p.detach()).pow(lambda_focal)
                ce = -((focal_weight * q) * log_p).sum(dim=-1).mean()
            else:
                ce = -(q * log_p).sum(dim=-1).mean()
        else:
            ce = torch.tensor(0.0, device=device)

        t_grid = (torch.arange(T, device=device, dtype=z.dtype) * dt)[None, :]
        t_hat = (p * t_grid).sum(dim=-1)
        rmse = F.mse_loss(t_hat, y_rel).sqrt()
        kl = (q * (torch.log(q + 1e-8) - log_p)).sum(dim=-1).mean() if lambda_kl else torch.tensor(0.0, device=device)
        wass = (torch.abs(torch.cumsum(p, -1) - torch.cumsum(q, -1)).sum(-1).mean()) * dt if lambda_wass else torch.tensor(0.0, device=device)

        loss = lambda_ce * ce + lambda_time * rmse + lambda_kl * kl + lambda_wass * wass + lambda_entropy * entropy
        (loss / grad_accum).backward()
        if (i + 1) % grad_accum == 0 or (i + 1) == n_batches:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += float(loss.detach())
        total_ce += float(ce.detach())
        total_rmse += float(rmse.detach())
        total_kl += float(kl.detach())
        total_wass += float(wass.detach())

        diff = t_hat.detach() - y_rel.detach()
        sse += torch.sum(diff * diff).item()
        n_samples += y_rel.numel()
        sum_y += y_rel.detach().sum().item()
        sum_y2 += torch.sum(y_rel.detach() * y_rel.detach()).item()

        if print_batch_stats:
            rmse_ds = (sse / max(n_samples, 1)) ** 0.5
            mean_y = sum_y / max(n_samples, 1)
            var_y = max((sum_y2 / max(n_samples, 1)) - mean_y**2, 1e-12)
            nrmse = rmse_ds / (var_y**0.5)
            progress.set_description(
                f"Epoch {epoch} [{i+1}/{n_batches}] Loss {loss:.4f} CE {ce:.4f} "
                f"RMSE {rmse:.4f} KL {kl:.4f} WASS {wass:.4f} NRMSE {nrmse:.4f}"
            )

    rmse_ds = (sse / max(n_samples, 1)) ** 0.5
    mean_y = sum_y / max(n_samples, 1)
    var_y = max((sum_y2 / max(n_samples, 1)) - mean_y**2, 1e-12)
    nrmse = rmse_ds / (var_y**0.5)
    avg_loss = total_loss / max(n_batches, 1)
    loss_dict = {
        "ce": total_ce / max(n_batches, 1),
        "rmse": total_rmse / max(n_batches, 1),
        "kl": total_kl / max(n_batches, 1),
        "wass": total_wass / max(n_batches, 1),
        "total": avg_loss,
    }
    print(f"NRMSE: {nrmse:.4f} Total loss: {total_loss:.4f} Total CE: {total_ce:.4f} Total RMSE: {total_rmse:.4f}")
    return avg_loss, nrmse, loss_dict


@torch.no_grad()
def valid_model(
    dataloader: DataLoader,
    model: Module,
    device,
    print_batch_stats: bool = True,
    channels_list: list = None,
    default_rmse: float = None,
    sigma: float = 0.10,
    temperature: float = 1.0,
    lambda_time: float = 1.0,
    win_offset: float = 0.5,
    use_soft_argmax: bool = True,
    plot_last_batch: bool = True,
    plot_save_path=None,
    plot_title_prefix: str = None,
    plot_show: bool = True,
):
    model.eval()
    n_batches = len(dataloader)
    it = tqdm(enumerate(dataloader), total=n_batches, disable=not print_batch_stats)

    total_loss = total_ce = total_rmse = 0.0
    sse_abs = 0.0
    n_samples = 0
    preds_abs = []

    for i, batch in it:
        X = batch[0].to(device).float()
        y = batch[1].to(device).float().view(-1)
        if channels_list is not None:
            X = X[:, channels_list, :]

        _, _, T = X.shape
        dt = 1.0 / float(model.sfreq)
        win_sec = T * dt
        y_rel = (y - win_offset).clamp(0.0, win_sec)
        q = soft_label_1d(y_rel, T=T, dt=dt, sigma=sigma, density=True)

        z = model(X).squeeze(1)
        log_p = F.log_softmax(z / temperature, dim=-1)
        ce = -(q * log_p).sum(dim=-1).mean()

        if use_soft_argmax:
            t_grid = (torch.arange(T, device=device, dtype=z.dtype) * dt)[None, :]
            p = torch.softmax(z / temperature, dim=-1)
            t_hat_abs = (p * t_grid).sum(dim=-1) + win_offset
        else:
            t_hat_abs = z.argmax(dim=-1) * dt + win_offset

        rmse_time = F.mse_loss(t_hat_abs, y).sqrt()
        loss = ce + lambda_time * rmse_time

        total_loss += float(loss)
        total_ce += float(ce)
        total_rmse += float(rmse_time)

        diff_abs = (t_hat_abs - y).detach()
        sse_abs += torch.sum(diff_abs * diff_abs).item()
        n_samples += y.numel()
        preds_abs.extend(t_hat_abs.cpu().numpy())

        if print_batch_stats:
            rmse = (sse_abs / max(n_samples, 1)) ** 0.5
            nrmse = rmse / default_rmse if default_rmse else rmse
            it.set_description(f"Val [{i+1}/{n_batches}] Loss {loss:.4f} CE {ce:.4f} RMSE {rmse_time:.4f} NRMSE {nrmse:.4f}")

    avg_loss = total_loss / n_batches if n_batches else float("nan")
    avg_ce = total_ce / n_batches if n_batches else float("nan")
    avg_rmse = total_rmse / n_batches if n_batches else float("nan")
    rmse = (sse_abs / max(n_samples, 1)) ** 0.5 * 1000
    nrmse = rmse / default_rmse if default_rmse else rmse

    print(f"Val NRMSE: {nrmse:.3f}, Val Loss: {avg_loss:.6f}, Val CE: {avg_ce:.6f}, Val RMSE: {avg_rmse:.6f}")
    if plot_last_batch:
        draw_batch(
            X=X.cpu(),
            y=y.cpu(),
            z=z.cpu(),
            t_hat_abs=t_hat_abs.cpu(),
            ce=float(ce),
            rmse_time=float(rmse_time),
            T=T,
            dt=dt,
            win_offset=win_offset,
            q=q.cpu(),
            temperature=temperature,
            save_path=plot_save_path,
            title_prefix=plot_title_prefix,
            show=plot_show,
        )
    return np.array(preds_abs), avg_loss, nrmse, avg_ce, avg_rmse


def run_experiment(
    experiment: dict[str, Any],
    data,
    device,
    *,
    notebook: str = "ayana_revision/segmentation.ipynb",
    print_batch_stats: bool = True,
) -> dict[str, Any]:
    experiment = deepcopy(experiment)
    train_config = experiment["train_config"]
    model_name = experiment["model"]
    experiment_name = experiment["name"]
    experiment_group = experiment.get("group", experiment_name)
    train_temperature = train_config.get("train_temperature", train_config["temperature"])
    eval_temperature = train_config.get("eval_temperature", train_config["temperature"])
    train_config.setdefault("train_lambda_ce", 1.0)
    train_config.setdefault("plateau_action", "stop")
    train_config.setdefault("lr_decay_factor", 0.5)
    train_config.setdefault("load_initial_checkpoint", False)
    train_config.setdefault("input_checkpoint_path", None)
    train_config.setdefault("require_initial_checkpoint", False)

    if device.type == "cpu":
        train_config = {**train_config, "n_epochs": 1, "early_stopping_patience": 1, "batch_size": min(16, train_config["batch_size"]), "eval_batch_size": min(32, train_config["eval_batch_size"])}
        print("CUDA is unavailable: using a 1-epoch CPU smoke test config.")

    set_seed(train_config.get("seed"))
    model = build_model(model_name, experiment["model_config"], device)
    channels_list = channels_for_model(model_name, model)

    resolved_initial_checkpoint = resolve_checkpoint_path(train_config.get("input_checkpoint_path"), data.project_root)
    loaded_initial_checkpoint = False
    if train_config.get("load_initial_checkpoint") and resolved_initial_checkpoint is not None:
        if resolved_initial_checkpoint.exists():
            print(f"Loading initial checkpoint: {resolved_initial_checkpoint}")
            model.load_state_dict(torch.load(resolved_initial_checkpoint, map_location=device))
            loaded_initial_checkpoint = True
        elif train_config.get("require_initial_checkpoint"):
            raise FileNotFoundError(f"Initial checkpoint not found: {resolved_initial_checkpoint}")
        else:
            print(f"Initial checkpoint not found, starting from scratch: {resolved_initial_checkpoint}")
    train_config["resolved_input_checkpoint_path"] = str(resolved_initial_checkpoint) if resolved_initial_checkpoint else None
    train_config["loaded_initial_checkpoint"] = loaded_initial_checkpoint

    num_workers = 0 if device.type == "cpu" else train_config.get("num_workers", 4)

    valid_loader = DataLoader(data.valid_dataset, batch_size=train_config["eval_batch_size"], shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(data.test_dataset, batch_size=train_config["eval_batch_size"], shuffle=False, num_workers=num_workers)
    train_dataset = TrainCroppingDataset(
        data.train_dataset,
        sfreq=100,
        crop_sec=2.0,
        sigma=train_config["sigma"],
        use_channels=channels_list,
        end_time=3,
        use_augmentation=train_config["use_augmentation"],
        cropping_offset=train_config["cropping_offset"],
        crop_proba=train_config["crop_proba"],
        dropout_proba=train_config["dropout_proba"],
        dropout_range=train_config["dropout_range"],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["batch_size"],
        shuffle=True,
        num_workers=(0 if device.type == "cpu" else train_config.get("train_num_workers", 8)),
    )

    paths, run_config, run_summary = logsave.create_run(
        artifacts_dir=data.artifacts_dir,
        experiment_name=experiment_name,
        experiment_group=experiment_group,
        model=model,
        model_name=model_name,
        model_config=experiment["model_config"],
        training_config=train_config,
        data_config=data.data_config,
        device=device,
        notebook=notebook,
        load_initial_checkpoint=loaded_initial_checkpoint,
        input_checkpoint_path=resolved_initial_checkpoint,
    )
    print(f"Artifacts for this run: {paths.run_dir}")

    optimizer_class = OPTIMIZERS[train_config["optimizer"]]
    current_lr = train_config["initial_lr"]
    optimizer = optimizer_class(
        model.parameters(),
        lr=current_lr,
        weight_decay=train_config["weight_decay"],
    )

    best_rmse = float("inf")
    best_epoch = None
    epochs_no_improve = 0
    epoch_history = []

    training_started_at = logsave.now_utc_iso()
    logsave.sync_if_cuda(device)
    training_start_perf = time.perf_counter()
    run_summary.update({"training_started_at": training_started_at, "status": "training_started"})
    run_config["timing"].update({"training_started_at": training_started_at})
    logsave.write_config(paths.config, run_config)
    logsave.save_summary(run_summary, paths)

    for epoch in range(1, train_config["n_epochs"] + 1):
        print(f"Epoch {epoch}/{train_config['n_epochs']}: ", end="")
        epoch_started_at = logsave.now_utc_iso()
        epoch_start_perf = time.perf_counter()

        logsave.sync_if_cuda(device)
        train_start_perf = time.perf_counter()
        train_loss, train_nrmse, train_loss_dict = train_one_epoch(
            train_loader,
            model,
            optimizer,
            epoch,
            device,
            print_batch_stats=print_batch_stats,
            mixup_p=train_config["mixup_p"],
            lambda_time=train_config["lambda_time"],
            lambda_ce=train_config["train_lambda_ce"],
            lambda_kl=train_config["lambda_kl"],
            lambda_wass=train_config["lambda_wass"],
            lambda_entropy=train_config["lambda_entropy"],
            lambda_focal=train_config["lambda_focal"],
            temperature=train_temperature,
            grad_accum=train_config["grad_accum"],
        )
        logsave.sync_if_cuda(device)
        train_seconds = time.perf_counter() - train_start_perf

        logsave.sync_if_cuda(device)
        valid_start_perf = time.perf_counter()
        val_preds, val_loss, val_nrmse, val_ce, val_mse = valid_model(
            valid_loader,
            model,
            device,
            print_batch_stats=print_batch_stats,
            sigma=train_config["sigma"],
            lambda_time=1.0,
            temperature=eval_temperature,
            channels_list=channels_list,
            default_rmse=data.default_rmse_valid,
            plot_last_batch=train_config["save_epoch_plots"],
            use_soft_argmax=True,
            plot_save_path=paths.val_img_dir / f"epoch_{epoch:04d}_r9_r10_val_batch.png",
            plot_title_prefix=f"{experiment_name} | epoch {epoch} | R9-R10 validation",
            plot_show=train_config["show_val_epoch_plots"],
        )
        logsave.sync_if_cuda(device)
        valid_seconds = time.perf_counter() - valid_start_perf

        epoch_seconds = time.perf_counter() - epoch_start_perf
        cumulative_training_seconds = time.perf_counter() - training_start_perf
        epoch_finished_at = logsave.now_utc_iso()
        epoch_row = {
            "epoch": epoch,
            "epoch_started_at": epoch_started_at,
            "epoch_finished_at": epoch_finished_at,
            "lr": current_lr,
            "train_seconds": train_seconds,
            "valid_seconds": valid_seconds,
            "epoch_seconds": epoch_seconds,
            "cumulative_training_seconds": cumulative_training_seconds,
            "train_loss": train_loss,
            "train_nrmse": train_nrmse,
            "val_loss": val_loss,
            "val_nrmse": val_nrmse,
            "val_ce": val_ce,
            "val_rmse": val_mse,
            **{f"train_{key}": value for key, value in train_loss_dict.items()},
        }
        epoch_history.append(epoch_row)
        logsave.append_metrics(paths.metrics, epoch_history)

        if val_nrmse < best_rmse:
            print(f"New best validation NRMSE: {val_nrmse:.6f} at epoch {epoch}")
            best_rmse = val_nrmse
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), paths.checkpoint)
            logsave.save_predictions(paths.best_val_predictions, val_preds, data.valid_metadata)
            run_summary.update(
                {
                    "best_epoch": best_epoch,
                    "best_val_nrmse": float(best_rmse),
                    "time_to_best_seconds": cumulative_training_seconds,
                    "best_epoch_finished_at": epoch_finished_at,
                    "status": "training_best_updated",
                }
            )
            logsave.save_summary(run_summary, paths)
        else:
            epochs_no_improve += 1
            print(f"No validation improvement for {epochs_no_improve}/{train_config['early_stopping_patience']} epochs.")
            if epochs_no_improve >= train_config["early_stopping_patience"]:
                if train_config["plateau_action"] == "halve_lr_reload_best":
                    reload_path = paths.checkpoint if paths.checkpoint.exists() else resolved_initial_checkpoint
                    if reload_path is not None and reload_path.exists():
                        print(f"Restart from checkpoint {reload_path}. Best Val NRMSE: {best_rmse:.6f} at epoch {best_epoch}.")
                        model.load_state_dict(torch.load(reload_path, map_location=device))
                    else:
                        print("No checkpoint is available for plateau reload; continuing from current weights.")
                    next_lr = current_lr * train_config["lr_decay_factor"]
                    print(f"Updating learning rate {current_lr} -> {next_lr}")
                    current_lr = next_lr
                    optimizer = optimizer_class(
                        model.parameters(),
                        lr=current_lr,
                        weight_decay=train_config["weight_decay"],
                    )
                    epochs_no_improve = 0
                    run_summary.update({"status": "training_lr_reduced", "lr": current_lr})
                    logsave.save_summary(run_summary, paths)
                else:
                    print(f"Early stopping. Best Val NRMSE: {best_rmse:.6f} at epoch {best_epoch}.")
                    break

    logsave.sync_if_cuda(device)
    training_wall_seconds = time.perf_counter() - training_start_perf
    logsave.finish_run(
        paths=paths,
        run_config=run_config,
        run_summary=run_summary,
        epoch_history=epoch_history,
        best_epoch=best_epoch,
        best_rmse=best_rmse,
        training_wall_seconds=training_wall_seconds,
    )

    if train_config.get("run_holdout_test", True):
        if not paths.checkpoint.exists():
            raise FileNotFoundError(f"No checkpoint from this run is available for holdout evaluation: {paths.checkpoint}")
        print(f"Loading best checkpoint for holdout test: {paths.checkpoint}")
        model.load_state_dict(torch.load(paths.checkpoint, map_location=device))

        holdout_started_at = logsave.now_utc_iso()
        logsave.sync_if_cuda(device)
        holdout_start_perf = time.perf_counter()
        test_preds, test_loss, test_nrmse, test_ce, test_rmse = valid_model(
            test_loader,
            model,
            device,
            print_batch_stats=print_batch_stats,
            sigma=train_config["sigma"],
            lambda_time=1.0,
            temperature=eval_temperature,
            channels_list=channels_list,
            default_rmse=data.default_rmse_test,
            plot_last_batch=True,
            use_soft_argmax=True,
            plot_save_path=paths.val_img_dir / "r11_holdout_batch.png",
            plot_title_prefix=f"{experiment_name} | R11 holdout",
            plot_show=train_config["show_val_epoch_plots"],
        )
        logsave.sync_if_cuda(device)
        holdout_seconds = time.perf_counter() - holdout_start_perf
        holdout_finished_at = logsave.now_utc_iso()

        logsave.save_predictions(paths.holdout_predictions, test_preds, data.test_metadata)
        run_summary.update(
            {
                "holdout_nrmse": float(test_nrmse),
                "holdout_loss": float(test_loss),
                "holdout_ce": float(test_ce),
                "holdout_rmse": float(test_rmse),
                "holdout_started_at": holdout_started_at,
                "holdout_finished_at": holdout_finished_at,
                "holdout_seconds": holdout_seconds,
                "status": "holdout_evaluated",
            }
        )
        run_config.setdefault("timing", {}).update(
            {
                "holdout_started_at": holdout_started_at,
                "holdout_finished_at": holdout_finished_at,
                "holdout_seconds": holdout_seconds,
            }
        )
        logsave.write_config(paths.config, run_config)
        logsave.save_summary(run_summary, paths)
        print(f"Holdout R11 NRMSE: {test_nrmse:.6f}")

    print(f"Run finished. Artifacts saved in: {paths.run_dir}")
    return {"paths": paths, "summary": run_summary, "config": run_config}
