from __future__ import annotations

import pickle
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


def find_project_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for path in [start, *start.parents]:
        if (path / "data").exists() and (path / "neurosned").exists():
            return path
        nested = path / "neurosned"
        if (nested / "data").exists() and (nested / "neurosned").exists():
            return nested
    raise FileNotFoundError("Could not find neurosned project root with data/ and neurosned/ directories.")


def ensure_project_on_path(project_root: Path) -> None:
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def choose_torch_device(prefer_cuda: bool = True, require_cuda: bool = False):
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        _ = torch.empty(1, device=device)
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        return device

    message = "CUDA is not usable. Falling back to CPU."
    if require_cuda:
        try:
            smi = subprocess.run(["nvidia-smi"], text=True, capture_output=True, timeout=10)
            message += "\n" + (smi.stdout or smi.stderr).strip()
        except Exception as exc:
            message += f"\nnvidia-smi failed: {type(exc).__name__}: {exc}"
        raise RuntimeError(message)
    print(message)
    return torch.device("cpu")


@dataclass
class SegmentationData:
    project_root: Path
    artifacts_dir: Path
    train_dataset: Dataset
    valid_dataset: Dataset
    test_dataset: Dataset
    train_metadata: object
    valid_metadata: object
    test_metadata: object
    default_rmse_train: float
    default_rmse_valid: float
    default_rmse_test: float
    data_config: dict


def load_segmentation_data(project_root: Path | None = None) -> SegmentationData:
    project_root = project_root or find_project_root()
    ensure_project_on_path(project_root)

    data_home = project_root / "data"
    new_validation_home = data_home / "new_validation"
    paths = {
        "train_5sec": new_validation_home / "r1_r8_train_5sec.pkl",
        "train_2sec": new_validation_home / "r1_r8_train.pkl",
        "valid_2sec": new_validation_home / "r9_r10_val.pkl",
        "valid_5sec": new_validation_home / "r9_r10_val_5sec.pkl",
        "test_2sec": new_validation_home / "r11_test.pkl",
    }
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing new-validation pickle files:\n" + "\n".join(str(path) for path in missing))

    with paths["train_5sec"].open("rb") as f:
        train_dataset = pickle.load(f)
    with paths["valid_2sec"].open("rb") as f:
        valid_dataset = pickle.load(f)
    with paths["test_2sec"].open("rb") as f:
        test_dataset = pickle.load(f)

    train_metadata = train_dataset.get_metadata()
    valid_metadata = valid_dataset.get_metadata()
    test_metadata = test_dataset.get_metadata()
    data_config = {name: str(path) for name, path in paths.items()}
    data_config["split"] = "R1-R8 train, R9-R10 validation, R11 holdout"

    print(f"Project root: {project_root}")
    print(f"Train dataset: {paths['train_5sec'].relative_to(data_home)} | windows={len(train_dataset):,}")
    print(f"Valid dataset: {paths['valid_2sec'].relative_to(data_home)} | windows={len(valid_dataset):,}")
    print(f"Holdout test dataset: {paths['test_2sec'].relative_to(data_home)} | windows={len(test_dataset):,}")

    return SegmentationData(
        project_root=project_root,
        artifacts_dir=project_root / "artifacts",
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
        train_metadata=train_metadata,
        valid_metadata=valid_metadata,
        test_metadata=test_metadata,
        default_rmse_train=train_metadata["target"].std(),
        default_rmse_valid=valid_metadata["target"].std(),
        default_rmse_test=test_metadata["target"].std(),
        data_config=data_config,
    )


def soft_label_1d(
    y_sec: torch.Tensor,
    T: int,
    dt: float,
    sigma: float | torch.Tensor = 0.12,
    density: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    y_sec = y_sec.to(torch.float32).view(-1)
    batch_size = y_sec.numel()
    device = y_sec.device
    u_star = (y_sec / dt).clamp(0.0, T - 1e-6).unsqueeze(1)
    grid = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(0)
    rel_sec = (grid - u_star) * dt
    if not torch.is_tensor(sigma):
        sigma = torch.tensor(float(sigma), device=device, dtype=torch.float32)
    sigma = sigma.to(torch.float32)
    sigma = sigma.view(-1, 1) if sigma.numel() > 1 else sigma.view(1, 1)
    if sigma.numel() == 1:
        sigma = sigma.expand(batch_size, 1)
    q = torch.exp(-0.5 * (rel_sec / sigma).pow(2))
    if density:
        return q / (q.sum(dim=1, keepdim=True) + eps)

    q = q / q.amax(dim=1, keepdim=True).clamp_min(eps)
    idx = u_star.round().long().clamp_(0, T - 1).squeeze(1)
    q[torch.arange(batch_size, device=device), idx] = 1.0
    return q.clamp_(0, 1)


class TrainCroppingDataset(Dataset):
    def __init__(
        self,
        base: Dataset,
        sfreq: float,
        crop_sec: float = 2.0,
        sigma: float = 0.12,
        cropping_offset: float = 0.2,
        crop_proba: float = 1.0,
        dropout_range: float = 0.2,
        dropout_proba: float = 0.5,
        use_channels: list | None = None,
        end_time: float | None = None,
        use_augmentation: bool = False,
    ):
        self.base = base
        self.sfreq = float(sfreq)
        self.crop_sec = float(crop_sec)
        self.T_crop = int(round(self.crop_sec * self.sfreq))
        self.sigma = float(sigma)
        self.cropping_offset = cropping_offset
        self.crop_proba = crop_proba
        self.dropout_proba = dropout_proba
        self.dropout_range = dropout_range
        self.use_channels = use_channels
        self.end_time = end_time
        self.use_augmentation = use_augmentation
        self.dt = 1.0 / self.sfreq

    def __len__(self):
        return len(self.base)

    def _crop_segment(self, X_full, y_abs):
        if torch.rand((), device=X_full.device) < self.crop_proba:
            start_second = 0.5 + random.uniform(-self.cropping_offset, self.cropping_offset)
            start_second = max(start_second, y_abs - 2.0)
            start_second = min(start_second, y_abs - self.cropping_offset)
            start_second = np.clip(start_second, 0.5 - self.cropping_offset, 0.5 + self.cropping_offset)
            start_second = np.clip(start_second, 0, self.end_time - self.crop_sec)
        else:
            start_second = 0.5

        start_point = int(round(start_second * self.sfreq))
        end_point = start_point + self.T_crop
        X_crop = X_full[:, start_point:end_point].contiguous().to(torch.float32)
        y_rel_sec = y_abs - start_second
        return X_crop, y_rel_sec

    def _augment_segment(self, X: torch.Tensor, y_rel_sec: float):
        channels, times = X.shape

        if torch.rand((), device=X.device) < 0.2:
            min_scale, max_scale = 0.8, 1.2
            lb = max(min_scale, (y_rel_sec / self.crop_sec) + 1e-6)
            scale = 1.0 if lb > max_scale else float(torch.empty((), device=X.device).uniform_(lb, max_scale))
            new_times = int(round(times * scale))
            X = F.interpolate(X.unsqueeze(0), size=new_times, mode="linear", align_corners=False).squeeze(0)
            X = X[:, :times] if new_times > times else F.pad(X, (0, times - new_times))
            y_rel_sec = y_rel_sec / scale

        if torch.rand((), device=X.device) < self.dropout_proba:
            drop = torch.rand((), device=X.device).item() * self.dropout_range
            if drop > 0:
                ch_mask = (torch.rand(channels, device=X.device) > drop).to(X.dtype).unsqueeze(-1)
                X = X * ch_mask

        if torch.rand((), device=X.device) < 0.25:
            seg_len = int(torch.randint(10, min(60, times) + 1, (1,), device=X.device))
            start = int(torch.randint(0, times - seg_len + 1, (1,), device=X.device))
            X[:, start:start + seg_len] = 0

        if torch.rand((), device=X.device) < 0.2:
            noise_std = 0.03 * torch.randn((), device=X.device).abs().item() + 0.01
            X = X + torch.randn_like(X) * noise_std

        return X.contiguous(), y_rel_sec

    def __getitem__(self, idx: int):
        X_full, y_abs = self.base[idx][0], self.base[idx][1]
        X_full = torch.as_tensor(X_full)
        y_abs = float(y_abs.item())

        if self.use_channels is not None:
            X_full = X_full[self.use_channels, :]
        if self.end_time is not None:
            t_end = min(X_full.shape[1], int(round(self.end_time * self.sfreq)))
            X_full = X_full[:, :t_end]

        X_crop, y_rel_sec = self._crop_segment(X_full, y_abs)
        if self.use_augmentation:
            X_crop, y_rel_sec = self._augment_segment(X_crop, y_rel_sec)

        q = soft_label_1d(
            torch.tensor([y_rel_sec], dtype=torch.float32),
            T=self.T_crop,
            dt=self.dt,
            sigma=self.sigma,
            density=True,
        ).squeeze(0)
        return X_crop, q, torch.tensor(y_rel_sec, dtype=torch.float32)
