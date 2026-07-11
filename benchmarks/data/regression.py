"""Fixed-window datasets for regression experiments."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class FixedWindowDataset(Dataset):
    """Wrap fixed-window EEG samples for regression-style training.

    The wrapped dataset is expected to yield `(X, y, ...)`, where `X` is already
    a fixed-length EEG window and `y` is the scalar target. This wrapper keeps
    the same `(X, y)` contract while optionally selecting channels and applying
    per-sample training transforms.

    Batch-level transforms such as mixup stay in the trainer.
    """

    def __init__(
        self,
        base: Dataset,
        use_channels: list | None = None,
        use_augmentation: bool = False,
        channel_dropout_proba: float = 1.0,
        channel_dropout_max_ratio: float = 0.0,
        cutout_proba: float = 0.0,
        cutout_min_len: int = 10,
        cutout_max_len: int = 100,
        noise_proba: float = 0.0,
        noise_base_std: float = 0.01,
        noise_random_std: float = 0.03,
    ):
        self.base = base
        self.use_channels = use_channels
        self.use_augmentation = use_augmentation
        self.channel_dropout_proba = channel_dropout_proba
        self.channel_dropout_max_ratio = channel_dropout_max_ratio
        self.cutout_proba = cutout_proba
        self.cutout_min_len = cutout_min_len
        self.cutout_max_len = cutout_max_len
        self.noise_proba = noise_proba
        self.noise_base_std = noise_base_std
        self.noise_random_std = noise_random_std

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        batch = self.base[idx]
        X, y = torch.as_tensor(batch[0]).float(), torch.as_tensor(batch[1]).float()

        if self.use_channels is not None:
            X = X[self.use_channels, :]

        if self.use_augmentation:
            X = self._augment_window(X)

        return X.contiguous(), y

    def _augment_window(self, X: torch.Tensor) -> torch.Tensor:
        channels, times = X.shape

        if self.channel_dropout_max_ratio > 0 and torch.rand(1).item() < self.channel_dropout_proba:
            channel_dropout_ratio = torch.rand(1).item() * self.channel_dropout_max_ratio
            if channel_dropout_ratio > 0:
                mask = (torch.rand(channels, device=X.device) > channel_dropout_ratio).to(X.dtype).unsqueeze(-1)
                X = X * mask

        if self.cutout_proba > 0 and torch.rand(1).item() < self.cutout_proba:
            max_len = min(self.cutout_max_len, times)
            if max_len >= self.cutout_min_len:
                seg_len = int(torch.randint(self.cutout_min_len, max_len + 1, (1,)).item())
                start = int(torch.randint(0, times - seg_len + 1, (1,)).item())
                X[:, start:start + seg_len] = 0

        if self.noise_proba > 0 and torch.rand(1).item() < self.noise_proba:
            noise_std = self.noise_random_std * torch.randn(1).abs().item() + self.noise_base_std
            X = X + torch.randn_like(X) * noise_std

        return X


class ShiftedFixedWindowDataset(FixedWindowDataset):
    """Create fixed-length regression crops from longer EEG windows.

    This wrapper is intended for temporal-shift diagnostics: the base dataset
    should yield a longer window, for example the prepared 5 s windows, and this
    wrapper extracts a deterministic crop starting at `crop_start` seconds. It
    keeps the regression `(X, y)` contract used by `RegrTrainer`.
    """

    def __init__(
        self,
        base: Dataset,
        sfreq: float = 100.0,
        crop_sec: float = 2.0,
        crop_start: float = 0.5,
        target_mode: str = "absolute",
        **fixed_window_kwargs,
    ):
        super().__init__(base, **fixed_window_kwargs)
        self.sfreq = float(sfreq)
        self.crop_sec = float(crop_sec)
        self.crop_start = float(crop_start)
        self.target_mode = target_mode
        self.crop_samples = int(round(self.crop_sec * self.sfreq))
        self.start_sample = int(round(self.crop_start * self.sfreq))
        if self.crop_samples <= 0:
            raise ValueError("crop_sec must produce at least one sample.")
        if self.start_sample < 0:
            raise ValueError("crop_start must be non-negative.")
        if self.target_mode not in {"absolute", "relative"}:
            raise ValueError("target_mode must be 'absolute' or 'relative'.")

    def __getitem__(self, idx: int):
        batch = self.base[idx]
        X_full = torch.as_tensor(batch[0]).float()
        y_abs = torch.as_tensor(batch[1]).float()

        if self.use_channels is not None:
            X_full = X_full[self.use_channels, :]

        end_sample = self.start_sample + self.crop_samples
        if end_sample > X_full.shape[-1]:
            raise ValueError(
                f"Requested crop [{self.start_sample}:{end_sample}] exceeds "
                f"window length {X_full.shape[-1]} for index {idx}."
            )

        X = X_full[:, self.start_sample:end_sample]
        if self.use_augmentation:
            X = self._augment_window(X)

        if self.target_mode == "relative":
            y = y_abs - self.crop_start
        else:
            y = y_abs

        return X.contiguous(), y
