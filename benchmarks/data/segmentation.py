"""Segmentation datasets used by benchmark notebooks."""

import random

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from benchmarks.training.labels import soft_label_1d


class TrainCroppingDataset(Dataset):
    """Turn full EEG windows into cropped segmentation training samples.

    The wrapped pickle dataset is expected to yield `(X_full, y_abs, ...)`,
    where `X_full` has shape `(channels, time)` and `y_abs` is the reaction time
    in seconds from stimulus onset. Each item returns:

    - `X_crop`: a fixed-length EEG crop with shape `(channels, T_crop)`.
    - `q`: a 1D Gaussian soft label centered at the reaction time inside crop.
    - `y_rel_sec`: the scalar reaction time relative to crop start.

    `crop_sec` controls the crop duration. `end_time` optionally trims late
    signal tails before cropping. With `crop_proba > 0`, the crop start is
    jittered around the default 0.5 s post-stimulus offset by
    `cropping_offset`, while preserving the target inside the crop.

    When `use_augmentation=True`, the dataset can apply time scaling, channel
    dropout, short temporal cutout, and Gaussian noise. These augmentations are
    training-only and are intentionally kept in this wrapper rather than in the
    validation loader.
    """

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
        """
        Crops a 2-second window that starts around the default 0.5 s offset
        with a random shift in [-cropping_offset, +cropping_offset] seconds.
        """
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
        C, T = X.shape

        if torch.rand((), device=X.device) < 0.2:
            min_scale, max_scale = 0.8, 1.2
            eps = 1e-6

            lb = max(min_scale, (y_rel_sec / self.crop_sec) + eps)
            ub = max_scale
            scale = 1.0 if lb > ub else float(torch.empty((), device=X.device).uniform_(lb, ub))
            new_T = int(round(T * scale))

            X = F.interpolate(X.unsqueeze(0), size=new_T, mode="linear", align_corners=False).squeeze(0)

            if new_T > T:
                X = X[:, :T]
            elif new_T < T:
                X = F.pad(X, (0, T - new_T))
            y_rel_sec = y_rel_sec / scale

        if torch.rand((), device=X.device) < self.dropout_proba:
            drop = torch.rand((), device=X.device).item() * self.dropout_range
            if drop > 0:
                ch_mask = (torch.rand(C, device=X.device) > drop).to(X.dtype).unsqueeze(-1)
                X = X * ch_mask

        if torch.rand((), device=X.device) < 0.25:
            seg_len = int(torch.randint(10, min(60, T) + 1, (1,), device=X.device))
            start = int(torch.randint(0, T - seg_len + 1, (1,), device=X.device))
            X[:, start:start + seg_len] = 0

        if torch.rand((), device=X.device) < 0.2:
            noise_std = 0.03 * torch.randn((), device=X.device).abs().item() + 0.01
            X = X + torch.randn_like(X) * noise_std

        return X.contiguous(), y_rel_sec

    def __getitem__(self, idx: int):
        batch = self.base[idx]
        X_full, y_abs = batch[0], batch[1]

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
