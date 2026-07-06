"""Segmentation datasets used by benchmark notebooks."""

import random

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from benchmarks.pkg.training.labels import soft_label_1d


class TrainCroppingDataset(Dataset):
    """Turn full EEG windows into cropped segmentation training samples.

    The wrapped pickle dataset is expected to yield `(X_full, y_abs, ...)`,
    where `X_full` has shape `(channels, time)` and `y_abs` is the reaction time
    in seconds from stimulus onset. Each item returns:

    - `X_crop`: a fixed-length EEG crop with shape `(channels, T_crop)`.
    - `q`: a 1D Gaussian soft label centered at the reaction time inside crop.
    - `y_rel_sec`: the scalar reaction time relative to crop start.

    `crop_sec` controls the crop duration. `end_time` trims late signal tails
    before cropping and defaults to the 5 s prepared-window length. With
    `crop_proba > 0`, the crop start is sampled from `crop_start_min` to
    `crop_start_max`, while preserving the target inside the crop. The legacy
    `cropping_offset` argument is a shorthand for `0.5 +/- cropping_offset`.

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
        cropping_offset: float | None = None,
        crop_start_min: float | None = None,
        crop_start_max: float | None = None,
        target_margin: float | None = None,
        crop_proba: float = 1.0,
        dropout_range: float = 0.2,
        dropout_proba: float = 0.5,
        scale_proba: float = 0.2,
        scale_min: float = 0.8,
        scale_max: float = 1.2,
        cutout_proba: float = 0.25,
        cutout_min_len: int = 10,
        cutout_max_len: int = 60,
        noise_proba: float = 0.2,
        noise_base_std: float = 0.01,
        noise_random_std: float = 0.03,
        use_channels: list | None = None,
        end_time: float | None = 5.0,
        use_augmentation: bool = False,
    ):
        self.base = base
        self.sfreq = float(sfreq)
        self.crop_sec = float(crop_sec)
        self.T_crop = int(round(self.crop_sec * self.sfreq))
        self.sigma = float(sigma)
        if cropping_offset is not None and (crop_start_min is not None or crop_start_max is not None):
            raise ValueError("Use either cropping_offset or crop_start_min/crop_start_max, not both.")
        if cropping_offset is not None:
            cropping_offset = float(cropping_offset)
            crop_start_min = 0.5 - cropping_offset
            crop_start_max = 0.5 + cropping_offset
        elif crop_start_min is None and crop_start_max is None:
            crop_start_min = crop_start_max = 0.5
        elif crop_start_min is None or crop_start_max is None:
            raise ValueError("crop_start_min and crop_start_max must be set together.")
        if float(crop_start_min) > float(crop_start_max):
            raise ValueError("crop_start_min must be <= crop_start_max.")
        if target_margin is not None and float(target_margin) < 0:
            raise ValueError("target_margin must be non-negative.")

        self.cropping_offset = cropping_offset
        self.crop_start_min = float(crop_start_min)
        self.crop_start_max = float(crop_start_max)
        self.target_margin = None if target_margin is None else float(target_margin)
        self.crop_proba = crop_proba
        self.dropout_proba = dropout_proba
        self.dropout_range = dropout_range
        self.scale_proba = scale_proba
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.cutout_proba = cutout_proba
        self.cutout_min_len = cutout_min_len
        self.cutout_max_len = cutout_max_len
        self.noise_proba = noise_proba
        self.noise_base_std = noise_base_std
        self.noise_random_std = noise_random_std
        self.use_channels = use_channels
        self.end_time = end_time
        self.use_augmentation = use_augmentation
        self.dt = 1.0 / self.sfreq

    def __len__(self):
        return len(self.base)

    def _crop_segment(self, X_full, y_abs):
        """
        Crops a fixed-length window from the configured start range.
        """
        if torch.rand((), device=X_full.device) < self.crop_proba:
            start_second = random.uniform(self.crop_start_min, self.crop_start_max)
            target_margin = 0.0 if self.target_margin is None else self.target_margin
            available_end = self.end_time if self.end_time is not None else X_full.shape[-1] / self.sfreq
            min_start = max(0.0, y_abs - self.crop_sec + target_margin)
            max_start = min(available_end - self.crop_sec, y_abs - target_margin)
            if min_start <= max_start:
                start_second = np.clip(start_second, min_start, max_start)
            else:
                start_second = np.clip(start_second, 0.0, available_end - self.crop_sec)
        else:
            start_second = 0.5
        start_point = int(round(start_second * self.sfreq))
        end_point = start_point + self.T_crop

        X_crop = X_full[:, start_point:end_point].contiguous().to(torch.float32)
        y_rel_sec = y_abs - start_second
        return X_crop, y_rel_sec

    def _augment_segment(self, X: torch.Tensor, y_rel_sec: float):
        C, T = X.shape

        if torch.rand((), device=X.device) < self.scale_proba:
            eps = 1e-6

            lb = max(self.scale_min, (y_rel_sec / self.crop_sec) + eps)
            ub = self.scale_max
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

        if self.cutout_proba > 0 and torch.rand((), device=X.device) < self.cutout_proba:
            max_len = min(self.cutout_max_len, T)
            if max_len >= self.cutout_min_len:
                seg_len = int(torch.randint(self.cutout_min_len, max_len + 1, (1,), device=X.device))
                start = int(torch.randint(0, T - seg_len + 1, (1,), device=X.device))
                X[:, start:start + seg_len] = 0

        if self.noise_proba > 0 and torch.rand((), device=X.device) < self.noise_proba:
            noise_std = self.noise_random_std * torch.randn((), device=X.device).abs().item() + self.noise_base_std
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
