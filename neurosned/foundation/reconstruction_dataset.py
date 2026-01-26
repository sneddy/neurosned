from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class ReconstructionPatchDataset(Dataset):
    """
    Dataset wrapper that turns full EEG trials into overlapping time patches for reconstruction
    pretraining (e.g., MAE-style objectives).

    Each base sample is expected to be either:
    - a tensor/ndarray shaped (C, T), or
    - a tuple where the first element is the EEG array (extra fields are ignored).

    Returns (patches, patch_starts_sec, x_full):
    - patches: (N_patches, C, patch_size)
    - patch_starts_sec: (N_patches,) start time of each patch in seconds
    - x_full: (C, T_aligned) EEG padded to the patch grid (right padding only)
    """

    def __init__(
        self,
        base: Dataset,
        sfreq: float,
        window_sec: float = 3.0,
        patch_size: int = 20,
        patch_overlap: int = 5,
        use_channels: Optional[Sequence[int]] = None,
        pad_to_window: bool = True,
        return_full: bool = True,
    ):
        if patch_size <= 0:
            raise ValueError("patch_size must be > 0.")
        if patch_overlap < 0 or patch_overlap >= patch_size:
            raise ValueError("patch_overlap must be in [0, patch_size).")

        self.base = base
        self.sfreq = float(sfreq)
        self.dt = 1.0 / self.sfreq
        self.window_sec = float(window_sec) if window_sec is not None else None
        self.window_samples = (
            int(round(self.window_sec * self.sfreq)) if self.window_sec is not None else None
        )
        self.patch_size = int(patch_size)
        self.patch_overlap = int(patch_overlap)
        self.stride = self.patch_size - self.patch_overlap
        self.use_channels = list(use_channels) if use_channels is not None else None
        self.pad_to_window = pad_to_window
        self.return_full = return_full

    def __len__(self) -> int:
        return len(self.base)

    def _align_window(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Trim/pad a trial to the target window length and report valid data length."""
        if self.window_samples is None:
            return x, x.shape[-1]

        target_len = self.window_samples
        current_len = x.shape[-1]

        if current_len >= target_len:
            return x[..., :target_len], target_len

        if not self.pad_to_window:
            return x, current_len

        pad = target_len - current_len
        x = F.pad(x, (0, pad))
        return x, current_len

    def _patchify(self, x: torch.Tensor, valid_len: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build overlapping patches (no masking; assumes fixed-length inputs).
        x: (C, T_aligned) after window alignment (may still be shorter than patch_size).
        valid_len: number of real (non-padded) time steps in x.
        """
        if x.shape[-1] < self.patch_size:
            pad_needed = self.patch_size - x.shape[-1]
            x = F.pad(x, (0, pad_needed))

        total_len = x.shape[-1]
        remainder = (total_len - self.patch_size) % self.stride
        if remainder != 0:
            pad_tail = self.stride - remainder
            x = F.pad(x, (0, pad_tail))
            total_len = x.shape[-1]

        num_patches = 1 + (total_len - self.patch_size) // self.stride
        patches = x.unfold(-1, self.patch_size, self.stride)  # (C, num_patches, patch_size)
        patches = patches.permute(1, 0, 2).contiguous()       # (num_patches, C, patch_size)

        starts = torch.arange(0, num_patches * self.stride, self.stride, device=x.device)
        patch_starts_sec = starts.to(torch.float32) * self.dt

        return patches, patch_starts_sec, x

    @staticmethod
    def collate_fn(batch):
        """
        Collate a list of samples into batch tensors.
        Expects each sample as (patches, patch_starts_sec, x_full).
        """
        patches, starts_sec, x_full = zip(*batch)
        return (
            torch.stack(patches),    # (B, P, C, T)
            torch.stack(starts_sec), # (B, P)
            torch.stack(x_full),     # (B, C, T_full)
        )

    def __getitem__(self, idx: int):
        sample = self.base[idx]
        x_full = sample[0] if isinstance(sample, (tuple, list)) else sample
        x_full = torch.as_tensor(x_full, dtype=torch.float32)

        if x_full.ndim != 2:
            raise ValueError(f"Expected EEG shaped (C, T), got shape {tuple(x_full.shape)}.")

        if self.use_channels is not None:
            x_full = x_full[self.use_channels, :]

        x_aligned, valid_len = self._align_window(x_full)
        patches, patch_starts_sec, x_grid = self._patchify(x_aligned, valid_len)

        if self.return_full:
            return patches, patch_starts_sec, x_grid

        return patches, patch_starts_sec
