"""Skip-free dilated TCN segmentation model for event-time benchmarks."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from benchmarks.pkg.models.layers import ChannelSqueeze, ResBlock, StdPerSample


def _make_dilation_schedule(dilations: Sequence[int], depth: int) -> tuple[int, ...]:
    """Repeat a dilation cycle until the requested depth is reached."""
    if depth <= 0:
        raise ValueError("depth must be positive.")
    if not dilations:
        raise ValueError("dilations must contain at least one value.")

    base = tuple(int(d) for d in dilations)
    if any(d <= 0 for d in base):
        raise ValueError("dilations must be positive integers.")

    schedule: list[int] = []
    while len(schedule) < depth:
        for dilation in base:
            schedule.append(dilation)
            if len(schedule) == depth:
                break
    return tuple(schedule)


class ETSTCN1D(nn.Module):
    """Resolution-preserving dilated TCN for event-time segmentation.

    The model uses the same benchmark-facing contract as ``EventTimeUNet1D``:
    an EEG window ``(B, C, T)`` is mapped to per-time logits ``(B, out_channels, T)``.
    Unlike the U-Net backbone, this architecture has no temporal downsampling,
    decoder, or skip fusion. It is intended as a strong skip-free control for
    testing whether event-time supervision depends on U-Net-style localization
    bias.
    """

    def __init__(
        self,
        n_chans: int,
        n_times: int,
        sfreq: float,
        c0: int = 192,
        depth: int = 20,
        dilations: Sequence[int] = (1, 2, 4, 8, 16),
        k: int = 15,
        dropout: float = 0.2,
        out_channels: int = 1,
        use_norm: bool = True,
    ):
        super().__init__()
        if k % 2 == 0:
            raise ValueError("ETSTCN1D requires an odd kernel size to preserve temporal length.")
        if c0 <= 0:
            raise ValueError("c0 must be positive.")
        if out_channels <= 0:
            raise ValueError("out_channels must be positive.")

        self.n_chans = int(n_chans)
        self.n_times = int(n_times)
        self.sfreq = float(sfreq)
        self.out_channels = int(out_channels)
        self.use_norm = bool(use_norm)
        self.c0 = int(c0)
        self.depth = int(depth)
        self.dilation_schedule = _make_dilation_schedule(dilations, self.depth)

        self.norm = StdPerSample()
        self.c_squeeze = ChannelSqueeze(self.n_chans, self.c0)
        self.blocks = nn.Sequential(
            *[
                ResBlock(self.c0, k=k, dropout=dropout, dilation=dilation)
                for dilation in self.dilation_schedule
            ]
        )
        self.head = nn.Sequential(
            nn.GroupNorm(1, self.c0),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(self.c0, self.out_channels, kernel_size=1, bias=True),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return full-resolution temporal features before the output head."""
        if x.ndim != 3:
            raise ValueError(f"Expected input with shape (B, C, T), got {tuple(x.shape)}.")
        if self.use_norm:
            x = self.norm(x)
        h = self.c_squeeze(x)
        return self.blocks(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-time logits."""
        _, _, time = x.shape
        h = self.forward_features(x)
        if h.shape[-1] != time:
            h = F.interpolate(h, size=time, mode="linear", align_corners=False)
        return self.head(h)

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        mode: str = "argmax",
        temperature: float = 1.0,
        window_sec: float = 2.0,
        return_var: bool = False,
    ):
        """Return predicted time in seconds relative to the window start."""
        logits = self.forward(x)
        if self.out_channels != 1:
            raise ValueError("predict() assumes out_channels==1 for time readout.")
        _, _, time = logits.shape
        dt = float(window_sec) / float(time)
        z = logits.squeeze(1)

        if mode == "argmax":
            idx = torch.argmax(z, dim=-1)
            t_hat = idx.to(z.dtype) * dt
            if not return_var:
                return t_hat
            var = torch.full_like(t_hat, fill_value=(dt**2))
            return t_hat, var

        if mode == "softargmax":
            prob = F.softmax(z / float(temperature), dim=-1)
            grid = torch.arange(time, device=z.device, dtype=z.dtype)[None, :]
            t_idx = (prob * grid).sum(dim=-1)
            t_hat = t_idx * dt
            if not return_var:
                return t_hat
            var = (prob * ((grid * dt - t_hat[:, None]) ** 2)).sum(dim=-1)
            return t_hat, var

        raise ValueError("mode must be 'argmax' or 'softargmax'.")

    @torch.no_grad()
    def predict_mask(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Return per-time probabilities from logits."""
        logits = self.forward(x)
        if self.out_channels != 1:
            raise ValueError("predict_mask() assumes out_channels==1.")
        return F.softmax(logits.squeeze(1) / float(temperature), dim=-1)
