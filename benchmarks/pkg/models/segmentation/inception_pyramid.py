"""Full-resolution Inception-style temporal pyramid for event-time benchmarks."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from benchmarks.pkg.models.layers import ChannelSqueeze, DropPath, ResBlock, StdPerSample


def _odd_kernel_from_seconds(scale_s: float, sfreq: float) -> int:
    """Convert a temporal scale in seconds to an odd sample kernel."""
    if scale_s <= 0:
        raise ValueError("All temporal scales must be positive.")
    samples = max(3, int(round(float(scale_s) * float(sfreq))))
    if samples % 2 == 0:
        samples += 1
    return samples


class TemporalScaleBranch(nn.Module):
    """Single full-resolution temporal branch for one receptive-field scale."""

    def __init__(self, ch: int, out_ch: int, kernel_size: int, dropout: float = 0.0):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve temporal length.")
        if ch <= 0 or out_ch <= 0:
            raise ValueError("ch and out_ch must be positive.")

        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.GroupNorm(1, ch),
            nn.GELU(),
            nn.Conv1d(
                ch,
                ch,
                kernel_size=kernel_size,
                padding=padding,
                groups=ch,
                bias=False,
            ),
            nn.Conv1d(ch, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(1, out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return branch features with the same temporal length as the input."""
        return self.net(x)


class InceptionPyramidBlock(nn.Module):
    """Parallel temporal scale bank followed by residual full-resolution refinement."""

    def __init__(
        self,
        ch: int,
        branch_ch: int,
        kernel_sizes: Sequence[int],
        refine_kernel: int = 11,
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        if not kernel_sizes:
            raise ValueError("kernel_sizes must contain at least one scale.")
        if refine_kernel % 2 == 0:
            raise ValueError("refine_kernel must be odd to preserve temporal length.")

        self.branches = nn.ModuleList(
            [TemporalScaleBranch(ch, branch_ch, kernel_size=k, dropout=dropout) for k in kernel_sizes]
        )
        fusion_ch = ch + branch_ch * len(self.branches)
        self.fuse = nn.Sequential(
            nn.Conv1d(fusion_ch, ch, kernel_size=1, bias=False),
            nn.GroupNorm(1, ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.drop_path = DropPath(drop_path)
        self.refine = ResBlock(ch, k=refine_kernel, dropout=dropout, dilation=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-scale filtering without changing temporal resolution."""
        multi_scale = [branch(x) for branch in self.branches]
        h = self.fuse(torch.cat([x, *multi_scale], dim=1))
        return self.refine(x + self.drop_path(h))


class ETSInceptionPyramid1D(nn.Module):
    """Full-resolution multi-scale temporal segmenter for event-time logits.

    The architecture keeps the benchmark-facing segmentation contract
    ``(B, C, T) -> (B, out_channels, T)``. It provides an explicit temporal
    scale bank at every block, but does not use temporal pooling, a decoder, or
    U-Net-style skip reconstruction. This makes it a direct architecture
    alternative to ETS-U-Net for testing whether event-time supervision depends
    on encoder-decoder localization bias.
    """

    def __init__(
        self,
        n_chans: int,
        n_times: int,
        sfreq: float,
        c0: int = 288,
        branch_ch: int = 96,
        depth: int = 6,
        scales_s: Sequence[float] = (0.05, 0.10, 0.25, 0.50),
        stem_kernel: int = 7,
        refine_kernel: int = 11,
        dropout: float = 0.2,
        drop_path: float = 0.0,
        out_channels: int = 1,
        use_norm: bool = True,
    ):
        super().__init__()
        if c0 <= 0:
            raise ValueError("c0 must be positive.")
        if branch_ch <= 0:
            raise ValueError("branch_ch must be positive.")
        if depth <= 0:
            raise ValueError("depth must be positive.")
        if stem_kernel % 2 == 0:
            raise ValueError("stem_kernel must be odd to preserve temporal length.")
        if out_channels <= 0:
            raise ValueError("out_channels must be positive.")
        if not scales_s:
            raise ValueError("scales_s must contain at least one temporal scale.")

        self.n_chans = int(n_chans)
        self.n_times = int(n_times)
        self.sfreq = float(sfreq)
        self.out_channels = int(out_channels)
        self.use_norm = bool(use_norm)
        self.c0 = int(c0)
        self.branch_ch = int(branch_ch)
        self.depth = int(depth)
        self.kernel_sizes = tuple(_odd_kernel_from_seconds(scale, self.sfreq) for scale in scales_s)

        self.norm = StdPerSample()
        self.c_squeeze = ChannelSqueeze(self.n_chans, self.c0)
        self.stem = ResBlock(self.c0, k=stem_kernel, dropout=dropout, dilation=1)

        drop_schedule = torch.linspace(0.0, float(drop_path), steps=self.depth).tolist()
        self.blocks = nn.Sequential(
            *[
                InceptionPyramidBlock(
                    self.c0,
                    branch_ch=self.branch_ch,
                    kernel_sizes=self.kernel_sizes,
                    refine_kernel=refine_kernel,
                    dropout=dropout,
                    drop_path=block_drop_path,
                )
                for block_drop_path in drop_schedule
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
        h = self.stem(h)
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
