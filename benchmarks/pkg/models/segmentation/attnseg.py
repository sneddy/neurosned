"""Attention-based temporal segmenter for event-time benchmarks."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from benchmarks.pkg.models.layers import ChannelSqueeze, DropPath, StdPerSample, TemporalSelfAttention


class AttnSegFeedForward(nn.Module):
    """Position-wise feed-forward branch for ``(B, C, T)`` features."""

    def __init__(self, ch: int, ff_mult: float = 2.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(ch * ff_mult)
        if hidden <= 0:
            raise ValueError("ff_mult produces a non-positive hidden dimension.")
        self.norm = nn.LayerNorm(ch)
        self.net = nn.Sequential(
            nn.Linear(ch, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, ch),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward branch and return ``(B, C, T)`` features."""
        xt = x.transpose(1, 2)
        xt = self.net(self.norm(xt))
        return xt.transpose(1, 2)


class AttnSegConvModule(nn.Module):
    """Depthwise temporal convolution branch from attention-convolution blocks."""

    def __init__(
        self,
        ch: int,
        kernel_size: int = 15,
        expansion: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve temporal length.")
        if expansion <= 0:
            raise ValueError("expansion must be positive.")

        inner_ch = ch * int(expansion)
        self.norm = nn.LayerNorm(ch)
        self.pointwise_in = nn.Conv1d(ch, inner_ch * 2, kernel_size=1, bias=True)
        self.glu = nn.GLU(dim=1)
        self.depthwise = nn.Conv1d(
            inner_ch,
            inner_ch,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=inner_ch,
            bias=False,
        )
        self.gn = nn.GroupNorm(1, inner_ch)
        self.act = nn.SiLU()
        self.pointwise_out = nn.Conv1d(inner_ch, ch, kernel_size=1, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply local temporal convolution and return ``(B, C, T)`` features."""
        h = self.norm(x.transpose(1, 2)).transpose(1, 2)
        h = self.pointwise_in(h)
        h = self.glu(h)
        h = self.depthwise(h)
        h = self.act(self.gn(h))
        h = self.pointwise_out(h)
        return self.dropout(h)


class TemporalAttnSegBlock(nn.Module):
    """Attention-convolution temporal block for full-resolution EEG features."""

    def __init__(
        self,
        ch: int,
        n_heads: int = 8,
        conv_kernel: int = 15,
        ff_mult: float = 2.0,
        conv_expansion: int = 2,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.ff1 = AttnSegFeedForward(ch, ff_mult=ff_mult, dropout=dropout)
        self.attn = TemporalSelfAttention(
            ch,
            n_heads=n_heads,
            attn_dropout=attn_dropout,
            drop_path=drop_path,
            pos_kernel=3,
        )
        self.conv = AttnSegConvModule(
            ch,
            kernel_size=conv_kernel,
            expansion=conv_expansion,
            dropout=dropout,
        )
        self.ff2 = AttnSegFeedForward(ch, ff_mult=ff_mult, dropout=dropout)
        self.drop_path = DropPath(drop_path)
        self.final_norm = nn.LayerNorm(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward, attention, convolution and final feed-forward branches."""
        x = x + 0.5 * self.drop_path(self.ff1(x))
        x = self.attn(x)
        x = x + self.drop_path(self.conv(x))
        x = x + 0.5 * self.drop_path(self.ff2(x))
        return self.final_norm(x.transpose(1, 2)).transpose(1, 2)


class ETSAttnSeg1D(nn.Module):
    """Full-resolution attention-based segmenter for event-time logits.

    The model keeps the benchmark-facing segmentation contract
    ``(B, C, T) -> (B, out_channels, T)`` while using a different inductive bias
    from ETS-U-Net. It has no encoder-decoder path or skip fusion; each block
    combines global temporal self-attention with local depthwise temporal
    convolution.
    """

    def __init__(
        self,
        n_chans: int,
        n_times: int,
        sfreq: float,
        c0: int = 160,
        depth: int = 8,
        n_heads: int = 8,
        conv_kernel: int = 15,
        ff_mult: float = 2.0,
        conv_expansion: int = 2,
        dropout: float = 0.2,
        attn_dropout: float = 0.1,
        drop_path: float = 0.0,
        out_channels: int = 1,
        use_norm: bool = True,
    ):
        super().__init__()
        if c0 <= 0:
            raise ValueError("c0 must be positive.")
        if depth <= 0:
            raise ValueError("depth must be positive.")
        if n_heads <= 0:
            raise ValueError("n_heads must be positive.")
        if c0 % n_heads != 0:
            raise ValueError(f"c0={c0} must be divisible by n_heads={n_heads}.")
        if out_channels <= 0:
            raise ValueError("out_channels must be positive.")

        self.n_chans = int(n_chans)
        self.n_times = int(n_times)
        self.sfreq = float(sfreq)
        self.out_channels = int(out_channels)
        self.use_norm = bool(use_norm)
        self.c0 = int(c0)
        self.depth = int(depth)

        self.norm = StdPerSample()
        self.c_squeeze = ChannelSqueeze(self.n_chans, self.c0)

        drop_schedule = torch.linspace(0.0, float(drop_path), steps=self.depth).tolist()
        self.blocks = nn.Sequential(
            *[
                TemporalAttnSegBlock(
                    self.c0,
                    n_heads=n_heads,
                    conv_kernel=conv_kernel,
                    ff_mult=ff_mult,
                    conv_expansion=conv_expansion,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
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
