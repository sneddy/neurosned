"""Temporal attention layers shared by benchmark-facing models."""

from __future__ import annotations

import torch
from torch import nn

from benchmarks.pkg.models.layers.core import DropPath


class TemporalSelfAttention(nn.Module):
    """Pre-norm multi-head self-attention over the temporal axis.

    Inputs and outputs use convolutional layout ``(B, C, T)``. Internally, the
    tensor is transposed to ``(B, T, C)`` for ``nn.MultiheadAttention``. A small
    depthwise positional convolution is applied in the original layout before
    attention, matching the positional-bias pattern used by the attention U-Net
    variant while keeping the block reusable outside U-Net-style models.
    """

    def __init__(
        self,
        ch: int,
        n_heads: int = 4,
        attn_dropout: float = 0.0,
        drop_path: float = 0.0,
        pos_kernel: int = 3,
    ):
        super().__init__()
        if ch <= 0:
            raise ValueError("ch must be positive.")
        if n_heads <= 0:
            raise ValueError("n_heads must be positive.")
        if ch % n_heads != 0:
            raise ValueError(f"ch={ch} must be divisible by n_heads={n_heads}.")
        if pos_kernel % 2 == 0:
            raise ValueError("pos_kernel must be odd to preserve temporal length.")

        self.pos = nn.Conv1d(ch, ch, kernel_size=pos_kernel, padding=pos_kernel // 2, groups=ch, bias=True)
        self.norm = nn.LayerNorm(ch)
        self.attn = nn.MultiheadAttention(
            embed_dim=ch,
            num_heads=n_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply temporal self-attention and return ``(B, C, T)`` features."""
        if x.ndim != 3:
            raise ValueError(f"Expected input with shape (B, C, T), got {tuple(x.shape)}.")
        x = x + self.pos(x)
        xt = x.transpose(1, 2)
        q = self.norm(xt)
        h, _ = self.attn(q, q, q, need_weights=False)
        xt = xt + self.drop_path(h)
        return xt.transpose(1, 2)


class TemporalMHSABlock(nn.Module):
    """Temporal MHSA block with a feed-forward residual branch.

    This is the reusable form of the bottleneck attention block originally used
    in the attention U-Net. It keeps the same ``(B, C, T)`` contract while
    exposing the block through the shared layers package.
    """

    def __init__(
        self,
        ch: int,
        n_heads: int = 4,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        ff_mult: float = 2.0,
        drop_path: float = 0.0,
        pos_kernel: int = 3,
    ):
        super().__init__()
        hidden = int(ch * ff_mult)
        if hidden <= 0:
            raise ValueError("ff_mult produces a non-positive hidden dimension.")

        self.attn = TemporalSelfAttention(
            ch,
            n_heads=n_heads,
            attn_dropout=attn_dropout,
            drop_path=drop_path,
            pos_kernel=pos_kernel,
        )
        self.norm = nn.LayerNorm(ch)
        self.ff = nn.Sequential(
            nn.Linear(ch, hidden),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(hidden, ch),
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention and feed-forward residual branches."""
        x = self.attn(x)
        xt = x.transpose(1, 2)
        h = self.ff(self.norm(xt))
        xt = xt + self.drop_path(h)
        return xt.transpose(1, 2)
