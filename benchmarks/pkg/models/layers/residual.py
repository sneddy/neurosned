"""Residual blocks shared by benchmark-facing models."""

import torch
from torch import nn

from benchmarks.pkg.models.layers.core import DSConv1d, DropPath


class ResBlock(nn.Module):
    """Depthwise separable residual block."""

    def __init__(self, ch: int, k: int = 7, dropout: float = 0.0, dilation: int = 1):
        super().__init__()
        self.conv1 = DSConv1d(ch, k=k, dilation=dilation)
        self.gn1 = nn.GroupNorm(1, ch)
        self.conv2 = DSConv1d(ch, k=k, dilation=1)
        self.gn2 = nn.GroupNorm(1, ch)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual depthwise separable convolutions."""
        residual = x
        x = self.conv1(x)
        x = self.act(self.gn1(x))
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.gn2(x)
        x = self.act(x + residual)
        return x


class DropPathResBlock(nn.Module):
    """Depthwise separable residual block with stochastic depth."""

    def __init__(self, ch: int, k: int = 7, dropout: float = 0.0, dilation: int = 1, drop_path: float = 0.0):
        super().__init__()
        self.conv1 = DSConv1d(ch, k=k, dilation=dilation)
        self.gn1 = nn.GroupNorm(1, ch)
        self.conv2 = DSConv1d(ch, k=k, dilation=1)
        self.gn2 = nn.GroupNorm(1, ch)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual convolutions with optional stochastic depth."""
        residual = x
        x = self.conv1(x)
        x = self.act(self.gn1(x))
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.gn2(x)
        x = residual + self.drop_path(x)
        x = self.act(x)
        return x
