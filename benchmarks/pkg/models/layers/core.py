"""Core layers shared by benchmark-facing models."""

import torch
from torch import nn


class DropPath(nn.Module):
    """Stochastic depth per sample."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly drop residual paths during training."""
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep) / keep
        return x * mask


class StdPerSample(nn.Module):
    """Normalize each sample/channel over time."""

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return time-standardized input."""
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True).clamp_min(self.eps)
        return (x - mean) / std


class DSConv1d(nn.Module):
    """Depthwise separable 1D convolution."""

    def __init__(self, ch: int, k: int = 7, stride: int = 1, dilation: int = 1, bias: bool = False):
        super().__init__()
        pad = ((k - 1) // 2) * dilation
        self.dw = nn.Conv1d(
            ch,
            ch,
            k,
            stride=stride,
            padding=pad,
            dilation=dilation,
            groups=ch,
            bias=bias,
        )
        self.pw = nn.Conv1d(ch, ch, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply depthwise then pointwise convolution."""
        return self.pw(self.dw(x))


class ChannelSqueeze(nn.Module):
    """Learnable 1x1 mixing over EEG channels."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.proj = nn.Conv1d(c_in, c_out, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.proj.weight, nonlinearity="linear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input channels to feature channels."""
        return self.proj(x)


class TimeDown(nn.Module):
    """Downsample time by 2 with fixed smoothing and average pooling."""

    def __init__(self, ch: int, k: int = 5):
        super().__init__()
        pad = (k - 1) // 2
        self.aa = nn.Conv1d(ch, ch, k, groups=ch, padding=pad, bias=False)
        with torch.no_grad():
            kernel = torch.tensor([1, 2, 3, 2, 1], dtype=torch.float32)
            kernel = (kernel / kernel.sum()).view(1, 1, -1).repeat(ch, 1, 1)
            self.aa.weight.copy_(kernel)
        for param in self.aa.parameters():
            param.requires_grad_(False)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Smooth and downsample the time axis."""
        return self.pool(self.aa(x))


class AntiAliasDown2(nn.Module):
    """Downsample time by 2 with configurable fixed smoothing."""

    def __init__(self, ch: int, k: int = 5):
        super().__init__()
        assert k in (3, 5, 7), "Use small odd k for stable smoothing"
        pad = (k - 1) // 2
        self.aa = nn.Conv1d(ch, ch, k, groups=ch, padding=pad, bias=False)
        with torch.no_grad():
            if k == 3:
                kernel = torch.tensor([1, 2, 1], dtype=torch.float32)
            elif k == 5:
                kernel = torch.tensor([1, 2, 3, 2, 1], dtype=torch.float32)
            else:
                kernel = torch.tensor([1, 2, 3, 3, 3, 2, 1], dtype=torch.float32)
            kernel = (kernel / kernel.sum()).view(1, 1, -1).repeat(ch, 1, 1)
            self.aa.weight.copy_(kernel)
        for param in self.aa.parameters():
            param.requires_grad_(False)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Smooth and downsample the time axis."""
        return self.pool(self.aa(x))
