"""Event-time 1D U-Net segmentation model for benchmark runs."""

import torch
import torch.nn.functional as F
from torch import nn

from benchmarks.pkg.models.layers import ChannelSqueeze, ResBlock, StdPerSample, TimeDown


class UpBlock(nn.Module):
    """Linear upsample, skip concat, 1x1 fuse and residual refinement."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, k: int = 7, dropout: float = 0.0):
        super().__init__()
        self.fuse = nn.Conv1d(in_ch + skip_ch, out_ch, kernel_size=1, bias=False)
        self.gn = nn.GroupNorm(1, out_ch)
        self.act = nn.GELU()
        self.refine = ResBlock(out_ch, k=k, dropout=dropout, dilation=1)

    def forward(self, x_low: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        """Upsample low-resolution features and merge with skip features."""
        x = F.interpolate(x_low, size=x_skip.shape[-1], mode="linear", align_corners=False)
        x = torch.cat([x, x_skip], dim=1)
        x = self.act(self.gn(self.fuse(x)))
        return self.refine(x)


class Encoder1D(nn.Module):
    """Event-time U-Net encoder with residual blocks and temporal downsampling."""

    def __init__(self, c0: int = 32, widen: int = 2, depth_per_stage: int = 2, k: int = 7, dropout: float = 0.1):
        super().__init__()
        self.chs = [c0, c0 * widen, c0 * widen * 2]
        encoder_blocks = []
        downs = []
        in_ch = c0
        for out_ch in self.chs:
            stage = []
            if out_ch != in_ch:
                stage += [nn.Conv1d(in_ch, out_ch, 1, bias=False), nn.GroupNorm(1, out_ch), nn.GELU()]
            for _ in range(depth_per_stage):
                stage.append(ResBlock(out_ch, k=k, dropout=dropout, dilation=1))
            encoder_blocks.append(nn.Sequential(*stage))
            downs.append(TimeDown(out_ch))
            in_ch = out_ch

        self.encoder_blocks = nn.ModuleList(encoder_blocks)
        self.downs = nn.ModuleList(downs)
        bottleneck_ch = self.chs[-1]
        self.bottleneck = nn.Sequential(
            ResBlock(bottleneck_ch, k=k, dropout=dropout, dilation=1),
            ResBlock(bottleneck_ch, k=k, dropout=dropout, dilation=2),
        )

    def forward(self, x: torch.Tensor):
        """Return bottleneck features and shallow-to-deep skips."""
        skips = []
        h = x
        for encoder_block, down in zip(self.encoder_blocks, self.downs):
            h = encoder_block(h)
            skips.append(h)
            h = down(h)
        h = self.bottleneck(h)
        return h, skips


class Decoder1D(nn.Module):
    """U-Net decoder over 1D feature maps."""

    def __init__(self, chs: list[int], k: int = 7, dropout: float = 0.1):
        super().__init__()
        c0, c1, c2 = chs[0], chs[1], chs[2]
        steps = [
            (c2, c2, c1),
            (c1, c1, c0),
            (c0, c0, c0),
        ]
        self.upblocks = nn.ModuleList(
            [UpBlock(in_ch, skip_ch, out_ch, k=k, dropout=dropout) for (in_ch, skip_ch, out_ch) in steps]
        )
        self.out_ch = c0

    def forward(self, h: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        """Decode bottleneck features with reversed skip connections."""
        for upblock, skip in zip(self.upblocks, reversed(skips)):
            h = upblock(h, skip)
        return h


class EventTimeUNet1D(nn.Module):
    """Encoder-decoder 1D segmentation model."""

    def __init__(
        self,
        n_chans: int,
        n_times: int,
        sfreq: int,
        c0: int = 32,
        widen: int = 2,
        depth_per_stage: int = 2,
        dropout: float = 0.1,
        k: int = 7,
        out_channels: int = 1,
        use_norm: bool = True,
    ):
        super().__init__()
        self.n_chans = n_chans
        self.n_times = n_times
        self.sfreq = sfreq
        self.out_channels = out_channels
        self.use_norm = use_norm

        self.norm = StdPerSample()
        self.c_squeeze = ChannelSqueeze(n_chans, c0)
        self.encoder = Encoder1D(c0=c0, widen=widen, depth_per_stage=depth_per_stage, k=k, dropout=dropout)
        self.decoder = Decoder1D(self.encoder.chs, k=k, dropout=dropout)
        self.head = nn.Conv1d(self.decoder.out_ch, out_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-time logits."""
        _, _, time = x.shape
        if self.use_norm:
            x = self.norm(x)
        x = self.c_squeeze(x)
        h_low, skips = self.encoder(x)
        h = self.decoder(h_low, skips)
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
        dt = window_sec / time
        z = logits.squeeze(1)

        if mode == "argmax":
            idx = torch.argmax(z, dim=-1)
            t_hat = idx.to(z.dtype) * dt
            if not return_var:
                return t_hat
            var = torch.full_like(t_hat, fill_value=(dt**2))
            return t_hat, var

        if mode == "softargmax":
            prob = F.softmax(z / temperature, dim=-1)
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
        return F.softmax(logits.squeeze(1) / temperature, dim=-1)
