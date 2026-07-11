"""ETR-CNN scalar RT regression baseline for benchmark runs."""

from __future__ import annotations

import torch
from torch import nn

from benchmarks.pkg.models.layers import AntiAliasDown2, ChannelSqueeze, ResBlock, SegmentStatPool, StdPerSample, TimeHead


def _make_schedule(base, depth: int | None):
    """Repeat a dilation schedule to the requested depth."""
    base = tuple(base)
    if depth is None:
        return base
    schedule = []
    while len(schedule) < depth:
        for dilation in base:
            schedule.append(dilation)
            if len(schedule) == depth:
                break
    return tuple(schedule)


class ETRCNN(nn.Module):
    """Event-time-readout CNN returning scalar RT seconds."""

    def __init__(
        self,
        n_chans: int,
        n_times: int,
        sfreq: float,
        n_outputs: int = 1,
        c0: int = 32,
        widen: int = 2,
        k: int = 7,
        dropout: float = 0.1,
        use_segpool: bool = False,
        segments=(4, 8, 16),
        hi_dilations=(1, 2, 4),
        hi_depth: int | None = None,
        lo_dilations=(1, 2, 4, 8),
        lo_depth: int = 4,
        num_lo_stacks: int = 1,
        downsample_once: bool = True,
        downsample_twice: bool = False,
        widen2: int = 2,
    ):
        super().__init__()
        assert n_outputs == 1, "ETRCNN is designed to output (B, 1) scalar RT predictions."
        self.n_chans = n_chans
        self.n_times = n_times
        self.sfreq = float(sfreq)
        self.window_sec = float(n_times) / float(sfreq)
        self.n_outputs = n_outputs

        self.norm = StdPerSample()
        self.c_squeeze = ChannelSqueeze(n_chans, c0)

        hi_sched = _make_schedule(hi_dilations, hi_depth)
        self.hi = nn.Sequential(*[ResBlock(c0, k=k, dropout=dropout, dilation=dilation) for dilation in hi_sched])

        self.downsample_once = downsample_once
        feat_ch = c0
        T_prime = n_times
        if downsample_once:
            self.down = AntiAliasDown2(feat_ch)
            T_prime //= 2
            c1 = feat_ch * widen
            self.expand = nn.Sequential(
                nn.Conv1d(feat_ch, c1, kernel_size=1, bias=False),
                nn.GroupNorm(1, c1),
                nn.GELU(),
            )
            feat_ch = c1
        else:
            self.down = nn.Identity()
            self.expand = nn.Identity()

        lo_sched = _make_schedule(lo_dilations, lo_depth)
        self.lo = nn.Sequential(*[ResBlock(feat_ch, k=k, dropout=dropout, dilation=dilation) for dilation in lo_sched])

        if num_lo_stacks > 1:
            extra = []
            for _ in range(num_lo_stacks - 1):
                extra.extend([ResBlock(feat_ch, k=k, dropout=dropout, dilation=dilation) for dilation in lo_sched])
            self.lo_extra = nn.Sequential(*extra)
        else:
            self.lo_extra = nn.Identity()

        self.downsample_twice = downsample_twice
        if downsample_twice:
            self.down2 = AntiAliasDown2(feat_ch)
            T_prime //= 2
            c2 = feat_ch * widen2
            self.expand2 = nn.Sequential(
                nn.Conv1d(feat_ch, c2, kernel_size=1, bias=False),
                nn.GroupNorm(1, c2),
                nn.GELU(),
            )
            feat_ch = c2
            self.lo2 = nn.Sequential(
                *[ResBlock(feat_ch, k=k, dropout=dropout, dilation=dilation) for dilation in lo_sched]
            )
        else:
            self.down2 = nn.Identity()
            self.expand2 = nn.Identity()
            self.lo2 = nn.Identity()

        self.T_prime = T_prime
        self.time_head = TimeHead(feat_ch, self.T_prime, use_context=True, ctx_dim=256)

        self.use_segpool = use_segpool
        if use_segpool:
            self.segpool = SegmentStatPool(segments=segments)
            with torch.no_grad():
                dummy = torch.zeros(1, feat_ch, self.T_prime)
                feat_dim = self.segpool(dummy).shape[1]
            self.ctx_head = nn.Sequential(
                nn.Linear(feat_dim, 128),
                nn.GELU(),
                nn.Dropout(0.30),
                nn.Linear(128, 32),
                nn.GELU(),
            )
            self.fuse = nn.Linear(32, self.T_prime)
            nn.init.zeros_(self.fuse.weight)
            nn.init.zeros_(self.fuse.bias)
        else:
            self.segpool = None
            self.ctx_head = None
            self.fuse = None

    @property
    def bin_sec(self) -> float:
        """Return the seconds represented by one output bin."""
        return self.window_sec / float(self.T_prime)

    def idx_to_sec(self, t_idx: torch.Tensor) -> torch.Tensor:
        """Convert soft time index to seconds."""
        return t_idx * self.bin_sec

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the final temporal feature map."""
        x = self.norm(x)
        x = self.c_squeeze(x)
        x = self.hi(x)
        x = self.down(x)
        x = self.expand(x)
        x = self.lo(x)
        x = self.lo_extra(x)
        x = self.down2(x)
        x = self.expand2(x)
        x = self.lo2(x)
        return x

    def forward(self, x: torch.Tensor, return_dict: bool = False):
        """Return seconds, or detailed intermediate outputs when requested."""
        feats = self.forward_features(x)
        logits, prob, t_idx = self.time_head(feats)

        if self.use_segpool:
            ctx_vec = self.segpool(feats)
            ctx = self.ctx_head(ctx_vec)
            bias = self.fuse(ctx)
            logits = logits + bias
            prob = torch.softmax(logits, dim=-1)
            idx = torch.arange(prob.size(-1), device=prob.device, dtype=prob.dtype)
            t_idx = (prob * idx).sum(dim=-1)

        t_sec = self.idx_to_sec(t_idx)

        if return_dict:
            return {
                "logits": logits,
                "prob": prob,
                "t_idx": t_idx,
                "t_sec": t_sec,
                "features": feats,
            }
        return t_sec.unsqueeze(-1)
