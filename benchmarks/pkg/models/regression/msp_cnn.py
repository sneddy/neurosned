"""MSP-CNN scalar RT regression baseline for benchmark runs."""

import torch
from torch import nn

from benchmarks.pkg.models.layers import ChannelSqueeze, ResBlock, SegmentStatPool, StdPerSample, TimeDown


class MSPCNN(nn.Module):
    """Multiscale segment-pooling CNN for scalar RT regression."""

    def __init__(
        self,
        n_chans: int,
        n_times: int,
        sfreq: int,
        n_outputs: int = 1,
        c0: int = 32,
        widen: int = 2,
        depth_per_stage: int = 2,
        dropout: float = 0.1,
        k: int = 7,
        use_norm: bool = True,
        segments=(2, 4),
    ):
        super().__init__()
        self.n_chans = n_chans
        self.n_times = n_times
        self.sfreq = sfreq
        self.n_outputs = n_outputs
        self.use_norm = use_norm

        self.norm = StdPerSample()
        self.c_squeeze = ChannelSqueeze(n_chans, c0)

        chs = [c0, c0 * widen, c0 * widen * 2]
        stages = []
        in_ch = c0
        for out_ch in chs:
            if out_ch != in_ch:
                stages.append(nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False))
                stages.append(nn.GroupNorm(1, out_ch))
                stages.append(nn.GELU())
            for _ in range(depth_per_stage):
                stages.append(ResBlock(out_ch, k=k, dropout=dropout, dilation=1))
            stages.append(TimeDown(out_ch))
            in_ch = out_ch
        self.backbone = nn.Sequential(*stages)

        self.segpool = SegmentStatPool(segments=segments)

        with torch.no_grad():
            dummy = torch.zeros(1, n_chans, n_times)
            features = self.forward_features(dummy)
            feat_dim = features.shape[1]

        self.head = nn.Sequential(
            nn.Linear(feat_dim, max(64, feat_dim // 2)),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(max(64, feat_dim // 2), n_outputs),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return pooled backbone features."""
        if self.use_norm:
            x = self.norm(x)
        x = self.c_squeeze(x)
        x = self.backbone(x)
        return self.segpool(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return direct scalar prediction."""
        features = self.forward_features(x)
        return self.head(features)
