# sneddy_net.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class StdPerSample(nn.Module):
    """Per-sample normalization: subtract mean over time and divide by std over time.
    Independent of dataset/batch. For stable initialization."""
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
    def forward(self, x):  # x:(B,C,T)
        mu = x.mean(dim=-1, keepdim=True)
        sd = x.std(dim=-1, keepdim=True).clamp_min(self.eps)
        return (x - mu) / sd


class DSConv1d(nn.Module):
    """Depthwise separable 1D: depthwise (along feature channels) + pointwise 1x1."""
    def __init__(self, ch, k=7, stride=1, dilation=1):
        super().__init__()
        pad = ((k - 1) // 2) * dilation
        self.dw = nn.Conv1d(ch, ch, k, stride=stride, padding=pad,
                            dilation=dilation, groups=ch, bias=False)
        self.pw = nn.Conv1d(ch, ch, 1, bias=False)
    def forward(self, x):
        return self.pw(self.dw(x))


class ResBlock(nn.Module):
    """Lightweight residual block: DSConv → GN → GELU → DSConv → GN + skip."""
    def __init__(self, ch, k=7, dropout=0.0, dilation=1):
        super().__init__()
        self.conv1 = DSConv1d(ch, k=k, dilation=dilation)
        self.gn1   = nn.GroupNorm(1, ch)
        self.conv2 = DSConv1d(ch, k=k, dilation=1)
        self.gn2   = nn.GroupNorm(1, ch)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
    def forward(self, x):
        r = x
        x = self.conv1(x)
        x = self.act(self.gn1(x))
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.act(self.gn2(x) + r)
        return x


class ChannelSqueeze(nn.Module):
    """Channel squeeze over electrodes: 1x1 conv C_in→C_out (learned channel mixing)."""
    def __init__(self, c_in, c_out):
        super().__init__()
        self.proj = nn.Conv1d(c_in, c_out, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.proj.weight, nonlinearity="linear")
    def forward(self, x):
        return self.proj(x)


class TimeDown(nn.Module):
    """Time downsampling: antialiasing (light depthwise) + AvgPool(stride=2)."""
    def __init__(self, ch, k=5):
        super().__init__()
        pad = (k - 1) // 2
        self.aa = nn.Conv1d(ch, ch, k, groups=ch, padding=pad, bias=False)
        with torch.no_grad():
            # triangle-shaped smoothing kernel
            w = torch.tensor([1, 2, 3, 2, 1], dtype=torch.float32)
            w = (w / w.sum()).view(1, 1, -1).repeat(ch, 1, 1)
            self.aa.weight.copy_(w)
        for p in self.aa.parameters():
            p.requires_grad_(False)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
    def forward(self, x):
        return self.pool(self.aa(x))


class SegmentStatPool(nn.Module):
    """Segmented pooling: splits time into S segments and returns mean/max for each.
    This highlights 'where' (beginning/end/middle)."""
    def __init__(self, segments=(2, 4)):
        super().__init__()
        self.segments = segments
    def forward(self, x):  # x:(B,F,T)
        B, F, T = x.shape
        outs = []
        for S in self.segments:
            seg_len = max(1, T // S)
            trimmed_T = seg_len * S
            if trimmed_T != T:
                x_cut = x[..., :trimmed_T]
            else:
                x_cut = x
            x_resh = x_cut.view(B, F, S, seg_len)  # (B,F,S,L)
            m = x_resh.mean(dim=-1)                # (B,F,S)
            a = x_resh.amax(dim=-1)                # (B,F,S)
            outs.append(m)
            outs.append(a)
        y = torch.cat([o.flatten(start_dim=1) for o in outs], dim=1)  # (B, F*sum(2*S_i))
        return y


class SneddyNet(nn.Module):
    """
    Simple baseline for (B, C, T), focused on 'where is the change' (beginning/end):
      - StdPerSample normalization
      - Channel squeeze (C -> c0)
      - 3 stages: [ResBlock x2 + TimeDown] with increased features and reduced time
      - SegmentStatPool on the last feature map (S=2,4)
      - MLP head -> n_outputs
    Interface compatible with braindecode models.
    """
    def __init__(
        self,
        n_chans: int,
        n_times: int,
        sfreq: int,
        n_outputs: int = 1,
        c0: int = 32,          # initial feature width after channel squeeze
        widen: int = 2,        # feature expansion coefficient per stage
        depth_per_stage: int = 2,
        dropout: float = 0.1,
        k: int = 7,
        use_norm: bool = True,
    ):
        super().__init__()
        self.n_chans = n_chans
        self.n_times = n_times
        self.sfreq   = sfreq
        self.n_outputs = n_outputs
        self.use_norm = use_norm

        self.norm = StdPerSample()

        # Channel squeeze: 129 -> 32 (by default)
        self.c_squeeze = ChannelSqueeze(n_chans, c0)

        # Pyramid stages: time //= 2 at each
        chs = [c0, c0 * widen, c0 * widen * 2]
        stages = []
        in_c = c0
        for out_c in chs:
            # expand features at the beginning of the stage
            if out_c != in_c:
                stages.append(nn.Conv1d(in_c, out_c, kernel_size=1, bias=False))
                stages.append(nn.GroupNorm(1, out_c))
                stages.append(nn.GELU())
            # residual blocks
            for d in range(depth_per_stage):
                stages.append(ResBlock(out_c, k=k, dropout=dropout, dilation=1))
            # time downsampling
            stages.append(TimeDown(out_c))  # stride=2
            in_c = out_c
        self.backbone = nn.Sequential(*stages)

        # Segmented pooling (explicitly highlight "early/late")
        self.segpool = SegmentStatPool(segments=(2, 4))  # gives mean/max per segment

        # Calculate feature size for head after the backbone
        with torch.no_grad():
            dummy = torch.zeros(1, n_chans, n_times)
            z = self.forward_features(dummy)
            feat_dim = z.shape[1]

        self.head = nn.Sequential(
            nn.Linear(feat_dim, max(64, feat_dim // 2)),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(max(64, feat_dim // 2), n_outputs),
        )

    def forward_features(self, x):
        if self.use_norm:
            x = self.norm(x)          # (B,C,T)
        x = self.c_squeeze(x)         # (B,c0,T)
        x = self.backbone(x)          # (B,F,T')
        z = self.segpool(x)           # (B, F * (2*sum(segments)))
        return z

    def forward(self, x):             # (B,C,T) -> (B,n_outputs)
        z = self.forward_features(x)
        y = self.head(z)
        return y
