# sneddy_net.py - point event detection as 1D segmentation
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class StdPerSample(nn.Module):
    """Per-sample normalization: subtract mean over time and divide by std over time.
    No dependence on dataset/batch. For stable startup."""
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
    def forward(self, x):  # x:(B,C,T)
        mu = x.mean(dim=-1, keepdim=True)
        sd = x.std(dim=-1, keepdim=True).clamp_min(self.eps)
        return (x - mu) / sd


class DSConv1d(nn.Module):
    """Depthwise separable 1D: depthwise (over feature channels) + pointwise 1x1."""
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
    """Channel squeeze: 1x1 conv C_in→C_out (learned channel mixing)."""
    def __init__(self, c_in, c_out):
        super().__init__()
        self.proj = nn.Conv1d(c_in, c_out, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.proj.weight, nonlinearity="linear")
    def forward(self, x):
        return self.proj(x)


class TimeDown(nn.Module):
    """Temporal downsampling: antialias (lightweight depthwise) + AvgPool(stride=2)."""
    def __init__(self, ch, k=5):
        super().__init__()
        pad = (k - 1) // 2
        self.aa = nn.Conv1d(ch, ch, k, groups=ch, padding=pad, bias=False)
        with torch.no_grad():
            # triangular smoothing kernel
            w = torch.tensor([1, 2, 3, 2, 1], dtype=torch.float32)
            w = (w / w.sum()).view(1, 1, -1).repeat(ch, 1, 1)
            self.aa.weight.copy_(w)
        for p in self.aa.parameters():
            p.requires_grad_(False)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
    def forward(self, x):
        return self.pool(self.aa(x))

class UpBlock(nn.Module):
    """Linear upsample + skip concat + fuse + DS refinement."""
    def __init__(self, in_ch, skip_ch, out_ch, k=7, dropout=0.0):
        super().__init__()
        self.fuse = nn.Conv1d(in_ch + skip_ch, out_ch, kernel_size=1, bias=False)
        self.gn   = nn.GroupNorm(1, out_ch)
        self.act  = nn.GELU()
        self.refine = ResBlock(out_ch, k=k, dropout=dropout, dilation=1)

    def forward(self, x_low, x_skip):
        x = F.interpolate(x_low, size=x_skip.shape[-1], mode='linear', align_corners=False)
        x = torch.cat([x, x_skip], dim=1)
        x = self.act(self.gn(self.fuse(x)))
        x = self.refine(x)
        return x


class Encoder1D(nn.Module):
    """Sneddy-style encoder: DS blocks + TimeDown, returns bottleneck and skips."""
    def __init__(self, c0=32, widen=2, depth_per_stage=2, k=7, dropout=0.1):
        super().__init__()
        self.chs = [c0, c0*widen, c0*widen*2]
        enc, downs = [], []
        in_c = c0
        for out_c in self.chs:
            stage = []
            if out_c != in_c:
                stage += [nn.Conv1d(in_c, out_c, 1, bias=False), nn.GroupNorm(1, out_c), nn.GELU()]
            for _ in range(depth_per_stage):
                stage.append(ResBlock(out_c, k=k, dropout=dropout, dilation=1))
            enc.append(nn.Sequential(*stage))
            downs.append(TimeDown(out_c))
            in_c = out_c
        self.encoder_blocks = nn.ModuleList(enc)
        self.downs = nn.ModuleList(downs)
        bottleneck_ch = self.chs[-1]
        self.bottleneck = nn.Sequential(
            ResBlock(bottleneck_ch, k=k, dropout=dropout, dilation=1),
            ResBlock(bottleneck_ch, k=k, dropout=dropout, dilation=2),
        )

    def forward(self, x):
        skips = []
        h = x
        for enc, down in zip(self.encoder_blocks, self.downs):
            h = enc(h)
            skips.append(h)
            h = down(h)
        h = self.bottleneck(h)
        return h, skips  # h at lowest scale, skips from shallow->deep


class Decoder1D(nn.Module):
    """U-Net-like decoder over 1D features."""
    def __init__(self, chs, k=7, dropout=0.1):
        super().__init__()
        # build 3 up steps to use all skips
        c0, c1, c2 = chs[0], chs[1], chs[2]
        steps = [
            (c2, c2, c1),  # r3->r2, use skip c2
            (c1, c1, c0),  # r2->r1, use skip c1
            (c0, c0, c0),  # r1->r0, use skip c0
        ]
        self.upblocks = nn.ModuleList([UpBlock(in_ch, skip_ch, out_ch, k=k, dropout=dropout)
                                       for (in_ch, skip_ch, out_ch) in steps])
        self.out_ch = c0

    def forward(self, h, skips):
        # skips: [r0(c0), r1(c1), r2(c2)]
        for up, skip in zip(self.upblocks, reversed(skips)):
            h = up(h, skip)
        return h  # at r0


class SneddySegUNet1D(nn.Module):
    """Encoder/Decoder 1D segmentation; forward -> logits only; time readout in predict()."""
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
        use_norm: bool = True
    ):
        super().__init__()
        self.n_chans = n_chans
        self.n_times = n_times
        self.sfreq   = sfreq
        self.out_channels = out_channels
        self.use_norm = use_norm

        self.norm = StdPerSample()
        self.c_squeeze = ChannelSqueeze(n_chans, c0)

        self.encoder = Encoder1D(c0=c0, widen=widen, depth_per_stage=depth_per_stage, k=k, dropout=dropout)
        self.decoder = Decoder1D(self.encoder.chs, k=k, dropout=dropout)

        self.head = nn.Conv1d(self.decoder.out_ch, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        B, C, T = x.shape
        if self.use_norm:
            x = self.norm(x)
        x = self.c_squeeze(x)
        h_low, skips = self.encoder(x)
        h = self.decoder(h_low, skips)
        if h.shape[-1] != T:
            h = F.interpolate(h, size=T, mode='linear', align_corners=False)
        logits = self.head(h)  # (B, out_channels, T)
        return logits

    @torch.no_grad()
    def predict(
        self,
        x,
        mode: str = "argmax",          # "argmax" | "softargmax"
        temperature: float = 1.0,      # used for softargmax
        window_sec: float = 2.0,       # your [0.5..2.5]s window length
        return_var: bool = False
    ):
        """
        Returns:
          - t_hat_sec: (B,) time in seconds relative to window start
          - var_sec (optional): (B,) predictive variance
        """
        logits = self.forward(x)
        if self.out_channels != 1:
            raise ValueError("predict() assumes out_channels==1 for time readout.")
        B, _, T = logits.shape
        dt = window_sec / T
        z = logits.squeeze(1)  # (B, T)

        if mode == "argmax":
            idx = torch.argmax(z, dim=-1)                       # (B,)
            t_hat = idx.to(z.dtype) * dt
            if not return_var:
                return t_hat
            # optional local quadratic fit variance (simple proxy)
            var = torch.full_like(t_hat, fill_value=(dt**2))
            return t_hat, var

        elif mode == "softargmax":
            p = F.softmax(z / temperature, dim=-1)              # (B, T)
            grid = torch.arange(T, device=z.device, dtype=z.dtype)[None, :]
            t_idx = (p * grid).sum(dim=-1)                      # (B,)
            t_hat = t_idx * dt
            if not return_var:
                return t_hat
            var = (p * ((grid * dt - t_hat[:, None])**2)).sum(dim=-1)
            return t_hat, var

        else:
            raise ValueError("mode must be 'argmax' or 'softargmax'.")

    @torch.no_grad()
    def predict_mask(self, x, temperature: float = 1.0):
        """Returns per-time probabilities (B, T) from logits (softmax over time)."""
        logits = self.forward(x)
        if self.out_channels != 1:
            raise ValueError("predict_mask() assumes out_channels==1.")
        z = logits.squeeze(1)
        return F.softmax(z / temperature, dim=-1)