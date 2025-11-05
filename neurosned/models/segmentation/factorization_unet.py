import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class StdPerSample(nn.Module):
    """Per-sample normalization over time."""
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
    def forward(self, x):  # x:(B,C,T)
        mu = x.mean(dim=-1, keepdim=True)
        sd = x.std(dim=-1, keepdim=True).clamp_min(self.eps)
        return (x - mu) / sd

class DSConv1d(nn.Module):
    """Depthwise separable 1D: depthwise + pointwise."""
    def __init__(self, ch, k=7, stride=1, dilation=1):
        super().__init__()
        pad = ((k - 1) // 2) * dilation
        self.dw = nn.Conv1d(ch, ch, k, stride=stride, padding=pad,
                            dilation=dilation, groups=ch, bias=False)
        self.pw = nn.Conv1d(ch, ch, 1, bias=False)
    def forward(self, x):
        return self.pw(self.dw(x))

class ResBlock(nn.Module):
    """Lightweight residual block with DSConv."""
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
    """Electrode mixing: 1x1 C_in→C_out."""
    def __init__(self, c_in, c_out):
        super().__init__()
        self.proj = nn.Conv1d(c_in, c_out, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.proj.weight, nonlinearity="linear")
    def forward(self, x):
        return self.proj(x)

class TimeDown(nn.Module):
    """Antialias + AvgPool(stride=2) over time."""
    def __init__(self, ch, k=5):
        super().__init__()
        pad = (k - 1) // 2
        self.aa = nn.Conv1d(ch, ch, k, groups=ch, padding=pad, bias=False)
        with torch.no_grad():
            w = torch.tensor([1, 2, 3, 2, 1], dtype=torch.float32)
            w = (w / w.sum()).view(1, 1, -1).repeat(ch, 1, 1)
            self.aa.weight.copy_(w)
        for p in self.aa.parameters():
            p.requires_grad_(False)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
    def forward(self, x):
        return self.pool(self.aa(x))

class UpBlock(nn.Module):
    """Linear upsample + skip concat + 1x1 fuse + ResBlock."""
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


# ---------- New: Factorized cross-channel interactions ----------
class FactorizedCrossChannel(nn.Module):
    """
    FM-like layer for cross-channel interactions.
    Input x: (B, C, T). Returns (B, F, T) — per factor.
    Implements 0.5 * sum_f[ (sum_c v_cf x_c)^2 - sum_c (v_cf^2 x_c^2) ] in factored form.
    """
    def __init__(self, n_chans: int, n_factors: int, dropout: float = 0.0, use_gn: bool = True):
        super().__init__()
        self.n_chans = n_chans
        self.n_factors = n_factors
        # Factor matrix V (C x F)
        self.V = nn.Parameter(torch.empty(n_chans, n_factors))
        nn.init.xavier_normal_(self.V)  # stable initialization
        self.in_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.out_norm = nn.GroupNorm(1, n_factors) if use_gn else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):  # x: (B, C, T)
        x = self.in_drop(x)
        # s = x^T V  -> (B, F, T)
        s = torch.einsum('bct,cf->bft', x, self.V)
        # sum_sq = (sum_c v_cf x_c)^2 ; sq_sum = sum_c (v_cf^2 x_c^2)
        sum_sq = s * s
        sq_sum = torch.einsum('bct,cf->bft', x * x, self.V * self.V)
        inter = 0.5 * (sum_sq - sq_sum)  # (B, F, T)
        return self.act(self.out_norm(inter))


# ---------- New: "Flexible" encoder/decoder for arbitrary number of stages ----------
class Encoder1DFlex(nn.Module):
    """
    Sneddy-style encoder with N stages: [enc(stage)->skip->down] * N, then bottleneck.
    Optionally, FM can be inserted at the output of each stage and fused with the features.
    """
    def __init__(self, c0=32, widen=2, n_stages=3, depth_per_stage=2, k=7, dropout=0.1,
                 use_stage_fm: bool = False, fm_factors: int = 32, fm_dropout: float = 0.0):
        super().__init__()
        assert n_stages >= 1, "n_stages must be >= 1"
        # Channels at each stage (before downsampling)
        chs = [c0]
        for _ in range(1, n_stages):
            chs.append(int(chs[-1] * widen))
        self.chs = chs

        enc, downs = [], []
        stage_fm, stage_fuse, stage_norm, stage_act = [], [], [], []
        in_c = c0
        for out_c in chs:
            stage = []
            if out_c != in_c:
                stage += [nn.Conv1d(in_c, out_c, 1, bias=False), nn.GroupNorm(1, out_c), nn.GELU()]
            for _ in range(depth_per_stage):
                stage.append(ResBlock(out_c, k=k, dropout=dropout, dilation=1))
            enc.append(nn.Sequential(*stage))
            # Optional FM insertion at the end of the stage
            if use_stage_fm:
                stage_fm.append(FactorizedCrossChannel(out_c, fm_factors, dropout=fm_dropout, use_gn=True))
                stage_fuse.append(nn.Conv1d(out_c + fm_factors, out_c, kernel_size=1, bias=False))
                stage_norm.append(nn.GroupNorm(1, out_c))
                stage_act.append(nn.GELU())
            else:
                stage_fm.append(None)
                stage_fuse.append(None)
                stage_norm.append(None)
                stage_act.append(None)
            downs.append(TimeDown(out_c))
            in_c = out_c

        self.encoder_blocks = nn.ModuleList(enc)
        self.downs = nn.ModuleList(downs)

        # Layers for FM insertion at each stage
        self.use_stage_fm = use_stage_fm
        if use_stage_fm:
            self.stage_fm = nn.ModuleList(stage_fm)
            self.stage_fuse = nn.ModuleList(stage_fuse)
            self.stage_norm = nn.ModuleList(stage_norm)
            self.stage_act = nn.ModuleList(stage_act)

        # Bottleneck at the lowest resolution
        bottleneck_ch = chs[-1]
        self.bottleneck = nn.Sequential(
            ResBlock(bottleneck_ch, k=k, dropout=dropout, dilation=1),
            ResBlock(bottleneck_ch, k=k, dropout=dropout, dilation=2),
        )

    def forward(self, x):
        skips = []
        h = x
        for i, (enc, down) in enumerate(zip(self.encoder_blocks, self.downs)):
            h = enc(h)
            if self.use_stage_fm:
                fm = self.stage_fm[i](h)  # (B, F, T_i)
                h = self.stage_act[i](self.stage_norm[i](self.stage_fuse[i](torch.cat([h, fm], dim=1))))
            skips.append(h)
            h = down(h)
        h = self.bottleneck(h)
        return h, skips  # h at lowest scale, skips from shallow to deep


class Decoder1DFlex(nn.Module):
    """U-Net-like decoder over N stages (symmetric to encoder)."""
    def __init__(self, chs, k=7, dropout=0.1):
        super().__init__()
        steps = []
        in_ch = chs[-1]  # input channel from bottleneck
        # upsample from deepest skip to shallow
        for i in reversed(range(len(chs))):
            skip_ch = chs[i]
            out_ch  = chs[i]  # keep same channels as skip
            steps.append(UpBlock(in_ch, skip_ch, out_ch, k=k, dropout=dropout))
            in_ch = out_ch
        self.upblocks = nn.ModuleList(steps)
        self.out_ch = chs[0]

    def forward(self, h, skips):
        for up, skip in zip(self.upblocks, reversed(skips)):
            h = up(h, skip)
        return h


# ---------- Frontend with FM concatenation and linear channel mixing ----------
class FactorizedFrontEnd(nn.Module):
    """
    1) ChannelSqueeze C->c0 (linear mixing of electrodes),
    2) FactorizedCrossChannel C->F (cross-channel interactions),
    3) fuse: concat along channels -> 1x1 conv back to c0.
    """
    def __init__(self, n_chans: int, c0: int, fm_factors: int, fm_dropout: float = 0.0):
        super().__init__()
        self.lin = ChannelSqueeze(n_chans, c0)
        self.fm  = FactorizedCrossChannel(n_chans, fm_factors, dropout=fm_dropout, use_gn=True)
        self.fuse = nn.Conv1d(c0 + fm_factors, c0, kernel_size=1, bias=False)
        self.gn = nn.GroupNorm(1, c0)
        self.act = nn.GELU()

    def forward(self, x):  # x: (B, C, T)
        a = self.lin(x)        # (B, c0, T)
        b = self.fm(x)         # (B, F,  T)
        y = torch.cat([a, b], dim=1)
        y = self.act(self.gn(self.fuse(y)))  # (B, c0, T)
        return y


# ---------- Main model ----------
class FactorizationSneddyUnet(nn.Module):
    """
    U-Net 1D with FM-style cross-channel interactions.
    Parameters allow to deepen the network (n_stages, depth_per_stage) and widen (widen, c0).
    """
    def __init__(
        self,
        n_chans: int,
        n_times: int,
        sfreq: int,
        c0: int = 32,
        widen: int = 2,
        n_stages: int = 3,
        depth_per_stage: int = 2,
        dropout: float = 0.1,
        k: int = 7,
        out_channels: int = 1,
        # FM config
        fm_factors_front: int = 32,
        fm_dropout_front: float = 0.0,
        use_stage_fm: bool = False,
        fm_factors_stage: int = 16,
        fm_dropout_stage: float = 0.0,
    ):
        super().__init__()
        self.n_chans = n_chans
        self.n_times = n_times
        self.sfreq   = sfreq
        self.out_channels = out_channels

        self.norm = StdPerSample()
        self.front = FactorizedFrontEnd(n_chans, c0, fm_factors=fm_factors_front, fm_dropout=fm_dropout_front)

        self.encoder = Encoder1DFlex(
            c0=c0, widen=widen, n_stages=n_stages, depth_per_stage=depth_per_stage,
            k=k, dropout=dropout,
            use_stage_fm=use_stage_fm, fm_factors=fm_factors_stage, fm_dropout=fm_dropout_stage
        )
        self.decoder = Decoder1DFlex(self.encoder.chs, k=k, dropout=dropout)

        self.head = nn.Conv1d(self.decoder.out_ch, out_channels, kernel_size=1, bias=True)

    def forward(self, x):  # x:(B,C,T)
        B, C, T = x.shape
        x = self.norm(x)
        x = self.front(x)                # (B, c0, T)
        h_low, skips = self.encoder(x)   # h_low at lowest time scale
        h = self.decoder(h_low, skips)   # (B, c0, ~T)
        if h.shape[-1] != T:
            h = F.interpolate(h, size=T, mode='linear', align_corners=False)
        logits = self.head(h)            # (B, out_channels, T)
        return logits

    @torch.no_grad()
    def predict(
        self,
        x,
        mode: str = "argmax",          # "argmax" | "softargmax"
        temperature: float = 1.0,      # used for softargmax
        window_sec: float = 2.0,       # window length corresponding to input
        return_var: bool = False
    ):
        """
        Returns:
          - t_hat_sec: (B,) time in seconds relative to window start
          - var_sec (optional): (B,) simple proxy for variance
        """
        logits = self.forward(x)
        if self.out_channels != 1:
            raise ValueError("predict() expects out_channels==1.")
        B, _, T = logits.shape
        dt = window_sec / T
        z = logits.squeeze(1)  # (B, T)

        if mode == "argmax":
            idx = torch.argmax(z, dim=-1)
            t_hat = idx.to(z.dtype) * dt
            if not return_var:
                return t_hat
            var = torch.full_like(t_hat, fill_value=(dt**2))
            return t_hat, var

        elif mode == "softargmax":
            p = F.softmax(z / temperature, dim=-1)
            grid = torch.arange(T, device=z.device, dtype=z.dtype)[None, :]
            t_idx = (p * grid).sum(dim=-1)
            t_hat = t_idx * dt
            if not return_var:
                return t_hat
            var = (p * ((grid * dt - t_hat[:, None])**2)).sum(dim=-1)
            return t_hat, var

        else:
            raise ValueError("mode must be 'argmax' or 'softargmax'.")

    @torch.no_grad()
    def predict_mask(self, x, temperature: float = 1.0):
        """Per-timestep probabilities (B, T) from logits (softmax over time)."""
        logits = self.forward(x)
        if self.out_channels != 1:
            raise ValueError("predict_mask() expects out_channels==1.")
        z = logits.squeeze(1)
        return F.softmax(z / temperature, dim=-1)
