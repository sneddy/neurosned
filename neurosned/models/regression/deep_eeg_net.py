import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Normalization
# =========================
class StdPerSample(nn.Module):
    """Per-sample, per-channel standardization across time.
    For each sample and channel: (x - mean_t) / std_t.
    """
    def __init__(self, eps: float = 1e-5, unbiased: bool = False):
        super().__init__()
        self.eps = eps
        self.unbiased = unbiased
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, T)
        mu = x.mean(dim=-1, keepdim=True)
        sd = x.std(dim=-1, keepdim=True, unbiased=self.unbiased).clamp_min(self.eps)
        return (x - mu) / sd


# =========================
# Convolutions / Blocks
# =========================
class DSConv1d(nn.Module):
    """Depthwise-separable 1D conv: depthwise + 1x1 pointwise."""
    def __init__(self, ch: int, k: int = 7, dilation: int = 1, bias: bool = False):
        super().__init__()
        pad = ((k - 1) // 2) * dilation
        self.dw = nn.Conv1d(ch, ch, k, padding=pad, dilation=dilation, groups=ch, bias=bias)
        self.pw = nn.Conv1d(ch, ch, 1, bias=bias)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class SE1d(nn.Module):
    """Squeeze-and-Excitation for 1D feature maps."""
    def __init__(self, ch: int, reduction: int = 8):
        super().__init__()
        hidden = max(8, ch // reduction)
        self.fc1 = nn.Linear(ch, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, ch, bias=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        s = x.mean(dim=-1)                # (B, C)
        s = F.gelu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))    # (B, C)
        s = s.unsqueeze(-1)               # (B, C, 1)
        return x * s


class DSResSEBlock(nn.Module):
    """Deeper residual block with DSConv → GN → GELU → Dropout → DSConv → GN → SE → LayerScale → skip.
    - GroupNorm(1) behaves like per-time LayerNorm across channels.
    - LayerScale stabilizes very deep stacks.
    """
    def __init__(self, ch: int, k: int = 7, dropout: float = 0.0, dilation: int = 1, layerscale_init: float = 1e-4):
        super().__init__()
        self.conv1 = DSConv1d(ch, k=k, dilation=dilation, bias=False)
        self.gn1   = nn.GroupNorm(1, ch)
        self.conv2 = DSConv1d(ch, k=k, dilation=1, bias=False)
        self.gn2   = nn.GroupNorm(1, ch)
        self.se    = SE1d(ch, reduction=8)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
        # LayerScale: one learnable scalar per channel
        self.gamma = nn.Parameter(torch.full((ch, 1), layerscale_init, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        x = self.conv1(x)
        x = self.act(self.gn1(x))
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.gn2(x)
        x = self.se(x)
        x = r + self.gamma * x
        x = self.act(x)
        return x


class ChannelSqueeze(nn.Module):
    """Learnable 1x1 mixing over electrodes: Conv1d(C_in → C_out)."""
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.proj = nn.Conv1d(c_in, c_out, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.proj.weight, nonlinearity="linear")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class AntiAliasDown2(nn.Module):
    """Time downsample ×2 with fixed triangular smoothing + AvgPool(stride=2)."""
    def __init__(self, ch: int, k: int = 5):
        super().__init__()
        assert k in (3, 5, 7), "Use small odd k for stable smoothing"
        pad = (k - 1) // 2
        self.aa = nn.Conv1d(ch, ch, k, groups=ch, padding=pad, bias=False)
        with torch.no_grad():
            if k == 3:
                w = torch.tensor([1, 2, 1], dtype=torch.float32)
            elif k == 5:
                w = torch.tensor([1, 2, 3, 2, 1], dtype=torch.float32)
            else:
                w = torch.tensor([1, 2, 3, 3, 3, 2, 1], dtype=torch.float32)
            w = (w / w.sum()).view(1, 1, -1).repeat(ch, 1, 1)
            self.aa.weight.copy_(w)
        for p in self.aa.parameters():
            p.requires_grad_(False)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.aa(x))


# =========================
# Pyramid, Transformer, Time head
# =========================
class MultiScaleFuse(nn.Module):
    """Fuse features from multiple scales by:
       AdaptiveAvgPool to T_out → concat along channels → 1x1 fuse.
    """
    def __init__(self, in_chs: list[int], out_ch: int, T_out: int):
        super().__init__()
        self.pools = nn.ModuleList([nn.AdaptiveAvgPool1d(T_out) for _ in in_chs])
        self.fuse  = nn.Sequential(
            nn.Conv1d(sum(in_chs), out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(1, out_ch),
            nn.GELU(),
        )
    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        # feats: [(B, C_i, T_i), ...]
        pooled = [pool(f) for pool, f in zip(self.pools, feats)]
        x = torch.cat(pooled, dim=1)   # (B, sum(C_i), T_out)
        return self.fuse(x)            # (B, out_ch, T_out)


class PositionalEncoding1D(nn.Module):
    """Sinusoidal positional encoding for sequences of length T and dim d_model."""
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe, persistent=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        T = x.size(1)
        return x + self.pe[:, :T, :]


class TemporalTransformer(nn.Module):
    """Lightweight Transformer at low temporal resolution."""
    def __init__(self, in_ch: int, d_model: int = 256, num_heads: int = 8, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.proj_in  = nn.Conv1d(in_ch, d_model, kernel_size=1, bias=False)
        self.pe       = PositionalEncoding1D(d_model)
        enc_layer     = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_model*4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder  = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.proj_out = nn.Conv1d(d_model, in_ch, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T_low)
        z = self.proj_in(x).transpose(1, 2)   # (B, T_low, d_model)
        z = self.pe(z)
        z = self.encoder(z)                   # (B, T_low, d_model)
        z = z.transpose(1, 2)
        return self.proj_out(z) + x           # residual


class SegmentStatPool(nn.Module):
    """Segment-wise stat pooling over time. Returns flat vector (B, F * sum(2*S))."""
    def __init__(self, segments=(4, 8, 16)):
        super().__init__()
        self.segments = tuple(segments)
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, F, T)
        B, F, T = x.shape
        outs = []
        for S in self.segments:
            seg_len = max(1, T // S)
            trimmed_T = seg_len * S
            x_cut = x[..., :trimmed_T]
            x_resh = x_cut.view(B, F, S, seg_len)  # (B, F, S, L)
            m = x_resh.mean(dim=-1)                # (B, F, S)
            a = x_resh.amax(dim=-1)                # (B, F, S)
            outs.append(m)
            outs.append(a)
        y = torch.cat([o.flatten(start_dim=1) for o in outs], dim=1)
        return y


class TimeHead(nn.Module):
    """Per-time logits + softmax + soft-argmax index on a (B,F,T') map."""
    def __init__(self, feat_ch: int, T_prime: int, use_context: bool = True, ctx_dim: int = 256):
        super().__init__()
        self.T_prime = T_prime
        self.use_context = use_context
        self.score = nn.Conv1d(feat_ch, 1, kernel_size=1)

        if use_context:
            self.ctx_gate = nn.Conv1d(feat_ch, 1, kernel_size=1)
            self.ctx_proj = nn.Sequential(
                nn.Linear(feat_ch, ctx_dim), nn.GELU(), nn.Linear(ctx_dim, feat_ch), nn.GELU()
            )
        else:
            self.ctx_gate = None
            self.ctx_proj = None

    def forward(self, x: torch.Tensor):
        B, C, T = x.shape
        assert T == self.T_prime, f"Expected T'={self.T_prime}, got {T}"

        if self.use_context:
            g = self.ctx_gate(x).squeeze(1)                    # (B, T')
            w = torch.softmax(g, dim=-1)                       # (B, T')
            f_ctx = torch.einsum('bct,bt->bc', x, w)           # (B, C)
            f_ctx = self.ctx_proj(f_ctx).unsqueeze(-1)         # (B, C, 1)
            x = x + f_ctx

        logits = self.score(x).squeeze(1)                      # (B, T')
        prob = torch.softmax(logits, dim=-1)                   # (B, T')
        idx = torch.arange(T, device=x.device, dtype=prob.dtype)
        t_idx = (prob * idx).sum(dim=-1)                       # (B,)
        return logits, prob, t_idx


# =========================
# Utilities
# =========================
def _make_schedule(base, depth: int | None):
    """Repeat `base` pattern to reach exactly `depth` elements. If depth is None,
    return `base` unchanged."""
    base = tuple(base)
    if depth is None:
        return base
    sched = []
    while len(sched) < depth:
        for d in base:
            sched.append(d)
            if len(sched) == depth:
                break
    return tuple(sched)


# =========================
# Deep model
# =========================
class DeepEEGRTNet(nn.Module):
    """Deeper multi-scale EEG RT model (returns (B,1) by default).
    - 3-stage temporal pyramid: 200 → 100 → 50 → 25
    - Multi-scale fusion to T'=25
    - Optional low-res Transformer for global time reasoning
    - TimeHead produces per-bin distribution and soft-argmax time.

    Call with return_dict=True for detailed outputs.
    """
    def __init__(
        self,
        n_chans: int,
        n_times: int,
        sfreq: float,
        n_outputs: int = 1,            # must be 1 (seconds)
        # stem / width
        c0: int = 96,                   # initial channel squeeze
        widen1: int = 3,                # after 1st downsample
        widen2: int = 2,                # after 2nd downsample
        widen3: int = 2,                # after 3rd downsample
        k: int = 15,
        dropout: float = 0.5,
        layerscale_init: float = 1e-4,
        # depth schedules
        hi_dilations=(1, 2, 4, 8, 16),
        hi_depth: int = 24,             # high-res stack at T=200
        lo1_dilations=(1, 2, 4, 8, 16),
        lo1_depth: int = 16,            # stack at T=100
        lo2_dilations=(1, 2, 4, 8, 16),
        lo2_depth: int = 16,            # stack at T=50
        lo3_dilations=(1, 2, 4, 8, 16),
        lo3_depth: int = 12,            # stack at T=25
        # transformer (low-res)
        use_transformer: bool = True,
        transformer_dim: int = 256,
        transformer_heads: int = 8,
        transformer_layers: int = 2,
        transformer_dropout: float = 0.1,
        # optional segmental context bias
        use_segpool: bool = False,
        segments=(4, 8, 16),
    ):
        super().__init__()
        assert n_outputs == 1, "This model predicts a single scalar time in seconds."
        self.n_chans = n_chans
        self.n_times = n_times
        self.sfreq = float(sfreq)
        self.window_sec = float(n_times) / float(sfreq)
        self.n_outputs = n_outputs

        # --- Stem: normalization + electrode mixing ---
        self.norm = StdPerSample()
        self.c_squeeze = ChannelSqueeze(n_chans, c0)   # (B, c0, T=200)

        # --- Stage 0: high-res deep stack (T = 200) ---
        hi_sched = _make_schedule(hi_dilations, hi_depth)
        self.hi = nn.Sequential(*[
            DSResSEBlock(c0, k=k, dropout=dropout, dilation=d, layerscale_init=layerscale_init)
            for d in hi_sched
        ])

        # --- Downsample 1: 200 → 100 + expand width ---
        self.down1 = AntiAliasDown2(c0)
        c1 = c0 * widen1
        self.expand1 = nn.Sequential(
            nn.Conv1d(c0, c1, kernel_size=1, bias=False),
            nn.GroupNorm(1, c1),
            nn.GELU(),
        )
        lo1_sched = _make_schedule(lo1_dilations, lo1_depth)
        self.lo1 = nn.Sequential(*[
            DSResSEBlock(c1, k=k, dropout=dropout, dilation=d, layerscale_init=layerscale_init)
            for d in lo1_sched
        ])

        # --- Downsample 2: 100 → 50 + expand width ---
        self.down2 = AntiAliasDown2(c1)
        c2 = c1 * widen2
        self.expand2 = nn.Sequential(
            nn.Conv1d(c1, c2, kernel_size=1, bias=False),
            nn.GroupNorm(1, c2),
            nn.GELU(),
        )
        lo2_sched = _make_schedule(lo2_dilations, lo2_depth)
        self.lo2 = nn.Sequential(*[
            DSResSEBlock(c2, k=k, dropout=dropout, dilation=d, layerscale_init=layerscale_init)
            for d in lo2_sched
        ])

        # --- Downsample 3: 50 → 25 + expand width ---
        self.down3 = AntiAliasDown2(c2)
        c3 = c2 * widen3
        self.expand3 = nn.Sequential(
            nn.Conv1d(c2, c3, kernel_size=1, bias=False),
            nn.GroupNorm(1, c3),
            nn.GELU(),
        )
        lo3_sched = _make_schedule(lo3_dilations, lo3_depth)
        self.lo3 = nn.Sequential(*[
            DSResSEBlock(c3, k=k, dropout=dropout, dilation=d, layerscale_init=layerscale_init)
            for d in lo3_sched
        ])

        # --- Multi-scale fusion to T' = 25 ---
        T_prime = n_times // 2 // 2 // 2  # 200→100→50→25 (assumes n_times divisible by 8; with 200 it is)
        self.T_prime = T_prime
        fuse_out = c3                       # keep final width after fusion
        self.fuse = MultiScaleFuse(in_chs=[c0, c1, c2, c3], out_ch=fuse_out, T_out=T_prime)

        # --- Optional temporal transformer at low-res ---
        self.use_transformer = use_transformer
        if use_transformer:
            self.temporal_transformer = TemporalTransformer(
                in_ch=fuse_out, d_model=transformer_dim, num_heads=transformer_heads,
                num_layers=transformer_layers, dropout=transformer_dropout
            )
        else:
            self.temporal_transformer = nn.Identity()

        # --- Time head ---
        self.time_head = TimeHead(fuse_out, self.T_prime, use_context=True, ctx_dim=256)

        # --- Optional global context via segmental stats ---
        self.use_segpool = use_segpool
        if use_segpool:
            self.segpool = SegmentStatPool(segments=segments)
            with torch.no_grad():
                dummy = torch.zeros(1, fuse_out, self.T_prime)
                feat_dim = self.segpool(dummy).shape[1]
            self.ctx_head = nn.Sequential(
                nn.Linear(feat_dim, 128),
                nn.GELU(),
                nn.Dropout(0.30),
                nn.Linear(128, 32),
                nn.GELU(),
            )
            self.fuse_bias = nn.Linear(32, self.T_prime)
            nn.init.zeros_(self.fuse_bias.weight)
            nn.init.zeros_(self.fuse_bias.bias)
        else:
            self.segpool = None
            self.ctx_head = None
            self.fuse_bias = None

    # Helper: bin size in seconds
    @property
    def bin_sec(self) -> float:
        return self.window_sec / float(self.T_prime)

    def idx_to_sec(self, t_idx: torch.Tensor) -> torch.Tensor:
        return t_idx * self.bin_sec

    # ------- Feature extractor up to fused low-res map -------
    def forward_features_all_scales(self, x: torch.Tensor):
        """Return list of stage features [S0@200, S1@100, S2@50, S3@25] and fused map @25."""
        # Stem
        x = self.norm(x)            # (B, C_in, T=200)
        x = self.c_squeeze(x)       # (B, c0, 200)

        # Stage 0 (high-res)
        s0 = self.hi(x)             # (B, c0, 200)

        # Stage 1
        x1 = self.expand1(self.down1(s0))   # (B, c1, 100)
        s1 = self.lo1(x1)                   # (B, c1, 100)

        # Stage 2
        x2 = self.expand2(self.down2(s1))   # (B, c2, 50)
        s2 = self.lo2(x2)                   # (B, c2, 50)

        # Stage 3
        x3 = self.expand3(self.down3(s2))   # (B, c3, 25)
        s3 = self.lo3(x3)                   # (B, c3, 25)

        # Fuse to T'=25
        fused = self.fuse([s0, s1, s2, s3]) # (B, c3, 25)

        # Optional transformer refinement
        fused = self.temporal_transformer(fused)  # (B, c3, 25)

        return (s0, s1, s2, s3), fused

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        _, fused = self.forward_features_all_scales(x)
        return fused

    # ------- Forward -------
    def forward(self, x: torch.Tensor, return_dict: bool = False):
        """Default output: tensor (B, 1) with seconds.
        Set return_dict=True for detailed outputs."""
        x = x[:, :self.n_chans, :self.n_times]  # defensive slice
        _, feats = self.forward_features_all_scales(x)          # (B, F, T')

        logits, prob, t_idx = self.time_head(feats)             # (B, T'), (B, T'), (B,)

        # Optional context bias over time
        if self.use_segpool:
            ctx_vec = self.segpool(feats)                       # (B, feat_dim)
            ctx = self.ctx_head(ctx_vec)                        # (B, 32)
            bias = self.fuse_bias(ctx)                          # (B, T')
            logits = logits + bias
            prob = torch.softmax(logits, dim=-1)
            idx = torch.arange(prob.size(-1), device=prob.device, dtype=prob.dtype)
            t_idx = (prob * idx).sum(dim=-1)

        t_sec = self.idx_to_sec(t_idx)                          # (B,)

        if return_dict:
            return {
                'logits': logits,      # (B, T')
                'prob': prob,          # (B, T')
                't_idx': t_idx,        # (B,)
                't_sec': t_sec,        # (B,)
                'features': feats,     # (B, F, T')
                'bin_sec': self.bin_sec,
            }
        else:
            return t_sec.unsqueeze(-1)


# =========================
# Optional: training helpers for distributional supervision
# =========================
def gaussian_targets(t_sec: torch.Tensor, T: int, bin_sec: float, sigma_bins: float = 1.0) -> torch.Tensor:
    """Make soft Gaussian targets over T bins centered at ground-truth seconds."""
    device = t_sec.device
    idx = torch.arange(T, device=device).float()  # (T,)
    mu = (t_sec / bin_sec).unsqueeze(-1)          # (B, 1) in bins
    # N(mu, sigma^2) in bins; clamp for stability
    sigma = torch.full_like(mu, sigma_bins).clamp_min(1e-3)
    dist = torch.exp(-0.5 * ((idx - mu) / sigma)**2)  # (B, T)
    dist = dist / (dist.sum(dim=-1, keepdim=True) + 1e-8)
    return dist

def soft_ce_loss(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy with soft targets over time bins."""
    logp = F.log_softmax(logits, dim=-1)
    loss = -(soft_targets * logp).sum(dim=-1).mean()
    return loss


# =========================
# Example instantiation (matches your input: 128x200 @ 100 Hz)
# =========================
# if __name__ == "__main__":
#     # Example: deeper but effective default config (≈3 pyramid stages + transformer)
#     model_deeper = DeepEEGRTNet(
#         n_chans=128, n_times=200, sfreq=100.0,
#         c0=96, widen1=3, widen2=2, widen3=2,
#         k=15, dropout=0.50, layerscale_init=1e-4,
#         hi_dilations=(1,2,4,8,16), hi_depth=24,
#         lo1_dilations=(1,2,4,8,16), lo1_depth=16,
#         lo2_dilations=(1,2,4,8,16), lo2_depth=16,
#         lo3_dilations=(1,2,4,8,16), lo3_depth=12,
#         use_transformer=True, transformer_dim=256, transformer_heads=8, transformer_layers=2,
#         use_segpool=False
#     ).cuda() if torch.cuda.is_available() else DeepEEGRTNet(
#         n_chans=128, n_times=200, sfreq=100.0
#     )

#     # Dummy forward
#     x = torch.randn(4, 128, 200)  # (B, C, T)
#     y = model_deeper(x)           # (B, 1) seconds
#     print("Pred (sec):", y.shape, y[:2].flatten())

#     # If you want distributional training:
#     out = model_deeper(x, return_dict=True)
#     # Suppose ground-truth RT in seconds for the batch:
#     t_gt = torch.tensor([0.42, 0.71, 0.33, 1.05], dtype=torch.float32, device=x.device)
#     soft_t = gaussian_targets(t_gt, T=out['logits'].size(-1), bin_sec=out['bin_sec'], sigma_bins=1.5)
#     loss = soft_ce_loss(out['logits'], soft_t)
#     print("Loss:", loss.item())
