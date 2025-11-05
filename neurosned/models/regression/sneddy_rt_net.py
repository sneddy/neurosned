import math
import torch
import torch.nn as nn

"""
SneddyRTNet — Reaction-Time oriented EEG model (compatible output with SneddyNet)
with **deeper capacity controls**.

Key points
----------
• Minimal time downsampling by default (×2) to keep temporal resolution; you can opt
  into a second downsample for deeper/larger models if you can tolerate coarser time.
• Dilated depthwise‑separable residual blocks grow receptive field without shrinking T.
• Time head produces a distribution over time (logits → softmax) and soft‑argmax RT.
• **API compatibility**: forward(x) → tensor **(B, 1)** like SneddyNet (n_outputs=1).
  Call forward(x, return_dict=True) for detailed outputs.

Deeper knobs
------------
• hi_dilations / hi_depth: control the number of high‑resolution residual blocks.
• lo_dilations / lo_depth: control the number of blocks after the first downsample.
• num_lo_stacks: repeat the low‑res block stack multiple times (same channels/T').
• downsample_twice (+ widen2): optional second downsample stage for much deeper nets.

Input:  (B, C, T)   — e.g., (B, 129, 200) for a 2s window at ~100 Hz.
Output: (B, 1)      — seconds (like SneddyNet with n_outputs=1).
"""


# =========================
# Basic building blocks
# =========================
class StdPerSample(nn.Module):
    """Per-sample, per-channel standardization across time.
    For each sample and channel: (x - mean_t) / std_t.
    """
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, T)
        mu = x.mean(dim=-1, keepdim=True)
        sd = x.std(dim=-1, keepdim=True).clamp_min(self.eps)
        return (x - mu) / sd


class DSConv1d(nn.Module):
    """Depthwise-separable 1D conv: depthwise (per feature channel) + 1x1 pointwise."""
    def __init__(self, ch: int, k: int = 7, dilation: int = 1):
        super().__init__()
        pad = ((k - 1) // 2) * dilation
        self.dw = nn.Conv1d(ch, ch, k, padding=pad, dilation=dilation, groups=ch, bias=False)
        self.pw = nn.Conv1d(ch, ch, 1, bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class ResBlock(nn.Module):
    """Lightweight residual block.
    DSConv(k, dilation) → GroupNorm(1) → GELU → Dropout → DSConv(k) → GroupNorm(1) + skip.
    """
    def __init__(self, ch: int, k: int = 7, dropout: float = 0.0, dilation: int = 1):
        super().__init__()
        self.conv1 = DSConv1d(ch, k=k, dilation=dilation)
        self.gn1   = nn.GroupNorm(1, ch)
        self.conv2 = DSConv1d(ch, k=k, dilation=1)
        self.gn2   = nn.GroupNorm(1, ch)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        x = self.conv1(x)
        x = self.act(self.gn1(x))
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.gn2(x)
        x = self.act(x + r)
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
    """Time downsample ×2 with fixed triangular smoothing + AvgPool(stride=2).
    This mitigates aliasing before decimation.
    """
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


class SegmentStatPool(nn.Module):
    """Segment-wise stat pooling over time.
    For each S in `segments`, splits T into S equal parts (trim remainder),
    computes mean and max per segment. Returns a flat vector (B, F * sum(2*S)).
    """
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


# =========================
# Time head: distribution over time + soft-argmax
# =========================
class TimeHead(nn.Module):
    """Produces per-time logits, probability via softmax, and soft-argmax index.
    Expects a feature map (B, F, T').
    """
    def __init__(self, feat_ch: int, T_prime: int, use_context: bool = True, ctx_dim: int = 256):
        super().__init__()
        self.T_prime = T_prime
        self.use_context = use_context

        # Local time scorer
        self.score = nn.Conv1d(feat_ch, 1, kernel_size=1)

        if use_context:
            # Lightweight global context that can bias per-time logits
            self.ctx_gate = nn.Conv1d(feat_ch, 1, kernel_size=1)
            self.ctx_proj = nn.Sequential(
                nn.Linear(feat_ch, ctx_dim), nn.GELU(), nn.Linear(ctx_dim, feat_ch), nn.GELU()
            )
        else:
            self.ctx_gate = None
            self.ctx_proj = None

    def forward(self, x: torch.Tensor):  # x: (B, F, T')
        B, C, T = x.shape
        assert T == self.T_prime, f"Expected T'={self.T_prime}, got {T}"

        if self.use_context:
            # attention-style pooling over time → global context → residual bias
            g = self.ctx_gate(x).squeeze(1)                   # (B, T')
            w = torch.softmax(g, dim=-1)                      # (B, T')
            f_ctx = torch.einsum('bct,bt->bc', x, w)          # (B, C)
            f_ctx = self.ctx_proj(f_ctx).unsqueeze(-1)        # (B, C, 1)
            x = x + f_ctx                                     # broadcast along T

        logits = self.score(x).squeeze(1)                     # (B, T')
        prob = torch.softmax(logits, dim=-1)                  # (B, T')
        idx = torch.arange(T, device=x.device, dtype=prob.dtype)
        t_idx = (prob * idx).sum(dim=-1)                      # (B,)
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
# Main model (API-compatible)
# =========================
class SneddyRTNet(nn.Module):
    """RT-focused network that **returns (B,1) by default**, just like SneddyNet.

    If you need detailed outputs (logits/prob/t_idx/features), call
    `forward(x, return_dict=True)`.
    """
    def __init__(
        self,
        n_chans: int,
        n_times: int,
        sfreq: float,
        n_outputs: int = 1,          # kept for API parity; must be 1
        c0: int = 32,
        widen: int = 2,
        k: int = 7,
        dropout: float = 0.1,
        use_segpool: bool = False,    # off by default for better generalization
        segments=(4, 8, 16),
        # depth knobs
        hi_dilations=(1, 2, 4),
        hi_depth: int | None = None,  # if set, repeats hi_dilations to this length
        lo_dilations=(1, 2, 4, 8),
        lo_depth: int = 4,            # number of low-res blocks after first downsample
        num_lo_stacks: int = 1,       # repeat the low-res stack this many times
        # downsampling knobs
        downsample_once: bool = True,
        downsample_twice: bool = False,
        widen2: int = 2,              # channel widen factor at the second stage
    ):
        super().__init__()
        assert n_outputs == 1, "This RT version is designed to output (B,1) like SneddyNet."
        self.n_chans = n_chans
        self.n_times = n_times
        self.sfreq = float(sfreq)
        self.window_sec = float(n_times) / float(sfreq)
        self.n_outputs = n_outputs

        # --- Normalization + electrode mixing ---
        self.norm = StdPerSample()
        self.c_squeeze = ChannelSqueeze(n_chans, c0)   # (B, c0, T)

        # --- High-resolution residual stack (no time shrink) ---
        hi_sched = _make_schedule(hi_dilations, hi_depth)
        self.hi = nn.Sequential(*[ResBlock(c0, k=k, dropout=dropout, dilation=d) for d in hi_sched])

        # --- First downsample ×2 + channel expansion ---
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

        # --- Low-res block stack(s) at T' ---
        lo_sched = _make_schedule(lo_dilations, lo_depth)
        lo_stack = [ResBlock(feat_ch, k=k, dropout=dropout, dilation=d) for d in lo_sched]
        self.lo = nn.Sequential(*lo_stack)

        if num_lo_stacks > 1:
            extra = []
            for _ in range(num_lo_stacks - 1):
                extra.extend([ResBlock(feat_ch, k=k, dropout=dropout, dilation=d) for d in lo_sched])
            self.lo_extra = nn.Sequential(*extra)
        else:
            self.lo_extra = nn.Identity()

        # --- Optional second downsample ×2 (T'/2) + expansion ---
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
            # another low-res stack after the second downsample
            lo2_stack = [ResBlock(feat_ch, k=k, dropout=dropout, dilation=d) for d in lo_sched]
            self.lo2 = nn.Sequential(*lo2_stack)
        else:
            self.down2 = nn.Identity()
            self.expand2 = nn.Identity()
            self.lo2 = nn.Identity()

        self.T_prime = T_prime

        # --- Time head ---
        self.time_head = TimeHead(feat_ch, self.T_prime, use_context=True, ctx_dim=256)

        # --- Optional global context via segmental stats ---
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

    # Helper: bin size in seconds
    @property
    def bin_sec(self) -> float:
        return self.window_sec / float(self.T_prime)

    def idx_to_sec(self, t_idx: torch.Tensor) -> torch.Tensor:
        return t_idx * self.bin_sec

    # Forward up to the feature map (B, F, T')
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)            # (B, C, T)
        x = self.c_squeeze(x)       # (B, c0, T)
        x = self.hi(x)              # (B, c0, T)
        x = self.down(x)            # (B, c0, T')
        x = self.expand(x)          # (B, c*, T')
        x = self.lo(x)              # (B, c*, T')
        x = self.lo_extra(x)        # (B, c*, T')
        x = self.down2(x)           # (B, c*, T''?)
        x = self.expand2(x)         # (B, c**, T''?)
        x = self.lo2(x)             # (B, c**, T''?)
        return x

    def forward(self, x: torch.Tensor, return_dict: bool = False):
        """Default output matches SneddyNet: tensor (B, 1) with seconds.
        Set `return_dict=True` to get a dict with detailed fields.
        """
        feats = self.forward_features(x)                       # (B, F, T')
        logits, prob, t_idx = self.time_head(feats)            # (B, T'), (B, T'), (B,)

        # Optional context bias over time
        if self.use_segpool:
            ctx_vec = self.segpool(feats)                      # (B, feat_dim)
            ctx = self.ctx_head(ctx_vec)                       # (B, 32)
            bias = self.fuse(ctx)                              # (B, T')
            logits = logits + bias
            prob = torch.softmax(logits, dim=-1)
            idx = torch.arange(prob.size(-1), device=prob.device, dtype=prob.dtype)
            t_idx = (prob * idx).sum(dim=-1)

        t_sec = self.idx_to_sec(t_idx)                         # (B,)

        if return_dict:
            return {
                'logits': logits,      # (B, T')
                'prob': prob,          # (B, T')
                't_idx': t_idx,        # (B,)
                't_sec': t_sec,        # (B,)
                'features': feats,     # (B, F, T')
            }
        else:
            # API parity with SneddyNet: return (B, 1)
            return t_sec.unsqueeze(-1)


# =========================
# Examples / Suggested presets
# =========================
# if __name__ == "__main__":
#     B, C, T = 4, 128, 200
#     sfreq = 100.0

#     # Small (≈0.5M params):
#     small = SneddyRTNet(n_chans=C, n_times=T, sfreq=sfreq,
#                         c0=32, widen=2, k=7, dropout=0.1,
#                         use_segpool=False,
#                         hi_dilations=(1,2,4), hi_depth=None,
#                         lo_dilations=(1,2,4,8), lo_depth=4, num_lo_stacks=1,
#                         downsample_once=True, downsample_twice=False)

#     # Medium‑deep (couple of M params, still T'=100):
#     medium = SneddyRTNet(n_chans=C, n_times=T, sfreq=sfreq,
#                          c0=64, widen=2, k=9, dropout=0.15,
#                          use_segpool=False,
#                          hi_dilations=(1,2,4,8), hi_depth=8,
#                          lo_dilations=(1,2,4,8), lo_depth=8, num_lo_stacks=2,
#                          downsample_once=True, downsample_twice=False)

#     # Large (deeper, allows second downsample → T'=50):
#     large = SneddyRTNet(n_chans=C, n_times=T, sfreq=sfreq,
#                         c0=64, widen=3, k= nine := 9 if False else 9, dropout=0.2,
#                         use_segpool=False,
#                         hi_dilations=(1,2,4,8), hi_depth=12,
#                         lo_dilations=(1,2,4,8), lo_depth=10, num_lo_stacks=3,
#                         downsample_once=True, downsample_twice=True, widen2=2)

#     x = torch.randn(B, C, T)
#     for name, m in {"small": small, "medium": medium, "large": large}.items():
#         y = m(x)
#         print(name, "→", y.shape)
