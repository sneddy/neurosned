import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# -----------------------
# Building blocks
# -----------------------

class StdPerSample(nn.Module):
    """Per-sample, per-channel standardization across time: (x - mean_t) / std_t."""
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        mu = x.mean(dim=-1, keepdim=True)
        sd = x.std(dim=-1, keepdim=True).clamp_min(self.eps)
        return (x - mu) / sd


class DSConv1d(nn.Module):
    """Depthwise-separable 1D conv: depthwise (per channel) + 1x1 pointwise."""
    def __init__(self, ch: int, k: int = 7, dilation: int = 1, bias: bool = False):
        super().__init__()
        pad = ((k - 1) // 2) * dilation
        self.dw = nn.Conv1d(ch, ch, k, padding=pad, dilation=dilation, groups=ch, bias=bias)
        self.pw = nn.Conv1d(ch, ch, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class ReZero(nn.Module):
    """ReZero-style residual scaling: y = x + alpha * f(x). alpha init=0 for stable deep/recursive nets."""
    def __init__(self, init: float = 0.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(init, dtype=torch.float32))

    def forward(self, x: torch.Tensor, f_out: torch.Tensor) -> torch.Tensor:
        return x + self.alpha * f_out


class ChannelSqueeze(nn.Module):
    """Learnable 1x1 mixing over electrodes: Conv1d(C_in → C_out)."""
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.proj = nn.Conv1d(c_in, c_out, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.proj.weight, nonlinearity="linear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class AntiAliasDown2(nn.Module):
    """
    Time downsample ×2 with fixed triangular smoothing + AvgPool(stride=2).
    Mitigates aliasing before decimation.
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


class FiLM1d(nn.Module):
    """
    Lightweight FiLM modulation over channels from a conditioning vector:
      y = x * (1 + gamma) + beta, where gamma,beta = MLP(cond)
    """
    def __init__(self, c: int, cond_dim: int, hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2 * c)
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B,C,T), cond: (B,cond_dim)
        gb = self.net(cond)  # (B, 2C)
        gamma, beta = gb.chunk(2, dim=-1)  # (B,C), (B,C)
        gamma = 1.0 + gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)
        return x * gamma + beta


# -----------------------
# Time head (soft-argmax)
# -----------------------

class TimeHead(nn.Module):
    """
    Produces per-time logits, probability via softmax, and soft-argmax index.
    Expects a feature map (B, F, T').
    """
    def __init__(self, feat_ch: int, T_prime: int, use_context: bool = True, ctx_dim: int = 256):
        super().__init__()
        self.T_prime = T_prime
        self.use_context = use_context

        self.score = nn.Conv1d(feat_ch, 1, kernel_size=1)

        if use_context:
            # attention-style pooling over time → global context → residual bias
            self.ctx_gate = nn.Conv1d(feat_ch, 1, kernel_size=1)
            self.ctx_proj = nn.Sequential(
                nn.Linear(feat_ch, ctx_dim), nn.GELU(),
                nn.Linear(ctx_dim, feat_ch), nn.GELU()
            )
        else:
            self.ctx_gate = None
            self.ctx_proj = None

    def forward(self, x: torch.Tensor):
        # x: (B, F, T')
        B, C, T = x.shape
        assert T == self.T_prime, f"Expected T'={self.T_prime}, got {T}"

        if self.use_context:
            g = self.ctx_gate(x).squeeze(1)          # (B, T')
            w = torch.softmax(g, dim=-1)             # (B, T')
            f_ctx = torch.einsum('bct,bt->bc', x, w) # (B, C)
            f_ctx = self.ctx_proj(f_ctx).unsqueeze(-1)  # (B, C, 1)
            x = x + f_ctx                            # broadcast along T

        logits = self.score(x).squeeze(1)            # (B, T')
        prob = torch.softmax(logits, dim=-1)         # (B, T')
        idx = torch.arange(T, device=x.device, dtype=prob.dtype)
        t_idx = (prob * idx).sum(dim=-1)             # (B,)
        return logits, prob, t_idx


# -----------------------
# Halting head (ACT-style)
# -----------------------

class HaltingHead(nn.Module):
    """
    Computes per-step halting probability p in (0,1) from features (global average pooled).
    """
    def __init__(self, feat_ch: int, hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_ch, hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F, T')
        g = x.mean(dim=-1)              # (B, F)
        p = torch.sigmoid(self.mlp(g))  # (B, 1)
        return p.squeeze(-1)            # (B,)


# -----------------------
# Tiny shared core (reused recursively)
# -----------------------

class TinyRecursiveCore(nn.Module):
    """
    A tiny 2-layer DSConv block with GroupNorm + GELU + ReZero.
    Shared weights across all recursion steps (parameter tying).
    """
    def __init__(self, ch: int, k: int = 7, dropout: float = 0.1, dilation: int = 1):
        super().__init__()
        self.conv1 = DSConv1d(ch, k=k, dilation=dilation, bias=False)
        self.gn1 = nn.GroupNorm(1, ch)
        self.conv2 = DSConv1d(ch, k=k, dilation=dilation, bias=False)
        self.gn2 = nn.GroupNorm(1, ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.rz1 = ReZero(0.0)
        self.rz2 = ReZero(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        x = self.conv1(x)
        x = self.act(self.gn1(x))
        x = self.drop(x)
        x = self.rz1(r, x)             # first residual
        r2 = x
        x = self.conv2(x)
        x = self.act(self.gn2(x))
        x = self.drop(x)
        x = self.rz2(r2, x)            # second residual
        return x


# -----------------------
# Main recursive model
# -----------------------

class RecurTinyEEGRT(nn.Module):
    """
    EEG Reaction-Time predictor with recursive tiny core + ACT-style halting.
    - Shares a small block across many steps (effective depth without many params)
    - Per-step time head; final prediction is a halting-weighted mixture
    - Returns (B,1) seconds by default (soft-argmax over time bins)

    Set return_dict=True to get detailed per-step fields & ponder loss.
    """
    def __init__(
        self,
        n_chans: int,
        n_times: int,
        sfreq: float,
        # stem
        c0: int = 64,
        k_stem: int = 7,
        # downsampling
        downsample_once: bool = True,
        downsample_twice: bool = True,
        # feature width after first/second stage
        widen1: int = 2,
        widen2: int = 2,
        # recursive core
        max_steps: int = 24,
        core_k: int = 7,
        core_dilation: int = 1,
        core_dropout: float = 0.10,
        # FiLM conditioning from step embeddings
        step_embed_dim: int = 64,
        film_hidden: int = 128,
        film_dropout: float = 0.0,
        # halting/ACT
        use_act: bool = True,
        ponder_cost: float = 0.01,   # add to loss if return_dict=True
        # time head
        use_time_context: bool = True,
        time_ctx_dim: int = 256,
    ):
        super().__init__()
        assert n_times > 0 and n_chans > 0
        self.n_chans = n_chans
        self.n_times = n_times
        self.sfreq = float(sfreq)
        self.window_sec = float(n_times) / float(sfreq)

        # ----- stem -----
        self.norm = StdPerSample()
        self.c_squeeze = ChannelSqueeze(n_chans, c0)  # (B, c0, T)
        self.stem = nn.Sequential(
            DSConv1d(c0, k=k_stem, dilation=1, bias=False),
            nn.GroupNorm(1, c0),
            nn.GELU(),
        )

        # ----- downsample to T' (and optionally T'' = T'/2) -----
        T_prime = n_times
        feat_ch = c0

        self.down1 = AntiAliasDown2(feat_ch) if downsample_once else nn.Identity()
        if downsample_once:
            T_prime //= 2
        c1 = feat_ch * widen1
        self.expand1 = nn.Sequential(
            nn.Conv1d(feat_ch, c1, kernel_size=1, bias=False),
            nn.GroupNorm(1, c1),
            nn.GELU(),
        )
        feat_ch = c1

        self.down2 = AntiAliasDown2(feat_ch) if downsample_twice else nn.Identity()
        if downsample_twice:
            T_prime //= 2
        c2 = feat_ch * widen2
        self.expand2 = nn.Sequential(
            nn.Conv1d(feat_ch, c2, kernel_size=1, bias=False),
            nn.GroupNorm(1, c2),
            nn.GELU(),
        )
        feat_ch = c2

        # ----- recursive tiny core (shared across steps) -----
        self.core = TinyRecursiveCore(feat_ch, k=core_k, dropout=core_dropout, dilation=core_dilation)
        self.max_steps = int(max_steps)

        # step embeddings + FiLM to let shared core "know" the step id
        self.step_embed = nn.Embedding(self.max_steps, step_embed_dim)
        nn.init.normal_(self.step_embed.weight, mean=0.0, std=0.02)
        self.film = FiLM1d(feat_ch, cond_dim=step_embed_dim, hidden=film_hidden, dropout=film_dropout)

        # ----- heads -----
        self.time_head = TimeHead(feat_ch, T_prime, use_context=use_time_context, ctx_dim=time_ctx_dim)
        self.halt_head = HaltingHead(feat_ch, hidden=128, dropout=0.0)

        # ACT / halting
        self.use_act = bool(use_act)
        self.ponder_cost = float(max(0.0, ponder_cost))
        self.T_prime = T_prime
        self.feat_ch = feat_ch

    # Helper: bin size in seconds
    @property
    def bin_sec(self) -> float:
        return self.window_sec / float(self.T_prime)

    def idx_to_sec(self, t_idx: torch.Tensor) -> torch.Tensor:
        return t_idx * self.bin_sec

    # ------------- forward -------------
    def forward(self, x: torch.Tensor, return_dict: bool = False):
        """
        Default: returns (B,1) with predicted RT in seconds.
        If return_dict=True: returns a dict with detailed per-step outputs and 'ponder_loss' (if ACT).
        """
        x = x[:, :self.n_chans, :self.n_times]  # safety clip

        # stem
        x = self.norm(x)           # (B, C, T)
        x = self.c_squeeze(x)      # (B, c0, T)
        x = self.stem(x)           # (B, c0, T)

        # down/expand
        x = self.down1(x)          # (B, c0, T')
        x = self.expand1(x)        # (B, c*, T')
        x = self.down2(x)          # (B, c*, T''?)
        feats = self.expand2(x)    # (B, F, T')

        B, F, T = feats.shape
        assert T == self.T_prime

        # recursive loop
        eps = 1e-6
        running_halt = torch.zeros(B, device=feats.device, dtype=feats.dtype)  # sum of halting weights
        final_prob = torch.zeros(B, self.T_prime, device=feats.device, dtype=feats.dtype)

        step_logits = []
        step_probs = []
        step_tidx = []
        step_weights = []
        step_halt_p = []

        for s in range(self.max_steps):
            # shared tiny core
            feats = self.core(feats)
            # step-aware FiLM
            step_ids = torch.full((B,), s, device=feats.device, dtype=torch.long)
            cond = self.step_embed(step_ids)  # (B, D)
            feats = self.film(feats, cond)    # (B, F, T')

            # time prediction at this step
            logits_s, prob_s, t_idx_s = self.time_head(feats)  # (B,T'), (B,T'), (B,)
            step_logits.append(logits_s)
            step_probs.append(prob_s)
            step_tidx.append(t_idx_s)

            # halting
            if self.use_act:
                p_s = self.halt_head(feats)           # (B,)
                remaining = (1.0 - running_halt).clamp_min(0.0)  # (B,)
                if s == self.max_steps - 1:
                    w_s = remaining                    # force finish
                else:
                    w_s = torch.minimum(p_s, remaining)
            else:
                # uniform weights over fixed steps
                w_s = torch.full((B,), 1.0 / self.max_steps, device=feats.device, dtype=feats.dtype)

            step_halt_p.append(w_s if self.use_act else torch.full_like(w_s, 1.0 / self.max_steps))
            step_weights.append(w_s)

            # accumulate mixture
            final_prob = final_prob + (w_s.unsqueeze(-1) * prob_s)  # weighted sum over steps
            running_halt = running_halt + w_s

            # optional early break if everyone halted
            if self.use_act and bool(torch.all(running_halt > 1.0 - 1e-4)):
                # pad bookkeeping for consistent lengths if we break early
                break

        # normalize (numerical safety if ACT didn't sum perfectly to 1.0)
        norm = running_halt.clamp_min(eps).unsqueeze(-1)  # (B,1)
        final_prob = final_prob / norm

        # final soft-argmax
        idx = torch.arange(self.T_prime, device=feats.device, dtype=final_prob.dtype)
        t_idx = (final_prob * idx).sum(dim=-1)  # (B,)
        t_sec = self.idx_to_sec(t_idx)          # (B,)

        if not return_dict:
            return t_sec.unsqueeze(-1)

        # build rich outputs
        out = {
            "t_sec": t_sec,                                  # (B,)
            "t_idx": t_idx,                                  # (B,)
            "prob": final_prob,                              # (B, T')
            "T_prime": self.T_prime,
            "bin_sec": self.bin_sec,
            "running_halt": running_halt,                    # (B,)
            "steps": len(step_weights),
            "step_logits": step_logits,                      # list[(B, T')]
            "step_probs": step_probs,                        # list[(B, T')]
            "step_t_idx": step_tidx,                         # list[(B, )]
            "step_weights": step_weights,                    # list[(B, )] actual halting weights w_s
            "step_halt_p": step_halt_p,                      # list[(B, )] same as weights if ACT, else uniform
            "features_final": feats,                         # (B, F, T')
        }

        # ACT ponder loss (encourage fewer steps)
        if self.use_act and self.ponder_cost > 0.0:
            # expected steps = sum of weights; average over batch
            expected_steps = running_halt.mean()
            out["ponder_loss"] = self.ponder_cost * expected_steps

        return out


# -----------------------
# Example: instantiate a deeper-but-efficient RT model
# -----------------------
def build_recur_tiny_eeg_rt(
    n_chans: int = 128, n_times: int = 200, sfreq: float = 100.0, device: Optional[torch.device] = None
) -> nn.Module:
    """
    Example configuration:
    - Two anti-aliased downsamples → T' = 50 bins (bin size = 2s/50 = 0.04 s)
    - Tiny shared core applied 24 recursive steps (effective depth ~ 48 conv layers due to 2 convs/step)
    - FiLM step modulation and ACT-style halting with a small ponder cost
    """
    model = RecurTinyEEGRT(
        n_chans=n_chans,
        n_times=n_times,
        sfreq=sfreq,
        c0=64,
        k_stem=9,
        downsample_once=True,
        downsample_twice=True,   # 200 → 100 → 50 time bins
        widen1=2,
        widen2=2,
        max_steps=24,
        core_k=9,
        core_dilation=1,         # receptive field grows via recursion
        core_dropout=0.10,
        step_embed_dim=64,
        film_hidden=128,
        film_dropout=0.0,
        use_act=True,
        ponder_cost=0.01,
        use_time_context=True,
        time_ctx_dim=256,
    )
    if device is not None:
        model = model.to(device)
    return model


# -----------------------
# Minimal usage
# -----------------------
# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = build_recur_tiny_eeg_rt(device=device)

#     B = 4
#     x = torch.randn(B, 128, 200, device=device)
#     # Default: (B,1) RT seconds
#     y = model(x)                 # (B,1)
#     print("y.shape:", y.shape, "example secs:", y[:2].flatten().tolist())

#     # Rich diagnostics (per-step)
#     outs = model(x, return_dict=True)
#     print("T':", outs["T_prime"], "bin_sec:", outs["bin_sec"], "steps used:", outs["steps"])
#     if "ponder_loss" in outs:
#         print("ponder_loss:", float(outs["ponder_loss"]))
