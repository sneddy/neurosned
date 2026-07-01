"""Temporal pooling and prediction heads."""

import torch
from torch import nn


class SegmentStatPool(nn.Module):
    """Segment-wise mean and max pooling over time."""

    def __init__(self, segments=(4, 8, 16)):
        super().__init__()
        self.segments = tuple(segments)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return flattened segment statistics."""
        batch_size, features, time = x.shape
        outputs = []
        for segment_count in self.segments:
            segment_len = max(1, time // segment_count)
            trimmed_time = segment_len * segment_count
            x_cut = x[..., :trimmed_time]
            x_reshaped = x_cut.view(batch_size, features, segment_count, segment_len)
            outputs.append(x_reshaped.mean(dim=-1))
            outputs.append(x_reshaped.amax(dim=-1))
        return torch.cat([output.flatten(start_dim=1) for output in outputs], dim=1)


class TimeHead(nn.Module):
    """Produce time logits, probabilities and soft-argmax index."""

    def __init__(self, feat_ch: int, T_prime: int, use_context: bool = True, ctx_dim: int = 256):
        super().__init__()
        self.T_prime = T_prime
        self.use_context = use_context
        self.score = nn.Conv1d(feat_ch, 1, kernel_size=1)

        if use_context:
            self.ctx_gate = nn.Conv1d(feat_ch, 1, kernel_size=1)
            self.ctx_proj = nn.Sequential(
                nn.Linear(feat_ch, ctx_dim),
                nn.GELU(),
                nn.Linear(ctx_dim, feat_ch),
                nn.GELU(),
            )
        else:
            self.ctx_gate = None
            self.ctx_proj = None

    def forward(self, x: torch.Tensor):
        """Return logits, probabilities and expected time index."""
        _, _, time = x.shape
        assert time == self.T_prime, f"Expected T'={self.T_prime}, got {time}"

        if self.use_context:
            gate = self.ctx_gate(x).squeeze(1)
            weights = torch.softmax(gate, dim=-1)
            context = torch.einsum("bct,bt->bc", x, weights)
            context = self.ctx_proj(context).unsqueeze(-1)
            x = x + context

        logits = self.score(x).squeeze(1)
        prob = torch.softmax(logits, dim=-1)
        idx = torch.arange(time, device=x.device, dtype=prob.dtype)
        t_idx = (prob * idx).sum(dim=-1)
        return logits, prob, t_idx
