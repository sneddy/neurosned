"""Soft target helpers for segmentation-style training."""

from __future__ import annotations

import torch


def soft_label_1d(
    y_sec: torch.Tensor,
    T: int,
    dt: float,
    sigma: float | torch.Tensor = 0.12,
    density: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Create Gaussian soft labels over a 1D time grid."""
    y_sec = y_sec.to(torch.float32).view(-1)
    batch_size = y_sec.numel()
    device = y_sec.device
    u_star = (y_sec / dt).clamp(0.0, T - 1e-6).unsqueeze(1)
    grid = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(0)
    rel_sec = (grid - u_star) * dt

    if not torch.is_tensor(sigma):
        sigma = torch.tensor(float(sigma), device=device, dtype=torch.float32)
    sigma = sigma.to(torch.float32)
    sigma = sigma.view(-1, 1) if sigma.numel() > 1 else sigma.view(1, 1)
    if sigma.numel() == 1:
        sigma = sigma.expand(batch_size, 1)

    q = torch.exp(-0.5 * (rel_sec / sigma).pow(2))
    if density:
        q = q / (q.sum(dim=1, keepdim=True) + eps)
    else:
        q = q / q.amax(dim=1, keepdim=True).clamp_min(eps)
        idx = u_star.round().long().clamp_(0, T - 1).squeeze(1)
        q[torch.arange(batch_size, device=device), idx] = 1.0
        q.clamp_(0, 1)
    return q
