"""Discrete-time hazard readouts for event-time models."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def hazard_logits_to_log_pmf(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    condition_inside: bool = True,
) -> torch.Tensor:
    """Convert hazard logits to log event-bin probabilities.

    ``condition_inside=True`` renormalizes the event-bin PMF by the total
    probability that the event occurs within the represented temporal window.
    """
    z = logits / float(temperature)
    log_hazard = F.logsigmoid(z)
    log_not_hazard = F.logsigmoid(-z)
    log_survival_through = torch.cumsum(log_not_hazard, dim=-1)
    log_survival_before = F.pad(log_survival_through[..., :-1], (1, 0), value=0.0)
    log_pmf = log_survival_before + log_hazard
    if not condition_inside:
        return log_pmf
    log_inside_mass = torch.logsumexp(log_pmf, dim=-1, keepdim=True)
    return log_pmf - log_inside_mass


def hazard_logits_to_pmf(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    condition_inside: bool = True,
) -> torch.Tensor:
    """Convert hazard logits to event-bin probabilities."""
    return hazard_logits_to_log_pmf(logits, temperature=temperature, condition_inside=condition_inside).exp()


def hazard_discrete_nll(
    logits: torch.Tensor,
    target_time: torch.Tensor,
    *,
    dt: float,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Discrete-time survival NLL for an exact event bin.

    Each logit parameterizes the conditional event probability
    ``h_t = P(T=t | T>=t, x)``. The target time is assigned to the nearest
    represented bin and optimized with ``-log S(t-) h_t``.
    """
    if logits.ndim != 2:
        raise ValueError(f"Expected logits with shape (batch, time), got {tuple(logits.shape)}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt!r}")

    T = logits.shape[-1]
    target_idx = torch.floor(target_time.to(logits.device) / float(dt) + 0.5).long()
    target_idx = target_idx.clamp(0, T - 1).view(-1, 1)
    log_pmf = hazard_logits_to_log_pmf(logits, temperature=temperature, condition_inside=False)
    return -log_pmf.gather(dim=-1, index=target_idx).squeeze(-1).mean()


def hazard_expected_time(
    logits: torch.Tensor,
    t_grid: torch.Tensor,
    *,
    temperature: float = 1.0,
    condition_inside: bool = True,
) -> torch.Tensor:
    """Return the posterior mean event time under a hazard-derived PMF."""
    probabilities = hazard_logits_to_pmf(logits, temperature=temperature, condition_inside=condition_inside)
    return (probabilities * t_grid).sum(dim=-1)
