"""Tensor losses for event-time segmentation objectives."""

from __future__ import annotations

import numpy as np
import torch


def expected_time(probabilities: torch.Tensor, t_grid: torch.Tensor) -> torch.Tensor:
    """Return the posterior mean time for a temporal probability map."""
    return (probabilities * t_grid).sum(dim=-1)


def posterior_entropy(probabilities: torch.Tensor, log_probabilities: torch.Tensor) -> torch.Tensor:
    """Return mean categorical entropy for temporal event probabilities."""
    return -(probabilities * log_probabilities).sum(dim=-1).mean()


def soft_label_cross_entropy(
    log_probabilities: torch.Tensor,
    target_probabilities: torch.Tensor,
    *,
    probabilities: torch.Tensor | None = None,
    focal_gamma: float = 0.0,
) -> torch.Tensor:
    """Return soft-label cross entropy with an optional detached focal weight."""
    if focal_gamma > 0:
        if probabilities is None:
            probabilities = log_probabilities.exp()
        focal_weight = (1.0 - probabilities.detach()).pow(float(focal_gamma))
        return -((focal_weight * target_probabilities) * log_probabilities).sum(dim=-1).mean()
    return -(target_probabilities * log_probabilities).sum(dim=-1).mean()


def posterior_kl(target_probabilities: torch.Tensor, log_probabilities: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Return KL(target || predicted) for temporal probability maps."""
    return (target_probabilities * (torch.log(target_probabilities + eps) - log_probabilities)).sum(dim=-1).mean()


def cdf_distance(probabilities: torch.Tensor, target_probabilities: torch.Tensor, dt: float) -> torch.Tensor:
    """Return a 1D CDF-distance loss scaled by temporal bin width."""
    return torch.abs(torch.cumsum(probabilities, -1) - torch.cumsum(target_probabilities, -1)).sum(-1).mean() * float(dt)


def posterior_width(probabilities: torch.Tensor, t_grid: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """Return posterior standard deviation in seconds for each sample."""
    mean = expected_time(probabilities, t_grid)
    variance = (probabilities * (t_grid - mean[:, None]).pow(2)).sum(dim=-1)
    return variance.clamp_min(float(eps)).sqrt()


def event_mixture_nll(
    log_event_probabilities: torch.Tensor,
    y_rel: torch.Tensor,
    t_grid: torch.Tensor,
    *,
    sigma: float | torch.Tensor,
    event_sigma: torch.Tensor | None = None,
    kernel: str = "gaussian",
    df: float = 3.0,
    mixture_weight: float = 0.1,
    mixture_sigma_narrow: float | None = None,
    mixture_sigma_wide: float | None = None,
) -> torch.Tensor:
    """Return negative log likelihood under a temporal event mixture.

    ``log_event_probabilities`` can come from any temporal event distribution
    parameterization, e.g. a softmax posterior or a hazard-derived PMF.
    """
    log_kernel = event_observation_log_kernel(
        y_rel,
        t_grid,
        sigma=sigma,
        event_sigma=event_sigma,
        kernel=kernel,
        df=df,
        mixture_weight=mixture_weight,
        mixture_sigma_narrow=mixture_sigma_narrow,
        mixture_sigma_wide=mixture_sigma_wide,
    )
    return -torch.logsumexp(log_event_probabilities + log_kernel, dim=-1).mean()


def event_observation_log_kernel(
    y_rel: torch.Tensor,
    t_grid: torch.Tensor,
    *,
    sigma: float | torch.Tensor,
    event_sigma: torch.Tensor | None = None,
    kernel: str = "gaussian",
    df: float = 3.0,
    mixture_weight: float = 0.1,
    mixture_sigma_narrow: float | None = None,
    mixture_sigma_wide: float | None = None,
) -> torch.Tensor:
    """Return log K(y | t) for supported event-time observation kernels."""
    base_sigma = _prepare_sigma(sigma, log_device=t_grid.device, log_dtype=t_grid.dtype, y_rel=y_rel, event_sigma=event_sigma)
    standardized_error = (t_grid - y_rel[:, None]) / base_sigma
    kernel_name = str(kernel).lower().replace("-", "_")

    if kernel_name in {"gaussian", "normal"}:
        return _gaussian_log_kernel(standardized_error, base_sigma)

    if kernel_name in {"laplace", "double_exponential"}:
        return -standardized_error.abs() - torch.log(2.0 * base_sigma)

    if kernel_name in {"student", "student_t", "t"}:
        degrees = torch.tensor(max(float(df), 1e-6), device=t_grid.device, dtype=t_grid.dtype)
        log_norm = (
            torch.lgamma(0.5 * (degrees + 1.0))
            - torch.lgamma(0.5 * degrees)
            - 0.5 * torch.log(degrees * torch.tensor(np.pi, device=t_grid.device, dtype=t_grid.dtype))
            - base_sigma.log()
        )
        return log_norm - 0.5 * (degrees + 1.0) * torch.log1p(standardized_error.pow(2) / degrees)

    if kernel_name in {"gaussian_mixture", "normal_mixture", "mixture_gaussian", "mixture"}:
        weight = float(mixture_weight)
        if not 0.0 < weight < 1.0:
            raise ValueError("mixture_weight must be in (0, 1).")
        default_sigma = _scalar_sigma_value(sigma)
        narrow_value = default_sigma if mixture_sigma_narrow is None else float(mixture_sigma_narrow)
        wide_value = 2.5 * default_sigma if mixture_sigma_wide is None else float(mixture_sigma_wide)
        sigma_narrow = torch.tensor(
            max(narrow_value, 1e-8),
            device=t_grid.device,
            dtype=t_grid.dtype,
        )
        sigma_wide = torch.tensor(
            max(wide_value, 1e-8),
            device=t_grid.device,
            dtype=t_grid.dtype,
        )
        log_narrow = _gaussian_log_kernel((t_grid - y_rel[:, None]) / sigma_narrow, sigma_narrow)
        log_wide = _gaussian_log_kernel((t_grid - y_rel[:, None]) / sigma_wide, sigma_wide)
        return torch.logaddexp(
            torch.log(torch.tensor(1.0 - weight, device=t_grid.device, dtype=t_grid.dtype)) + log_narrow,
            torch.log(torch.tensor(weight, device=t_grid.device, dtype=t_grid.dtype)) + log_wide,
        )

    raise ValueError(f"Unknown event_nll_kernel: {kernel!r}")


def _gaussian_log_kernel(standardized_error: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Return Gaussian log density for a standardized residual and scale."""
    return -0.5 * standardized_error.pow(2) - sigma.log() - 0.5 * np.log(2.0 * np.pi)


def _prepare_sigma(
    sigma: float | torch.Tensor,
    *,
    log_device: torch.device,
    log_dtype: torch.dtype,
    y_rel: torch.Tensor,
    event_sigma: torch.Tensor | None,
) -> torch.Tensor:
    """Return a scalar or per-sample observation scale with broadcastable shape."""
    if event_sigma is None:
        scale = torch.as_tensor(sigma, device=log_device, dtype=log_dtype)
        if scale.ndim > 1:
            scale = scale.view(scale.shape[0], -1).squeeze(-1)
        if scale.ndim > 1:
            raise ValueError(f"sigma must be scalar or one value per sample, got shape {tuple(scale.shape)}.")
        if scale.ndim == 1:
            if scale.numel() == 1:
                scale = scale.squeeze(0)
            elif scale.numel() == y_rel.numel():
                scale = scale[:, None]
            else:
                raise ValueError(f"sigma has {scale.numel()} values for batch size {y_rel.numel()}.")
        return scale.clamp_min(1e-8)

    scale = event_sigma.to(device=log_device, dtype=log_dtype)
    if scale.ndim > 1:
        scale = scale.view(scale.shape[0], -1).squeeze(-1)
    if scale.ndim > 1:
        raise ValueError(f"event_sigma must be scalar or one value per sample, got shape {tuple(event_sigma.shape)}.")
    if scale.ndim == 1:
        if scale.numel() != y_rel.numel():
            raise ValueError(f"event_sigma has {scale.numel()} values for batch size {y_rel.numel()}.")
        scale = scale[:, None]
    return scale.clamp_min(1e-8)


def _scalar_sigma_value(sigma: float | torch.Tensor) -> float:
    """Return sigma as a scalar float for kernels with fixed mixture scales."""
    if torch.is_tensor(sigma):
        if sigma.numel() != 1:
            raise ValueError("gaussian_mixture default scales require scalar sigma.")
        return float(sigma.detach().cpu().item())
    return float(sigma)
