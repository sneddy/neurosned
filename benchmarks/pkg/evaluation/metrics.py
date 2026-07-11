"""Scalar and distributional evaluation metrics."""

from __future__ import annotations

import numpy as np


def rmse(predictions, targets) -> float:
    """Return root mean squared error."""
    prediction = np.asarray(predictions, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    return float(np.sqrt(np.mean((prediction - target) ** 2)))


def nrmse(predictions, targets) -> float:
    """Return RMSE normalized by target standard deviation."""
    target = np.asarray(targets, dtype=np.float64)
    denominator = float(np.std(target, ddof=1)) if len(target) > 1 else 0.0
    value = rmse(predictions, target)
    return value / denominator if denominator else value


def normalize_probabilities(probabilities) -> np.ndarray:
    """Return row-normalized posterior probabilities."""
    posterior = np.asarray(probabilities, dtype=np.float64)
    denominator = posterior.sum(axis=-1, keepdims=True)
    return np.divide(posterior, denominator, out=np.zeros_like(posterior), where=denominator > 0)


def crps_discrete(probabilities, grid, targets, *, reduction: str = "mean"):
    """Return CRPS for a discrete predictive distribution on a sorted grid.

    The score is computed as E|X-y| - 0.5 E|X-X'|, so its unit is the same as
    the event-time grid. Lower values are better.
    """
    posterior = normalize_probabilities(probabilities)
    time_grid = np.asarray(grid, dtype=np.float64).reshape(-1)
    target = np.asarray(targets, dtype=np.float64).reshape(-1)
    if posterior.ndim != 2:
        raise ValueError(f"Expected probabilities with shape (n, time), got {posterior.shape!r}.")
    if posterior.shape[-1] != time_grid.size:
        raise ValueError("Probability width and grid length must match.")
    if posterior.shape[0] != target.size:
        raise ValueError("Probability rows and targets length must match.")

    first_term = np.sum(posterior * np.abs(time_grid[None, :] - target[:, None]), axis=-1)
    if time_grid.size <= 1:
        scores = first_term
    else:
        cdf = np.cumsum(posterior, axis=-1)[:, :-1]
        delta = np.diff(time_grid)
        expected_pairwise_abs = 2.0 * np.sum(cdf * (1.0 - cdf) * delta[None, :], axis=-1)
        scores = first_term - 0.5 * expected_pairwise_abs
    return _reduce(scores, reduction)


def fixed_kernel_event_nll(
    probabilities,
    grid,
    targets,
    *,
    sigma: float = 0.15,
    reduction: str = "mean",
    eps: float = 1e-12,
):
    """Return Gaussian shared-kernel RT NLL for event-time posteriors.

    This evaluates the likelihood p(y|x)=sum_t p(t|x) N(y;t,sigma^2) using the
    same observation kernel for every model. Lower values are better.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma!r}.")
    posterior = normalize_probabilities(probabilities)
    time_grid = np.asarray(grid, dtype=np.float64).reshape(-1)
    target = np.asarray(targets, dtype=np.float64).reshape(-1)
    if posterior.ndim != 2:
        raise ValueError(f"Expected probabilities with shape (n, time), got {posterior.shape!r}.")
    if posterior.shape[-1] != time_grid.size:
        raise ValueError("Probability width and grid length must match.")
    if posterior.shape[0] != target.size:
        raise ValueError("Probability rows and targets length must match.")

    log_weights = np.log(np.maximum(posterior, eps))
    z = (target[:, None] - time_grid[None, :]) / float(sigma)
    log_kernel = -0.5 * z * z - np.log(float(sigma) * np.sqrt(2.0 * np.pi))
    log_terms = log_weights + log_kernel
    max_log = np.max(log_terms, axis=-1, keepdims=True)
    log_likelihood = max_log[:, 0] + np.log(np.sum(np.exp(log_terms - max_log), axis=-1))
    return _reduce(-log_likelihood, reduction)


def posterior_distributional_metrics(
    probabilities,
    grid,
    targets,
    *,
    event_nll_sigma: float = 0.15,
    reduction: str = "mean",
) -> dict[str, float | np.ndarray]:
    """Return proper distributional scores for event-time posteriors."""
    crps = crps_discrete(probabilities, grid, targets, reduction=reduction)
    event_nll = fixed_kernel_event_nll(
        probabilities,
        grid,
        targets,
        sigma=event_nll_sigma,
        reduction=reduction,
    )
    if reduction == "none":
        return {
            "posterior_crps": crps,
            "posterior_fixed_kernel_event_nll": event_nll,
            "posterior_fixed_kernel_sigma": float(event_nll_sigma),
        }
    return {
        "posterior_crps": float(crps),
        "posterior_fixed_kernel_event_nll": float(event_nll),
        "posterior_fixed_kernel_sigma": float(event_nll_sigma),
    }


def _reduce(values: np.ndarray, reduction: str):
    if reduction == "none":
        return values
    if reduction == "mean":
        return float(np.mean(values))
    raise ValueError(f"Unknown reduction: {reduction!r}.")
