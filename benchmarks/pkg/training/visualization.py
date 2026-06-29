"""Benchmark visualization helpers."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def draw_batch(
    X,
    y,
    z,
    t_hat_abs,
    ce,
    rmse_time,
    T,
    dt,
    win_offset,
    q,
    temperature,
    n_samples=4,
    rows=2,
    save_path=None,
    title_prefix=None,
    show=True,
):
    """Visualize batch predictions and optionally save the figure."""
    b = min(n_samples, X.shape[0])
    idxs = np.random.choice(X.shape[0], size=b, replace=False)
    cols = int(np.ceil(b / rows))
    fig = plt.figure(figsize=(cols * 5, rows * 2))
    t_grid = (np.arange(T) * dt) + win_offset
    for plot_idx, i in enumerate(idxs):
        plt.subplot(rows, cols, plot_idx + 1)
        p = torch.softmax(z[i] / temperature, dim=-1).cpu().numpy()
        qi = q[i].cpu().numpy()
        plt.plot(t_grid, p, label="Predicted p(t)", color="blue")
        plt.plot(t_grid, qi, label="Soft label $q(t)$", color="green", linestyle="--", alpha=0.8)
        plt.axvline([t_hat_abs[i].cpu().item()], color="red", label="Predicted time", linestyle="--")
        plt.axvline([y[i].cpu().item()], color="black", label="Actual time", linestyle="--")
        plt.title(f"Sample {i} | CE: {ce:.3f} | RMSE: {rmse_time:.3f}")
        plt.xlabel("Time (sec)")
        plt.ylabel("Probability / Weight")
        plt.legend()
        plt.tight_layout()
    if title_prefix:
        fig.suptitle(title_prefix)
        fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
