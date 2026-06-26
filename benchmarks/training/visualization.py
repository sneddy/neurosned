"""Benchmark visualization helpers."""

import matplotlib.pyplot as plt
import numpy as np
import torch


def draw_batch(X, y, z, t_hat_abs, ce, rmse_time, T, dt, win_offset, q, temperature, n_samples=4, rows=2):
    """
    Visualizes batch outputs: predicted distribution, actual value, groundtruth soft label, and losses.

    Args:
        X: Input batch (B, C, T)
        y: Groundtruth absolute times (B,)
        z: logits.squeeze(1) (B, T)
        t_hat_abs: predicted absolute times (B,)
        ce: cross-entropy loss (float)
        rmse_time: regression loss (float)
        T: int, window size (time bins)
        dt: time step (float)
        win_offset: float, window offset
        y_rel: relative click (B,)
        q: soft labels (B, T)
        temperature: softmax temperature
        n_samples: number of samples to plot from batch
        rows: number of rows in grid plot
    """
    b = min(n_samples, X.shape[0])
    idxs = np.random.choice(X.shape[0], size=b, replace=False)
    cols = int(np.ceil(b / rows))
    plt.figure(figsize=(cols * 5, rows * 2))
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
    plt.show()

