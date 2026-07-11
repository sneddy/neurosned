"""Benchmark visualization helpers."""

from pathlib import Path

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/neurosned-matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib

matplotlib.use("Agg", force=True)
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
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
    probabilities=None,
    readout_label=None,
    n_samples=4,
    rows=2,
    save_path=None,
    title_prefix=None,
    show=True,
):
    """Visualize batch predictions and optionally save the figure."""
    b = min(n_samples, X.shape[0])
    if b <= 0:
        return None
    idxs = np.random.choice(X.shape[0], size=b, replace=False)
    rows = max(1, int(rows))
    cols = max(1, int(np.ceil(b / rows)))
    fig = Figure(figsize=(cols * 5, rows * 2))
    FigureCanvas(fig)
    axes = np.asarray(fig.subplots(rows, cols, squeeze=False)).ravel()
    t_grid = (np.arange(T) * dt) + win_offset
    label = readout_label or "Predicted p(t)"
    for plot_idx, i in enumerate(idxs):
        ax = axes[plot_idx]
        if probabilities is None:
            p = torch.softmax(z[i] / temperature, dim=-1).cpu().numpy()
        else:
            p = probabilities[i].cpu().numpy()
        qi = q[i].cpu().numpy()
        ax.plot(t_grid, p, label=label, color="blue")
        ax.plot(t_grid, qi, label="Soft label $q(t)$", color="green", linestyle="--", alpha=0.8)
        ax.axvline(t_hat_abs[i].cpu().item(), color="red", label="Predicted time", linestyle="--")
        ax.axvline(y[i].cpu().item(), color="black", label="Actual time", linestyle="--")
        ax.set_title(f"Sample {i} | CE: {ce:.3f} | RMSE: {rmse_time:.3f}")
        ax.set_xlabel("Time (sec)")
        ax.set_ylabel("Probability / Weight")
        ax.legend()
    for ax in axes[b:]:
        ax.set_visible(False)
    fig.tight_layout()
    if title_prefix:
        fig.suptitle(title_prefix)
        fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig
