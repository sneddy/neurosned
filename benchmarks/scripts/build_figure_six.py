"""Build the final five-panel posterior-geometry figure for the paper.

Running this script without arguments reads the six repeated segmentation
experiments and writes only
``benchmarks/experiments/paper_figures/final_figure_six.png``.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ["MNE_DONTWRITE_HOME"] = "true"
os.environ["NUMBA_DISABLE_JIT"] = "1"
os.environ["MPLCONFIGDIR"] = "/tmp/neurosned-matplotlib"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from benchmarks.pkg.config import ExperimentConfig
from benchmarks.pkg.evaluation.factory import build_temperature_readout
from benchmarks.pkg.paper_figures import (
    canonical_name,
    color_for,
    discover_posterior_groups,
    infer_grid,
    label_for,
    load_snapshot,
    read_temperature,
)


DEFAULT_EXPERIMENT_DIR = PROJECT_ROOT / "benchmarks/experiments/02_segmentation_ablations"
DEFAULT_OUTPUT = PROJECT_ROOT / "benchmarks/experiments/paper_figures/final_figure_six.png"
PAPER_WIDTH_IN = 14.65 / 2.54
FIGURE_WIDTH_IN = 8.2
BASE_FONT_SIZE = 10.0 * FIGURE_WIDTH_IN / PAPER_WIDTH_IN
COVERAGE_LEVELS = np.asarray([0.50, 0.60, 0.70, 0.80, 0.90], dtype=np.float64)
NEAR_TARGET_MS = 150.0
ALIGN_WINDOW_MS = 1000.0


@dataclass(frozen=True)
class ObjectiveSpec:
    """Fixed visual identity for one event-time objective."""

    name: str
    short_label: str
    marker: str


OBJECTIVES = (
    ObjectiveSpec("ets_unet_ce", "CE", "o"),
    ObjectiveSpec("ets_unet_event_nll", "EventNLL", "o"),
    ObjectiveSpec("ets_unet_event_nll_mixture", "Mixture", "o"),
    ObjectiveSpec("ets_unet_hazard_event_nll", "Hazard", "o"),
    ObjectiveSpec("ets_unet_time_only", "RT-only", "o"),
    ObjectiveSpec("ets_unet_wasserstein", "Wasserstein", "o"),
)


@dataclass
class SeedSummary:
    """Posterior quantities needed from one trained seed."""

    aligned_time_ms: np.ndarray
    aligned_density: np.ndarray
    coverage: np.ndarray
    width80_median_ms: float
    near_target_mass_mean: float
    mode_mean_gap_median_ms: float


@dataclass
class ObjectiveSummary:
    """All seed summaries and plotting metadata for one objective."""

    spec: ObjectiveSpec
    color: str
    runs: list[SeedSummary]


def parse_args() -> argparse.Namespace:
    """Parse the intentionally small standalone CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR,
        help="Directory containing the six repeated segmentation experiments.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="PNG path to write.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Raster resolution.")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    """Resolve project-relative command-line paths."""
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def discover_required_groups(experiment_dir: Path) -> list[tuple[ObjectiveSpec, Path, list[Path]]]:
    """Return the latest complete repeated run for every plotted objective."""
    expected = {spec.name: spec for spec in OBJECTIVES}
    candidates: dict[str, list[tuple[Path, list[Path]]]] = {name: [] for name in expected}
    for name, group_dir, run_dirs in discover_posterior_groups(experiment_dir, split="test"):
        clean_name = canonical_name(name)
        if clean_name in expected:
            candidates[clean_name].append((group_dir, run_dirs))

    selected = []
    for spec in OBJECTIVES:
        matches = candidates[spec.name]
        if not matches:
            raise FileNotFoundError(f"No saved test posteriors found for {spec.name} in {experiment_dir}")
        group_dir, run_dirs = sorted(matches, key=lambda item: item[0].name)[-1]
        if len(run_dirs) != 5:
            raise RuntimeError(f"Expected five seeds for {spec.name}, found {len(run_dirs)} in {group_dir}")
        selected.append((spec, group_dir, sorted(run_dirs)))
    return selected


def quantile_indices(cdf: np.ndarray, quantile: float) -> np.ndarray:
    """Return the first discrete bin reaching a posterior quantile."""
    return np.argmax(cdf >= float(quantile), axis=1)


def aligned_posterior_density(
    probabilities: np.ndarray,
    targets: np.ndarray,
    grid: np.ndarray,
    *,
    window_ms: float,
    chunk_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Average posterior density after aligning each trial to its observed RT."""
    dt = float(np.median(np.diff(grid)))
    half_window = float(window_ms) / 1000.0
    relative_time = np.arange(-half_window, half_window + dt / 2.0, dt, dtype=np.float64)
    density_sum = np.zeros(relative_time.size, dtype=np.float64)
    density_count = np.zeros(relative_time.size, dtype=np.int64)

    for start in range(0, len(targets), chunk_size):
        stop = min(start + chunk_size, len(targets))
        chunk = probabilities[start:stop]
        query = targets[start:stop, None] + relative_time[None, :]
        position = (query - grid[0]) / dt
        valid = (query >= grid[0]) & (query <= grid[-1])
        left = np.floor(position).astype(np.int64)
        left = np.clip(left, 0, grid.size - 1)
        right = np.minimum(left + 1, grid.size - 1)
        fraction = position - left
        rows = np.arange(stop - start, dtype=np.int64)[:, None]
        interpolated = chunk[rows, left] * (1.0 - fraction) + chunk[rows, right] * fraction
        interpolated /= dt
        density_sum += np.where(valid, interpolated, 0.0).sum(axis=0)
        density_count += valid.sum(axis=0)

    density = np.divide(
        density_sum,
        density_count,
        out=np.full_like(density_sum, np.nan),
        where=density_count > 0,
    )
    return relative_time * 1000.0, density


def load_seed_summary(run_dir: Path, spec: ObjectiveSpec) -> SeedSummary:
    """Load one seed and reduce its posterior to Figure 6 quantities."""
    snapshot = load_snapshot(run_dir)
    if "config" not in snapshot:
        raise ValueError(f"Missing config section in {run_dir / 'config.yaml'}")
    config = ExperimentConfig.model_validate(snapshot["config"])
    temperature, calibration = read_temperature(run_dir, snapshot, "calibrated", None)

    logits = np.load(run_dir / "predictions/test_logits.npy")
    metadata = pd.read_csv(run_dir / "predictions/test_predictions.csv", usecols=["target"])
    targets = metadata["target"].to_numpy(dtype=np.float64)
    if logits.shape[0] != targets.size:
        raise ValueError(f"Logit/target row mismatch in {run_dir}")

    grid = infer_grid(snapshot, calibration, logits)
    sfreq = 1.0 / float(np.median(np.diff(grid)))
    win_offset = float(grid[0])
    readout = build_temperature_readout(config, sfreq=sfreq, win_offset=win_offset)
    expected_readout = "hazard" if spec.name == "ets_unet_hazard_event_nll" else "softmax"
    if readout.name != expected_readout:
        raise ValueError(
            f"Unexpected readout for {spec.name}: expected {expected_readout}, found {readout.name}"
        )
    probabilities = np.asarray(readout.probability_fn(logits, temperature), dtype=np.float64)

    row_sums = probabilities.sum(axis=1)
    if not np.all(np.isfinite(probabilities)) or not np.allclose(row_sums, 1.0, atol=2e-5):
        raise ValueError(f"Invalid posterior probabilities in {run_dir}")

    representable = (targets >= grid[0]) & (targets <= grid[-1])
    probabilities = probabilities[representable]
    targets = targets[representable]
    if targets.size == 0:
        raise ValueError(f"No representable targets in {run_dir}")

    cdf = np.cumsum(probabilities, axis=1)
    cdf[:, -1] = 1.0
    q10 = grid[quantile_indices(cdf, 0.10)]
    q90 = grid[quantile_indices(cdf, 0.90)]
    width80_median_ms = float(np.median(q90 - q10) * 1000.0)

    near_sec = NEAR_TARGET_MS / 1000.0
    near = np.abs(grid[None, :] - targets[:, None]) <= near_sec
    near_target_mass_mean = float(np.mean(np.sum(probabilities * near, axis=1)))

    posterior_mean = probabilities @ grid
    posterior_mode = grid[np.argmax(probabilities, axis=1)]
    mode_mean_gap_median_ms = float(np.median(np.abs(posterior_mode - posterior_mean)) * 1000.0)

    coverage = []
    for level in COVERAGE_LEVELS:
        tail = (1.0 - float(level)) / 2.0
        lower = grid[quantile_indices(cdf, tail)]
        upper = grid[quantile_indices(cdf, 1.0 - tail)]
        coverage.append(float(np.mean((targets >= lower) & (targets <= upper))))

    aligned_time_ms, aligned_density = aligned_posterior_density(
        probabilities,
        targets,
        grid,
        window_ms=ALIGN_WINDOW_MS,
    )
    return SeedSummary(
        aligned_time_ms=aligned_time_ms,
        aligned_density=aligned_density,
        coverage=np.asarray(coverage, dtype=np.float64),
        width80_median_ms=width80_median_ms,
        near_target_mass_mean=near_target_mass_mean,
        mode_mean_gap_median_ms=mode_mean_gap_median_ms,
    )


def load_objective_summaries(experiment_dir: Path) -> list[ObjectiveSummary]:
    """Load and summarize all six objectives."""
    summaries = []
    for index, (spec, _group_dir, run_dirs) in enumerate(discover_required_groups(experiment_dir)):
        paper_label = label_for(spec.name)
        color = color_for(paper_label, index)
        runs = []
        for run_dir in run_dirs:
            print(f"Loading {spec.short_label}: {run_dir.name}", flush=True)
            runs.append(load_seed_summary(run_dir, spec))
        summaries.append(ObjectiveSummary(spec=spec, color=color, runs=runs))
    return summaries


def mean_and_sample_std(values: list[np.ndarray] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return mean and sample standard deviation across seeds."""
    array = np.asarray(values, dtype=np.float64)
    return np.nanmean(array, axis=0), np.nanstd(array, axis=0, ddof=1)


def style_axis(ax, *, grid_axis: str = "y") -> None:
    """Apply consistent journal-scale styling."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, color="#e8e8e8", linewidth=0.65, zorder=0)
    ax.tick_params(axis="both", labelsize=BASE_FONT_SIZE, length=3.0, width=0.8)
    ax.xaxis.label.set_size(BASE_FONT_SIZE)
    ax.yaxis.label.set_size(BASE_FONT_SIZE)


def metric_values(
    summaries: list[ObjectiveSummary],
    attribute: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return objective-level means and seed standard deviations for one scalar metric."""
    means = []
    stds = []
    for summary in summaries:
        values = np.asarray([getattr(run, attribute) for run in summary.runs], dtype=np.float64)
        means.append(float(np.mean(values)))
        stds.append(float(np.std(values, ddof=1)))
    return np.asarray(means), np.asarray(stds)


def plot_figure(summaries: list[ObjectiveSummary], output: Path, *, dpi: int) -> None:
    """Render the final 2+3 Figure 6 layout."""
    plt.rcParams.update(
        {
            "font.size": BASE_FONT_SIZE,
            "axes.titlesize": BASE_FONT_SIZE,
            "axes.labelsize": BASE_FONT_SIZE,
            "xtick.labelsize": BASE_FONT_SIZE,
            "ytick.labelsize": BASE_FONT_SIZE,
            "legend.fontsize": BASE_FONT_SIZE,
        }
    )

    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, 5.8), facecolor="white")
    top_grid = fig.add_gridspec(
        1,
        2,
        left=0.095,
        right=0.99,
        bottom=0.56,
        top=0.89,
        wspace=0.30,
    )
    bottom_grid = fig.add_gridspec(
        1,
        3,
        left=0.16,
        right=0.99,
        bottom=0.10,
        top=0.39,
        wspace=0.10,
    )
    ax_aligned = fig.add_subplot(top_grid[0, 0])
    ax_coverage = fig.add_subplot(top_grid[0, 1])
    ax_width = fig.add_subplot(bottom_grid[0, 0])
    ax_mass = fig.add_subplot(bottom_grid[0, 1], sharey=ax_width)
    ax_gap = fig.add_subplot(bottom_grid[0, 2], sharey=ax_width)

    legend_handles = []
    for summary in summaries:
        curves = [run.aligned_density for run in summary.runs]
        mean, std = mean_and_sample_std(curves)
        x_ms = summary.runs[0].aligned_time_ms
        ax_aligned.fill_between(
            x_ms,
            np.maximum(mean - std, 0.0),
            mean + std,
            color=summary.color,
            alpha=0.12,
            linewidth=0,
            zorder=1,
        )
        ax_aligned.plot(
            x_ms,
            mean,
            color=summary.color,
            linewidth=1.0,
            zorder=2,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=summary.color,
                linewidth=1.0,
                marker="o",
                markersize=6.2,
                markeredgewidth=0.5,
                label=summary.spec.short_label,
            )
        )

    ax_aligned.axvline(0.0, color="#333333", linestyle="--", linewidth=0.8, zorder=1)
    ax_aligned.set_xlim(-1000.0, 1000.0)
    ax_aligned.set_xticks([-1000, -500, 0, 500, 1000])
    aligned_tick_labels = ax_aligned.get_xticklabels()
    aligned_tick_labels[-1].set_horizontalalignment("right")
    ax_aligned.set_ylim(bottom=0.0)
    ax_aligned.set_xlabel("Relative to observed RT (ms)")
    ax_aligned.set_ylabel("Posterior density")
    ax_aligned.set_title("A. Target-aligned posterior", fontweight="normal", pad=7)
    style_axis(ax_aligned)

    ax_coverage.plot(
        [0.48, 0.92],
        [0.48, 0.92],
        color="#777777",
        linestyle="--",
        linewidth=0.8,
        zorder=1,
    )
    for summary in summaries:
        mean, std = mean_and_sample_std([run.coverage for run in summary.runs])
        ax_coverage.fill_between(
            COVERAGE_LEVELS,
            np.maximum(mean - std, 0.0),
            np.minimum(mean + std, 1.0),
            color=summary.color,
            alpha=0.23,
            linewidth=0,
            zorder=1,
        )
        ax_coverage.plot(
            COVERAGE_LEVELS,
            mean,
            color=summary.color,
            linewidth=1.0,
            marker="o",
            markersize=4.6,
            markeredgewidth=0.4,
            zorder=2,
        )
    ax_coverage.plot(
        [0.79, 0.83],
        [0.07, 0.07],
        color="#777777",
        linestyle="--",
        linewidth=0.8,
    )
    ax_coverage.text(
        0.84,
        0.07,
        "ideal",
        color="#666666",
        fontsize=BASE_FONT_SIZE,
        va="center",
    )
    ax_coverage.set_xlim(0.48, 0.92)
    ax_coverage.set_ylim(0.0, 1.0)
    ax_coverage.set_xticks(COVERAGE_LEVELS)
    ax_coverage.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
    ax_coverage.set_xlabel("Nominal central interval")
    ax_coverage.set_ylabel("Observed-RT coverage")
    ax_coverage.set_title("B. Posterior coverage", fontweight="normal", pad=7)
    style_axis(ax_coverage)

    width_mean, width_std = metric_values(summaries, "width80_median_ms")
    mass_mean, mass_std = metric_values(summaries, "near_target_mass_mean")
    gap_mean, gap_std = metric_values(summaries, "mode_mean_gap_median_ms")
    y = np.arange(len(summaries), dtype=np.float64)
    metric_panels = (
        (ax_width, width_mean, width_std, "C. Width80", "Width (ms)"),
        (ax_mass, mass_mean, mass_std, "D. Mass $\\pm$150 ms", "Probability"),
        (ax_gap, gap_mean, gap_std, "E. Mode-mean", "Gap (ms)"),
    )
    for ax, means, stds, title, xlabel in metric_panels:
        for index, summary in enumerate(summaries):
            ax.errorbar(
                means[index],
                y[index],
                xerr=stds[index],
                fmt="o",
                color=summary.color,
                ecolor="#111111",
                elinewidth=1.05,
                capsize=4.0,
                capthick=1.05,
                markersize=6.6,
                markeredgewidth=0.4,
                zorder=3,
            )
        ax.set_title(title, fontweight="normal", pad=7)
        ax.set_xlabel(xlabel)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        style_axis(ax, grid_axis="x")

    ax_width.set_yticks(y, [summary.spec.short_label for summary in summaries])
    ax_width.invert_yaxis()
    ax_mass.tick_params(axis="y", left=False, labelleft=False)
    ax_gap.tick_params(axis="y", left=False, labelleft=False)
    ax_mass.spines["left"].set_visible(False)
    ax_gap.spines["left"].set_visible(False)
    ax_mass.set_xlim(0.28, 0.53)

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=len(legend_handles),
        frameon=False,
        fontsize=BASE_FONT_SIZE,
        handlelength=1.35,
        handletextpad=0.35,
        columnspacing=0.82,
        borderaxespad=0.0,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, facecolor="white")
    plt.close(fig)


def main() -> None:
    """Build only the final Figure 6 PNG."""
    args = parse_args()
    experiment_dir = resolve_path(args.experiment_dir)
    output = resolve_path(args.output)
    summaries = load_objective_summaries(experiment_dir)
    plot_figure(summaries, output, dpi=args.dpi)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
