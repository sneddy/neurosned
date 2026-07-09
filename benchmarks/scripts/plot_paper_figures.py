"""Generate paper-facing figures from benchmark artifacts."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["MNE_DONTWRITE_HOME"] = "true"
os.environ["NUMBA_DISABLE_JIT"] = "1"
os.environ["MPLCONFIGDIR"] = "/tmp/neurosned-matplotlib"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmarks.pkg.config import resolve_path
from benchmarks.pkg.paper_figures import (
    PerformanceRow,
    PosteriorGroup,
    aligned_posterior_frame,
    best_regression_row,
    coverage_summary_frame,
    group_summary_frame,
    load_json,
    load_posterior_groups,
    palette_frame,
    path_text,
    per_seed_summary_frame,
    representative_seed_frame,
    save_figure,
    segmentation_performance_rows,
)


plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8,
        "figure.titlesize": 12,
    }
)

DEFAULT_FIGURES = [
    "posterior_geometry",
    "posterior_pareto",
    "trial_raster",
    "performance_forest",
    "window_support",
    "temperature_sensitivity",
]


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description="Create paper-facing benchmark figures from saved artifacts.")
    parser.add_argument(
        "segmentation_dir",
        type=Path,
        help="Segmentation experiment directory, usually benchmarks/experiments/02_segmentation_ablations.",
    )
    parser.add_argument(
        "--regression-dir",
        type=Path,
        default=Path("benchmarks/experiments/01_regression_baselines"),
        help="Regression baseline directory used for the scalar baseline row.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/experiments/paper_figures"),
        help="Output directory for figures and source CSVs.",
    )
    parser.add_argument("--split", default="test", help="Saved prediction split prefix.")
    parser.add_argument("--readout", choices=("calibrated", "base"), default="calibrated")
    parser.add_argument("--temperature", type=float, default=None, help="Override all softmax temperatures.")
    parser.add_argument(
        "--target-filter",
        choices=("representable", "all"),
        default="representable",
        help="Rows used for posterior-geometry summaries.",
    )
    parser.add_argument("--near-ms", type=float, default=150.0, help="Near-target posterior mass window.")
    parser.add_argument("--score-sigma", type=float, default=0.12, help="Gaussian sigma for fixed EventNLL scoring.")
    parser.add_argument("--align-window-ms", type=float, default=1000.0)
    parser.add_argument("--coverage-levels", nargs="+", type=float, default=[0.50, 0.60, 0.70, 0.80, 0.90])
    parser.add_argument("--include", action="append", default=[], help="Only include segmentation groups containing text.")
    parser.add_argument("--exclude", action="append", default=[], help="Exclude segmentation groups containing text.")
    parser.add_argument(
        "--figures",
        nargs="+",
        default=DEFAULT_FIGURES,
        choices=(
            "all",
            "posterior_geometry",
            "posterior_pareto",
            "representative_posteriors",
            "trial_raster",
            "performance_forest",
            "window_support",
            "temperature_sensitivity",
        ),
        help="Figure groups to generate.",
    )
    parser.add_argument("--max-examples", type=int, default=3, help="Representative posterior examples to plot.")
    parser.add_argument("--formats", nargs="+", default=["png"], choices=("png", "svg", "pdf"))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--raster-bins", type=int, default=260)
    return parser


def resolve_required(path: Path) -> Path:
    """Resolve a project-relative path."""
    resolved = resolve_path(path, PROJECT_ROOT)
    if resolved is None:
        raise ValueError("Path cannot be None.")
    return resolved.resolve()


def csv_path(output_dir: Path, filename: str) -> Path:
    """Return a CSV output path under the paper figure CSV folder."""
    path = output_dir / "csv" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def style_axes(ax) -> None:
    """Apply compact paper-facing axis styling."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color="#e8e8e8", linewidth=0.8)
    ax.tick_params(axis="both", labelsize=8)


def display_label(label: str) -> str:
    """Return a compact multi-line label for dense figures."""
    replacements = {
        "ETS-U-Net CE": "CE",
        "ETS-U-Net EventNLL": "EventNLL",
        "ETS-U-Net mixture EventNLL": "Mixture EventNLL",
        "ETS-U-Net hazard EventNLL": "Hazard EventNLL",
        "ETS-U-Net time-only": "Soft-argmax\nRT loss",
        "Soft-argmax RT loss": "Soft-argmax\nRT loss",
        "ETS-U-Net Wasserstein": "Wasserstein",
    }
    return replacements.get(label, label)


def display_label_singleline(label: str) -> str:
    """Return a compact one-line label for point annotations."""
    return display_label(label).replace("\n", " ")


def selected(figures: list[str], name: str) -> bool:
    """Return whether a figure set is selected."""
    selected_figures = DEFAULT_FIGURES if "all" in figures else figures
    return name in selected_figures


def group_metric(seed_summary: pd.DataFrame, metric: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return labels, means, and stds for one seed-level metric."""
    labels = []
    means = []
    stds = []
    for label, frame in seed_summary.groupby("label", sort=False):
        labels.append(str(label))
        values = frame[metric].to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        means.append(float(np.mean(values)) if values.size else np.nan)
        stds.append(float(np.std(values, ddof=1)) if values.size > 1 else 0.0)
    return labels, np.asarray(means), np.asarray(stds)


def plot_posterior_geometry(
    groups: list[PosteriorGroup],
    seed_summary: pd.DataFrame,
    coverage: pd.DataFrame,
    output_dir: Path,
    *,
    formats: list[str],
    dpi: int,
    near_ms: float,
) -> list[Path]:
    """Save the main multi-panel posterior-geometry figure."""
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.4))
    ax_perf, ax_aligned, ax_width, ax_mass, ax_gap, ax_coverage = axes.ravel()

    labels = [display_label(group.label) for group in groups]
    colors = [group.color for group in groups]
    x = np.arange(len(groups), dtype=np.float64)

    perf_means = np.array([np.mean([run.nrmse_value for run in group.runs]) for group in groups])
    perf_stds = np.array([
        np.std([run.nrmse_value for run in group.runs], ddof=1) if len(group.runs) > 1 else 0.0
        for group in groups
    ])
    ax_perf.errorbar(x, perf_means, yerr=perf_stds, fmt="o", color="#222222", ecolor="#222222", capsize=3)
    ax_perf.scatter(x, perf_means, color=colors, s=50, zorder=3)
    ax_perf.set_xticks(x, labels, rotation=25, ha="right")
    ax_perf.set_ylabel("R11 nRMSE")
    ax_perf.set_title("A. Scalar readout")
    ax_perf.set_ylim(float(np.nanmin(perf_means - perf_stds)) - 0.012, float(np.nanmax(perf_means + perf_stds)) + 0.012)
    style_axes(ax_perf)

    aligned = aligned_posterior_frame(groups)
    for group in groups:
        frame = aligned[
            (aligned["label"] == group.label)
            & (aligned["curve_type"] == "group_mean")
        ]
        if frame.empty:
            continue
        x_ms = frame["relative_time_ms"].to_numpy(dtype=np.float64)
        mean = frame["mean_posterior_density"].to_numpy(dtype=np.float64)
        std = frame.get("std_posterior_density", pd.Series(np.zeros(len(frame)))).to_numpy(dtype=np.float64)
        ax_aligned.fill_between(x_ms, mean - std, mean + std, color=group.color, alpha=0.08, linewidth=0)
        ax_aligned.plot(x_ms, mean, color=group.color, linewidth=1.25, label=display_label(group.label))
    ax_aligned.axvline(0.0, color="#222222", linestyle="--", linewidth=1.0)
    ax_aligned.set_xlabel("Time relative to true RT (ms)")
    ax_aligned.set_ylabel("Mean posterior density")
    ax_aligned.set_title("B. Target-aligned posterior")
    ax_aligned.legend(frameon=False, fontsize=7)
    style_axes(ax_aligned)

    labels_width, width_mean, width_std = group_metric(seed_summary, "width80_median_ms")
    ax_width.errorbar(x, width_mean, yerr=width_std, fmt="o", color="#222222", capsize=3)
    ax_width.scatter(x, width_mean, color=colors, s=48, zorder=3)
    ax_width.set_xticks(x, [display_label(label) for label in labels_width], rotation=25, ha="right")
    ax_width.set_ylabel("80% width (ms)")
    ax_width.set_title("C. Posterior width")
    style_axes(ax_width)

    mass_col = f"mass_within_{near_ms:g}ms_mean"
    labels_mass, mass_mean, mass_std = group_metric(seed_summary, mass_col)
    ax_mass.errorbar(x, mass_mean, yerr=mass_std, fmt="o", color="#222222", capsize=3)
    ax_mass.scatter(x, mass_mean, color=colors, s=48, zorder=3)
    ax_mass.set_xticks(x, [display_label(label) for label in labels_mass], rotation=25, ha="right")
    ax_mass.set_ylabel(f"Mass within +/-{near_ms:g} ms")
    ax_mass.set_title("D. Near-target mass")
    style_axes(ax_mass)

    labels_gap, gap_mean, gap_std = group_metric(seed_summary, "mode_mean_gap_median_ms")
    ax_gap.errorbar(x, gap_mean, yerr=gap_std, fmt="o", color="#222222", capsize=3)
    ax_gap.scatter(x, gap_mean, color=colors, s=48, zorder=3)
    ax_gap.set_xticks(x, [display_label(label) for label in labels_gap], rotation=25, ha="right")
    ax_gap.set_ylabel("|mode - mean| (ms)")
    ax_gap.set_title("E. Mode-mean gap")
    style_axes(ax_gap)

    ax_coverage.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1.0, label="ideal")
    for group in groups:
        frame = coverage[
            (coverage["label"] == group.label)
            & (coverage["curve_type"] == "group_mean")
        ].sort_values("nominal")
        if frame.empty:
            continue
        nominal = frame["nominal"].to_numpy(dtype=np.float64)
        mean = frame["coverage"].to_numpy(dtype=np.float64)
        std = frame.get("coverage_std", pd.Series(np.zeros(len(frame)))).to_numpy(dtype=np.float64)
        ax_coverage.fill_between(nominal, mean - std, mean + std, color=group.color, alpha=0.22, linewidth=0)
        ax_coverage.plot(
            nominal,
            mean,
            marker="o",
            markersize=4.0,
            color=group.color,
            linewidth=1.15,
            label=display_label(group.label),
        )
    ax_coverage.set_xlim(0.45, 0.95)
    ax_coverage.set_ylim(0.0, 1.0)
    ax_coverage.set_xlabel("Nominal central interval")
    ax_coverage.set_ylabel("Empirical coverage")
    ax_coverage.set_title("F. Posterior coverage")
    ax_coverage.legend(frameon=False, fontsize=7)
    style_axes(ax_coverage)

    fig.suptitle("Segmentation losses produce distinct event-time posteriors", fontsize=13, y=1.02)
    fig.tight_layout()
    paths = save_figure(fig, output_dir, "posterior_geometry_main", formats, dpi=dpi)
    plt.close(fig)
    return paths


def metric_mean_std(frame: pd.DataFrame, column: str) -> tuple[float, float]:
    """Return mean and sample std for one metric column."""
    values = frame[column].to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        return float(values[0]), 0.0
    return float(np.mean(values)), float(np.std(values, ddof=1))


def pareto_frame(groups: list[PosteriorGroup], seed_summary: pd.DataFrame, *, near_ms: float) -> pd.DataFrame:
    """Return source values for the posterior-geometry Pareto plot."""
    mass_col = f"mass_within_{near_ms:g}ms_mean"
    rows = []
    for group in groups:
        frame = seed_summary[seed_summary["name"] == group.name]
        nrmse_mean, nrmse_std = metric_mean_std(frame, "nrmse")
        mass_mean, mass_std = metric_mean_std(frame, mass_col)
        width_mean, width_std = metric_mean_std(frame, "width80_median_ms")
        coverage_mean, coverage_std = metric_mean_std(frame, "coverage_mae")
        rows.append(
            {
                "name": group.name,
                "label": group.label,
                "display_label": display_label_singleline(group.label),
                "color": group.color,
                "n_runs": int(len(frame)),
                "nrmse_mean": nrmse_mean,
                "nrmse_std": nrmse_std,
                "near_target_mass_mean": mass_mean,
                "near_target_mass_std": mass_std,
                "width80_median_ms_mean": width_mean,
                "width80_median_ms_std": width_std,
                "coverage_mae_mean": coverage_mean,
                "coverage_mae_std": coverage_std,
            }
        )
    return pd.DataFrame(rows)


def annotate_points(ax, frame: pd.DataFrame, *, y_column: str) -> None:
    """Annotate Pareto points without adding a separate legend."""
    for _, row in frame.iterrows():
        ax.annotate(
            str(row["display_label"]),
            (float(row["nrmse_mean"]), float(row[y_column])),
            xytext=(5, 3),
            textcoords="offset points",
            fontsize=7.2,
            color="#333333",
        )


def plot_posterior_pareto(
    groups: list[PosteriorGroup],
    seed_summary: pd.DataFrame,
    output_dir: Path,
    *,
    formats: list[str],
    dpi: int,
    near_ms: float,
) -> tuple[list[Path], pd.DataFrame]:
    """Save scalar-vs-posterior geometry Pareto plots."""
    frame = pareto_frame(groups, seed_summary, near_ms=near_ms)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), sharex=True)
    ax_mass, ax_coverage = axes

    x = frame["nrmse_mean"].to_numpy(dtype=np.float64)
    xerr = frame["nrmse_std"].to_numpy(dtype=np.float64)
    colors = frame["color"].tolist()

    y_mass = frame["near_target_mass_mean"].to_numpy(dtype=np.float64)
    y_mass_err = frame["near_target_mass_std"].to_numpy(dtype=np.float64)
    for index, row in frame.iterrows():
        ax_mass.errorbar(
            row["nrmse_mean"],
            row["near_target_mass_mean"],
            xerr=row["nrmse_std"],
            yerr=row["near_target_mass_std"],
            fmt="o",
            color=colors[index],
            ecolor="#333333",
            elinewidth=0.9,
            capsize=2.5,
            markersize=6,
            zorder=3,
        )
    annotate_points(ax_mass, frame, y_column="near_target_mass_mean")
    ax_mass.set_xlabel("R11 nRMSE")
    ax_mass.set_ylabel(f"Mass within +/-{near_ms:g} ms")
    ax_mass.set_title("A. Accuracy vs target concentration")
    ax_mass.set_xlim(float(np.nanmin(x - xerr)) - 0.004, float(np.nanmax(x + xerr)) + 0.004)
    ax_mass.set_ylim(float(np.nanmin(y_mass - y_mass_err)) - 0.025, float(np.nanmax(y_mass + y_mass_err)) + 0.025)
    style_axes(ax_mass)
    ax_mass.grid(True, axis="x", color="#e8e8e8", linewidth=0.8)

    y_cov = frame["coverage_mae_mean"].to_numpy(dtype=np.float64)
    y_cov_err = frame["coverage_mae_std"].to_numpy(dtype=np.float64)
    for index, row in frame.iterrows():
        ax_coverage.errorbar(
            row["nrmse_mean"],
            row["coverage_mae_mean"],
            xerr=row["nrmse_std"],
            yerr=row["coverage_mae_std"],
            fmt="o",
            color=colors[index],
            ecolor="#333333",
            elinewidth=0.9,
            capsize=2.5,
            markersize=6,
            zorder=3,
        )
    annotate_points(ax_coverage, frame, y_column="coverage_mae_mean")
    ax_coverage.set_xlabel("R11 nRMSE")
    ax_coverage.set_ylabel("Coverage MAE")
    ax_coverage.set_title("B. Accuracy vs calibration error")
    ax_coverage.set_ylim(float(np.nanmin(y_cov - y_cov_err)) - 0.018, float(np.nanmax(y_cov + y_cov_err)) + 0.018)
    style_axes(ax_coverage)
    ax_coverage.grid(True, axis="x", color="#e8e8e8", linewidth=0.8)

    fig.suptitle("Similar scalar accuracy can imply different posterior geometry", fontsize=12, y=1.03)
    fig.tight_layout()
    paths = save_figure(fig, output_dir, "posterior_geometry_pareto", formats, dpi=dpi)
    plt.close(fig)
    return paths, frame


def temperature_sensitivity_frame(groups: list[PosteriorGroup]) -> pd.DataFrame:
    """Return temperature-calibration curves from seed-run calibration JSON files."""
    rows = []
    for group in groups:
        for run in group.runs:
            path = run.run_dir / "calibration" / "temperature.json"
            if not path.exists():
                continue
            calibration = load_json(path)
            for item in calibration.get("results", []):
                rows.append(
                    {
                        "name": group.name,
                        "label": group.label,
                        "display_label": display_label(group.label),
                        "seed": run.seed,
                        "temperature": float(item["temperature"]),
                        "valid_nrmse": float(item["nrmse"]),
                        "valid_rmse": float(item["rmse"]) if item.get("rmse") is not None else np.nan,
                        "best_temperature": float(calibration["best_temperature"]),
                        "best_valid_nrmse": float(calibration["best_nrmse"]),
                        "curve_type": "seed",
                    }
                )
    if not rows:
        return pd.DataFrame()

    seed_frame = pd.DataFrame(rows)
    aggregate_rows = []
    for (name, label, display, temperature), frame in seed_frame.groupby(
        ["name", "label", "display_label", "temperature"],
        sort=False,
    ):
        aggregate_rows.append(
            {
                "name": name,
                "label": label,
                "display_label": display,
                "seed": "mean",
                "temperature": float(temperature),
                "valid_nrmse": float(frame["valid_nrmse"].mean()),
                "valid_nrmse_std": float(frame["valid_nrmse"].std(ddof=1)) if len(frame) > 1 else 0.0,
                "valid_rmse": float(frame["valid_rmse"].mean()),
                "valid_rmse_std": float(frame["valid_rmse"].std(ddof=1)) if len(frame) > 1 else 0.0,
                "best_temperature": float(frame["best_temperature"].mean()),
                "best_temperature_std": float(frame["best_temperature"].std(ddof=1)) if len(frame) > 1 else 0.0,
                "best_valid_nrmse": float(frame["best_valid_nrmse"].mean()),
                "best_valid_nrmse_std": float(frame["best_valid_nrmse"].std(ddof=1)) if len(frame) > 1 else 0.0,
                "curve_type": "group_mean",
            }
        )
    return pd.concat([seed_frame, pd.DataFrame(aggregate_rows)], ignore_index=True)


def plot_temperature_sensitivity(
    groups: list[PosteriorGroup],
    output_dir: Path,
    *,
    formats: list[str],
    dpi: int,
) -> tuple[list[Path], pd.DataFrame]:
    """Save temperature-sensitivity curves aggregated across seeds."""
    frame = temperature_sensitivity_frame(groups)
    if frame.empty:
        return [], frame

    fig, ax = plt.subplots(figsize=(7.8, 4.7))
    mean_frames = []
    for group in groups:
        group_frame = frame[
            (frame["label"] == group.label)
            & (frame["curve_type"] == "group_mean")
        ].sort_values("temperature")
        if group_frame.empty:
            continue
        mean_frames.append(group_frame)
        temperature = group_frame["temperature"].to_numpy(dtype=np.float64)
        mean = group_frame["valid_nrmse"].to_numpy(dtype=np.float64)
        std = group_frame["valid_nrmse_std"].to_numpy(dtype=np.float64)
        ax.fill_between(temperature, mean - std, mean + std, color=group.color, alpha=0.12, linewidth=0)
        ax.plot(temperature, mean, color=group.color, linewidth=1.35, label=display_label(group.label))
        best_tau = float(group_frame["best_temperature"].iloc[0])
        ax.axvline(best_tau, color=group.color, linewidth=0.9, alpha=0.22)

    ax.set_xlabel("Softmax temperature")
    ax.set_ylabel("Development nRMSE")
    ax.set_title("Temperature sensitivity of event-time readout")
    if mean_frames:
        means = np.concatenate([item["valid_nrmse"].to_numpy(dtype=np.float64) for item in mean_frames])
        best_values = np.array([item["best_valid_nrmse"].iloc[0] for item in mean_frames], dtype=np.float64)
        y_low = max(0.0, float(np.nanmin(best_values)) - 0.025)
        y_high = float(np.nanmax(best_values)) + 0.16
        if np.nanmax(means) > y_high:
            ax.set_ylim(y_low, y_high)
            ax.text(
                0.99,
                0.04,
                f"curves clipped above {y_high:.2f}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=7.5,
                color="#555555",
            )
    ax.legend(frameon=False, fontsize=8, ncol=2)
    style_axes(ax)
    ax.grid(True, axis="x", color="#e8e8e8", linewidth=0.8)
    fig.tight_layout()
    paths = save_figure(fig, output_dir, "temperature_sensitivity", formats, dpi=dpi)
    plt.close(fig)
    return paths, frame


def common_representative_mask(groups: list[PosteriorGroup]) -> np.ndarray:
    """Return common mask over representative seed-runs."""
    mask = groups[0].representative.mask.copy()
    for group in groups[1:]:
        mask &= group.representative.mask
    return mask


def choose_example_indices(groups: list[PosteriorGroup], max_examples: int) -> list[tuple[str, int]]:
    """Choose deterministic representative trials from representative seed-runs."""
    if not groups or max_examples <= 0:
        return []
    mask = common_representative_mask(groups)
    valid = np.where(mask)[0]
    if valid.size == 0:
        return []

    reps = [group.representative for group in groups]
    predictions = np.vstack([run.predictions for run in reps])
    widths = np.vstack([run.width80 for run in reps])
    gaps = np.vstack([run.mode_mean_gap for run in reps])
    errors = np.vstack([run.abs_error for run in reps])

    choices: list[tuple[str, int]] = []
    pred_spread = np.ptp(predictions[:, valid], axis=0)
    width_spread = np.ptp(widths[:, valid], axis=0)
    good_scalar = np.nanmedian(errors[:, valid], axis=0) < 0.18
    score = np.where(good_scalar, width_spread - pred_spread, width_spread - pred_spread - 1.0)
    choices.append(("similar scalar, different width", int(valid[np.argmax(score)])))

    worst_gap_run = int(np.argmax(np.nanmedian(gaps[:, valid], axis=1)))
    candidates = valid[errors[worst_gap_run, valid] < 0.18]
    if candidates.size == 0:
        candidates = valid
    choices.append(("large mode-mean gap", int(candidates[np.argmax(gaps[worst_gap_run, candidates])])))
    choices.append(("largest model disagreement", int(valid[np.argmax(pred_spread)])))

    unique = []
    used = set()
    for label, index in choices:
        if index in used:
            continue
        unique.append((label, index))
        used.add(index)
        if len(unique) >= max_examples:
            break
    return unique


def plot_representative_posteriors(
    groups: list[PosteriorGroup],
    output_dir: Path,
    *,
    formats: list[str],
    dpi: int,
    max_examples: int,
) -> tuple[list[Path], pd.DataFrame | None]:
    """Save representative posterior overlays from representative seed-runs."""
    examples = choose_example_indices(groups, max_examples)
    if not examples:
        return [], None
    reps = [group.representative for group in groups]
    fig, axes = plt.subplots(len(examples), 1, figsize=(8.8, 2.55 * len(examples)), sharex=True)
    if len(examples) == 1:
        axes = [axes]
    rows = []
    dt = float(np.median(np.diff(reps[0].grid)))
    for ax, (example_label, index) in zip(axes, examples, strict=True):
        target = float(reps[0].targets[index])
        row: dict[str, object] = {"example": example_label, "row_index": int(index), "target": target}
        if "row_id" in reps[0].metadata:
            row["row_id"] = reps[0].metadata.loc[index, "row_id"]
        if "subject" in reps[0].metadata:
            row["subject"] = reps[0].metadata.loc[index, "subject"]
        for group, run in zip(groups, reps, strict=True):
            ax.plot(run.grid, run.probabilities[index] / dt, color=group.color, linewidth=1.7, label=display_label(group.label))
            ax.axvline(run.predictions[index], color=group.color, linestyle=":", linewidth=1.0, alpha=0.9)
            row[f"{group.name}_seed"] = run.seed
            row[f"{group.name}_prediction"] = float(run.predictions[index])
            row[f"{group.name}_width80_ms"] = float(run.width80[index] * 1000.0)
            row[f"{group.name}_mode_mean_gap_ms"] = float(run.mode_mean_gap[index] * 1000.0)
        ax.axvline(target, color="#111111", linestyle="--", linewidth=1.2, label="target")
        ax.set_ylabel("Posterior density")
        ax.set_title(f"{example_label}: target={target:.3f}s")
        style_axes(ax)
        rows.append(row)
    axes[-1].set_xlabel("Time from stimulus onset (s)")
    axes[0].legend(frameon=False, fontsize=8, ncol=min(len(groups) + 1, 4))
    fig.tight_layout()
    paths = save_figure(fig, output_dir, "representative_posteriors", formats, dpi=dpi)
    plt.close(fig)
    return paths, pd.DataFrame(rows)


def binned_representative_posteriors(
    groups: list[PosteriorGroup],
    *,
    max_bins: int,
) -> tuple[pd.DataFrame, list[np.ndarray], np.ndarray]:
    """Return RT-sorted binned posterior densities for representative seed-runs."""
    base = groups[0].representative
    mask = common_representative_mask(groups)
    sorted_idx = np.where(mask)[0][np.argsort(base.targets[mask])]
    if sorted_idx.size == 0:
        raise RuntimeError("No representative trials are available for the posterior raster.")
    n_bins = sorted_idx.size if max_bins <= 0 else min(int(max_bins), sorted_idx.size)
    chunks = [chunk for chunk in np.array_split(sorted_idx, n_bins) if chunk.size]
    display_rows = np.arange(len(chunks), dtype=np.float64) + 0.5
    rows = []
    for display_row, chunk in zip(display_rows, chunks, strict=True):
        rows.append(
            {
                "display_row": int(display_row - 0.5),
                "row_start": int(chunk[0]),
                "row_end": int(chunk[-1]),
                "n_trials": int(chunk.size),
                "target_mean": float(np.mean(base.targets[chunk])),
                "target_median": float(np.median(base.targets[chunk])),
                "target_min": float(np.min(base.targets[chunk])),
                "target_max": float(np.max(base.targets[chunk])),
                "n_representable_trials": int(sorted_idx.size),
                "n_displayed_bins": int(len(chunks)),
            }
        )
    densities = []
    for group in groups:
        run = group.representative
        dt = float(np.median(np.diff(run.grid)))
        densities.append(np.vstack([np.mean(run.probabilities[chunk], axis=0) / dt for chunk in chunks]))
    return pd.DataFrame(rows), densities, display_rows


def plot_trial_raster(
    groups: list[PosteriorGroup],
    output_dir: Path,
    *,
    formats: list[str],
    dpi: int,
    raster_bins: int,
) -> tuple[list[Path], pd.DataFrame]:
    """Save trial-sorted posterior raster panels using representative seeds."""
    raster_frame, densities, display_rows = binned_representative_posteriors(groups, max_bins=raster_bins)
    targets = raster_frame["target_mean"].to_numpy(dtype=np.float64)
    transformed = [np.log1p(values) for values in densities]
    vmax = float(np.quantile(np.concatenate([values.ravel() for values in transformed]), 0.995))
    vmax = max(vmax, 1e-6)

    fig = plt.figure(figsize=(7.2, 0.86 * len(groups) + 0.70))
    grid_spec = fig.add_gridspec(
        len(groups),
        2,
        width_ratios=[1.0, 0.026],
        left=0.055,
        right=0.955,
        bottom=0.10,
        top=0.975,
        hspace=0.10,
        wspace=0.035,
    )
    axes = [fig.add_subplot(grid_spec[i, 0]) for i in range(len(groups))]
    color_axis = fig.add_subplot(grid_spec[:, 1])
    image = None
    for index, (ax, group, values) in enumerate(zip(axes, groups, transformed, strict=True)):
        run = group.representative
        image = ax.imshow(
            values,
            aspect="auto",
            origin="lower",
            extent=(run.grid[0], run.grid[-1], 0, len(raster_frame)),
            cmap="magma",
            vmin=0.0,
            vmax=vmax,
            interpolation="nearest",
        )
        line = ax.plot(targets, display_rows, color="#111111", linewidth=1.15, label="Observed RT")[0]
        line.set_path_effects([pe.Stroke(linewidth=2.5, foreground="white"), pe.Normal()])
        ax.text(
            0.012,
            0.92,
            display_label(group.label).replace("\n", " "),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.2,
            fontweight="semibold",
            color="white",
        )
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.grid(False)
        if index < len(axes) - 1:
            ax.tick_params(axis="x", labelbottom=False)
    axes[-1].set_xlabel("Time from stimulus onset (s)")
    if image is not None:
        color_bar = fig.colorbar(image, cax=color_axis)
        color_bar.set_label("log(1 + density)")
        color_bar.ax.tick_params(labelsize=8)
    paths = save_figure(fig, output_dir, "trial_sorted_posterior_raster", formats, dpi=dpi)
    plt.close(fig)
    return paths, raster_frame


def performance_frame(rows: list[PerformanceRow]) -> pd.DataFrame:
    """Return scalar-performance rows as a dataframe."""
    return pd.DataFrame(
        [
            {
                "label": row.label,
                "group": row.group,
                "nrmse_mean": row.nrmse_mean,
                "nrmse_std": row.nrmse_std,
                "n_runs": row.n_runs,
                "source": row.source,
                "color": row.color,
            }
            for row in rows
        ]
    )


def plot_performance_forest(
    rows: list[PerformanceRow],
    output_dir: Path,
    *,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    """Save horizontal R11 nRMSE forest plot."""
    if not rows:
        return []
    fig, ax = plt.subplots(figsize=(7.9, max(3.0, 0.43 * len(rows) + 1.15)))
    y = np.arange(len(rows), dtype=np.float64)
    values = np.array([row.nrmse_mean for row in rows], dtype=np.float64)
    errors = np.array([0.0 if row.nrmse_std is None else row.nrmse_std for row in rows], dtype=np.float64)
    for index, row in enumerate(rows):
        ax.errorbar(
            row.nrmse_mean,
            y[index],
            xerr=errors[index],
            fmt="o",
            color=row.color,
            ecolor="#222222",
            elinewidth=1.1,
            capsize=3,
            markersize=6,
            zorder=3,
        )
    direct = next((row for row in rows if row.group == "Scalar regression"), None)
    if direct is not None:
        ax.axvline(direct.nrmse_mean, color="#777777", linestyle="--", linewidth=1.0, alpha=0.75)
    ax.set_yticks(y, [display_label(row.label) for row in rows])
    ax.invert_yaxis()
    ax.set_xlabel("R11 nRMSE (lower is better)")
    ax.set_title("R11 scalar readout across seeds")
    x_low = float(np.nanmin(values - errors)) - 0.010
    data_high = float(np.nanmax(values + errors))
    x_high = data_high + 0.006
    ax.set_xlim(x_low, x_high)
    text_transform = ax.get_yaxis_transform()
    for index, row in enumerate(rows):
        err = 0.0 if row.nrmse_std is None else row.nrmse_std
        ax.text(
            1.03,
            y[index],
            f"{row.nrmse_mean:.3f} +/- {err:.3f}",
            transform=text_transform,
            va="center",
            ha="left",
            fontsize=7.5,
            clip_on=False,
        )
    style_axes(ax)
    ax.grid(True, axis="x", color="#e8e8e8", linewidth=0.8)
    ax.grid(False, axis="y")
    fig.tight_layout(rect=(0.0, 0.0, 0.84, 1.0))
    paths = save_figure(fig, output_dir, "r11_performance_forest", formats, dpi=dpi)
    plt.close(fig)
    return paths


def plot_window_support(
    groups: list[PosteriorGroup],
    output_dir: Path,
    *,
    formats: list[str],
    dpi: int,
) -> tuple[list[Path], pd.DataFrame]:
    """Save target distribution relative to representative event-time support."""
    run = groups[0].representative
    targets = run.targets
    grid_start = float(run.grid[0])
    grid_end = float(run.grid[-1])
    inside = (targets >= grid_start) & (targets <= grid_end)
    below = targets < grid_start
    above = targets > grid_end

    fig, ax = plt.subplots(figsize=(7.2, 3.45))
    bins = np.linspace(float(np.nanmin(targets)), max(float(np.nanmax(targets)), grid_end), 60)
    ax.hist(targets, bins=bins, color="#bdbdbd", edgecolor="white", linewidth=0.35, label="All R11 targets")
    ax.hist(targets[~inside], bins=bins, color="#d62728", alpha=0.85, edgecolor="white", linewidth=0.35, label="Outside support")
    ax.axvspan(grid_start, grid_end, color="#1f77b4", alpha=0.12, label="Modeled event-time support")
    ax.axvline(grid_start, color="#1f77b4", linestyle="--", linewidth=1.1)
    ax.axvline(grid_end, color="#1f77b4", linestyle="--", linewidth=1.1)
    note = (
        f"inside support: {int(inside.sum()):,}/{len(targets):,}\n"
        f"outside: {int((~inside).sum()):,} "
        f"(<{grid_start:.2f}s: {int(below.sum()):,}, >{grid_end:.2f}s: {int(above.sum()):,})"
    )
    ax.text(
        0.985,
        0.93,
        note,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.95},
    )
    ax.set_xlabel("Observed RT from stimulus onset (s)")
    ax.set_ylabel("R11 trials")
    ax.set_title("Observed R11 RT support for posterior-geometry analysis")
    ax.legend(frameon=False, loc="upper left")
    style_axes(ax)
    fig.tight_layout()
    paths = save_figure(fig, output_dir, "window_support_diagnostic", formats, dpi=dpi)
    plt.close(fig)

    frame = pd.DataFrame(
        [
            {
                "rows_total": int(len(targets)),
                "rows_inside_support": int(inside.sum()),
                "rows_outside_support": int((~inside).sum()),
                "rows_below_support": int(below.sum()),
                "rows_above_support": int(above.sum()),
                "support_start_s": grid_start,
                "support_end_s": grid_end,
                "target_min_s": float(np.min(targets)),
                "target_max_s": float(np.max(targets)),
            }
        ]
    )
    return paths, frame


def run_cli(args: argparse.Namespace) -> None:
    """Run the paper-figure generation pipeline."""
    segmentation_dir = resolve_required(args.segmentation_dir)
    regression_dir = resolve_required(args.regression_dir)
    output_dir = resolve_required(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = load_posterior_groups(
        segmentation_dir,
        split=args.split,
        readout=args.readout,
        temperature_override=args.temperature,
        target_filter=args.target_filter,
        near_ms=args.near_ms,
        score_sigma=args.score_sigma,
        coverage_levels=args.coverage_levels,
        align_window_ms=args.align_window_ms,
        include=args.include,
        exclude=args.exclude,
    )
    if not groups:
        raise RuntimeError(f"No segmentation posterior groups found in {segmentation_dir}.")

    print("\n=== Paper figure generation ===")
    print(f"Segmentation directory: {path_text(segmentation_dir)}")
    print(f"Regression directory: {path_text(regression_dir)}")
    print(f"Output directory: {path_text(output_dir)}")
    print(f"Readout: {args.readout}")
    print(f"Groups: {len(groups)}")
    for group in groups:
        seeds = ", ".join(str(run.seed) for run in group.runs)
        print(f"- {group.label}: {len(group.runs)} seed-run(s) [{seeds}]")

    written: list[Path] = []
    data_paths: list[Path] = []

    seed_summary = per_seed_summary_frame(groups, near_ms=args.near_ms)
    group_summary = group_summary_frame(seed_summary)
    coverage = coverage_summary_frame(groups)
    aligned = aligned_posterior_frame(groups)

    seed_summary_path = csv_path(output_dir, "posterior_geometry_seed_summary.csv")
    group_summary_path = csv_path(output_dir, "posterior_geometry_group_summary.csv")
    coverage_path = csv_path(output_dir, "posterior_coverage.csv")
    aligned_path = csv_path(output_dir, "target_aligned_posterior.csv")
    palette_path = csv_path(output_dir, "posterior_color_palette.csv")
    representatives_path = csv_path(output_dir, "representative_seeds.csv")
    seed_summary.to_csv(seed_summary_path, index=False)
    group_summary.to_csv(group_summary_path, index=False)
    coverage.to_csv(coverage_path, index=False)
    aligned.to_csv(aligned_path, index=False)
    palette_frame(groups).to_csv(palette_path, index=False)
    representative_seed_frame(groups).to_csv(representatives_path, index=False)
    data_paths.extend([
        seed_summary_path,
        group_summary_path,
        coverage_path,
        aligned_path,
        palette_path,
        representatives_path,
    ])

    if selected(args.figures, "posterior_geometry"):
        written.extend(
            plot_posterior_geometry(
                groups,
                seed_summary,
                coverage,
                output_dir,
                formats=args.formats,
                dpi=args.dpi,
                near_ms=args.near_ms,
            )
        )

    if selected(args.figures, "posterior_pareto"):
        pareto_paths, pareto = plot_posterior_pareto(
            groups,
            seed_summary,
            output_dir,
            formats=args.formats,
            dpi=args.dpi,
            near_ms=args.near_ms,
        )
        written.extend(pareto_paths)
        pareto_path = csv_path(output_dir, "posterior_geometry_pareto.csv")
        pareto.to_csv(pareto_path, index=False)
        data_paths.append(pareto_path)

    if selected(args.figures, "representative_posteriors"):
        example_paths, examples = plot_representative_posteriors(
            groups,
            output_dir,
            formats=args.formats,
            dpi=args.dpi,
            max_examples=args.max_examples,
        )
        written.extend(example_paths)
        if examples is not None:
            examples_path = csv_path(output_dir, "representative_posteriors.csv")
            examples.to_csv(examples_path, index=False)
            data_paths.append(examples_path)

    if selected(args.figures, "trial_raster"):
        raster_paths, raster_index = plot_trial_raster(
            groups,
            output_dir,
            formats=args.formats,
            dpi=args.dpi,
            raster_bins=args.raster_bins,
        )
        written.extend(raster_paths)
        raster_index_path = csv_path(output_dir, "trial_sorted_posterior_raster_index.csv")
        raster_index.to_csv(raster_index_path, index=False)
        data_paths.append(raster_index_path)

    if selected(args.figures, "performance_forest"):
        rows: list[PerformanceRow] = []
        best_regression = best_regression_row(regression_dir)
        if best_regression is not None:
            rows.append(best_regression)
        rows.extend(segmentation_performance_rows(groups))
        performance = performance_frame(rows)
        performance_path = csv_path(output_dir, "r11_performance_forest.csv")
        performance.to_csv(performance_path, index=False)
        data_paths.append(performance_path)
        written.extend(plot_performance_forest(rows, output_dir, formats=args.formats, dpi=args.dpi))

    if selected(args.figures, "window_support"):
        support_paths, support = plot_window_support(groups, output_dir, formats=args.formats, dpi=args.dpi)
        written.extend(support_paths)
        support_path = csv_path(output_dir, "window_support_diagnostic.csv")
        support.to_csv(support_path, index=False)
        data_paths.append(support_path)

    if selected(args.figures, "temperature_sensitivity"):
        temperature_paths, temperature = plot_temperature_sensitivity(
            groups,
            output_dir,
            formats=args.formats,
            dpi=args.dpi,
        )
        written.extend(temperature_paths)
        if not temperature.empty:
            temperature_path = csv_path(output_dir, "temperature_sensitivity.csv")
            temperature.to_csv(temperature_path, index=False)
            data_paths.append(temperature_path)

    print("\nSaved data")
    for path in data_paths:
        print(f"- {path_text(path)}")
    print("\nSaved figures")
    for path in written:
        print(f"- {path_text(path)}")


def main(argv: list[str] | None = None) -> int:
    """Run CLI."""
    args = build_parser().parse_args(argv)
    run_cli(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
