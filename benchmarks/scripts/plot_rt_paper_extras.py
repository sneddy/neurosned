"""Generate additional paper-facing RT event-time visualizations."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

from benchmarks.pkg.config import resolve_path
from benchmarks.scripts.plot_segmentation_posteriors import (
    RunPosterior,
    color_for,
    compute_aligned_posterior,
    find_run_dirs,
    format_float,
    label_for,
    load_json,
    load_run_posterior,
    load_snapshot,
    order_key,
    path_text,
    run_name_from_snapshot,
    save_figure,
    style_axes,
    write_text,
)


@dataclass
class PerformanceRow:
    """One R11 performance entry for the forest plot."""

    label: str
    group: str
    nrmse: float
    ci_low: float | None
    ci_high: float | None
    source: str
    color: str


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for paper extra figures."""
    parser = argparse.ArgumentParser(
        description="Create extra paper-ready RT figures without changing existing posterior-geometry figures."
    )
    parser.add_argument(
        "experiment_dir",
        type=Path,
        help="Segmentation ablation directory containing run subdirectories.",
    )
    parser.add_argument(
        "--regression-dir",
        type=Path,
        default=Path("benchmarks/experiments/01_regression_baselines"),
        help="Regression baseline directory used to auto-select the strongest R11 direct-regression baseline.",
    )
    parser.add_argument("--split", default="test", help="Evaluation split prefix to read, usually 'test'.")
    parser.add_argument(
        "--readout",
        choices=("calibrated", "base"),
        default="calibrated",
        help="Readout used for segmentation posterior probabilities.",
    )
    parser.add_argument("--temperature", type=float, default=None, help="Override temperature for every segmentation run.")
    parser.add_argument(
        "--target-filter",
        choices=("representable", "all"),
        default="representable",
        help="Rows used for posterior raster summaries.",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Only include segmentation run directories whose name contains this text. Can be passed multiple times.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Exclude segmentation run directories whose name contains this text. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <experiment_dir>/figures/posterior_geometry_<readout>.",
    )
    parser.add_argument("--formats", nargs="+", default=["png"], choices=("png", "svg"), help="Figure formats.")
    parser.add_argument("--dpi", type=int, default=300, help="Raster figure DPI.")
    parser.add_argument(
        "--raster-max-trials",
        type=int,
        default=None,
        help="Deprecated alias for --raster-bins.",
    )
    parser.add_argument(
        "--raster-bins",
        type=int,
        default=260,
        help="Number of RT-sorted quantile bins displayed in the posterior raster.",
    )
    parser.add_argument(
        "--extra-performance",
        action="append",
        nargs=4,
        metavar=("LABEL", "NRMSE", "CI_LOW", "CI_HIGH"),
        default=[],
        help="Optional extra forest-plot row, e.g. a final R11 stack: LABEL NRMSE CI_LOW CI_HIGH.",
    )
    parser.add_argument(
        "--skip-captions",
        action="store_true",
        help="Do not write markdown caption drafts.",
    )
    return parser


def resolve_required(path: Path) -> Path:
    """Resolve a path relative to the project root."""
    resolved = resolve_path(path, PROJECT_ROOT)
    if resolved is None:
        raise ValueError("Path cannot be None.")
    return resolved.resolve()


def load_runs(args: argparse.Namespace) -> list[RunPosterior]:
    """Load segmentation posterior runs in stable ablation order."""
    experiment_dir = resolve_required(args.experiment_dir)
    run_dirs = find_run_dirs(experiment_dir, include=args.include, exclude=args.exclude, split=args.split)
    if not run_dirs:
        raise RuntimeError(f"No segmentation runs with saved {args.split} logits found in {experiment_dir}.")
    snapshots = {run_dir: load_snapshot(run_dir) for run_dir in run_dirs}
    run_dirs = sorted(
        run_dirs,
        key=lambda path: order_key(path, run_name_from_snapshot(path, snapshots.get(path, {}))),
    )

    runs = []
    for run_dir in run_dirs:
        run = load_run_posterior(
            run_dir,
            split=args.split,
            readout=args.readout,
            temperature_override=args.temperature,
            target_filter=args.target_filter,
            near_ms=150.0,
            coverage_levels=[0.50, 0.60, 0.70, 0.80, 0.90],
        )
        compute_aligned_posterior(run, align_window_ms=1000.0)
        runs.append(run)
    return runs


def output_dir_for(args: argparse.Namespace) -> Path:
    """Return output directory for extra figures."""
    if args.output_dir is not None:
        output_dir = resolve_required(args.output_dir)
    else:
        output_dir = resolve_required(args.experiment_dir) / "figures" / f"posterior_geometry_{args.readout}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def common_mask(runs: list[RunPosterior]) -> np.ndarray:
    """Return a common analysis mask across segmentation runs."""
    if not runs:
        raise ValueError("At least one run is required.")
    mask = runs[0].mask.copy()
    for run in runs[1:]:
        mask &= run.mask
    return mask


def sorted_trial_indices(runs: list[RunPosterior]) -> np.ndarray:
    """Return representable trial indices sorted by observed RT."""
    mask = common_mask(runs)
    return np.where(mask)[0][np.argsort(runs[0].targets[mask])]


def binned_sorted_posteriors(
    runs: list[RunPosterior],
    *,
    max_bins: int,
) -> tuple[pd.DataFrame, list[np.ndarray], np.ndarray]:
    """Average posterior densities in RT-sorted quantile bins."""
    sorted_idx = sorted_trial_indices(runs)
    if sorted_idx.size == 0:
        raise RuntimeError("No representable trials are available for the posterior raster.")
    n_bins = sorted_idx.size if max_bins <= 0 else min(int(max_bins), sorted_idx.size)
    chunks = [chunk for chunk in np.array_split(sorted_idx, n_bins) if chunk.size]
    display_rows = np.arange(len(chunks), dtype=np.float64) + 0.5
    targets = runs[0].targets

    bin_rows = []
    for display_row, chunk in zip(display_rows, chunks, strict=True):
        bin_rows.append(
            {
                "display_row": int(display_row - 0.5),
                "row_start": int(chunk[0]),
                "row_end": int(chunk[-1]),
                "n_trials": int(chunk.size),
                "target_mean": float(np.mean(targets[chunk])),
                "target_median": float(np.median(targets[chunk])),
                "target_min": float(np.min(targets[chunk])),
                "target_max": float(np.max(targets[chunk])),
                "n_representable_trials": int(sorted_idx.size),
                "n_displayed_bins": int(len(chunks)),
            }
        )

    dt = float(np.median(np.diff(runs[0].grid)))
    densities = []
    for run in runs:
        densities.append(np.vstack([np.mean(run.probabilities[chunk], axis=0) / dt for chunk in chunks]))
    return pd.DataFrame(bin_rows), densities, display_rows


def plot_trial_sorted_raster(
    runs: list[RunPosterior],
    output_dir: Path,
    *,
    formats: list[str],
    dpi: int,
    max_bins: int,
) -> tuple[list[Path], pd.DataFrame]:
    """Save trial-sorted posterior raster panels."""
    raster_frame, densities, display_rows = binned_sorted_posteriors(runs, max_bins=max_bins)
    targets = raster_frame["target_mean"].to_numpy(dtype=np.float64)
    transformed = [np.log1p(density) for density in densities]
    vmax = float(np.quantile(np.concatenate([arr.ravel() for arr in transformed]), 0.995))
    vmax = max(vmax, 1e-6)

    fig = plt.figure(figsize=(8.8, 1.30 * len(runs) + 1.10))
    grid_spec = fig.add_gridspec(
        len(runs),
        2,
        width_ratios=[1.0, 0.032],
        left=0.16,
        right=0.91,
        bottom=0.08,
        top=0.92,
        hspace=0.13,
        wspace=0.06,
    )
    axes = [fig.add_subplot(grid_spec[i, 0]) for i in range(len(runs))]
    color_axis = fig.add_subplot(grid_spec[:, 1])
    image = None
    for index, (ax, run, values) in enumerate(zip(axes, runs, transformed, strict=True)):
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
        ax.set_ylabel(run.label, rotation=0, ha="right", va="center", labelpad=32, color=color_for(run, index))
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.grid(False)
        if index < len(axes) - 1:
            ax.tick_params(axis="x", labelbottom=False)
    axes[-1].set_xlabel("Time from stimulus onset (s)")
    fig.suptitle("R11 event-time posteriors sorted by observed RT", y=0.985)
    fig.text(0.025, 0.5, "RT-sorted quantile bins", rotation=90, va="center", ha="center")
    if image is not None:
        cbar = fig.colorbar(image, cax=color_axis)
        cbar.set_label("log(1 + posterior density)")
        cbar.ax.tick_params(labelsize=8)
    paths = save_figure(fig, output_dir, "trial_sorted_posterior_raster", formats, dpi=dpi)

    return paths, raster_frame


def metric_from_json(path: Path) -> tuple[float, float | None, float | None]:
    """Read nRMSE and optional CI from a metrics JSON file."""
    payload = load_json(path)
    metrics = payload.get("metrics", {})
    ci = payload.get("confidence_interval", {}) or {}
    return (
        float(metrics["nrmse"]),
        float(ci["nrmse_ci_low"]) if ci.get("nrmse_ci_low") is not None else None,
        float(ci["nrmse_ci_high"]) if ci.get("nrmse_ci_high") is not None else None,
    )


def compact_regression_label(run_dir: Path) -> str:
    """Return a compact display label for a regression baseline."""
    name = run_dir.name.split("__")[0]
    labels = {
        "sneddy_rt_net": "Best direct regression\nSneddyRTNet",
        "sneddy_net": "Best direct regression\nSneddyNet",
        "sneddy_rt_net_larger": "Best direct regression\nSneddyRTNet-L",
        "sneddy_net_larger": "Best direct regression\nSneddyNet-L",
    }
    return labels.get(name, f"Best direct regression\n{name.replace('_', '-')}")


def best_regression_row(regression_dir: Path) -> PerformanceRow | None:
    """Return the best R11 direct-regression row available on disk."""
    if not regression_dir.exists():
        return None
    candidates = []
    for path in sorted(regression_dir.glob("*/test_metrics.json")):
        try:
            nrmse_value, ci_low, ci_high = metric_from_json(path)
        except (KeyError, ValueError, json.JSONDecodeError):
            continue
        candidates.append((nrmse_value, path.parent, ci_low, ci_high))
    if not candidates:
        return None
    nrmse_value, run_dir, ci_low, ci_high = min(candidates, key=lambda item: item[0])
    return PerformanceRow(
        label=compact_regression_label(run_dir),
        group="Direct",
        nrmse=nrmse_value,
        ci_low=ci_low,
        ci_high=ci_high,
        source=path_text(run_dir),
        color="#555555",
    )


def segmentation_rows(runs: list[RunPosterior]) -> list[PerformanceRow]:
    """Return forest rows for segmentation ablations."""
    rows = []
    for index, run in enumerate(runs):
        rows.append(
            PerformanceRow(
                label=run.label,
                group="Event-time",
                nrmse=run.nrmse_value,
                ci_low=run.ci_low,
                ci_high=run.ci_high,
                source=path_text(run.run_dir),
                color=color_for(run, index),
            )
        )
    return rows


def parse_extra_rows(extra: list[list[str]]) -> list[PerformanceRow]:
    """Parse optional user-supplied performance rows."""
    rows = []
    for label, nrmse_value, ci_low, ci_high in extra:
        low = None if ci_low.lower() in {"none", "nan", "na"} else float(ci_low)
        high = None if ci_high.lower() in {"none", "nan", "na"} else float(ci_high)
        rows.append(
            PerformanceRow(
                label=label,
                group="Extra",
                nrmse=float(nrmse_value),
                ci_low=low,
                ci_high=high,
                source="user-supplied",
                color="#111111",
            )
        )
    return rows


def performance_frame(rows: list[PerformanceRow]) -> pd.DataFrame:
    """Return performance rows as a dataframe."""
    return pd.DataFrame(
        [
            {
                "label": row.label.replace("\n", " "),
                "group": row.group,
                "nrmse": row.nrmse,
                "nrmse_ci_low": row.ci_low,
                "nrmse_ci_high": row.ci_high,
                "source": row.source,
                "color": row.color,
            }
            for row in rows
        ]
    )


def plot_r11_performance_forest(
    rows: list[PerformanceRow],
    output_dir: Path,
    *,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    """Save horizontal R11 nRMSE forest plot."""
    if not rows:
        return []
    fig, ax = plt.subplots(figsize=(7.7, max(3.0, 0.43 * len(rows) + 1.15)))
    y = np.arange(len(rows))
    values = np.array([row.nrmse for row in rows], dtype=np.float64)
    lower = np.array([row.nrmse - row.ci_low if row.ci_low is not None else 0.0 for row in rows])
    upper = np.array([row.ci_high - row.nrmse if row.ci_high is not None else 0.0 for row in rows])
    for i, row in enumerate(rows):
        ax.errorbar(
            row.nrmse,
            y[i],
            xerr=np.array([[lower[i]], [upper[i]]]),
            fmt="o",
            color=row.color,
            ecolor="#222222",
            elinewidth=1.1,
            capsize=3,
            markersize=6,
            zorder=3,
        )
    direct = next((row for row in rows if row.group == "Direct"), None)
    if direct is not None:
        ax.axvline(direct.nrmse, color="#777777", linestyle="--", linewidth=1.0, alpha=0.75)
    if any(row.group == "Direct" for row in rows) and any(row.group != "Direct" for row in rows):
        ax.axhline(0.5, color="#dddddd", linewidth=0.8, zorder=0)
    ax.set_yticks(y, [row.label for row in rows])
    ax.invert_yaxis()
    ax.set_xlabel("R11 nRMSE (lower is better)")
    ax.set_title("R11 scalar readout with subject-bootstrap confidence intervals")
    x_low = float(np.nanmin(values - lower)) - 0.01
    data_high = float(np.nanmax(values + upper))
    x_high = data_high + 0.026
    ax.set_xlim(x_low, x_high)
    text_x = data_high + 0.003
    for i, row in enumerate(rows):
        if row.ci_low is None or row.ci_high is None:
            text = f"{row.nrmse:.3f}"
        else:
            text = f"{row.nrmse:.3f} [{row.ci_low:.3f}, {row.ci_high:.3f}]"
        ax.text(text_x, y[i], text, va="center", ha="left", fontsize=7.5, color="#333333")
    style_axes(ax)
    ax.grid(True, axis="x", color="#e8e8e8", linewidth=0.8)
    ax.grid(False, axis="y")
    fig.tight_layout()
    return save_figure(fig, output_dir, "r11_performance_forest", formats, dpi=dpi)


def plot_window_support(
    run: RunPosterior,
    output_dir: Path,
    *,
    formats: list[str],
    dpi: int,
) -> tuple[list[Path], pd.DataFrame]:
    """Save RT target distribution with event-time support shading."""
    targets = run.targets
    grid_start = float(run.grid[0])
    grid_end = float(run.grid[-1])
    inside = (targets >= grid_start) & (targets <= grid_end)
    below = targets < grid_start
    above = targets > grid_end

    fig, ax = plt.subplots(figsize=(7.2, 3.45))
    bins = np.linspace(float(np.nanmin(targets)), max(float(np.nanmax(targets)), grid_end), 60)
    ax.hist(targets, bins=bins, color="#bdbdbd", edgecolor="white", linewidth=0.35, label="All R11 targets")
    ax.hist(targets[below | above], bins=bins, color="#d62728", alpha=0.85, edgecolor="white", linewidth=0.35, label="Outside support")
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


def write_caption_files(
    *,
    output_dir: Path,
    runs: list[RunPosterior],
    raster_index: pd.DataFrame,
    performance: pd.DataFrame,
    support: pd.DataFrame,
    formats: list[str],
) -> list[Path]:
    """Write markdown captions for paper-extra figures."""
    caption_dir = output_dir / "captions"
    support_row = support.iloc[0]
    perf_rows = "\n".join(
        f"- {row['label']}: nRMSE={format_float(row['nrmse'], 3)}, "
        f"CI=[{format_float(row['nrmse_ci_low'], 3)}, {format_float(row['nrmse_ci_high'], 3)}]."
        for _, row in performance.iterrows()
    )
    run_labels = ", ".join(run.label for run in runs)
    files = {
        "trial_sorted_posterior_raster.md": f"""# trial_sorted_posterior_raster

## Draft Caption

Trial-sorted event-time posterior maps on R11 for matched segmentation losses ({run_labels}). Rows are quantile bins of representable R11 trials sorted by observed RT; the x-axis is time from stimulus onset; color shows log-transformed posterior density, `log(1 + p(t|x)/dt)`, averaged within each bin. The black curve marks the mean observed RT in each bin. A coherent event-time localization model should form posterior mass near this curve. The raster displays {len(raster_index):,} RT-sorted bins formed from {int(raster_index['n_representable_trials'].iloc[0]):,} representable R11 trials.

## Interpretation

This figure is qualitative support for the aggregate posterior-geometry panels: scalar RT error can be similar even when the learned temporal evidence map is sharp, diffuse, shifted, or multimodal.
""",
        "r11_performance_forest.md": f"""# r11_performance_forest

## Draft Caption

R11 scalar readout performance with subject-bootstrap confidence intervals. The direct-regression row is automatically selected as the best available R11 direct-regression baseline from `benchmarks/experiments/01_regression_baselines`; event-time rows use the calibrated segmentation readout from saved logits. Lower nRMSE is better.

## Values

{perf_rows}

## Interpretation

This plot anchors the output-geometry analysis: several event-time losses are close in scalar R11 nRMSE, motivating posterior-level visualization rather than relying only on point-prediction error.
""",
        "window_support_diagnostic.md": f"""# window_support_diagnostic

## Draft Caption

Observed R11 reaction-time distribution relative to the event-time support of the segmentation posterior. The modeled posterior grid spans [{support_row['support_start_s']:.2f}, {support_row['support_end_s']:.2f}] s from stimulus onset. Posterior-geometry summaries therefore use {int(support_row['rows_inside_support']):,}/{int(support_row['rows_total']):,} trials whose observed RT lies inside this support; {int(support_row['rows_outside_support']):,} trials are outside the event-time window ({int(support_row['rows_below_support']):,} below, {int(support_row['rows_above_support']):,} above).

## Interpretation

This diagnostic separates scalar R11 evaluation from posterior-geometry analysis. Scalar nRMSE is computed on all R11 trials, while posterior-shape summaries are restricted to targets that are representable on the model's event-time grid.
""",
        "README_paper_extras.md": f"""# Paper Extra Figure Captions

Generated by `benchmarks/scripts/plot_rt_paper_extras.py`.

Figure format(s): {", ".join(formats)}.

Files:

- `trial_sorted_posterior_raster.md`
- `r11_performance_forest.md`
- `window_support_diagnostic.md`
""",
    }
    paths = []
    for name, text in files.items():
        path = caption_dir / name
        write_text(path, text)
        paths.append(path)
    return paths


def run_cli(args: argparse.Namespace) -> None:
    """Run extra paper figure generation."""
    output_dir = output_dir_for(args)
    regression_dir = resolve_required(args.regression_dir)
    runs = load_runs(args)

    print("\n=== RT paper extra figures ===")
    print(f"Segmentation directory: {path_text(resolve_required(args.experiment_dir))}")
    print(f"Regression directory: {path_text(regression_dir)}")
    print(f"Output directory: {path_text(output_dir)}")
    print(f"Runs: {', '.join(run.label for run in runs)}")

    written: list[Path] = []
    data_paths: list[Path] = []

    raster_paths, raster_index = plot_trial_sorted_raster(
        runs,
        output_dir,
        formats=args.formats,
        dpi=args.dpi,
        max_bins=args.raster_max_trials if args.raster_max_trials is not None else args.raster_bins,
    )
    written.extend(raster_paths)
    raster_index_path = output_dir / "trial_sorted_posterior_raster_index.csv"
    raster_index.to_csv(raster_index_path, index=False)
    data_paths.append(raster_index_path)

    rows: list[PerformanceRow] = []
    best_reg = best_regression_row(regression_dir)
    if best_reg is not None:
        rows.append(best_reg)
    rows.extend(segmentation_rows(runs))
    rows.extend(parse_extra_rows(args.extra_performance))
    perf = performance_frame(rows)
    perf_path = output_dir / "r11_performance_forest.csv"
    perf.to_csv(perf_path, index=False)
    data_paths.append(perf_path)
    written.extend(plot_r11_performance_forest(rows, output_dir, formats=args.formats, dpi=args.dpi))

    support_paths, support = plot_window_support(runs[0], output_dir, formats=args.formats, dpi=args.dpi)
    written.extend(support_paths)
    support_path = output_dir / "window_support_diagnostic.csv"
    support.to_csv(support_path, index=False)
    data_paths.append(support_path)

    caption_paths: list[Path] = []
    if not args.skip_captions:
        caption_paths = write_caption_files(
            output_dir=output_dir,
            runs=runs,
            raster_index=raster_index,
            performance=perf,
            support=support,
            formats=args.formats,
        )

    print("\nSaved data")
    for path in data_paths:
        print(f"- {path_text(path)}")
    print("\nSaved figures")
    for path in written:
        print(f"- {path_text(path)}")
    if caption_paths:
        print("\nSaved captions")
        for path in caption_paths:
            print(f"- {path_text(path)}")


def main(argv: list[str] | None = None) -> int:
    """Run CLI."""
    args = build_parser().parse_args(argv)
    run_cli(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
