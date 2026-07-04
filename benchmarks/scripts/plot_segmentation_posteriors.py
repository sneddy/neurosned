"""Plot output-geometry diagnostics for segmentation benchmark runs."""

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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from benchmarks.pkg.config import resolve_path
from benchmarks.pkg.evaluation.metrics import crps_discrete, fixed_kernel_event_nll, nrmse, rmse


ORDER_HINTS = {
    "ce_only": 0,
    "comboloss": 1,
    "combo": 1,
    "event_nll": 2,
    "time_only": 3,
    "wass_only": 4,
    "wasserstein": 4,
}

LABELS = {
    "unet_deeper_ce_only": "CE",
    "ce_only": "CE",
    "unet_deeper_comboloss": "CE+time",
    "comboloss": "CE+time",
    "combo": "CE+time",
    "unet_deeper_event_nll": "EventNLL",
    "event_nll": "EventNLL",
    "unet_deeper_time_only": "Time-only",
    "time_only": "Time-only",
    "unet_deeper_wass_only": "Wasserstein",
    "wass_only": "Wasserstein",
    "wasserstein": "Wasserstein",
}

MODEL_COLORS = {
    "CE": "#1f77b4",
    "CE+time": "#ff7f0e",
    "EventNLL": "#2ca02c",
    "Time-only": "#d62728",
    "Wasserstein": "#9467bd",
}
FALLBACK_COLORS = (
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)


@dataclass
class RunPosterior:
    """Computed posterior geometry for one run."""

    name: str
    label: str
    run_dir: Path
    temperature: float
    grid: np.ndarray
    metadata: pd.DataFrame
    targets: np.ndarray
    mask: np.ndarray
    probabilities: np.ndarray
    predictions: np.ndarray
    modes: np.ndarray
    std: np.ndarray
    entropy: np.ndarray
    width50: np.ndarray
    width80: np.ndarray
    width90: np.ndarray
    mass_near: np.ndarray
    mode_mean_gap: np.ndarray
    abs_error: np.ndarray
    crps: np.ndarray
    fixed_kernel_event_nll: np.ndarray
    fixed_kernel_event_nll_sigma: float
    nrmse_value: float
    rmse_value: float
    ci_low: float | None
    ci_high: float | None
    aligned_x_ms: np.ndarray | None = None
    aligned_density: np.ndarray | None = None
    coverage: pd.DataFrame | None = None


def build_parser() -> argparse.ArgumentParser:
    """Build the posterior-geometry plotting CLI parser."""
    parser = argparse.ArgumentParser(
        description="Plot posterior-geometry diagnostics for segmentation runs in one experiment folder."
    )
    parser.add_argument(
        "experiment_dir",
        type=Path,
        help="Experiment directory containing run subdirectories with predictions/test_logits.npy.",
    )
    parser.add_argument("--split", default="test", help="Evaluation split prefix to read, usually 'test'.")
    parser.add_argument(
        "--readout",
        choices=("calibrated", "base"),
        default="calibrated",
        help="Temperature used to convert logits into posterior probabilities.",
    )
    parser.add_argument("--temperature", type=float, default=None, help="Override temperature for every run.")
    parser.add_argument(
        "--target-filter",
        choices=("representable", "all"),
        default="representable",
        help="Rows used for posterior-geometry summaries.",
    )
    parser.add_argument("--near-ms", type=float, default=150.0, help="Mass-near-target half-window in milliseconds.")
    parser.add_argument(
        "--score-sigma",
        type=float,
        default=0.15,
        help="Gaussian sigma in seconds for fixed-kernel EventNLL scoring.",
    )
    parser.add_argument(
        "--align-window-ms",
        type=float,
        default=1000.0,
        help="Target-aligned posterior plot half-window in milliseconds.",
    )
    parser.add_argument(
        "--coverage-levels",
        type=float,
        nargs="+",
        default=[0.50, 0.60, 0.70, 0.80, 0.90],
        help="Central posterior interval levels for empirical coverage curves.",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Only include run directories whose name contains this text. Can be passed multiple times.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Exclude run directories whose name contains this text. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated figures and CSV files. Defaults to <experiment_dir>/figures/posterior_geometry_<readout>.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        choices=("png", "svg"),
        help="Figure image formats to save.",
    )
    parser.add_argument("--dpi", type=int, default=220, help="Raster figure DPI.")
    parser.add_argument("--max-examples", type=int, default=3, help="Number of representative examples to plot.")
    parser.add_argument(
        "--save-per-run",
        action="store_true",
        help="Also save a compact single-run diagnostic figure for each run.",
    )
    parser.add_argument(
        "--save-trial-csv",
        action="store_true",
        help="Also save long-form per-trial posterior metrics. This can be large.",
    )
    parser.add_argument(
        "--skip-captions",
        action="store_true",
        help="Do not write markdown caption drafts next to the generated figures.",
    )
    return parser


def path_text(path: Path | None) -> str:
    """Return a readable project-relative path when possible."""
    if path is None:
        return "None"
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_snapshot(run_dir: Path) -> dict[str, Any]:
    """Load a run config snapshot."""
    path = run_dir / "config.yaml"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_name_from_snapshot(run_dir: Path, snapshot: dict[str, Any]) -> str:
    """Return the configured experiment name for one run."""
    config = snapshot.get("config", {})
    name = config.get("name")
    if name:
        return str(name)
    return run_dir.name.split("__")[0]


def label_for(name: str) -> str:
    """Return a compact paper-facing label."""
    if name in LABELS:
        return LABELS[name]
    simplified = name.replace("unet_deeper_", "")
    return LABELS.get(simplified, simplified.replace("_", " "))


def color_for(run: "RunPosterior", index: int = 0) -> str:
    """Return the fixed display color for one run."""
    if run.label in MODEL_COLORS:
        return MODEL_COLORS[run.label]
    return FALLBACK_COLORS[index % len(FALLBACK_COLORS)]


def color_map_for(runs: list["RunPosterior"]) -> dict[str, str]:
    """Return a stable run-name to display-color mapping."""
    return {run.name: color_for(run, index) for index, run in enumerate(runs)}


def order_key(run_dir: Path, name: str) -> tuple[int, str]:
    """Sort common ablations in a stable paper-facing order."""
    text = f"{name} {run_dir.name}".lower()
    for key, value in ORDER_HINTS.items():
        if key in text:
            return value, name
    return 100, name


def find_run_dirs(experiment_dir: Path, *, include: list[str], exclude: list[str], split: str) -> list[Path]:
    """Find run directories with saved logits and predictions."""
    logits_name = f"{split}_logits.npy"
    pred_name = f"{split}_predictions.csv"
    candidates = []
    for path in sorted(experiment_dir.iterdir()):
        if not path.is_dir():
            continue
        text = path.name
        if include and not any(token in text for token in include):
            continue
        if exclude and any(token in text for token in exclude):
            continue
        if (path / "predictions" / logits_name).exists() and (path / "predictions" / pred_name).exists():
            candidates.append(path)
    return candidates


def configured_eval_temperature(snapshot: dict[str, Any]) -> float:
    """Return the configured eval temperature from a run snapshot."""
    params = snapshot.get("config", {}).get("trainer", {}).get("params", {})
    return float(params.get("eval_temperature", params.get("temperature", 1.0)))


def infer_grid(snapshot: dict[str, Any], calibration: dict[str, Any] | None, logits: np.ndarray) -> np.ndarray:
    """Infer the absolute time grid used by the segmentation readout."""
    config = snapshot.get("config", {})
    model_params = config.get("model", {}).get("params", {})
    sfreq = float(model_params.get("sfreq", 100.0))
    win_offset = float(config.get("trainer", {}).get("params", {}).get("win_offset", 0.5))
    if calibration:
        sfreq = float(calibration.get("sfreq", sfreq))
        win_offset = float(calibration.get("win_offset", win_offset))
    return np.arange(logits.shape[-1], dtype=np.float64) / sfreq + win_offset


def read_temperature(run_dir: Path, snapshot: dict[str, Any], readout: str, override: float | None) -> tuple[float, dict[str, Any] | None]:
    """Return the softmax temperature and optional calibration object."""
    calibration_path = run_dir / "calibration" / "temperature.json"
    calibration = load_json(calibration_path) if calibration_path.exists() else None
    if override is not None:
        return float(override), calibration
    if readout == "calibrated":
        if calibration is None:
            raise FileNotFoundError(f"Missing calibrated temperature: {calibration_path}")
        return float(calibration["best_temperature"]), calibration
    return configured_eval_temperature(snapshot), calibration


def metrics_path_for(run_dir: Path, split: str, readout: str) -> Path:
    """Return the metrics JSON path for a split/readout pair."""
    if readout == "calibrated":
        return run_dir / f"{split}_tau_metrics.json"
    return run_dir / f"{split}_metrics.json"


def read_saved_metrics(run_dir: Path, split: str, readout: str) -> tuple[float | None, float | None, float | None]:
    """Read NRMSE and bootstrap interval from a saved metrics JSON file."""
    path = metrics_path_for(run_dir, split, readout)
    if not path.exists():
        return None, None, None
    payload = load_json(path)
    metrics = payload.get("metrics", {})
    ci = payload.get("confidence_interval", {}) or {}
    return (
        float(metrics["nrmse"]) if metrics.get("nrmse") is not None else None,
        float(ci["nrmse_ci_low"]) if ci.get("nrmse_ci_low") is not None else None,
        float(ci["nrmse_ci_high"]) if ci.get("nrmse_ci_high") is not None else None,
    )


def softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Return temperature-scaled softmax probabilities."""
    values = logits.astype(np.float64, copy=False) / float(temperature)
    values = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(values)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def posterior_quantiles(probabilities: np.ndarray, grid: np.ndarray, quantiles: list[float]) -> list[np.ndarray]:
    """Return discrete posterior quantiles on the configured grid."""
    cdf = np.cumsum(probabilities, axis=1)
    output = []
    for q in quantiles:
        idx = np.argmax(cdf >= q, axis=1)
        output.append(grid[idx])
    return output


def compute_mask(targets: np.ndarray, grid: np.ndarray, target_filter: str) -> np.ndarray:
    """Return the analysis-row mask."""
    if target_filter == "all":
        return np.ones_like(targets, dtype=bool)
    return (targets >= grid[0]) & (targets <= grid[-1])


def load_run_posterior(
    run_dir: Path,
    *,
    split: str,
    readout: str,
    temperature_override: float | None,
    target_filter: str,
    near_ms: float,
    score_sigma: float,
    coverage_levels: list[float],
) -> RunPosterior:
    """Load saved logits and compute posterior geometry for one run."""
    snapshot = load_snapshot(run_dir)
    name = run_name_from_snapshot(run_dir, snapshot)
    temperature, calibration = read_temperature(run_dir, snapshot, readout, temperature_override)
    logits = np.load(run_dir / "predictions" / f"{split}_logits.npy")
    predictions_frame = pd.read_csv(run_dir / "predictions" / f"{split}_predictions.csv")
    targets = predictions_frame["target"].to_numpy(dtype=np.float64)
    grid = infer_grid(snapshot, calibration, logits)
    probabilities = softmax(logits, temperature)

    predictions = probabilities @ grid
    modes = grid[np.argmax(probabilities, axis=1)]
    var = np.sum(probabilities * (grid[None, :] - predictions[:, None]) ** 2, axis=1)
    std = np.sqrt(np.maximum(var, 0.0))
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-12), axis=1)
    q05, q10, q25, q75, q90, q95 = posterior_quantiles(probabilities, grid, [0.05, 0.10, 0.25, 0.75, 0.90, 0.95])
    width50 = q75 - q25
    width80 = q90 - q10
    width90 = q95 - q05
    near_sec = float(near_ms) / 1000.0
    mass_near = np.sum(probabilities * (np.abs(grid[None, :] - targets[:, None]) <= near_sec), axis=1)
    mode_mean_gap = np.abs(modes - predictions)
    abs_error = np.abs(predictions - targets)
    crps = crps_discrete(probabilities, grid, targets, reduction="none")
    scored_event_nll = fixed_kernel_event_nll(
        probabilities,
        grid,
        targets,
        sigma=score_sigma,
        reduction="none",
    )

    saved_nrmse, ci_low, ci_high = read_saved_metrics(run_dir, split, readout)
    nrmse_value = saved_nrmse if saved_nrmse is not None else nrmse(predictions, targets)
    rmse_value = rmse(predictions, targets)

    mask = compute_mask(targets, grid, target_filter)
    posterior = RunPosterior(
        name=name,
        label=label_for(name),
        run_dir=run_dir,
        temperature=temperature,
        grid=grid,
        metadata=predictions_frame,
        targets=targets,
        mask=mask,
        probabilities=probabilities,
        predictions=predictions,
        modes=modes,
        std=std,
        entropy=entropy,
        width50=width50,
        width80=width80,
        width90=width90,
        mass_near=mass_near,
        mode_mean_gap=mode_mean_gap,
        abs_error=abs_error,
        crps=crps,
        fixed_kernel_event_nll=scored_event_nll,
        fixed_kernel_event_nll_sigma=float(score_sigma),
        nrmse_value=float(nrmse_value),
        rmse_value=float(rmse_value),
        ci_low=ci_low,
        ci_high=ci_high,
    )
    posterior.coverage = coverage_frame(posterior, coverage_levels)
    return posterior


def coverage_frame(run: RunPosterior, levels: list[float]) -> pd.DataFrame:
    """Return empirical central-interval coverage for one run."""
    rows = []
    probabilities = run.probabilities[run.mask]
    targets = run.targets[run.mask]
    for level in levels:
        tail = (1.0 - float(level)) / 2.0
        qlo, qhi = posterior_quantiles(probabilities, run.grid, [tail, 1.0 - tail])
        rows.append(
            {
                "name": run.name,
                "label": run.label,
                "temperature": run.temperature,
                "nominal": float(level),
                "coverage": float(np.mean((targets >= qlo) & (targets <= qhi))),
                "median_width_ms": float(np.median(qhi - qlo) * 1000.0),
            }
        )
    return pd.DataFrame(rows)


def compute_aligned_posterior(run: RunPosterior, *, align_window_ms: float) -> None:
    """Attach target-aligned mean posterior density to one run."""
    dt = float(np.median(np.diff(run.grid)))
    half_sec = float(align_window_ms) / 1000.0
    rel_grid = np.arange(-half_sec, half_sec + dt / 2.0, dt)
    selected_probabilities = run.probabilities[run.mask]
    selected_targets = run.targets[run.mask]

    aligned = np.full((selected_probabilities.shape[0], rel_grid.size), np.nan, dtype=np.float64)
    for idx, (prob, target) in enumerate(zip(selected_probabilities, selected_targets, strict=True)):
        aligned[idx] = np.interp(rel_grid, run.grid - target, prob / dt, left=np.nan, right=np.nan)

    run.aligned_x_ms = rel_grid * 1000.0
    run.aligned_density = np.nanmean(aligned, axis=0)


def subject_groups(run: RunPosterior) -> pd.Series:
    """Return subject labels for masked rows."""
    if "subject" in run.metadata:
        return run.metadata.loc[run.mask, "subject"].astype(str)
    return pd.Series(np.arange(int(run.mask.sum())), index=run.metadata.index[run.mask]).astype(str)


def subject_metric_values(run: RunPosterior, values: np.ndarray, reducer: str) -> np.ndarray:
    """Aggregate one per-trial vector to subject-level values."""
    frame = pd.DataFrame({"subject": subject_groups(run).to_numpy(), "value": values[run.mask]})
    grouped = frame.groupby("subject", sort=False)["value"]
    if reducer == "median":
        return grouped.median().to_numpy(dtype=np.float64)
    if reducer == "mean":
        return grouped.mean().to_numpy(dtype=np.float64)
    raise ValueError(f"Unknown reducer: {reducer}")


def summary_frame(runs: list[RunPosterior], near_ms: float) -> pd.DataFrame:
    """Return one row per run with posterior-geometry summary metrics."""
    rows = []
    for run in runs:
        mask = run.mask
        coverage_mae = np.nan
        coverage80 = np.nan
        if run.coverage is not None and not run.coverage.empty:
            coverage_mae = float(np.mean(np.abs(run.coverage["coverage"] - run.coverage["nominal"])))
            closest80 = run.coverage.loc[(run.coverage["nominal"] - 0.80).abs().idxmin()]
            coverage80 = float(closest80["coverage"])
        rows.append(
            {
                "name": run.name,
                "label": run.label,
                "run_dir": path_text(run.run_dir),
                "temperature": run.temperature,
                "rows_total": int(len(mask)),
                "rows_used": int(mask.sum()),
                "rows_excluded": int((~mask).sum()),
                "rmse": run.rmse_value,
                "nrmse": run.nrmse_value,
                "nrmse_ci_low": run.ci_low,
                "nrmse_ci_high": run.ci_high,
                "bias_ms": float(np.mean(run.predictions[mask] - run.targets[mask]) * 1000.0),
                "mae_ms": float(np.mean(run.abs_error[mask]) * 1000.0),
                "posterior_std_median_ms": float(np.median(run.std[mask]) * 1000.0),
                "posterior_std_mean_ms": float(np.mean(run.std[mask]) * 1000.0),
                "width50_median_ms": float(np.median(run.width50[mask]) * 1000.0),
                "width80_median_ms": float(np.median(run.width80[mask]) * 1000.0),
                "width90_median_ms": float(np.median(run.width90[mask]) * 1000.0),
                f"mass_within_{near_ms:g}ms_mean": float(np.mean(run.mass_near[mask])),
                "mode_mean_gap_median_ms": float(np.median(run.mode_mean_gap[mask]) * 1000.0),
                "uncertainty_error_corr": float(np.corrcoef(run.std[mask], run.abs_error[mask])[0, 1]),
                "entropy_median": float(np.median(run.entropy[mask])),
                "crps_mean": float(np.mean(run.crps[mask])),
                "crps_mean_ms": float(np.mean(run.crps[mask]) * 1000.0),
                "fixed_kernel_event_nll_mean": float(np.mean(run.fixed_kernel_event_nll[mask])),
                "fixed_kernel_event_nll_sigma_s": run.fixed_kernel_event_nll_sigma,
                "coverage80": coverage80,
                "coverage_mae": coverage_mae,
            }
        )
    return pd.DataFrame(rows)


def trial_metrics_frame(runs: list[RunPosterior]) -> pd.DataFrame:
    """Return long-form per-trial posterior metrics."""
    frames = []
    for run in runs:
        frame = run.metadata.copy()
        frame["name"] = run.name
        frame["label"] = run.label
        frame["temperature"] = run.temperature
        frame["used_in_geometry"] = run.mask
        frame["posterior_mean"] = run.predictions
        frame["posterior_mode"] = run.modes
        frame["abs_error"] = run.abs_error
        frame["posterior_std"] = run.std
        frame["width50"] = run.width50
        frame["width80"] = run.width80
        frame["width90"] = run.width90
        frame["mass_near"] = run.mass_near
        frame["mode_mean_gap"] = run.mode_mean_gap
        frame["entropy"] = run.entropy
        frame["crps"] = run.crps
        frame["fixed_kernel_event_nll"] = run.fixed_kernel_event_nll
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def palette_frame(runs: list[RunPosterior]) -> pd.DataFrame:
    """Return the fixed color palette used by comparison figures."""
    return pd.DataFrame(
        [
            {
                "name": run.name,
                "label": run.label,
                "color": color_for(run, index),
            }
            for index, run in enumerate(runs)
        ]
    )


def style_axes(ax) -> None:
    """Apply compact paper-facing axis styling."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color="#e8e8e8", linewidth=0.8)
    ax.tick_params(axis="both", labelsize=8)


def add_jittered_points(ax, values: list[np.ndarray], positions: np.ndarray, colors: list, *, seed: int = 2026) -> None:
    """Overlay deterministic jittered subject-level points on a boxplot."""
    rng = np.random.default_rng(seed)
    for pos, vals, color in zip(positions, values, colors, strict=True):
        if len(vals) == 0:
            continue
        jitter = rng.uniform(-0.11, 0.11, size=len(vals))
        ax.scatter(
            np.full(len(vals), pos) + jitter,
            vals,
            s=8,
            color=color,
            alpha=0.22,
            linewidths=0,
            zorder=1,
        )


def add_clipped_jittered_points(
    ax,
    values: list[np.ndarray],
    positions: np.ndarray,
    colors: list,
    *,
    y_max: float,
    seed: int = 2026,
) -> int:
    """Overlay jittered points and mark values above y_max with triangles."""
    rng = np.random.default_rng(seed)
    n_clipped = 0
    for pos, vals, color in zip(positions, values, colors, strict=True):
        if len(vals) == 0:
            continue
        vals = np.asarray(vals, dtype=np.float64)
        jitter = rng.uniform(-0.11, 0.11, size=len(vals))
        x_pos = np.full(len(vals), pos) + jitter
        in_range = vals <= y_max
        if np.any(in_range):
            ax.scatter(
                x_pos[in_range],
                vals[in_range],
                s=8,
                color=color,
                alpha=0.22,
                linewidths=0,
                zorder=1,
            )
        clipped = ~in_range
        if np.any(clipped):
            n_clipped += int(np.sum(clipped))
            ax.scatter(
                x_pos[clipped],
                np.full(int(np.sum(clipped)), y_max),
                s=16,
                color=color,
                marker="^",
                alpha=0.55,
                linewidths=0,
                zorder=3,
                clip_on=False,
            )
    return n_clipped


def save_figure(fig, output_dir: Path, stem: str, formats: list[str], *, dpi: int) -> list[Path]:
    """Save one matplotlib figure in all requested formats."""
    paths = []
    for fmt in formats:
        path = output_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def plot_main_figure(runs: list[RunPosterior], output_dir: Path, *, formats: list[str], dpi: int, near_ms: float) -> list[Path]:
    """Save the multi-panel posterior-geometry summary figure."""
    color_map = color_map_for(runs)

    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.2))
    ax_perf, ax_aligned, ax_width, ax_mass, ax_gap, ax_coverage = axes.ravel()

    labels = [run.label for run in runs]
    x = np.arange(len(runs))
    nrmse_values = np.array([run.nrmse_value for run in runs])
    lower = np.array([
        run.nrmse_value - run.ci_low if run.ci_low is not None else 0.0
        for run in runs
    ])
    upper = np.array([
        run.ci_high - run.nrmse_value if run.ci_high is not None else 0.0
        for run in runs
    ])
    point_colors = [color_map[run.name] for run in runs]
    if np.any(lower > 0) or np.any(upper > 0):
        for i, run in enumerate(runs):
            ax_perf.errorbar(
                x[i],
                nrmse_values[i],
                yerr=np.array([[lower[i]], [upper[i]]]),
                fmt="o",
                color=point_colors[i],
                ecolor="#222222",
                elinewidth=1.1,
                capsize=3,
                markersize=6,
                zorder=3,
            )
    else:
        ax_perf.scatter(x, nrmse_values, color=point_colors, s=45, zorder=3)
    ax_perf.set_xticks(x, labels, rotation=25, ha="right")
    ax_perf.set_ylabel("R11 NRMSE")
    ax_perf.set_title("A. Scalar readout")
    y_low = max(0.0, float(np.nanmin(nrmse_values - lower)) - 0.015)
    y_high = float(np.nanmax(nrmse_values + upper)) + 0.015
    ax_perf.set_ylim(y_low, y_high)
    style_axes(ax_perf)

    for run in runs:
        if run.aligned_x_ms is None or run.aligned_density is None:
            continue
        ax_aligned.plot(run.aligned_x_ms, run.aligned_density, label=run.label, color=color_map[run.name], linewidth=2.0)
    ax_aligned.axvline(0.0, color="#222222", linestyle="--", linewidth=1.0)
    ax_aligned.set_xlabel("Time relative to true RT (ms)")
    ax_aligned.set_ylabel("Mean posterior density")
    ax_aligned.set_title("B. Target-aligned posterior")
    ax_aligned.legend(frameon=False, fontsize=7, ncol=1)
    style_axes(ax_aligned)

    width_values = [subject_metric_values(run, run.width80 * 1000.0, "median") for run in runs]
    box = ax_width.boxplot(width_values, tick_labels=labels, patch_artist=True, showfliers=False)
    for patch, run in zip(box["boxes"], runs, strict=True):
        patch.set_facecolor(color_map[run.name])
        patch.set_alpha(0.55)
    add_jittered_points(ax_width, width_values, x + 1, point_colors)
    ax_width.set_xticks(x + 1, labels, rotation=25, ha="right")
    ax_width.set_ylabel("80% width (ms)")
    ax_width.set_title("C. Posterior width")
    style_axes(ax_width)

    mass_values = [subject_metric_values(run, run.mass_near, "mean") for run in runs]
    box = ax_mass.boxplot(mass_values, tick_labels=labels, patch_artist=True, showfliers=False)
    for patch, run in zip(box["boxes"], runs, strict=True):
        patch.set_facecolor(color_map[run.name])
        patch.set_alpha(0.55)
    add_jittered_points(ax_mass, mass_values, x + 1, point_colors)
    ax_mass.set_xticks(x + 1, labels, rotation=25, ha="right")
    ax_mass.set_ylabel(f"Mass within +/-{near_ms:g} ms")
    ax_mass.set_title("D. Near-target mass")
    style_axes(ax_mass)

    gap_values = [subject_metric_values(run, run.mode_mean_gap * 1000.0, "median") for run in runs]
    box = ax_gap.boxplot(gap_values, tick_labels=labels, patch_artist=True, showfliers=False)
    for patch, run in zip(box["boxes"], runs, strict=True):
        patch.set_facecolor(color_map[run.name])
        patch.set_alpha(0.55)
    gap_ymax = 300.0
    n_clipped_gap = add_clipped_jittered_points(ax_gap, gap_values, x + 1, point_colors, y_max=gap_ymax)
    ax_gap.set_xticks(x + 1, labels, rotation=25, ha="right")
    ax_gap.set_ylabel("|mode - mean| (ms)")
    ax_gap.set_title("E. Mode-mean gap")
    ax_gap.set_ylim(0.0, gap_ymax)
    if n_clipped_gap:
        ax_gap.text(
            0.98,
            0.96,
            f"triangles: >{gap_ymax:.0f} ms",
            transform=ax_gap.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            color="#555555",
        )
    style_axes(ax_gap)

    ax_coverage.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1.0, label="ideal")
    for run in runs:
        if run.coverage is None:
            continue
        ax_coverage.plot(
            run.coverage["nominal"],
            run.coverage["coverage"],
            marker="o",
            color=color_map[run.name],
            linewidth=1.8,
            label=run.label,
        )
    ax_coverage.set_xlim(0.45, 0.95)
    ax_coverage.set_ylim(0.0, 1.0)
    ax_coverage.set_xlabel("Nominal central interval")
    ax_coverage.set_ylabel("Empirical coverage")
    ax_coverage.set_title("F. Posterior coverage")
    ax_coverage.legend(frameon=False, fontsize=7, ncol=1)
    style_axes(ax_coverage)

    fig.suptitle("Segmentation losses produce distinct event-time posteriors", fontsize=13, y=1.02)
    fig.tight_layout()
    return save_figure(fig, output_dir, "posterior_geometry_main", formats, dpi=dpi)


def plot_temperature_curves(runs: list[RunPosterior], output_dir: Path, *, formats: list[str], dpi: int) -> list[Path]:
    """Save validation NRMSE versus temperature curves when calibration files exist."""
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    plotted = 0
    for i, run in enumerate(runs):
        path = run.run_dir / "calibration" / "temperature.json"
        if not path.exists():
            continue
        calibration = load_json(path)
        results = calibration.get("results", [])
        if not results:
            continue
        frame = pd.DataFrame(results)
        color = color_for(run, i)
        ax.plot(frame["temperature"], frame["nrmse"], marker="o", markersize=3.5, linewidth=1.5, label=run.label, color=color)
        ax.axvline(float(calibration["best_temperature"]), color=color, alpha=0.25, linewidth=1.2)
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return []
    ax.set_xlabel("Softmax temperature")
    ax.set_ylabel("Validation NRMSE")
    ax.set_title("Temperature sensitivity on development split")
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax)
    fig.tight_layout()
    return save_figure(fig, output_dir, "temperature_sensitivity", formats, dpi=dpi)


def choose_example_indices(runs: list[RunPosterior], max_examples: int) -> list[tuple[str, int]]:
    """Choose representative examples using deterministic posterior-geometry criteria."""
    if not runs or max_examples <= 0:
        return []
    base = runs[0]
    mask = base.mask.copy()
    for run in runs[1:]:
        mask &= run.mask
    valid = np.where(mask)[0]
    if valid.size == 0:
        return []

    predictions = np.vstack([run.predictions for run in runs])
    widths = np.vstack([run.width80 for run in runs])
    gaps = np.vstack([run.mode_mean_gap for run in runs])
    errors = np.vstack([run.abs_error for run in runs])

    choices: list[tuple[str, int]] = []

    pred_spread = np.ptp(predictions[:, valid], axis=0)
    width_spread = np.ptp(widths[:, valid], axis=0)
    good_scalar = np.nanmedian(errors[:, valid], axis=0) < 0.18
    score = width_spread - pred_spread
    score = np.where(good_scalar, score, score - 1.0)
    choices.append(("similar scalar, different width", int(valid[np.argmax(score)])))

    worst_gap_run = int(np.argmax(np.nanmedian(gaps[:, valid], axis=1)))
    candidates = valid[errors[worst_gap_run, valid] < 0.18]
    if candidates.size == 0:
        candidates = valid
    choices.append(("large mode-mean gap", int(candidates[np.argmax(gaps[worst_gap_run, candidates])])))

    choices.append(("largest model disagreement", int(valid[np.argmax(pred_spread)])))

    unique = []
    used = set()
    for label, idx in choices:
        if idx in used:
            continue
        unique.append((label, idx))
        used.add(idx)
        if len(unique) >= max_examples:
            break
    return unique


def plot_examples(runs: list[RunPosterior], output_dir: Path, *, formats: list[str], dpi: int, max_examples: int) -> tuple[list[Path], pd.DataFrame | None]:
    """Save representative posterior overlays."""
    examples = choose_example_indices(runs, max_examples)
    if not examples:
        return [], None
    rows = []
    fig, axes = plt.subplots(len(examples), 1, figsize=(8.8, 2.55 * len(examples)), sharex=True)
    if len(examples) == 1:
        axes = [axes]
    dt = float(np.median(np.diff(runs[0].grid)))

    for ax, (example_label, idx) in zip(axes, examples, strict=True):
        target = float(runs[0].targets[idx])
        row = {
            "example": example_label,
            "row_index": idx,
            "target": target,
        }
        if "row_id" in runs[0].metadata:
            row["row_id"] = runs[0].metadata.loc[idx, "row_id"]
        if "subject" in runs[0].metadata:
            row["subject"] = runs[0].metadata.loc[idx, "subject"]
        for i, run in enumerate(runs):
            color = color_for(run, i)
            ax.plot(run.grid, run.probabilities[idx] / dt, color=color, linewidth=1.7, label=run.label)
            ax.axvline(run.predictions[idx], color=color, linestyle=":", linewidth=1.0, alpha=0.9)
            row[f"{run.name}_prediction"] = float(run.predictions[idx])
            row[f"{run.name}_width80_ms"] = float(run.width80[idx] * 1000.0)
            row[f"{run.name}_mode_mean_gap_ms"] = float(run.mode_mean_gap[idx] * 1000.0)
        ax.axvline(target, color="#111111", linestyle="--", linewidth=1.2, label="target")
        ax.set_ylabel("Posterior density")
        ax.set_title(f"{example_label}: target={target:.3f}s")
        style_axes(ax)
        rows.append(row)
    axes[-1].set_xlabel("Time from stimulus onset (s)")
    axes[0].legend(frameon=False, fontsize=8, ncol=min(len(runs) + 1, 4))
    fig.tight_layout()
    paths = save_figure(fig, output_dir, "representative_posteriors", formats, dpi=dpi)
    return paths, pd.DataFrame(rows)


def plot_single_run(run: RunPosterior, output_dir: Path, *, formats: list[str], dpi: int, near_ms: float) -> list[Path]:
    """Save a compact diagnostic figure for one run."""
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.2))
    ax_aligned, ax_width, ax_cov = axes
    if run.aligned_x_ms is not None and run.aligned_density is not None:
        ax_aligned.plot(run.aligned_x_ms, run.aligned_density, color="#1f77b4", linewidth=2.0)
    ax_aligned.axvline(0.0, color="#222222", linestyle="--", linewidth=1.0)
    ax_aligned.set_title("Target-aligned posterior")
    ax_aligned.set_xlabel("Relative time (ms)")
    ax_aligned.set_ylabel("Mean density")
    style_axes(ax_aligned)

    ax_width.hist(run.width80[run.mask] * 1000.0, bins=32, color="#2ca02c", alpha=0.75)
    ax_width.set_title("80% posterior width")
    ax_width.set_xlabel("Width (ms)")
    ax_width.set_ylabel("Trials")
    style_axes(ax_width)

    if run.coverage is not None:
        ax_cov.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1.0)
        ax_cov.plot(run.coverage["nominal"], run.coverage["coverage"], marker="o", color="#d62728")
    ax_cov.set_title(f"Coverage; mass +/-{near_ms:g} ms={np.mean(run.mass_near[run.mask]):.3f}")
    ax_cov.set_xlabel("Nominal")
    ax_cov.set_ylabel("Empirical")
    ax_cov.set_xlim(0.45, 0.95)
    ax_cov.set_ylim(0.0, 1.0)
    style_axes(ax_cov)

    fig.suptitle(f"{run.label} posterior diagnostics", fontsize=12, y=1.02)
    fig.tight_layout()
    stem = f"posterior_geometry_{run.name}"
    return save_figure(fig, output_dir, stem, formats, dpi=dpi)


def format_float(value: Any, digits: int = 3) -> str:
    """Format a scalar for captions."""
    if value is None:
        return "NA"
    try:
        if pd.isna(value):
            return "NA"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def caption_metric_line(row: pd.Series, near_ms: float) -> str:
    """Return one compact metric line for a caption."""
    return (
        f"- {row['label']}: NRMSE={format_float(row['nrmse'], 3)}, "
        f"80% width={format_float(row['width80_median_ms'], 0)} ms, "
        f"mass +/-{near_ms:g} ms={format_float(row[f'mass_within_{near_ms:g}ms_mean'], 3)}, "
        f"mode-mean gap={format_float(row['mode_mean_gap_median_ms'], 0)} ms."
    )


def quantitative_table_frame(summary: pd.DataFrame, near_ms: float) -> pd.DataFrame:
    """Return a compact paper-facing posterior-geometry table."""
    mass_col = f"mass_within_{near_ms:g}ms_mean"
    columns = [
        "label",
        "nrmse",
        "mae_ms",
        "crps_mean_ms",
        "fixed_kernel_event_nll_mean",
        "width80_median_ms",
        mass_col,
        "mode_mean_gap_median_ms",
        "coverage80",
        "coverage_mae",
    ]
    available = [column for column in columns if column in summary]
    table = summary.loc[:, available].copy()
    return table.rename(
        columns={
            "label": "Model",
            "nrmse": "nRMSE",
            "mae_ms": "MAE ms",
            "crps_mean_ms": "CRPS ms",
            "fixed_kernel_event_nll_mean": "Fixed-kernel EventNLL",
            "width80_median_ms": "Width80 ms",
            mass_col: f"Mass +/-{near_ms:g} ms",
            "mode_mean_gap_median_ms": "Mode-mean gap ms",
            "coverage80": "Coverage80",
            "coverage_mae": "Coverage MAE",
        }
    )


def write_csv_and_markdown_table(
    table: pd.DataFrame,
    output_dir: Path,
    *,
    score_sigma: float,
    write_markdown: bool,
) -> tuple[Path, Path | None]:
    """Save the quantitative posterior-geometry table as CSV and Markdown."""
    csv_path = output_dir / "quantitative_posterior_geometry_table.csv"
    table.to_csv(csv_path, index=False)
    md_path = None
    if write_markdown:
        md_path = output_dir / "captions" / "quantitative_posterior_geometry_table.md"
        write_text(md_path, quantitative_table_caption(table, score_sigma=score_sigma))
    return csv_path, md_path


def markdown_table(table: pd.DataFrame) -> str:
    """Render a small DataFrame as a GitHub-flavored Markdown table."""
    headers = [str(column) for column in table.columns]
    rows = []
    for _, row in table.iterrows():
        rendered = []
        for column in table.columns:
            value = row[column]
            if column == "Model":
                rendered.append(str(value))
            elif column == "nRMSE" or column.startswith("Mass +/-") or column in {"Coverage80", "Coverage MAE"}:
                rendered.append(format_float(value, 3))
            elif column == "Fixed-kernel EventNLL":
                rendered.append(format_float(value, 3))
            else:
                rendered.append(format_float(value, 0))
        rows.append(rendered)

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join([":---" if header == "Model" else "---:" for header in headers]) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def quantitative_table_caption(table: pd.DataFrame, *, score_sigma: float) -> str:
    """Return the markdown caption and paper-ready summary for the quantitative table."""
    table_text = markdown_table(table)
    return f"""# quantitative_posterior_geometry_table

## Draft Caption

Quantitative posterior geometry on R11 for matched event-time segmentation losses. Scalar nRMSE and MAE summarize point-readout accuracy; CRPS and fixed-kernel EventNLL are proper distributional scores computed from the full posterior; width, near-target mass, mode-mean gap, and empirical coverage quantify posterior concentration and calibration. CRPS is reported in milliseconds. Fixed-kernel EventNLL uses the same Gaussian observation kernel for all models (`sigma={score_sigma:.2f} s`), so lower values indicate that the observed RT has higher likelihood under the predicted event-time mixture.

{table_text}

## Camera-Ready Summary

EventNLL produces the sharpest and most target-concentrated event-time posteriors, but these posteriors are under-calibrated as uncertainty estimates. Thus, EventNLL is better interpreted as a localization objective, whereas coverage-based metrics quantify whether posterior concentration corresponds to calibrated uncertainty.
"""


def write_text(path: Path, text: str) -> None:
    """Write one UTF-8 text file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def find_summary_row(summary: pd.DataFrame, label: str) -> pd.Series | None:
    """Return the first summary row with a display label."""
    rows = summary.loc[summary["label"] == label]
    if rows.empty:
        return None
    return rows.iloc[0]


def caption_filter_note(summary: pd.DataFrame, target_filter: str, runs: list[RunPosterior]) -> str:
    """Return a short note describing which rows enter posterior geometry."""
    if summary.empty:
        return ""
    first = summary.iloc[0]
    if target_filter == "all":
        return f"All {int(first['rows_total']):,} trials are included in posterior-geometry summaries."
    grid = runs[0].grid if runs else np.array([np.nan, np.nan])
    return (
        f"Posterior-geometry summaries use {int(first['rows_used']):,}/{int(first['rows_total']):,} "
        f"representable trials with targets in [{grid[0]:.2f}, {grid[-1]:.2f}] s; "
        f"{int(first['rows_excluded']):,} trials outside this event-time window are excluded from these summaries."
    )


def main_figure_caption(
    *,
    summary: pd.DataFrame,
    runs: list[RunPosterior],
    readout: str,
    target_filter: str,
    near_ms: float,
) -> str:
    """Return the markdown caption for the main posterior-geometry figure."""
    key_rows = []
    for label in ("CE", "CE+time", "EventNLL", "Time-only", "Wasserstein"):
        row = find_summary_row(summary, label)
        if row is not None:
            key_rows.append(caption_metric_line(row, near_ms))
    key_text = "\n".join(key_rows)
    filter_note = caption_filter_note(summary, target_filter, runs)
    event_row = find_summary_row(summary, "EventNLL")
    time_row = find_summary_row(summary, "Time-only")
    event_note = ""
    if event_row is not None and time_row is not None:
        event_note = (
            "In this run set, EventNLL produces a more concentrated posterior "
            f"(80% width {format_float(event_row['width80_median_ms'], 0)} ms; "
            f"mass +/-{near_ms:g} ms {format_float(event_row[f'mass_within_{near_ms:g}ms_mean'], 3)}) "
            "whereas time-only training has the largest mode-mean gap "
            f"({format_float(time_row['mode_mean_gap_median_ms'], 0)} ms)."
        )

    return f"""# posterior_geometry_main

## Draft Caption

Output geometry of event-time posteriors learned by matched segmentation losses. All panels use the `{readout}` readout from saved temporal logits. (A) R11 scalar NRMSE anchors the comparison and shows that several losses have similar scalar RT error. (B) Predicted posterior distributions are aligned to each trial's observed RT and averaged, so zero on the x-axis denotes the true RT. (C-E) Subject-level distributions of posterior width, near-target mass, and mode-mean gap summarize whether the scalar prediction is supported by a localized temporal event distribution. In panel E, triangles at the upper axis boundary mark subject-level mode-mean gaps above 300 ms, clipped for readability. (F) Empirical coverage of central posterior intervals evaluates whether posterior concentration should be interpreted as calibrated uncertainty. {event_note}

## Analysis Notes

{filter_note}

Key summary values:

{key_text}

Interpretation: scalar RT error alone hides output geometry. EventNLL does not primarily improve scalar NRMSE here; it changes the learned posterior by concentrating probability mass near the observed RT. Time-only training can recover the posterior mean while leaving a broader or less coherent temporal event map.
"""


def examples_caption(examples: pd.DataFrame | None) -> str:
    """Return the markdown caption for representative posterior examples."""
    rows = []
    if examples is not None:
        for _, row in examples.iterrows():
            bits = [
                f"- {row.get('example', 'example')}: target={format_float(row.get('target'), 3)} s",
            ]
            if "row_id" in row:
                bits.append(f"row_id={row['row_id']}")
            if "subject" in row:
                bits.append(f"subject={row['subject']}")
            rows.append(", ".join(bits) + ".")
    selected = "\n".join(rows) if rows else "- No representative examples were selected."
    return f"""# representative_posteriors

## Draft Caption

Representative R11 trials illustrating how similar scalar predictions can arise from different event-time posterior shapes. Solid curves show posterior density over event time for each loss; colored dotted lines mark posterior means used as scalar RT readouts; the black dashed line marks the observed RT. Examples are selected deterministically from posterior-geometry criteria rather than at random: similar scalar readout but different posterior width, large mode-mean gap, and largest disagreement among calibrated model predictions.

## Selected Trials

{selected}

Interpretation: these examples are qualitative support for the aggregate geometry panels. They should not be used as standalone evidence, but they help show why a scalar RT prediction can be insufficient to describe what the model learned temporally.
"""


def temperature_caption(runs: list[RunPosterior]) -> str:
    """Return the markdown caption for the temperature-sensitivity figure."""
    rows = []
    boundary_rows = []
    for run in runs:
        path = run.run_dir / "calibration" / "temperature.json"
        if not path.exists():
            continue
        calibration = load_json(path)
        best = float(calibration.get("best_temperature", run.temperature))
        best_nrmse = calibration.get("best_nrmse")
        grid = calibration.get("grid", {})
        grid_min = grid.get("min")
        grid_max = grid.get("max")
        rows.append(
            f"- {run.label}: tau={format_float(best, 2)}, "
            f"development NRMSE={format_float(best_nrmse, 4)}."
        )
        if grid_min is not None and abs(best - float(grid_min)) < 1e-9:
            boundary_rows.append(f"{run.label} selected the lower grid boundary tau={best:.2f}")
        if grid_max is not None and abs(best - float(grid_max)) < 1e-9:
            boundary_rows.append(f"{run.label} selected the upper grid boundary tau={best:.2f}")
    row_text = "\n".join(rows) if rows else "- No calibration files were found."
    boundary_text = "; ".join(boundary_rows) + "." if boundary_rows else "No selected temperature is on the grid boundary."
    return f"""# temperature_sensitivity

## Draft Caption

Development-set temperature sensitivity for post-hoc temporal softmax calibration. Curves show NRMSE on the development split as logits are converted to event-time posteriors with different softmax temperatures. Vertical guide lines mark the selected temperature for each run. The plot is a diagnostic for how strongly each learned logit field must be sharpened or smoothed before scalar readout.

## Selected Temperatures

{row_text}

Diagnostic note: {boundary_text} Boundary selections should be interpreted as a warning that the calibration grid may be too narrow or that the corresponding loss produced logits whose posterior geometry requires strong post-hoc correction.
"""


def captions_readme(output_dir: Path, readout: str, formats: list[str]) -> str:
    """Return a README for the captions folder."""
    figure_formats = ", ".join(formats)
    return f"""# Caption Drafts

These markdown files are generated by `benchmarks/scripts/plot_segmentation_posteriors.py`.

They are intentionally stored outside the figure images so manuscript wording can be edited without changing the plotted data. The current run used `{readout}` readout and saved figure format(s): {figure_formats}.

Figures live in:

`{path_text(output_dir)}`

The fixed color palette is saved as `posterior_color_palette.csv` in the same directory.

Files:

- `posterior_geometry_main.md`: main multi-panel output-geometry caption.
- `quantitative_posterior_geometry_table.md`: compact paper-facing posterior-geometry table and summary.
- `representative_posteriors.md`: qualitative posterior overlay caption.
- `temperature_sensitivity.md`: calibration-temperature diagnostic caption.
"""


def write_caption_files(
    *,
    output_dir: Path,
    summary: pd.DataFrame,
    runs: list[RunPosterior],
    examples: pd.DataFrame | None,
    readout: str,
    target_filter: str,
    near_ms: float,
    formats: list[str],
) -> list[Path]:
    """Write markdown caption drafts for generated figures."""
    caption_dir = output_dir / "captions"
    files = {
        "README.md": captions_readme(output_dir, readout, formats),
        "posterior_geometry_main.md": main_figure_caption(
            summary=summary,
            runs=runs,
            readout=readout,
            target_filter=target_filter,
            near_ms=near_ms,
        ),
        "representative_posteriors.md": examples_caption(examples),
        "temperature_sensitivity.md": temperature_caption(runs),
    }
    paths = []
    for name, text in files.items():
        path = caption_dir / name
        write_text(path, text)
        paths.append(path)
    return paths


def run_cli(args: argparse.Namespace) -> list[RunPosterior]:
    """Run the plotting pipeline."""
    experiment_dir = resolve_path(args.experiment_dir, PROJECT_ROOT)
    if experiment_dir is None:
        raise ValueError("experiment_dir cannot be None.")
    experiment_dir = experiment_dir.resolve()
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Missing experiment directory: {experiment_dir}")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = experiment_dir / "figures" / f"posterior_geometry_{args.readout}"
    else:
        output_dir = resolve_path(output_dir, PROJECT_ROOT)
        if output_dir is None:
            raise ValueError("output_dir cannot be None.")
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = find_run_dirs(experiment_dir, include=args.include, exclude=args.exclude, split=args.split)
    if not run_dirs:
        raise RuntimeError(f"No segmentation run directories with saved {args.split} logits found in {experiment_dir}.")

    snapshots = {run_dir: load_snapshot(run_dir) for run_dir in run_dirs}
    run_dirs = sorted(
        run_dirs,
        key=lambda path: order_key(path, run_name_from_snapshot(path, snapshots.get(path, {}))),
    )

    print("\n=== Segmentation posterior geometry ===")
    print(f"Experiment directory: {path_text(experiment_dir)}")
    print(f"Output directory: {path_text(output_dir)}")
    print(f"Split: {args.split}")
    print(f"Readout: {args.readout}")
    print(f"Target filter: {args.target_filter}")
    print(f"Runs: {len(run_dirs)}")

    runs = []
    for run_dir in run_dirs:
        run = load_run_posterior(
            run_dir,
            split=args.split,
            readout=args.readout,
            temperature_override=args.temperature,
            target_filter=args.target_filter,
            near_ms=args.near_ms,
            score_sigma=args.score_sigma,
            coverage_levels=args.coverage_levels,
        )
        compute_aligned_posterior(run, align_window_ms=args.align_window_ms)
        runs.append(run)
        print(
            f"- {run.label}: tau={run.temperature:.4f}, "
            f"rows={int(run.mask.sum()):,}/{len(run.mask):,}, "
            f"nrmse={run.nrmse_value:.6f}"
        )

    summary = summary_frame(runs, args.near_ms)
    summary_path = output_dir / "posterior_geometry_summary.csv"
    summary.to_csv(summary_path, index=False)

    data_paths = [summary_path]
    quantitative_table = quantitative_table_frame(summary, args.near_ms)
    table_csv_path, table_caption_path = write_csv_and_markdown_table(
        quantitative_table,
        output_dir,
        score_sigma=args.score_sigma,
        write_markdown=not args.skip_captions,
    )
    data_paths.append(table_csv_path)
    palette_path = output_dir / "posterior_color_palette.csv"
    palette_frame(runs).to_csv(palette_path, index=False)
    data_paths.append(palette_path)
    if args.save_trial_csv:
        trial_path = output_dir / "posterior_geometry_trials.csv"
        trial_metrics_frame(runs).to_csv(trial_path, index=False)
        data_paths.append(trial_path)

    coverage = pd.concat([run.coverage for run in runs if run.coverage is not None], ignore_index=True)
    coverage_path = output_dir / "posterior_coverage.csv"
    coverage.to_csv(coverage_path, index=False)
    data_paths.append(coverage_path)

    aligned_rows = []
    for run in runs:
        if run.aligned_x_ms is None or run.aligned_density is None:
            continue
        aligned_rows.append(
            pd.DataFrame(
                {
                    "name": run.name,
                    "label": run.label,
                    "relative_time_ms": run.aligned_x_ms,
                    "mean_posterior_density": run.aligned_density,
                }
            )
        )
    aligned_path = output_dir / "target_aligned_posterior.csv"
    pd.concat(aligned_rows, ignore_index=True).to_csv(aligned_path, index=False)
    data_paths.append(aligned_path)

    written = []
    written.extend(plot_main_figure(runs, output_dir, formats=args.formats, dpi=args.dpi, near_ms=args.near_ms))
    written.extend(plot_temperature_curves(runs, output_dir, formats=args.formats, dpi=args.dpi))
    example_paths, examples = plot_examples(runs, output_dir, formats=args.formats, dpi=args.dpi, max_examples=args.max_examples)
    written.extend(example_paths)
    if examples is not None:
        examples.to_csv(output_dir / "representative_posteriors.csv", index=False)
    if args.save_per_run:
        for run in runs:
            written.extend(plot_single_run(run, output_dir, formats=args.formats, dpi=args.dpi, near_ms=args.near_ms))
    caption_paths = []
    if not args.skip_captions:
        caption_paths = write_caption_files(
            output_dir=output_dir,
            summary=summary,
            runs=runs,
            examples=examples,
            readout=args.readout,
            target_filter=args.target_filter,
            near_ms=args.near_ms,
            formats=args.formats,
        )
        if table_caption_path is not None:
            caption_paths.append(table_caption_path)

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
    return runs


def main(argv: list[str] | None = None) -> int:
    """Run the posterior plotting CLI."""
    args = build_parser().parse_args(argv)
    run_cli(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
