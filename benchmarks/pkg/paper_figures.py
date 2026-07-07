"""Paper-facing figure helpers for benchmark experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from benchmarks.pkg.evaluation.metrics import crps_discrete, fixed_kernel_event_nll, nrmse, rmse
from benchmarks.pkg.runtime import PROJECT_ROOT, path_text


PAPER_ORDER = {
    "ets_unet_ce": 0,
    "unet_deeper_ce_only": 0,
    "ce_only": 0,
    "ets_unet_event_nll": 1,
    "unet_deeper_event_nll": 1,
    "event_nll": 1,
    "ets_unet_event_nll_mixture": 2,
    "event_nll_mixture": 2,
    "ets_unet_hazard_event_nll": 3,
    "hazard_event_nll": 3,
    "ets_unet_time_only": 4,
    "unet_deeper_time_only": 4,
    "time_only": 4,
    "ets_unet_wasserstein": 5,
    "unet_deeper_wass_only": 5,
    "wass_only": 5,
    "wasserstein": 5,
}

PAPER_LABELS = {
    "ets_unet_ce": "ETS-U-Net CE",
    "unet_deeper_ce_only": "ETS-U-Net CE",
    "ce_only": "ETS-U-Net CE",
    "ets_unet_event_nll": "ETS-U-Net EventNLL",
    "unet_deeper_event_nll": "ETS-U-Net EventNLL",
    "event_nll": "ETS-U-Net EventNLL",
    "ets_unet_event_nll_mixture": "ETS-U-Net mixture EventNLL",
    "event_nll_mixture": "ETS-U-Net mixture EventNLL",
    "ets_unet_hazard_event_nll": "ETS-U-Net hazard EventNLL",
    "hazard_event_nll": "ETS-U-Net hazard EventNLL",
    "ets_unet_time_only": "ETS-U-Net time-only",
    "unet_deeper_time_only": "ETS-U-Net time-only",
    "time_only": "ETS-U-Net time-only",
    "ets_unet_wasserstein": "ETS-U-Net Wasserstein",
    "unet_deeper_wass_only": "ETS-U-Net Wasserstein",
    "wass_only": "ETS-U-Net Wasserstein",
    "wasserstein": "ETS-U-Net Wasserstein",
}

PAPER_COLORS = {
    "ETS-U-Net CE": "#1f77b4",
    "ETS-U-Net EventNLL": "#2ca02c",
    "ETS-U-Net mixture EventNLL": "#17becf",
    "ETS-U-Net hazard EventNLL": "#9467bd",
    "ETS-U-Net time-only": "#d62728",
    "ETS-U-Net Wasserstein": "#8c564b",
}

FALLBACK_COLORS = (
    "#4c78a8",
    "#f58518",
    "#54a24b",
    "#e45756",
    "#b279a2",
    "#72b7b2",
)


@dataclass
class RunPosterior:
    """Computed posterior geometry for one seed-run."""

    name: str
    label: str
    color: str
    run_dir: Path
    group_dir: Path
    seed: int | None
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


@dataclass
class PosteriorGroup:
    """One model/objective with one or more seed-runs."""

    name: str
    label: str
    color: str
    group_dir: Path
    runs: list[RunPosterior]

    @property
    def representative(self) -> RunPosterior:
        """Return the seed closest to the group's mean nRMSE."""
        if len(self.runs) == 1:
            return self.runs[0]
        values = np.array([run.nrmse_value for run in self.runs], dtype=np.float64)
        mean_value = float(np.nanmean(values))
        index = int(np.nanargmin(np.abs(values - mean_value)))
        return self.runs[index]


@dataclass
class PerformanceRow:
    """One scalar-performance row for forest plots."""

    label: str
    group: str
    nrmse_mean: float
    nrmse_std: float | None
    n_runs: int
    source: str
    color: str


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML object."""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_snapshot(run_dir: Path) -> dict[str, Any]:
    """Load a saved run config snapshot."""
    path = run_dir / "config.yaml"
    if not path.exists():
        return {}
    return load_yaml(path)


def experiment_name_from_snapshot(run_dir: Path, snapshot: dict[str, Any]) -> str:
    """Return the configured experiment name for a run directory."""
    config = snapshot.get("config", {})
    name = config.get("name")
    if name:
        return str(name)
    return run_dir.name.split("__")[0]


def canonical_name(name: str) -> str:
    """Normalize historical segmentation config names."""
    name = name.replace("_repeated", "")
    if name.startswith("unet_deeper_"):
        name = name.replace("unet_deeper_", "", 1)
    if name == "ce":
        return "ets_unet_ce"
    return name


def label_for(name: str) -> str:
    """Return the paper-facing display label."""
    clean = canonical_name(name)
    if clean in PAPER_LABELS:
        return PAPER_LABELS[clean]
    if name in PAPER_LABELS:
        return PAPER_LABELS[name]
    return clean.replace("_", " ")


def color_for(label: str, index: int) -> str:
    """Return a stable paper-facing color."""
    return PAPER_COLORS.get(label, FALLBACK_COLORS[index % len(FALLBACK_COLORS)])


def order_key(name: str, path: Path) -> tuple[int, str]:
    """Return paper-facing sort order for segmentation objectives."""
    text = f"{name} {path.name}".lower()
    for key, order in PAPER_ORDER.items():
        if key in text:
            return order, name
    return 100, name


def has_predictions(run_dir: Path, split: str) -> bool:
    """Return true when a run directory contains saved segmentation predictions."""
    return (
        (run_dir / "predictions" / f"{split}_logits.npy").exists()
        and (run_dir / "predictions" / f"{split}_predictions.csv").exists()
    )


def discover_posterior_groups(
    experiment_dir: Path,
    *,
    split: str,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> list[tuple[str, Path, list[Path]]]:
    """Find single-run and repeated-run segmentation groups."""
    include = include or []
    exclude = exclude or []
    discovered: list[tuple[str, Path, list[Path]]] = []
    for child in sorted(experiment_dir.iterdir()):
        if not child.is_dir():
            continue
        summary_path = child / "repeated_summary.json"
        run_dirs: list[Path] = []
        name = child.name.split("__")[0]
        if summary_path.exists():
            summary = load_json(summary_path)
            name = str(summary.get("config") or name)
            for item in summary.get("runs", []):
                run_dir = PROJECT_ROOT / str(item.get("run_dir", ""))
                if has_predictions(run_dir, split):
                    run_dirs.append(run_dir)
            if not run_dirs:
                run_dirs = [path for path in sorted(child.glob("seed*")) if has_predictions(path, split)]
        elif has_predictions(child, split):
            snapshot = load_snapshot(child)
            name = experiment_name_from_snapshot(child, snapshot)
            run_dirs = [child]
        if not run_dirs:
            continue
        text = f"{name} {child.name}"
        if include and not any(token in text for token in include):
            continue
        if exclude and any(token in text for token in exclude):
            continue
        discovered.append((name, child, run_dirs))
    return sorted(discovered, key=lambda item: order_key(item[0], item[1]))


def configured_eval_temperature(snapshot: dict[str, Any]) -> float:
    """Return the configured evaluation temperature."""
    params = snapshot.get("config", {}).get("trainer", {}).get("params", {})
    return float(params.get("eval_temperature", params.get("temperature", 1.0)))


def infer_grid(snapshot: dict[str, Any], calibration: dict[str, Any] | None, logits: np.ndarray) -> np.ndarray:
    """Infer the absolute event-time grid for saved logits."""
    config = snapshot.get("config", {})
    model_params = config.get("model", {}).get("params", {})
    sfreq = float(model_params.get("sfreq", 100.0))
    win_offset = float(config.get("trainer", {}).get("params", {}).get("win_offset", 0.5))
    if calibration:
        sfreq = float(calibration.get("sfreq", sfreq))
        win_offset = float(calibration.get("win_offset", win_offset))
    return np.arange(logits.shape[-1], dtype=np.float64) / sfreq + win_offset


def read_temperature(
    run_dir: Path,
    snapshot: dict[str, Any],
    readout: str,
    override: float | None,
) -> tuple[float, dict[str, Any] | None]:
    """Return the temporal softmax temperature and calibration object."""
    calibration_path = run_dir / "calibration" / "temperature.json"
    calibration = load_json(calibration_path) if calibration_path.exists() else None
    if override is not None:
        return float(override), calibration
    if readout == "calibrated":
        if calibration is None:
            raise FileNotFoundError(f"Missing calibrated temperature: {calibration_path}")
        return float(calibration["best_temperature"]), calibration
    return configured_eval_temperature(snapshot), calibration


def read_saved_metrics(run_dir: Path, split: str, readout: str) -> tuple[float | None, float | None, float | None]:
    """Read saved nRMSE and optional bootstrap interval."""
    path = run_dir / (f"{split}_tau_metrics.json" if readout == "calibrated" else f"{split}_metrics.json")
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
    values -= np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(values)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def posterior_quantiles(probabilities: np.ndarray, grid: np.ndarray, quantiles: list[float]) -> list[np.ndarray]:
    """Return discrete posterior quantiles."""
    cdf = np.cumsum(probabilities, axis=1)
    output = []
    for quantile in quantiles:
        indices = np.argmax(cdf >= float(quantile), axis=1)
        output.append(grid[indices])
    return output


def compute_mask(targets: np.ndarray, grid: np.ndarray, target_filter: str) -> np.ndarray:
    """Return the analysis-row mask."""
    if target_filter == "all":
        return np.ones_like(targets, dtype=bool)
    return (targets >= grid[0]) & (targets <= grid[-1])


def coverage_frame(run: RunPosterior, levels: list[float]) -> pd.DataFrame:
    """Return empirical central-interval coverage for one seed-run."""
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
                "seed": run.seed,
                "temperature": run.temperature,
                "nominal": float(level),
                "coverage": float(np.mean((targets >= qlo) & (targets <= qhi))),
                "median_width_ms": float(np.median(qhi - qlo) * 1000.0),
            }
        )
    return pd.DataFrame(rows)


def compute_aligned_posterior(run: RunPosterior, *, align_window_ms: float) -> None:
    """Attach target-aligned mean posterior density to one seed-run."""
    dt = float(np.median(np.diff(run.grid)))
    half_sec = float(align_window_ms) / 1000.0
    rel_grid = np.arange(-half_sec, half_sec + dt / 2.0, dt)
    selected_probabilities = run.probabilities[run.mask]
    selected_targets = run.targets[run.mask]
    aligned = np.full((selected_probabilities.shape[0], rel_grid.size), np.nan, dtype=np.float64)
    for idx, (probability, target) in enumerate(zip(selected_probabilities, selected_targets, strict=True)):
        aligned[idx] = np.interp(rel_grid, run.grid - target, probability / dt, left=np.nan, right=np.nan)
    run.aligned_x_ms = rel_grid * 1000.0
    run.aligned_density = np.nanmean(aligned, axis=0)


def seed_from_run_dir(run_dir: Path) -> int | None:
    """Parse a seedXXXX directory name."""
    if run_dir.name.startswith("seed"):
        try:
            return int(run_dir.name.replace("seed", ""))
        except ValueError:
            return None
    return None


def load_run_posterior(
    run_dir: Path,
    *,
    group_dir: Path,
    name: str,
    label: str,
    color: str,
    split: str,
    readout: str,
    temperature_override: float | None,
    target_filter: str,
    near_ms: float,
    score_sigma: float,
    coverage_levels: list[float],
    align_window_ms: float,
) -> RunPosterior:
    """Load saved logits and compute posterior geometry for one seed-run."""
    snapshot = load_snapshot(run_dir)
    temperature, calibration = read_temperature(run_dir, snapshot, readout, temperature_override)
    logits = np.load(run_dir / "predictions" / f"{split}_logits.npy")
    metadata = pd.read_csv(run_dir / "predictions" / f"{split}_predictions.csv")
    targets = metadata["target"].to_numpy(dtype=np.float64)
    grid = infer_grid(snapshot, calibration, logits)
    probabilities = softmax(logits, temperature)

    predictions = probabilities @ grid
    modes = grid[np.argmax(probabilities, axis=1)]
    var = np.sum(probabilities * (grid[None, :] - predictions[:, None]) ** 2, axis=1)
    std = np.sqrt(np.maximum(var, 0.0))
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-12), axis=1)
    q05, q10, q25, q75, q90, q95 = posterior_quantiles(
        probabilities,
        grid,
        [0.05, 0.10, 0.25, 0.75, 0.90, 0.95],
    )
    width50 = q75 - q25
    width80 = q90 - q10
    width90 = q95 - q05
    near_sec = float(near_ms) / 1000.0
    mass_near = np.sum(probabilities * (np.abs(grid[None, :] - targets[:, None]) <= near_sec), axis=1)
    mode_mean_gap = np.abs(modes - predictions)
    abs_error = np.abs(predictions - targets)
    crps = crps_discrete(probabilities, grid, targets, reduction="none")
    scored_event_nll = fixed_kernel_event_nll(probabilities, grid, targets, sigma=score_sigma, reduction="none")
    saved_nrmse, ci_low, ci_high = read_saved_metrics(run_dir, split, readout)
    nrmse_value = saved_nrmse if saved_nrmse is not None else nrmse(predictions, targets)
    mask = compute_mask(targets, grid, target_filter)

    run = RunPosterior(
        name=name,
        label=label,
        color=color,
        run_dir=run_dir,
        group_dir=group_dir,
        seed=seed_from_run_dir(run_dir),
        temperature=temperature,
        grid=grid,
        metadata=metadata,
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
        rmse_value=rmse(predictions, targets),
        ci_low=ci_low,
        ci_high=ci_high,
    )
    run.coverage = coverage_frame(run, coverage_levels)
    compute_aligned_posterior(run, align_window_ms=align_window_ms)
    return run


def load_posterior_groups(
    experiment_dir: Path,
    *,
    split: str,
    readout: str,
    temperature_override: float | None,
    target_filter: str,
    near_ms: float,
    score_sigma: float,
    coverage_levels: list[float],
    align_window_ms: float,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> list[PosteriorGroup]:
    """Load all requested segmentation posterior groups."""
    groups = []
    discovered = discover_posterior_groups(experiment_dir, split=split, include=include, exclude=exclude)
    for index, (name, group_dir, run_dirs) in enumerate(discovered):
        label = label_for(name)
        color = color_for(label, index)
        runs = [
            load_run_posterior(
                run_dir,
                group_dir=group_dir,
                name=name,
                label=label,
                color=color,
                split=split,
                readout=readout,
                temperature_override=temperature_override,
                target_filter=target_filter,
                near_ms=near_ms,
                score_sigma=score_sigma,
                coverage_levels=coverage_levels,
                align_window_ms=align_window_ms,
            )
            for run_dir in sorted(run_dirs)
        ]
        groups.append(PosteriorGroup(name=name, label=label, color=color, group_dir=group_dir, runs=runs))
    return groups


def mean_std(values: list[float] | np.ndarray) -> tuple[float, float]:
    """Return mean and sample std, ignoring NaNs."""
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan"), float("nan")
    if array.size == 1:
        return float(array[0]), 0.0
    return float(np.mean(array)), float(np.std(array, ddof=1))


def subject_groups(run: RunPosterior) -> pd.Series:
    """Return subject labels for masked rows."""
    if "subject" in run.metadata:
        return run.metadata.loc[run.mask, "subject"].astype(str)
    return pd.Series(np.arange(int(run.mask.sum())), index=run.metadata.index[run.mask]).astype(str)


def subject_metric_values(run: RunPosterior, values: np.ndarray, reducer: str) -> np.ndarray:
    """Aggregate per-trial values to subject-level values."""
    frame = pd.DataFrame({"subject": subject_groups(run).to_numpy(), "value": values[run.mask]})
    grouped = frame.groupby("subject", sort=False)["value"]
    if reducer == "median":
        return grouped.median().to_numpy(dtype=np.float64)
    if reducer == "mean":
        return grouped.mean().to_numpy(dtype=np.float64)
    raise ValueError(f"Unknown reducer: {reducer}")


def per_seed_summary_frame(groups: list[PosteriorGroup], *, near_ms: float) -> pd.DataFrame:
    """Return one posterior-geometry row per seed-run."""
    rows = []
    mass_col = f"mass_within_{near_ms:g}ms_mean"
    for group in groups:
        for run in group.runs:
            mask = run.mask
            coverage_mae = np.nan
            coverage80 = np.nan
            if run.coverage is not None and not run.coverage.empty:
                coverage_mae = float(np.mean(np.abs(run.coverage["coverage"] - run.coverage["nominal"])))
                closest80 = run.coverage.loc[(run.coverage["nominal"] - 0.80).abs().idxmin()]
                coverage80 = float(closest80["coverage"])
            rows.append(
                {
                    "name": group.name,
                    "label": group.label,
                    "seed": run.seed,
                    "run_dir": path_text(run.run_dir),
                    "temperature": run.temperature,
                    "rows_total": int(len(mask)),
                    "rows_used": int(mask.sum()),
                    "rows_excluded": int((~mask).sum()),
                    "rmse": run.rmse_value,
                    "nrmse": run.nrmse_value,
                    "masked_nrmse": nrmse(run.predictions[mask], run.targets[mask]),
                    "mae_ms": float(np.mean(run.abs_error[mask]) * 1000.0),
                    "bias_ms": float(np.mean(run.predictions[mask] - run.targets[mask]) * 1000.0),
                    "posterior_std_median_ms": float(np.median(run.std[mask]) * 1000.0),
                    "posterior_std_mean_ms": float(np.mean(run.std[mask]) * 1000.0),
                    "width50_median_ms": float(np.median(run.width50[mask]) * 1000.0),
                    "width80_median_ms": float(np.median(run.width80[mask]) * 1000.0),
                    "width90_median_ms": float(np.median(run.width90[mask]) * 1000.0),
                    mass_col: float(np.mean(run.mass_near[mask])),
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


def group_summary_frame(seed_summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-seed posterior metrics to one row per model."""
    id_columns = ["name", "label"]
    numeric_columns = [
        column
        for column in seed_summary.columns
        if column not in {*id_columns, "seed", "run_dir"} and pd.api.types.is_numeric_dtype(seed_summary[column])
    ]
    rows = []
    for (name, label), frame in seed_summary.groupby(id_columns, sort=False):
        row: dict[str, Any] = {"name": name, "label": label, "n_runs": int(len(frame))}
        for column in numeric_columns:
            mean, std = mean_std(frame[column].to_numpy(dtype=np.float64))
            row[f"{column}_mean"] = mean
            row[f"{column}_std"] = std
        rows.append(row)
    return pd.DataFrame(rows)


def palette_frame(groups: list[PosteriorGroup]) -> pd.DataFrame:
    """Return the fixed model color palette."""
    return pd.DataFrame(
        [{"name": group.name, "label": group.label, "color": group.color} for group in groups]
    )


def representative_seed_frame(groups: list[PosteriorGroup]) -> pd.DataFrame:
    """Return the selected representative seed per model."""
    rows = []
    for group in groups:
        run = group.representative
        rows.append(
            {
                "name": group.name,
                "label": group.label,
                "seed": run.seed,
                "nrmse": run.nrmse_value,
                "run_dir": path_text(run.run_dir),
            }
        )
    return pd.DataFrame(rows)


def aligned_posterior_frame(groups: list[PosteriorGroup]) -> pd.DataFrame:
    """Return per-seed and aggregate aligned posterior curves."""
    rows = []
    for group in groups:
        seed_curves = []
        x_values = None
        for run in group.runs:
            if run.aligned_x_ms is None or run.aligned_density is None:
                continue
            x_values = run.aligned_x_ms
            seed_curves.append(run.aligned_density)
            rows.append(
                pd.DataFrame(
                    {
                        "name": group.name,
                        "label": group.label,
                        "seed": run.seed,
                        "relative_time_ms": run.aligned_x_ms,
                        "mean_posterior_density": run.aligned_density,
                        "curve_type": "seed",
                    }
                )
            )
        if seed_curves and x_values is not None:
            stacked = np.vstack(seed_curves)
            rows.append(
                pd.DataFrame(
                    {
                        "name": group.name,
                        "label": group.label,
                        "seed": "mean",
                        "relative_time_ms": x_values,
                        "mean_posterior_density": np.nanmean(stacked, axis=0),
                        "std_posterior_density": np.nanstd(stacked, axis=0, ddof=1)
                        if len(seed_curves) > 1
                        else np.zeros_like(x_values),
                        "curve_type": "group_mean",
                    }
                )
            )
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def coverage_summary_frame(groups: list[PosteriorGroup]) -> pd.DataFrame:
    """Return seed-level and group-level coverage curves."""
    frames = []
    for group in groups:
        seed_frames = [run.coverage for run in group.runs if run.coverage is not None]
        seed_frames = [frame for frame in seed_frames if frame is not None and not frame.empty]
        if not seed_frames:
            continue
        seeds = pd.concat(seed_frames, ignore_index=True)
        frames.append(seeds.assign(curve_type="seed"))
        for nominal, frame in seeds.groupby("nominal", sort=True):
            coverage_mean, coverage_std = mean_std(frame["coverage"].to_numpy(dtype=np.float64))
            width_mean, width_std = mean_std(frame["median_width_ms"].to_numpy(dtype=np.float64))
            frames.append(
                pd.DataFrame(
                    [
                        {
                            "name": group.name,
                            "label": group.label,
                            "seed": "mean",
                            "temperature": np.nan,
                            "nominal": float(nominal),
                            "coverage": coverage_mean,
                            "coverage_std": coverage_std,
                            "median_width_ms": width_mean,
                            "median_width_ms_std": width_std,
                            "curve_type": "group_mean",
                        }
                    ]
                )
            )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def regression_label(name: str) -> str:
    """Return paper-facing labels for regression baselines."""
    labels = {
        "msp_cnn": "MSP-CNN",
        "etr_cnn": "ETR-CNN",
        "etr_cnn_large": "ETR-CNN large",
        "tidnet_wrapped": "TIDNet wrapped",
        "eegconformer_wrapped": "EEGConformer wrapped",
        "eegnet_wrapped": "EEGNet wrapped",
        "deep4net_wrapped": "Deep4Net wrapped",
        "shallowfbcspnet_wrapped": "ShallowFBCSPNet wrapped",
        "atcnet_wrapped": "ATCNet wrapped",
        "labram_wrapped": "LaBraM wrapped",
        "eegpt_wrapped": "EEGPT wrapped",
        "medformer_wrapped": "MedFormer wrapped",
    }
    return labels.get(name, name.replace("_", " "))


def repeated_performance_rows(experiment_dir: Path, *, group: str, color: str) -> list[PerformanceRow]:
    """Read repeated_summary.json files as scalar performance rows."""
    rows = []
    for summary_path in sorted(experiment_dir.glob("*/repeated_summary.json")):
        payload = load_json(summary_path)
        config_name = str(payload.get("config") or summary_path.parent.name.split("__")[0])
        nrmse_mean = payload.get("test_tau_nrmse_mean")
        nrmse_std = payload.get("test_tau_nrmse_std")
        if nrmse_mean is None:
            nrmse_mean = payload.get("test_nrmse_mean")
            nrmse_std = payload.get("test_nrmse_std")
        if nrmse_mean is None:
            continue
        rows.append(
            PerformanceRow(
                label=regression_label(config_name),
                group=group,
                nrmse_mean=float(nrmse_mean),
                nrmse_std=float(nrmse_std) if nrmse_std is not None else None,
                n_runs=int(payload.get("n_runs") or len(payload.get("runs", [])) or 1),
                source=path_text(summary_path.parent),
                color=color,
            )
        )
    return sorted(rows, key=lambda row: row.nrmse_mean)


def best_regression_row(regression_dir: Path) -> PerformanceRow | None:
    """Return the strongest repeated regression baseline by mean R11 nRMSE."""
    rows = repeated_performance_rows(regression_dir, group="Scalar regression", color="#555555")
    return rows[0] if rows else None


def segmentation_performance_rows(groups: list[PosteriorGroup]) -> list[PerformanceRow]:
    """Return segmentation group rows for scalar forest plots."""
    rows = []
    for group in groups:
        values = [run.nrmse_value for run in group.runs]
        mean, std = mean_std(values)
        rows.append(
            PerformanceRow(
                label=group.label,
                group="Event-time",
                nrmse_mean=mean,
                nrmse_std=std,
                n_runs=len(group.runs),
                source=path_text(group.group_dir),
                color=group.color,
            )
        )
    return rows


def save_figure(fig, output_dir: Path, stem: str, formats: list[str], *, dpi: int) -> list[Path]:
    """Save a matplotlib figure in all requested formats."""
    paths = []
    for fmt in formats:
        format_dir = output_dir / fmt
        format_dir.mkdir(parents=True, exist_ok=True)
        path = format_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)
    return paths
