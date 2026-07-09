"""Post-hoc observation-noise calibration for event-time posteriors."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.special import logsumexp, ndtr

os.environ["MNE_DONTWRITE_HOME"] = "true"
os.environ["NUMBA_DISABLE_JIT"] = "1"
os.environ["MPLCONFIGDIR"] = "/tmp/neurosned-matplotlib"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.pkg.config import ExperimentConfig, resolve_path
from benchmarks.pkg.evaluation.factory import build_temperature_readout
from benchmarks.pkg.paper_figures import label_for, order_key


DEFAULT_SEGMENTATION_DIR = Path("benchmarks/experiments/02_segmentation_ablations")
DEFAULT_OUTPUT_TABLE = Path("benchmarks/experiments/paper_tables/appendix_04_observation_noise_calibration.md")
DEFAULT_OUTPUT_CSV_DIR = Path("benchmarks/experiments/paper_figures/csv")
DEFAULT_COVERAGE_LEVELS = [0.50, 0.60, 0.70, 0.80, 0.90]


@dataclass(frozen=True)
class ObservationKernel:
    """EventNLL observation model with a post-hoc multiplicative scale."""

    kind: str
    sigma: float
    mixture_weight: float = 0.10
    mixture_sigma_narrow: float | None = None
    mixture_sigma_wide: float | None = None

    @property
    def display(self) -> str:
        """Return a compact display label for the kernel."""
        if self.kind == "gaussian_mixture":
            return (
                f"mixture({self.mixture_sigma_narrow:.2f},"
                f"{self.mixture_sigma_wide:.2f}; w={self.mixture_weight:.2f})"
            )
        return f"gaussian({self.sigma:.2f})"

    def scaled_sigmas(self, scale: float) -> tuple[float, float | None, float | None]:
        """Return Gaussian or mixture component scales after calibration."""
        factor = float(scale)
        if factor <= 0:
            raise ValueError(f"scale must be positive, got {scale!r}.")
        if self.kind == "gaussian_mixture":
            narrow = self.mixture_sigma_narrow if self.mixture_sigma_narrow is not None else self.sigma
            wide = self.mixture_sigma_wide if self.mixture_sigma_wide is not None else 2.5 * self.sigma
            return factor * self.sigma, factor * narrow, factor * wide
        return factor * self.sigma, None, None

    def max_scaled_sigma(self, scale: float) -> float:
        """Return the largest active Gaussian scale."""
        sigma, narrow, wide = self.scaled_sigmas(scale)
        return max(value for value in (sigma, narrow, wide) if value is not None)


@dataclass(frozen=True)
class RunArrays:
    """Loaded posterior probabilities and metadata for one split."""

    probabilities: np.ndarray
    grid: np.ndarray
    targets: np.ndarray
    mask: np.ndarray


@dataclass(frozen=True)
class RunPair:
    """Validation and holdout arrays for one trained seed."""

    name: str
    label: str
    seed: int
    run_dir: Path
    temperature: float
    kernel: ObservationKernel
    valid: RunArrays
    test: RunArrays


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate the RT observation-noise scale for EventNLL-family event-time "
            "posteriors using saved R9-R10 logits and evaluate on R11."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[DEFAULT_SEGMENTATION_DIR],
        help="Run, repeated-run, or experiment directories to scan.",
    )
    parser.add_argument("--split", default="test", help="Holdout split prefix, usually 'test'.")
    parser.add_argument(
        "--readout",
        choices=("calibrated", "base"),
        default="calibrated",
        help="Use saved readout temperature or configured base temperature.",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=None,
        help="Only include groups/runs whose name contains this token. Defaults to event_nll.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Exclude groups/runs whose name contains this token.",
    )
    parser.add_argument(
        "--selection-objective",
        choices=("nll", "coverage_mae", "coverage80_abs"),
        default="coverage_mae",
        help="R9-R10 objective used to select the observation-noise scale.",
    )
    parser.add_argument("--scale-min", type=float, default=0.5, help="Minimum multiplicative noise scale.")
    parser.add_argument("--scale-max", type=float, default=6.0, help="Maximum multiplicative noise scale.")
    parser.add_argument("--scale-step", type=float, default=0.05, help="Noise-scale grid step.")
    parser.add_argument(
        "--coverage-levels",
        nargs="+",
        type=float,
        default=DEFAULT_COVERAGE_LEVELS,
        help="Central interval levels used for coverage MAE.",
    )
    parser.add_argument("--target-filter", choices=("representable", "all"), default="representable")
    parser.add_argument("--quantile-iterations", type=int, default=32, help="Bisection iterations for interval widths.")
    parser.add_argument(
        "--output-table",
        type=Path,
        default=DEFAULT_OUTPUT_TABLE,
        help="Markdown appendix table output path.",
    )
    parser.add_argument(
        "--output-csv-dir",
        type=Path,
        default=DEFAULT_OUTPUT_CSV_DIR,
        help="Directory for seed-level and group-level CSV outputs.",
    )
    return parser


def resolve_required(path: Path) -> Path:
    """Resolve a path relative to the project root and ensure it exists."""
    resolved = resolve_path(path, PROJECT_ROOT)
    if resolved is None:
        raise ValueError("Path cannot be None.")
    resolved = resolved.resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping."""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON mapping."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def seed_from_run_dir(run_dir: Path) -> int:
    """Parse seedXXXX directory names."""
    if run_dir.name.startswith("seed"):
        try:
            return int(run_dir.name.replace("seed", ""))
        except ValueError:
            pass
    return -1


def discover_run_groups(
    paths: list[Path],
    *,
    split: str,
    include: list[str],
    exclude: list[str],
) -> list[tuple[str, Path, list[Path]]]:
    """Find repeated EventNLL-family groups with saved validation and holdout logits."""
    groups: list[tuple[str, Path, list[Path]]] = []
    for raw in paths:
        path = resolve_required(raw)
        candidates = [path] if (path / "config.yaml").exists() or (path / "repeated_summary.json").exists() else []
        if not candidates:
            candidates = [child for child in sorted(path.iterdir()) if child.is_dir()]
        for candidate in candidates:
            name = candidate.name.split("__")[0]
            run_dirs: list[Path] = []
            summary_path = candidate / "repeated_summary.json"
            if summary_path.exists():
                summary = load_json(summary_path)
                name = str(summary.get("config") or name)
                for item in summary.get("runs", []):
                    value = item.get("run_dir")
                    if value:
                        run_dirs.append(resolve_required(Path(value)))
                if not run_dirs:
                    run_dirs = sorted(child for child in candidate.glob("seed*") if child.is_dir())
            elif (candidate / "config.yaml").exists():
                snapshot = load_yaml(candidate / "config.yaml")
                name = str(snapshot.get("config", {}).get("name") or name)
                run_dirs = [candidate]

            if not run_dirs:
                continue
            text = f"{name} {candidate.name}".lower()
            if include and not any(token.lower() in text for token in include):
                continue
            if exclude and any(token.lower() in text for token in exclude):
                continue
            usable = [
                run_dir
                for run_dir in run_dirs
                if (run_dir / "predictions" / "best_logits.npy").exists()
                and (run_dir / "predictions" / "best_val_predictions.csv").exists()
                and (run_dir / "predictions" / f"{split}_logits.npy").exists()
                and (run_dir / "predictions" / f"{split}_predictions.csv").exists()
            ]
            if usable:
                groups.append((name, candidate, sorted(usable, key=seed_from_run_dir)))
    return sorted(groups, key=lambda item: order_key(item[0], item[1]))


def read_temperature(snapshot: dict[str, Any], run_dir: Path, readout: str) -> tuple[float, dict[str, Any] | None]:
    """Return the event-time readout temperature and calibration metadata."""
    calibration_path = run_dir / "calibration" / "temperature.json"
    calibration = load_json(calibration_path) if calibration_path.exists() else None
    if readout == "calibrated":
        if calibration is None:
            raise FileNotFoundError(calibration_path)
        return float(calibration["best_temperature"]), calibration
    params = snapshot.get("config", {}).get("trainer", {}).get("params", {})
    return float(params.get("eval_temperature", params.get("temperature", 1.0))), calibration


def infer_grid(snapshot: dict[str, Any], calibration: dict[str, Any] | None, logits: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Infer the absolute event-time grid, sampling rate, and window offset."""
    config = snapshot.get("config", {})
    model_params = config.get("model", {}).get("params", {})
    trainer_params = config.get("trainer", {}).get("params", {})
    sfreq = float(model_params.get("sfreq", 100.0))
    win_offset = float(trainer_params.get("win_offset", 0.5))
    if calibration:
        sfreq = float(calibration.get("sfreq", sfreq))
        win_offset = float(calibration.get("win_offset", win_offset))
    grid = np.arange(logits.shape[-1], dtype=np.float64) / sfreq + win_offset
    return grid, sfreq, win_offset


def observation_kernel_from_snapshot(snapshot: dict[str, Any]) -> ObservationKernel:
    """Read the EventNLL observation kernel from a run snapshot."""
    params = snapshot.get("config", {}).get("trainer", {}).get("params", {})
    kind = str(params.get("event_nll_kernel", "gaussian")).lower().replace("-", "_")
    sigma = float(params.get("sigma", 0.15))
    weight = float(params.get("event_nll_mixture_weight", 0.10))
    narrow = params.get("event_nll_mixture_sigma_narrow")
    wide = params.get("event_nll_mixture_sigma_wide")
    if kind in {"normal_mixture", "mixture_gaussian", "mixture"}:
        kind = "gaussian_mixture"
    if kind in {"normal"}:
        kind = "gaussian"
    if kind not in {"gaussian", "gaussian_mixture"}:
        raise ValueError(f"Unsupported observation kernel for calibration: {kind!r}")
    return ObservationKernel(
        kind=kind,
        sigma=sigma,
        mixture_weight=weight,
        mixture_sigma_narrow=float(narrow) if narrow is not None else None,
        mixture_sigma_wide=float(wide) if wide is not None else None,
    )


def compute_mask(targets: np.ndarray, grid: np.ndarray, target_filter: str) -> np.ndarray:
    """Return the rows used for calibration/evaluation."""
    if target_filter == "all":
        return np.ones_like(targets, dtype=bool)
    dt = float(np.median(np.diff(grid))) if grid.size > 1 else 0.0
    return (targets >= grid[0] - 0.5 * dt) & (targets <= grid[-1] + 0.5 * dt)


def load_split_arrays(
    run_dir: Path,
    *,
    split: str,
    snapshot: dict[str, Any],
    config: ExperimentConfig,
    temperature: float,
    calibration: dict[str, Any] | None,
    sfreq: float,
    win_offset: float,
    target_filter: str,
) -> RunArrays:
    """Load logits/metadata and compute posterior probabilities for one split."""
    if split == "valid":
        logits_path = run_dir / "predictions" / "best_logits.npy"
        metadata_path = run_dir / "predictions" / "best_val_predictions.csv"
    else:
        logits_path = run_dir / "predictions" / f"{split}_logits.npy"
        metadata_path = run_dir / "predictions" / f"{split}_predictions.csv"
    logits = np.load(logits_path)
    metadata = pd.read_csv(metadata_path)
    grid, grid_sfreq, grid_win_offset = infer_grid(snapshot, calibration, logits)
    if abs(grid_sfreq - sfreq) > 1e-9 or abs(grid_win_offset - win_offset) > 1e-9:
        raise ValueError(f"Inconsistent time grid in {run_dir}: {split}")
    readout = build_temperature_readout(config, sfreq=sfreq, win_offset=win_offset)
    probabilities = readout.probability_fn(logits, temperature)
    targets = metadata["target"].to_numpy(dtype=np.float64)
    mask = compute_mask(targets, grid, target_filter)
    return RunArrays(
        probabilities=np.asarray(probabilities, dtype=np.float64),
        grid=grid,
        targets=targets,
        mask=mask,
    )


def load_run_pair(
    name: str,
    label: str,
    run_dir: Path,
    *,
    split: str,
    readout: str,
    target_filter: str,
) -> RunPair:
    """Load validation and holdout posteriors for one seed-run."""
    snapshot = load_yaml(run_dir / "config.yaml")
    if "config" not in snapshot:
        raise ValueError(f"Missing config section in {run_dir / 'config.yaml'}")
    config = ExperimentConfig.model_validate(snapshot["config"])
    temperature, calibration = read_temperature(snapshot, run_dir, readout)
    probe_logits = np.load(run_dir / "predictions" / f"{split}_logits.npy")
    _, sfreq, win_offset = infer_grid(snapshot, calibration, probe_logits)
    kernel = observation_kernel_from_snapshot(snapshot)
    valid = load_split_arrays(
        run_dir,
        split="valid",
        snapshot=snapshot,
        config=config,
        temperature=temperature,
        calibration=calibration,
        sfreq=sfreq,
        win_offset=win_offset,
        target_filter=target_filter,
    )
    test = load_split_arrays(
        run_dir,
        split=split,
        snapshot=snapshot,
        config=config,
        temperature=temperature,
        calibration=calibration,
        sfreq=sfreq,
        win_offset=win_offset,
        target_filter=target_filter,
    )
    return RunPair(
        name=name,
        label=label,
        seed=seed_from_run_dir(run_dir),
        run_dir=run_dir,
        temperature=temperature,
        kernel=kernel,
        valid=valid,
        test=test,
    )


def gaussian_log_kernel(targets: np.ndarray, grid: np.ndarray, sigma: float) -> np.ndarray:
    """Return log N(y; grid, sigma^2)."""
    z = (targets[:, None] - grid[None, :]) / float(sigma)
    return -0.5 * z * z - np.log(float(sigma) * np.sqrt(2.0 * np.pi))


def predictive_logpdf_and_pit(
    probabilities: np.ndarray,
    grid: np.ndarray,
    targets: np.ndarray,
    kernel: ObservationKernel,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return predictive log density and PIT values at the observed RT."""
    posterior = np.asarray(probabilities, dtype=np.float64)
    posterior = posterior / np.clip(posterior.sum(axis=1, keepdims=True), 1e-12, None)
    log_weights = np.log(np.maximum(posterior, 1e-12))
    sigma, narrow, wide = kernel.scaled_sigmas(scale)
    if kernel.kind == "gaussian_mixture":
        assert narrow is not None and wide is not None
        log_narrow = gaussian_log_kernel(targets, grid, narrow)
        log_wide = gaussian_log_kernel(targets, grid, wide)
        log_kernel = np.logaddexp(
            np.log1p(-kernel.mixture_weight) + log_narrow,
            np.log(kernel.mixture_weight) + log_wide,
        )
        cdf_kernel = (
            (1.0 - kernel.mixture_weight) * ndtr((targets[:, None] - grid[None, :]) / narrow)
            + kernel.mixture_weight * ndtr((targets[:, None] - grid[None, :]) / wide)
        )
    else:
        log_kernel = gaussian_log_kernel(targets, grid, sigma)
        cdf_kernel = ndtr((targets[:, None] - grid[None, :]) / sigma)
    logpdf = logsumexp(log_weights + log_kernel, axis=1)
    pit = np.sum(posterior * cdf_kernel, axis=1)
    return logpdf, np.clip(pit, 0.0, 1.0)


def coverage_from_pit(pit: np.ndarray, levels: list[float]) -> dict[str, float]:
    """Return central-interval coverage metrics from PIT values."""
    values: dict[str, float] = {}
    errors = []
    for level in levels:
        tail = (1.0 - float(level)) / 2.0
        coverage = float(np.mean((pit >= tail) & (pit <= 1.0 - tail)))
        values[f"coverage{int(round(level * 100)):02d}"] = coverage
        errors.append(abs(coverage - float(level)))
    values["coverage_mae"] = float(np.mean(errors))
    return values


def latent_interval_metrics(
    probabilities: np.ndarray,
    grid: np.ndarray,
    targets: np.ndarray,
    levels: list[float],
) -> dict[str, float]:
    """Return central-interval coverage and width for the latent event posterior."""
    cdf = np.cumsum(probabilities, axis=1)
    values: dict[str, float] = {}
    errors = []
    width80 = np.nan
    for level in levels:
        tail = (1.0 - float(level)) / 2.0
        lo_index = np.argmax(cdf >= tail, axis=1)
        hi_index = np.argmax(cdf >= 1.0 - tail, axis=1)
        qlo = grid[lo_index]
        qhi = grid[hi_index]
        coverage = float(np.mean((targets >= qlo) & (targets <= qhi)))
        values[f"latent_coverage{int(round(level * 100)):02d}"] = coverage
        errors.append(abs(coverage - float(level)))
        if abs(float(level) - 0.80) < 1e-9:
            width80 = float(np.median(qhi - qlo) * 1000.0)
    values["latent_coverage_mae"] = float(np.mean(errors))
    values["latent_width80_ms"] = width80
    return values


def predictive_cdf_at_values(
    probabilities: np.ndarray,
    grid: np.ndarray,
    values: np.ndarray,
    kernel: ObservationKernel,
    scale: float,
) -> np.ndarray:
    """Return predictive CDF F(values | X) for one value per row."""
    posterior = probabilities / np.clip(probabilities.sum(axis=1, keepdims=True), 1e-12, None)
    sigma, narrow, wide = kernel.scaled_sigmas(scale)
    if kernel.kind == "gaussian_mixture":
        assert narrow is not None and wide is not None
        cdf_kernel = (
            (1.0 - kernel.mixture_weight) * ndtr((values[:, None] - grid[None, :]) / narrow)
            + kernel.mixture_weight * ndtr((values[:, None] - grid[None, :]) / wide)
        )
    else:
        cdf_kernel = ndtr((values[:, None] - grid[None, :]) / sigma)
    return np.clip(np.sum(posterior * cdf_kernel, axis=1), 0.0, 1.0)


def predictive_quantile(
    probabilities: np.ndarray,
    grid: np.ndarray,
    kernel: ObservationKernel,
    scale: float,
    quantile: float,
    *,
    iterations: int,
) -> np.ndarray:
    """Return per-row predictive RT quantiles by vectorized bisection."""
    margin = max(1.0, 8.0 * kernel.max_scaled_sigma(scale))
    lo = np.full(probabilities.shape[0], float(grid[0] - margin), dtype=np.float64)
    hi = np.full(probabilities.shape[0], float(grid[-1] + margin), dtype=np.float64)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        cdf = predictive_cdf_at_values(probabilities, grid, mid, kernel, scale)
        lo = np.where(cdf < quantile, mid, lo)
        hi = np.where(cdf >= quantile, mid, hi)
    return 0.5 * (lo + hi)


def predictive_width80_ms(
    probabilities: np.ndarray,
    grid: np.ndarray,
    kernel: ObservationKernel,
    scale: float,
    *,
    iterations: int,
) -> float:
    """Return the median central 80% predictive interval width in milliseconds."""
    q10 = predictive_quantile(probabilities, grid, kernel, scale, 0.10, iterations=iterations)
    q90 = predictive_quantile(probabilities, grid, kernel, scale, 0.90, iterations=iterations)
    return float(np.median(q90 - q10) * 1000.0)


def scale_grid(min_value: float, max_value: float, step: float) -> np.ndarray:
    """Return an inclusive scale grid."""
    if min_value <= 0 or max_value <= 0 or step <= 0:
        raise ValueError("scale min/max/step must be positive.")
    if max_value < min_value:
        raise ValueError("--scale-max must be >= --scale-min.")
    count = int(np.floor((max_value - min_value) / step + 1e-9)) + 1
    grid = min_value + step * np.arange(count + 1, dtype=np.float64)
    return grid[grid <= max_value + 1e-9]


def select_scale(
    run: RunPair,
    scales: np.ndarray,
    *,
    objective: str,
    coverage_levels: list[float],
) -> dict[str, float]:
    """Select a post-hoc observation-noise scale on R9-R10."""
    probabilities = run.valid.probabilities[run.valid.mask]
    targets = run.valid.targets[run.valid.mask]
    grid = run.valid.grid
    rows = []
    for scale in scales:
        logpdf, pit = predictive_logpdf_and_pit(probabilities, grid, targets, run.kernel, float(scale))
        coverage = coverage_from_pit(pit, coverage_levels)
        row = {
            "scale": float(scale),
            "nll": float(-np.mean(logpdf)),
            "coverage80_abs": abs(coverage.get("coverage80", np.nan) - 0.80),
            "coverage_mae": coverage["coverage_mae"],
            "coverage80": coverage.get("coverage80", np.nan),
        }
        rows.append(row)
    frame = pd.DataFrame(rows)
    best_index = frame[objective].idxmin()
    best = frame.loc[best_index].to_dict()
    return {f"valid_{key}": float(value) for key, value in best.items()}


def evaluate_run(
    run: RunPair,
    selected: dict[str, float],
    *,
    coverage_levels: list[float],
    quantile_iterations: int,
) -> dict[str, Any]:
    """Evaluate raw and calibrated predictive interval behavior on R11."""
    scale = float(selected["valid_scale"])
    probabilities = run.test.probabilities[run.test.mask]
    targets = run.test.targets[run.test.mask]
    grid = run.test.grid

    latent = latent_interval_metrics(probabilities, grid, targets, coverage_levels)
    base_logpdf, base_pit = predictive_logpdf_and_pit(probabilities, grid, targets, run.kernel, 1.0)
    cal_logpdf, cal_pit = predictive_logpdf_and_pit(probabilities, grid, targets, run.kernel, scale)
    base_coverage = coverage_from_pit(base_pit, coverage_levels)
    cal_coverage = coverage_from_pit(cal_pit, coverage_levels)

    row: dict[str, Any] = {
        "name": run.name,
        "label": run.label,
        "seed": run.seed,
        "run_dir": str(run.run_dir.relative_to(PROJECT_ROOT)),
        "readout_temperature": run.temperature,
        "kernel": run.kernel.display,
        "rows_valid": int(run.valid.mask.sum()),
        "rows_test": int(run.test.mask.sum()),
        "selected_scale": scale,
        "selected_at_grid_min": bool(np.isclose(scale, selected["valid_scale_min"]))
        if "valid_scale_min" in selected
        else False,
        "selected_at_grid_max": bool(np.isclose(scale, selected["valid_scale_max"]))
        if "valid_scale_max" in selected
        else False,
        "test_base_predictive_nll": float(-np.mean(base_logpdf)),
        "test_calibrated_predictive_nll": float(-np.mean(cal_logpdf)),
        "test_base_predictive_width80_ms": predictive_width80_ms(
            probabilities,
            grid,
            run.kernel,
            1.0,
            iterations=quantile_iterations,
        ),
        "test_calibrated_predictive_width80_ms": predictive_width80_ms(
            probabilities,
            grid,
            run.kernel,
            scale,
            iterations=quantile_iterations,
        ),
    }
    row.update(selected)
    row.update({f"test_{key}": value for key, value in latent.items()})
    row.update({f"test_base_predictive_{key}": value for key, value in base_coverage.items()})
    row.update({f"test_calibrated_predictive_{key}": value for key, value in cal_coverage.items()})
    return row


def summarize_groups(seed_frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate seed-level calibration rows to one row per objective."""
    id_columns = ["name", "label", "kernel"]
    numeric_columns = [
        column
        for column in seed_frame.columns
        if column not in {*id_columns, "seed", "run_dir"}
        and pd.api.types.is_numeric_dtype(seed_frame[column])
        and not pd.api.types.is_bool_dtype(seed_frame[column])
    ]
    rows = []
    for keys, frame in seed_frame.groupby(id_columns, sort=False):
        name, label, kernel = keys
        row: dict[str, Any] = {"name": name, "label": label, "kernel": kernel, "n_runs": int(len(frame))}
        for column in numeric_columns:
            values = frame[column].to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]
            if values.size == 0:
                row[f"{column}_mean"] = np.nan
                row[f"{column}_std"] = np.nan
            elif values.size == 1:
                row[f"{column}_mean"] = float(values[0])
                row[f"{column}_std"] = 0.0
            else:
                row[f"{column}_mean"] = float(np.mean(values))
                row[f"{column}_std"] = float(np.std(values, ddof=1))
        row["selected_at_grid_min_any"] = bool(frame.get("selected_at_grid_min", pd.Series(False)).any())
        row["selected_at_grid_max_any"] = bool(frame.get("selected_at_grid_max", pd.Series(False)).any())
        rows.append(row)
    return pd.DataFrame(rows)


def display_label(label: str) -> str:
    """Return compact objective labels."""
    return (
        label.replace("ETS-U-Net ", "")
        .replace("mixture", "Mixture")
        .replace("hazard", "Hazard")
    )


def fmt_pm(mean: float, sd: float, decimals: int = 3) -> str:
    """Format mean +/- SD."""
    return f"{mean:.{decimals}f} +/- {sd:.{decimals}f}"


def write_markdown_table(
    path: Path,
    group_summary: pd.DataFrame,
    *,
    selection_objective: str,
    coverage_levels: list[float],
) -> None:
    """Write a compact appendix-ready Markdown table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Observation-Noise Calibration for EventNLL-Family Posteriors",
        "",
        (
            "Intended placement: appendix or a short supporting table for "
            "`Posterior Geometry Diagnostics`."
        ),
        "",
        (
            "Caption draft: Post-hoc RT observation-noise calibration for EventNLL-family "
            "models. The trained EEG model, event-time posterior, and posterior-mean scalar "
            "readout are fixed. A single multiplicative scale on the RT observation kernel is "
            f"selected on R9-R10 by `{selection_objective}` and applied unchanged to R11. "
            "Each model block separates the latent event-time posterior from the behavioral-RT "
            "predictive distribution obtained by convolving that posterior with the observation "
            "kernel. Values are mean +/- sample standard deviation across seeds."
        ),
        "",
    ]
    for _, row in group_summary.iterrows():
        lines.extend(
            [
                f"### {display_label(str(row['label']))}",
                "",
                "| Distribution | Scale c | Coverage MAE | Coverage80 | Width80 ms | Predictive NLL |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
                "| "
                + " | ".join(
                    [
                        "Latent event-time posterior",
                        "-",
                        fmt_pm(row["test_latent_coverage_mae_mean"], row["test_latent_coverage_mae_std"], 3),
                        fmt_pm(row["test_latent_coverage80_mean"], row["test_latent_coverage80_std"], 3),
                        fmt_pm(row["test_latent_width80_ms_mean"], row["test_latent_width80_ms_std"], 0),
                        "-",
                    ]
                )
                + " |",
                "| "
                + " | ".join(
                    [
                        "Base predictive RT",
                        "1.00",
                        fmt_pm(row["test_base_predictive_coverage_mae_mean"], row["test_base_predictive_coverage_mae_std"], 3),
                        fmt_pm(row["test_base_predictive_coverage80_mean"], row["test_base_predictive_coverage80_std"], 3),
                        fmt_pm(row["test_base_predictive_width80_ms_mean"], row["test_base_predictive_width80_ms_std"], 0),
                        fmt_pm(row["test_base_predictive_nll_mean"], row["test_base_predictive_nll_std"], 3),
                    ]
                )
                + " |",
                "| "
                + " | ".join(
                    [
                        "Calibrated predictive RT",
                        fmt_pm(row["selected_scale_mean"], row["selected_scale_std"], 2),
                        fmt_pm(
                            row["test_calibrated_predictive_coverage_mae_mean"],
                            row["test_calibrated_predictive_coverage_mae_std"],
                            3,
                        ),
                        fmt_pm(
                            row["test_calibrated_predictive_coverage80_mean"],
                            row["test_calibrated_predictive_coverage80_std"],
                            3,
                        ),
                        fmt_pm(
                            row["test_calibrated_predictive_width80_ms_mean"],
                            row["test_calibrated_predictive_width80_ms_std"],
                            0,
                        ),
                        fmt_pm(row["test_calibrated_predictive_nll_mean"], row["test_calibrated_predictive_nll_std"], 3),
                    ]
                )
                + " |",
                "",
            ]
        )
    levels = ", ".join(f"{level:.2f}" for level in coverage_levels)
    lines.extend(
        [
            f"Coverage MAE is averaged over central interval levels `{levels}`.",
            (
                "Interpretation note: this calibration changes only the RT observation-noise "
                "layer used for probabilistic prediction. It does not change trained weights, "
                "posterior-mean RT predictions, or tau-nRMSE."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Run the observation-noise calibration CLI."""
    args = build_parser().parse_args(argv)
    include = args.include if args.include is not None else ["event_nll"]
    scales = scale_grid(args.scale_min, args.scale_max, args.scale_step)
    groups = discover_run_groups(args.paths, split=args.split, include=include, exclude=args.exclude)
    if not groups:
        raise RuntimeError("No matching EventNLL-family runs with saved logits were found.")

    rows = []
    for name, group_dir, run_dirs in groups:
        label = label_for(name)
        print(f"\n{name} ({len(run_dirs)} runs)")
        for run_dir in run_dirs:
            run = load_run_pair(
                name,
                label,
                run_dir,
                split=args.split,
                readout=args.readout,
                target_filter=args.target_filter,
            )
            selected = select_scale(
                run,
                scales,
                objective=args.selection_objective,
                coverage_levels=args.coverage_levels,
            )
            selected["valid_scale_min"] = float(scales.min())
            selected["valid_scale_max"] = float(scales.max())
            row = evaluate_run(
                run,
                selected,
                coverage_levels=args.coverage_levels,
                quantile_iterations=args.quantile_iterations,
            )
            rows.append(row)
            print(
                f"  seed{run.seed}: c={row['selected_scale']:.2f}, "
                f"R11 Cov80 {row['test_base_predictive_coverage80']:.3f} -> "
                f"{row['test_calibrated_predictive_coverage80']:.3f}, "
                f"CovMAE {row['test_base_predictive_coverage_mae']:.3f} -> "
                f"{row['test_calibrated_predictive_coverage_mae']:.3f}"
            )

    seed_frame = pd.DataFrame(rows)
    group_summary = summarize_groups(seed_frame)

    output_csv_dir = resolve_path(args.output_csv_dir, PROJECT_ROOT)
    if output_csv_dir is None:
        raise ValueError("--output-csv-dir cannot be None.")
    output_csv_dir.mkdir(parents=True, exist_ok=True)
    seed_path = output_csv_dir / "observation_noise_calibration_seed_summary.csv"
    group_path = output_csv_dir / "observation_noise_calibration_group_summary.csv"
    seed_frame.to_csv(seed_path, index=False)
    group_summary.to_csv(group_path, index=False)

    output_table = resolve_path(args.output_table, PROJECT_ROOT)
    if output_table is None:
        raise ValueError("--output-table cannot be None.")
    write_markdown_table(
        output_table,
        group_summary,
        selection_objective=args.selection_objective,
        coverage_levels=args.coverage_levels,
    )

    print("\nWrote:")
    print(f"  {seed_path.relative_to(PROJECT_ROOT)}")
    print(f"  {group_path.relative_to(PROJECT_ROOT)}")
    print(f"  {output_table.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
