"""Summarize paired effect sizes for paper-facing RT decoding comparisons."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

os.environ["MNE_DONTWRITE_HOME"] = "true"
os.environ["NUMBA_DISABLE_JIT"] = "1"
os.environ["MPLCONFIGDIR"] = "/tmp/neurosned-matplotlib"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.pkg.config import resolve_path
from benchmarks.pkg.evaluation.metrics import nrmse, rmse


DEFAULT_SCALAR_DIR = Path(
    "benchmarks/experiments/01_regression_baselines/etr_cnn_large_repeated__20260706_200136"
)
DEFAULT_EVENT_DIRS = [
    Path("benchmarks/experiments/02_segmentation_ablations/ets_unet_ce_repeated__20260707_110450"),
    Path("benchmarks/experiments/02_segmentation_ablations/ets_unet_event_nll_mixture_repeated__20260707_180808"),
]
DEFAULT_EVENT_LABELS = ["CE", "Mixture EventNLL"]


@dataclass(frozen=True)
class ModelPredictions:
    """Per-seed prediction frames for one model family."""

    label: str
    run_dir: Path
    frames: dict[int, pd.DataFrame]


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute matched-seed effect sizes and subject-bootstrap CIs for "
            "the strongest scalar baseline versus event-time objectives."
        )
    )
    parser.add_argument(
        "--scalar-run-dir",
        type=Path,
        default=DEFAULT_SCALAR_DIR,
        help="Repeated scalar baseline run directory. Defaults to the ETR-CNN large run.",
    )
    parser.add_argument("--scalar-label", default="ETR-CNN large", help="Display label for the scalar baseline.")
    parser.add_argument(
        "--event-run-dir",
        type=Path,
        action="append",
        default=None,
        help="Repeated event-time run directory. Can be passed multiple times.",
    )
    parser.add_argument(
        "--event-label",
        action="append",
        default=None,
        help="Display label for an event-time run. Must match --event-run-dir order if provided.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[2025, 2026, 2027, 2028, 2029],
        help="Seeds to compare as matched pairs.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=5000,
        help="Number of subject-level bootstrap resamples.",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=2025, help="Subject-bootstrap RNG seed.")
    parser.add_argument(
        "--output-table",
        type=Path,
        default=Path("benchmarks/experiments/paper_tables/main_05_effect_size.md"),
        help="Markdown table output path.",
    )
    parser.add_argument(
        "--output-csv-dir",
        type=Path,
        default=Path("benchmarks/experiments/paper_figures/csv"),
        help="Directory for source CSV outputs.",
    )
    return parser


def resolve_required(path: Path) -> Path:
    """Resolve a project-relative path and ensure it exists."""
    resolved = resolve_path(path, PROJECT_ROOT)
    if resolved is None:
        raise ValueError("Path cannot be None.")
    resolved = resolved.resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved


def prediction_path(run_dir: Path, seed: int, *, event_time: bool) -> Path:
    """Return the expected prediction CSV for a run/seed."""
    filename = "test_tau_predictions.csv" if event_time else "test_predictions.csv"
    return run_dir / f"seed{seed}" / "predictions" / filename


def load_model_predictions(label: str, run_dir: Path, seeds: list[int], *, event_time: bool) -> ModelPredictions:
    """Load prediction frames for one repeated run."""
    resolved = resolve_required(run_dir)
    frames: dict[int, pd.DataFrame] = {}
    for seed in seeds:
        path = prediction_path(resolved, seed, event_time=event_time)
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        required = {"row_id", "subject", "target", "prediction"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        frames[seed] = frame[["row_id", "subject", "target", "prediction"]].copy()
    return ModelPredictions(label=label, run_dir=resolved, frames=frames)


def assert_matched_rows(left: pd.DataFrame, right: pd.DataFrame, *, context: str) -> None:
    """Raise if prediction rows cannot be paired exactly."""
    columns = ["row_id", "subject", "target"]
    if len(left) != len(right):
        raise ValueError(f"{context}: row counts differ ({len(left)} vs {len(right)}).")
    for column in columns:
        if not np.array_equal(left[column].to_numpy(), right[column].to_numpy()):
            raise ValueError(f"{context}: column {column!r} does not match.")


def scalar_metrics(frame: pd.DataFrame) -> dict[str, float]:
    """Return scalar RT metrics for one prediction frame."""
    target = frame["target"].to_numpy(dtype=np.float64)
    prediction = frame["prediction"].to_numpy(dtype=np.float64)
    return {
        "nrmse": nrmse(prediction, target),
        "rmse_ms": 1000.0 * rmse(prediction, target),
        "mae_ms": 1000.0 * float(np.mean(np.abs(prediction - target))),
    }


def summarize_seed_metrics(model: ModelPredictions) -> dict[str, float]:
    """Return mean and sample SD over per-seed scalar metrics."""
    rows = [scalar_metrics(frame) for frame in model.frames.values()]
    summary: dict[str, float] = {}
    for metric in ["nrmse", "rmse_ms", "mae_ms"]:
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        summary[f"{metric}_mean"] = float(np.mean(values))
        summary[f"{metric}_sd"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    return summary


def per_seed_deltas(scalar: ModelPredictions, event: ModelPredictions, seeds: list[int]) -> pd.DataFrame:
    """Return matched-seed scalar-minus-event deltas."""
    rows = []
    for seed in seeds:
        scalar_frame = scalar.frames[seed]
        event_frame = event.frames[seed]
        assert_matched_rows(scalar_frame, event_frame, context=f"{scalar.label} vs {event.label}, seed{seed}")
        scalar_row = scalar_metrics(scalar_frame)
        event_row = scalar_metrics(event_frame)
        row = {
            "comparison": f"{scalar.label} -> {event.label}",
            "seed": seed,
            "scalar_nrmse": scalar_row["nrmse"],
            "event_nrmse": event_row["nrmse"],
            "delta_nrmse": scalar_row["nrmse"] - event_row["nrmse"],
            "scalar_rmse_ms": scalar_row["rmse_ms"],
            "event_rmse_ms": event_row["rmse_ms"],
            "delta_rmse_ms": scalar_row["rmse_ms"] - event_row["rmse_ms"],
            "scalar_mae_ms": scalar_row["mae_ms"],
            "event_mae_ms": event_row["mae_ms"],
            "delta_mae_ms": scalar_row["mae_ms"] - event_row["mae_ms"],
        }
        rows.append(row)
    return pd.DataFrame(rows)


def seed_averaged_error_frame(scalar: ModelPredictions, event: ModelPredictions, seeds: list[int]) -> pd.DataFrame:
    """Return one frame with seed-averaged per-row squared and absolute errors."""
    base = scalar.frames[seeds[0]][["row_id", "subject", "target"]].copy()
    scalar_sq = []
    scalar_abs = []
    event_sq = []
    event_abs = []
    for seed in seeds:
        scalar_frame = scalar.frames[seed]
        event_frame = event.frames[seed]
        assert_matched_rows(base, scalar_frame, context=f"{scalar.label}, seed{seed}")
        assert_matched_rows(base, event_frame, context=f"{event.label}, seed{seed}")
        target = base["target"].to_numpy(dtype=np.float64)
        scalar_error = scalar_frame["prediction"].to_numpy(dtype=np.float64) - target
        event_error = event_frame["prediction"].to_numpy(dtype=np.float64) - target
        scalar_sq.append(scalar_error**2)
        scalar_abs.append(np.abs(scalar_error))
        event_sq.append(event_error**2)
        event_abs.append(np.abs(event_error))

    frame = base.copy()
    frame["scalar_sqerr"] = np.mean(np.vstack(scalar_sq), axis=0)
    frame["event_sqerr"] = np.mean(np.vstack(event_sq), axis=0)
    frame["scalar_abserr"] = np.mean(np.vstack(scalar_abs), axis=0)
    frame["event_abserr"] = np.mean(np.vstack(event_abs), axis=0)
    return frame


def bootstrap_effect_ci(
    error_frame: pd.DataFrame,
    *,
    n_samples: int,
    resampling_seed: int,
) -> dict[str, float]:
    """Return subject-bootstrap CIs for scalar-minus-event deltas."""
    rng = np.random.default_rng(resampling_seed)
    subjects = error_frame["subject"].drop_duplicates().to_numpy()
    grouped = {subject: group for subject, group in error_frame.groupby("subject", sort=False)}
    values = {
        "delta_nrmse": [],
        "delta_rmse_ms": [],
        "delta_mae_ms": [],
    }
    for _ in range(n_samples):
        sampled_subjects = rng.choice(subjects, size=len(subjects), replace=True)
        sample = pd.concat([grouped[subject] for subject in sampled_subjects], ignore_index=True)
        target = sample["target"].to_numpy(dtype=np.float64)
        denominator = float(np.std(target, ddof=1)) if len(target) > 1 else 0.0
        scalar_rmse = float(np.sqrt(np.mean(sample["scalar_sqerr"].to_numpy(dtype=np.float64))))
        event_rmse = float(np.sqrt(np.mean(sample["event_sqerr"].to_numpy(dtype=np.float64))))
        delta_rmse = scalar_rmse - event_rmse
        scalar_mae = float(np.mean(sample["scalar_abserr"].to_numpy(dtype=np.float64)))
        event_mae = float(np.mean(sample["event_abserr"].to_numpy(dtype=np.float64)))
        values["delta_rmse_ms"].append(1000.0 * delta_rmse)
        values["delta_mae_ms"].append(1000.0 * (scalar_mae - event_mae))
        values["delta_nrmse"].append(delta_rmse / denominator if denominator else delta_rmse)

    result: dict[str, float] = {
        "bootstrap_samples": float(n_samples),
        "bootstrap_subjects": float(len(subjects)),
        "bootstrap_rows": float(len(error_frame)),
    }
    for metric, metric_values in values.items():
        array = np.asarray(metric_values, dtype=np.float64)
        result[f"{metric}_boot_mean"] = float(np.mean(array))
        result[f"{metric}_ci_low"] = float(np.quantile(array, 0.025))
        result[f"{metric}_ci_high"] = float(np.quantile(array, 0.975))
    return result


def fmt_pm(mean: float, sd: float, decimals: int = 4) -> str:
    """Format mean +/- SD."""
    return f"{mean:.{decimals}f} +/- {sd:.{decimals}f}"


def fmt_ci(mean: float, low: float, high: float, decimals: int) -> str:
    """Format mean [low, high]."""
    return f"{mean:.{decimals}f} [{low:.{decimals}f}, {high:.{decimals}f}]"


def write_markdown_table(path: Path, summary: pd.DataFrame, *, scalar_label: str) -> None:
    """Write a paper-facing Markdown table and caption notes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Effect Size Summary: Practical Magnitude of Event-Time Supervision",
        "",
        (
            "Intended placement: supporting paragraph or compact table near "
            "`Event-Time Posterior Formulation / Formulation Comparison and Robustness`."
        ),
        "",
        (
            "Caption draft: Practical magnitude of the gain over the strongest scalar baseline. "
            "Scalar and event-time values are mean +/- sample standard deviation over matched seeds "
            "2025-2029. Delta columns report scalar-minus-event improvement with subject-bootstrap "
            "95% confidence intervals over R11 subjects, using seed-averaged per-trial errors. "
            "Positive deltas indicate better event-time performance."
        ),
        "",
        "| Comparison | Scalar tau nRMSE | Event tau nRMSE | Delta tau nRMSE | Relative gain | Delta RMSE ms | Delta MAE ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{scalar_label} -> {row['event_label']}",
                    fmt_pm(row["scalar_nrmse_mean"], row["scalar_nrmse_sd"]),
                    fmt_pm(row["event_nrmse_mean"], row["event_nrmse_sd"]),
                    fmt_ci(row["delta_nrmse_boot_mean"], row["delta_nrmse_ci_low"], row["delta_nrmse_ci_high"], 4),
                    f"{row['relative_gain_pct']:.1f}%",
                    fmt_ci(row["delta_rmse_ms_boot_mean"], row["delta_rmse_ms_ci_low"], row["delta_rmse_ms_ci_high"], 2),
                    fmt_ci(row["delta_mae_ms_boot_mean"], row["delta_mae_ms_ci_low"], row["delta_mae_ms_ci_high"], 2),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            (
                "Paper note: This table is intended to calibrate the practical size of the main scalar "
                "accuracy gain. The absolute improvement is moderate in milliseconds but consistent "
                "across matched seeds and R11 subjects."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    event_run_dirs = args.event_run_dir or DEFAULT_EVENT_DIRS
    event_labels = args.event_label or DEFAULT_EVENT_LABELS
    if len(event_run_dirs) != len(event_labels):
        raise ValueError("--event-run-dir and --event-label must have the same length.")

    scalar = load_model_predictions(args.scalar_label, args.scalar_run_dir, args.seeds, event_time=False)
    scalar_summary = summarize_seed_metrics(scalar)
    summary_rows = []
    seed_delta_frames = []

    for event_dir, event_label in zip(event_run_dirs, event_labels, strict=True):
        event = load_model_predictions(event_label, event_dir, args.seeds, event_time=True)
        event_summary = summarize_seed_metrics(event)
        seed_deltas = per_seed_deltas(scalar, event, args.seeds)
        seed_delta_frames.append(seed_deltas)
        error_frame = seed_averaged_error_frame(scalar, event, args.seeds)
        ci = bootstrap_effect_ci(
            error_frame,
            n_samples=args.bootstrap_samples,
            resampling_seed=args.bootstrap_seed,
        )
        delta_seed_mean = seed_deltas[["delta_nrmse", "delta_rmse_ms", "delta_mae_ms"]].mean()
        delta_seed_sd = seed_deltas[["delta_nrmse", "delta_rmse_ms", "delta_mae_ms"]].std(ddof=1)
        row = {
            "scalar_label": scalar.label,
            "event_label": event.label,
            "scalar_run_dir": str(scalar.run_dir.relative_to(PROJECT_ROOT)),
            "event_run_dir": str(event.run_dir.relative_to(PROJECT_ROOT)),
            "n_seeds": len(args.seeds),
            "seeds": ",".join(str(seed) for seed in args.seeds),
            "scalar_nrmse_mean": scalar_summary["nrmse_mean"],
            "scalar_nrmse_sd": scalar_summary["nrmse_sd"],
            "event_nrmse_mean": event_summary["nrmse_mean"],
            "event_nrmse_sd": event_summary["nrmse_sd"],
            "relative_gain_pct": 100.0 * delta_seed_mean["delta_nrmse"] / scalar_summary["nrmse_mean"],
            "delta_nrmse_seed_mean": delta_seed_mean["delta_nrmse"],
            "delta_nrmse_seed_sd": delta_seed_sd["delta_nrmse"],
            "delta_rmse_ms_seed_mean": delta_seed_mean["delta_rmse_ms"],
            "delta_rmse_ms_seed_sd": delta_seed_sd["delta_rmse_ms"],
            "delta_mae_ms_seed_mean": delta_seed_mean["delta_mae_ms"],
            "delta_mae_ms_seed_sd": delta_seed_sd["delta_mae_ms"],
        }
        row.update(ci)
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    seed_deltas_all = pd.concat(seed_delta_frames, ignore_index=True)

    output_csv_dir = resolve_path(args.output_csv_dir, PROJECT_ROOT)
    if output_csv_dir is None:
        raise ValueError("--output-csv-dir cannot be None.")
    output_csv_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_csv_dir / "effect_size_summary.csv"
    seed_csv = output_csv_dir / "effect_size_seed_deltas.csv"
    summary.to_csv(summary_csv, index=False)
    seed_deltas_all.to_csv(seed_csv, index=False)

    output_table = resolve_path(args.output_table, PROJECT_ROOT)
    if output_table is None:
        raise ValueError("--output-table cannot be None.")
    write_markdown_table(output_table, summary, scalar_label=scalar.label)

    print(f"Wrote {output_table}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {seed_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
