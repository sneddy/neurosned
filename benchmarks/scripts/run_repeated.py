"""Run one benchmark config for all configured evaluation seeds."""

from __future__ import annotations

import argparse
import os
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ["MNE_DONTWRITE_HOME"] = "true"
os.environ["NUMBA_DISABLE_JIT"] = "1"
os.environ["MPLCONFIGDIR"] = "/tmp/neurosned-matplotlib"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from benchmarks.pkg.config import load_experiment_config, resolve_path
from benchmarks.pkg.runtime import choose_device, path_text
from benchmarks.scripts.run import DEFAULT_OUTPUT_DIR, run_config


def build_parser() -> argparse.ArgumentParser:
    """Build the repeated-run CLI parser."""
    parser = argparse.ArgumentParser(description="Run one benchmark YAML config for repeated evaluation seeds.")
    parser.add_argument("config", type=Path, help="Path to experiment YAML config.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Artefact root directory.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto", help="Torch device selection.")
    parser.add_argument("--show-plots", action="store_true", help="Show diagnostic plots in addition to saving them.")
    parser.add_argument("--skip-initial-validation", action="store_true", help="Skip validation before training.")
    return parser


def now_stamp() -> str:
    """Return a compact UTC timestamp for aggregate directories."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def safe_slug(value: Any) -> str:
    """Return a filesystem-safe slug."""
    text = str(value).strip().replace(".", "p")
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", text)
    return text.strip("_") or "run"


def numeric_summary(values: pd.Series) -> dict[str, float | None]:
    """Return mean/std for a numeric result column."""
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return {"mean": None, "std": None}
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0}


def run_record(seed: int, manager) -> dict[str, Any]:
    """Return one compact repeated-run summary row."""
    summary = manager.summary
    return {
        "seed": seed,
        "run_name": summary.get("run_name"),
        "run_dir": summary.get("run_dir"),
        "status": summary.get("status"),
        "best_epoch": summary.get("best_epoch"),
        "best_metric": summary.get("best_metric"),
        "test_nrmse": summary.get("test_nrmse"),
        "test_nrmse_ci_low": summary.get("test_nrmse_ci_low"),
        "test_nrmse_ci_high": summary.get("test_nrmse_ci_high"),
        "test_posterior_crps": summary.get("test_posterior_crps"),
        "test_posterior_fixed_kernel_event_nll": summary.get("test_posterior_fixed_kernel_event_nll"),
        "test_tau_nrmse": summary.get("test_tau_nrmse"),
        "test_tau_nrmse_ci_low": summary.get("test_tau_nrmse_ci_low"),
        "test_tau_nrmse_ci_high": summary.get("test_tau_nrmse_ci_high"),
        "test_tau_posterior_crps": summary.get("test_tau_posterior_crps"),
        "test_tau_posterior_fixed_kernel_event_nll": summary.get("test_tau_posterior_fixed_kernel_event_nll"),
    }


def save_repeated_summary(aggregate_dir: Path, *, config_path: Path, config, records: list[dict[str, Any]]) -> None:
    """Save repeated-run CSV and aggregate JSON."""
    aggregate_dir.mkdir(parents=True, exist_ok=False)
    frame = pd.DataFrame(records)
    csv_path = aggregate_dir / "repeated_summary.csv"
    json_path = aggregate_dir / "repeated_summary.json"
    frame.to_csv(csv_path, index=False)

    valid = numeric_summary(frame["best_metric"]) if "best_metric" in frame else {"mean": None, "std": None}
    test = numeric_summary(frame["test_nrmse"]) if "test_nrmse" in frame else {"mean": None, "std": None}
    aggregate_metrics = {}
    for column in (
        "test_posterior_crps",
        "test_posterior_fixed_kernel_event_nll",
        "test_tau_nrmse",
        "test_tau_posterior_crps",
        "test_tau_posterior_fixed_kernel_event_nll",
    ):
        if column in frame:
            stats = numeric_summary(frame[column])
            aggregate_metrics[f"{column}_mean"] = stats["mean"]
            aggregate_metrics[f"{column}_std"] = stats["std"]
    summary = {
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config_path": path_text(config_path),
        "experiment": config.experiment,
        "config": config.name,
        "seeds": [record["seed"] for record in records],
        "n_runs": len(records),
        "valid_nrmse_mean": valid["mean"],
        "valid_nrmse_std": valid["std"],
        "test_nrmse_mean": test["mean"],
        "test_nrmse_std": test["std"],
        **aggregate_metrics,
        "runs": records,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Repeated summary CSV: {path_text(csv_path)}")
    print(f"Repeated summary JSON: {path_text(json_path)}")
    if valid["mean"] is not None:
        print(f"Valid NRMSE mean/std: {valid['mean']:.6f} / {valid['std']:.6f}")
    if test["mean"] is not None:
        print(f"Test NRMSE mean/std: {test['mean']:.6f} / {test['std']:.6f}")


def main(argv: list[str] | None = None) -> int:
    """Run repeated benchmark seeds."""
    args = build_parser().parse_args(argv)
    config_path = args.config.resolve()
    config = load_experiment_config(config_path)
    repeated = config.evaluation.repeated_runs
    if not repeated.enabled or not repeated.seeds:
        raise RuntimeError("evaluation.repeated_runs.enabled must be true and seeds must be non-empty.")

    device = choose_device(args.device)
    output_dir = resolve_path(args.output_dir, PROJECT_ROOT)
    aggregate_dir = (
        output_dir
        / safe_slug(config.experiment)
        / f"{safe_slug(config.name)}_repeated__{now_stamp()}"
    )

    print(f"Repeated config: {path_text(config_path)}")
    print(f"Seeds: {', '.join(str(seed) for seed in repeated.seeds)}")
    print(f"Aggregate directory: {path_text(aggregate_dir)}")

    records = []
    for seed in repeated.seeds:
        print(f"\n=== Seed {seed} ===")
        manager = run_config(
            config_path,
            output_dir=args.output_dir,
            device=device,
            show_plots=args.show_plots,
            skip_initial_validation=args.skip_initial_validation,
            seed_override=seed,
            name_suffix=f"seed{seed}",
        )
        records.append(run_record(seed, manager))

    save_repeated_summary(aggregate_dir, config_path=config_path, config=config, records=records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
