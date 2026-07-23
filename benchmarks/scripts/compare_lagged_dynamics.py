"""Compare lagged-dynamics runs with completed scalar-regression baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE_DIR = PROJECT_ROOT / "benchmarks" / "experiments" / "01_regression_baselines"
DEFAULT_LAGGED_DIR = PROJECT_ROOT / "benchmarks" / "experiments" / "07_lagged_dynamics"
CONTRASTS = (
    ("lagged_dynamics_full", "etr_cnn_large"),
    ("lagged_dynamics_full", "msp_cnn"),
    ("lagged_dynamics_full", "lagged_dynamics_covariance_only"),
    ("lagged_dynamics_full", "lagged_dynamics_lagged_only"),
)


def build_parser() -> argparse.ArgumentParser:
    """Build the comparison CLI."""
    parser = argparse.ArgumentParser(
        description="Build a Markdown comparison from the latest repeated-run summaries."
    )
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--lagged-dir", type=Path, default=DEFAULT_LAGGED_DIR)
    parser.add_argument("--output", type=Path, default=None, help="Optional Markdown output path.")
    return parser


def latest_summaries(directory: Path) -> dict[str, dict[str, Any]]:
    """Return the newest repeated summary for every config in a group."""
    candidates: dict[str, list[tuple[str, Path, dict[str, Any]]]] = {}
    if not directory.exists():
        return {}
    for path in directory.glob("*_repeated__*/repeated_summary.json"):
        with path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        name = str(summary["config"])
        created_at = str(summary.get("created_at", ""))
        candidates.setdefault(name, []).append((created_at, path, summary))
    return {
        name: max(items, key=lambda item: (item[0], str(item[1])))[2]
        for name, items in candidates.items()
    }


def metric_text(summary: dict[str, Any], prefix: str) -> str:
    """Format one mean and seed-level standard deviation."""
    mean = summary.get(f"{prefix}_mean")
    std = summary.get(f"{prefix}_std")
    if mean is None:
        return "--"
    if std is None:
        return f"{float(mean):.4f}"
    return f"{float(mean):.4f} +/- {float(std):.4f}"


def seed_scores(summary: dict[str, Any]) -> dict[int, float]:
    """Return finite per-seed holdout nRMSE values."""
    scores = {}
    for record in summary.get("runs", []):
        seed = record.get("seed")
        value = record.get("test_nrmse")
        if seed is not None and value is not None:
            scores[int(seed)] = float(value)
    return scores


def sample_mean_std(values: list[float]) -> tuple[float, float]:
    """Return mean and sample standard deviation without external dependencies."""
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, variance**0.5


def build_markdown(
    baseline: dict[str, dict[str, Any]],
    lagged: dict[str, dict[str, Any]],
) -> str:
    """Build the scalar table and paired-seed architectural contrasts."""
    rows = [("baseline", name, summary) for name, summary in baseline.items()]
    rows.extend(("lagged dynamics", name, summary) for name, summary in lagged.items())
    rows.sort(
        key=lambda item: (
            item[2].get("test_nrmse_mean") is None,
            item[2].get("test_nrmse_mean") or float("inf"),
            item[1],
        )
    )

    lines = [
        "# Lagged-dynamics regression comparison",
        "",
        "Lower nRMSE is better. Each row is the latest repeated summary found for that config.",
        "",
        "| family | config | seeds | valid nRMSE | holdout nRMSE |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for family, name, summary in rows:
        lines.append(
            f"| {family} | `{name}` | {summary.get('n_runs', 0)} | "
            f"{metric_text(summary, 'valid_nrmse')} | {metric_text(summary, 'test_nrmse')} |"
        )

    combined = {**baseline, **lagged}
    contrast_lines = []
    for left_name, right_name in CONTRASTS:
        if left_name not in combined or right_name not in combined:
            continue
        left_scores = seed_scores(combined[left_name])
        right_scores = seed_scores(combined[right_name])
        common_seeds = sorted(set(left_scores) & set(right_scores))
        if not common_seeds:
            continue
        deltas = [left_scores[seed] - right_scores[seed] for seed in common_seeds]
        mean, std = sample_mean_std(deltas)
        contrast_lines.append(
            f"| `{left_name}` - `{right_name}` | {len(common_seeds)} | {mean:+.4f} +/- {std:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Paired-seed contrasts",
            "",
            "A negative delta favors the first model.",
            "",
            "| contrast | paired seeds | delta holdout nRMSE |",
            "| --- | ---: | ---: |",
        ]
    )
    lines.extend(contrast_lines or ["| -- | 0 | -- |"])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Load summaries, render Markdown, and optionally save it."""
    args = build_parser().parse_args(argv)
    baseline = latest_summaries(args.baseline_dir.resolve())
    lagged = latest_summaries(args.lagged_dir.resolve())
    markdown = build_markdown(baseline, lagged)
    print(markdown, end="")
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(markdown, encoding="utf-8")
        print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
