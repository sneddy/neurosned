"""Compare raw, matrix-only, and dual-view repeated regression runs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.scripts.compare_lagged_dynamics import (
    latest_summaries,
    metric_text,
    sample_mean_std,
    seed_scores,
)


DEFAULT_BASELINE_DIR = PROJECT_ROOT / "benchmarks" / "experiments" / "01_regression_baselines"
DEFAULT_MATRIX_DIR = PROJECT_ROOT / "benchmarks" / "experiments" / "07_lagged_dynamics"
DEFAULT_DUAL_DIR = PROJECT_ROOT / "benchmarks" / "experiments" / "08_dual_view_lagged_dynamics"
CONTRASTS = (
    ("dual_view_full", "raw_view_only"),
    ("dual_view_full", "lagged_dynamics_full"),
    ("dual_view_full", "dual_view_covariance_only"),
    ("dual_view_covariance_only", "raw_view_only"),
    ("dual_view_full", "etr_cnn_large"),
)


def build_parser() -> argparse.ArgumentParser:
    """Build the comparison CLI."""
    parser = argparse.ArgumentParser(description="Build a dual-view regression comparison table.")
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    parser.add_argument("--dual-dir", type=Path, default=DEFAULT_DUAL_DIR)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def build_markdown(groups: list[tuple[str, dict]]) -> str:
    """Render all available summaries and paired-seed contrasts."""
    rows = []
    combined = {}
    for family, summaries in groups:
        combined.update(summaries)
        rows.extend((family, name, summary) for name, summary in summaries.items())
    rows.sort(
        key=lambda item: (
            item[2].get("test_nrmse_mean") is None,
            item[2].get("test_nrmse_mean") or float("inf"),
            item[1],
        )
    )

    lines = [
        "# Dual-view lagged-dynamics comparison",
        "",
        "Lower nRMSE is better. Each row is the latest repeated summary for that config.",
        "",
        "| family | config | seeds | valid nRMSE | holdout nRMSE |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for family, name, summary in rows:
        lines.append(
            f"| {family} | `{name}` | {summary.get('n_runs', 0)} | "
            f"{metric_text(summary, 'valid_nrmse')} | {metric_text(summary, 'test_nrmse')} |"
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
    contrast_rows = []
    for left_name, right_name in CONTRASTS:
        if left_name not in combined or right_name not in combined:
            continue
        left = seed_scores(combined[left_name])
        right = seed_scores(combined[right_name])
        seeds = sorted(set(left) & set(right))
        if not seeds:
            continue
        mean, std = sample_mean_std([left[seed] - right[seed] for seed in seeds])
        contrast_rows.append(
            f"| `{left_name}` - `{right_name}` | {len(seeds)} | {mean:+.4f} +/- {std:.4f} |"
        )
    lines.extend(contrast_rows or ["| -- | 0 | -- |"])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Load the latest summaries and print or save the comparison."""
    args = build_parser().parse_args(argv)
    markdown = build_markdown(
        [
            ("baseline", latest_summaries(args.baseline_dir.resolve())),
            ("matrix only", latest_summaries(args.matrix_dir.resolve())),
            ("dual-view group", latest_summaries(args.dual_dir.resolve())),
        ]
    )
    print(markdown, end="")
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(markdown, encoding="utf-8")
        print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
