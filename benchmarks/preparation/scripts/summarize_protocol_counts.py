#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import pickle
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/neurosned-matplotlib")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from benchmarks.preparation.config import DEFAULT_SPLIT_OUTPUT_DIR, resolve_output_dir
from benchmarks.preparation.data import package_versions


DEFAULT_PROTOCOL_SUMMARY_DIR = PROJECT_ROOT / "benchmarks" / "experiments" / "00_data_protocol"


@dataclass(frozen=True)
class SplitSpec:
    """One prepared split used in the paper protocol."""

    name: str
    label: str
    releases: str
    role: str
    primary_file: str
    long_file: str | None = None


SPLITS = (
    SplitSpec(
        name="r1_r8_train",
        label="Train",
        releases="R1--R8",
        role="model fitting",
        primary_file="r1_r8_train.pkl",
        long_file="r1_r8_train_5sec.pkl",
    ),
    SplitSpec(
        name="r9_r10_val",
        label="Development",
        releases="R9--R10",
        role="early stopping, checkpoint selection, temperature tuning",
        primary_file="r9_r10_val.pkl",
        long_file="r9_r10_val_5sec.pkl",
    ),
    SplitSpec(
        name="r11_test",
        label="Test",
        releases="R11",
        role="one-shot final holdout evaluation",
        primary_file="r11_test.pkl",
        long_file="r11_test_5sec.pkl",
    ),
)


def load_metadata(path: Path) -> pd.DataFrame:
    """Load metadata from a prepared Braindecode pickle dataset."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        with path.open("rb") as f:
            dataset = pickle.load(f)
        return dataset.get_metadata()


def target_mask(metadata: pd.DataFrame, target_min: float | None, target_max: float | None) -> pd.Series:
    """Return rows whose target lies inside the requested support."""
    target = metadata["target"]
    mask = target.notna()
    if target_min is not None:
        mask &= target >= float(target_min)
    if target_max is not None:
        mask &= target <= float(target_max)
    return mask


def summarize_metadata(
    metadata: pd.DataFrame,
    *,
    target_min: float,
    target_max: float,
    shifted_min: float,
    shifted_max: float,
) -> dict[str, Any]:
    """Summarize prepared, analyzed, and shifted-crop rows for one split."""
    main_mask = target_mask(metadata, target_min, target_max)
    shifted_mask = target_mask(metadata, shifted_min, shifted_max)
    target = metadata["target"]

    return {
        "prepared_trials": int(len(metadata)),
        "prepared_subjects": int(metadata["subject"].nunique()),
        "target_min": float(target.min()),
        "target_max": float(target.max()),
        "below_main_support": int((target < target_min).sum()),
        "above_main_support": int((target > target_max).sum()),
        "analyzed_trials": int(main_mask.sum()),
        "analyzed_subjects": int(metadata.loc[main_mask, "subject"].nunique()),
        "shifted_subset_trials": int(shifted_mask.sum()),
        "shifted_subset_subjects": int(metadata.loc[shifted_mask, "subject"].nunique()),
    }


def split_summary(
    spec: SplitSpec,
    output_dir: Path,
    *,
    target_min: float,
    target_max: float,
    shifted_min: float,
    shifted_max: float,
) -> tuple[dict[str, Any], set[str]]:
    """Summarize primary and optional long-window files for one split."""
    primary_path = output_dir / spec.primary_file
    if not primary_path.exists():
        raise FileNotFoundError(f"Missing prepared split: {primary_path}")

    primary_metadata = load_metadata(primary_path)
    main_mask = target_mask(primary_metadata, target_min, target_max)
    subject_set = set(primary_metadata.loc[main_mask, "subject"].astype(str))

    row = {
        "split": spec.name,
        "label": spec.label,
        "releases": spec.releases,
        "role": spec.role,
        "primary_file": str(primary_path),
        "primary": summarize_metadata(
            primary_metadata,
            target_min=target_min,
            target_max=target_max,
            shifted_min=shifted_min,
            shifted_max=shifted_max,
        ),
    }

    if spec.long_file is not None:
        long_path = output_dir / spec.long_file
        row["long_file"] = str(long_path)
        if long_path.exists():
            long_metadata = load_metadata(long_path)
            row["long"] = summarize_metadata(
                long_metadata,
                target_min=target_min,
                target_max=target_max,
                shifted_min=shifted_min,
                shifted_max=shifted_max,
            )
        else:
            row["long"] = None

    return row, subject_set


def subject_overlaps(subjects_by_split: dict[str, set[str]]) -> list[dict[str, Any]]:
    """Compute pairwise overlap between analyzed subject sets."""
    names = list(subjects_by_split)
    rows = []
    for idx, left in enumerate(names):
        for right in names[idx + 1 :]:
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "overlap_subjects": len(subjects_by_split[left] & subjects_by_split[right]),
                }
            )
    return rows


def format_int(value: int) -> str:
    """Format an integer with comma separators for manuscript-facing output."""
    return f"{int(value):,}"


def markdown_table(rows: list[dict[str, Any]]) -> str:
    """Return a Markdown table for the primary 2 s protocol."""
    lines = [
        "| Partition | Releases | Role | Prepared 2 s trials | Analyzed trials | Subjects |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        primary = row["primary"]
        lines.append(
            "| {label} | {releases} | {role} | {prepared} | {analyzed} | {subjects} |".format(
                label=row["label"],
                releases=row["releases"],
                role=row["role"],
                prepared=format_int(primary["prepared_trials"]),
                analyzed=format_int(primary["analyzed_trials"]),
                subjects=format_int(primary["analyzed_subjects"]),
            )
        )
    return "\n".join(lines)


def shifted_subset_table(rows: list[dict[str, Any]]) -> str:
    """Return a Markdown table for shifted-crop common-inside counts."""
    lines = [
        "| Partition | 2 s shifted-subset trials | 2 s shifted-subset subjects | 5 s shifted-subset trials |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        primary = row["primary"]
        long = row.get("long") or {}
        lines.append(
            "| {label} | {primary_trials} | {primary_subjects} | {long_trials} |".format(
                label=row["label"],
                primary_trials=format_int(primary["shifted_subset_trials"]),
                primary_subjects=format_int(primary["shifted_subset_subjects"]),
                long_trials=format_int(long["shifted_subset_trials"]) if long else "NA",
            )
        )
    return "\n".join(lines)


def overlap_table(overlaps: list[dict[str, Any]]) -> str:
    """Return a Markdown table for analyzed-subject overlap."""
    lines = [
        "| Split pair | Overlap subjects |",
        "| --- | ---: |",
    ]
    for row in overlaps:
        lines.append(
            "| {left} vs {right} | {overlap} |".format(
                left=row["left"],
                right=row["right"],
                overlap=format_int(row["overlap_subjects"]),
            )
        )
    return "\n".join(lines)


def package_versions_block() -> str:
    """Return a compact Markdown block with package versions used to load data."""
    versions = package_versions(("braindecode", "eegdash", "mne", "numpy", "pandas", "torch"))
    lines = ["| Package | Version |", "| --- | --- |"]
    for package, version in versions.items():
        lines.append(f"| {package} | {version} |")
    return "\n".join(lines)


def markdown_report(
    rows: list[dict[str, Any]],
    overlaps: list[dict[str, Any]],
    *,
    output_dir: Path,
    target_min: float,
    target_max: float,
    shifted_min: float,
    shifted_max: float,
) -> str:
    """Return the complete paper-facing protocol summary in Markdown."""
    return "\n\n".join(
        [
            "# Release-Separated CCD Protocol Summary",
            f"Created: {datetime.now(timezone.utc).isoformat()}",
            f"Prepared split directory: `{output_dir}`",
            (
                "Main analyzed-trial support: "
                f"`{target_min:.2f} <= RT <= {target_max:.2f}` seconds."
            ),
            markdown_table(rows),
            (
                "Prepared trials are CCD windows with a stimulus and response annotation. "
                "Analyzed trials additionally satisfy the main RT-support filter, matching "
                "the fixed 2 s inference window."
            ),
            "## Subject Overlap After Filtering",
            overlap_table(overlaps),
            (
                "The train, development, and R11 test partitions are subject-disjoint "
                "after the main support filter."
            ),
            "## Shifted-Crop Common-Inside Subset",
            (
                "The shifted-crop diagnostic uses the common-inside subset "
                f"`{shifted_min:.2f} <= RT <= {shifted_max:.2f}` seconds."
            ),
            shifted_subset_table(rows),
            "## Package Versions",
            package_versions_block(),
            "",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for protocol count summaries."""
    parser = argparse.ArgumentParser(
        description="Summarize release-separated dataset counts for the paper protocol.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_SPLIT_OUTPUT_DIR),
        help="Directory with prepared split pickle files. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--summary-dir",
        default=str(DEFAULT_PROTOCOL_SUMMARY_DIR),
        help="Directory for generated protocol summaries. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--target-min",
        type=float,
        default=0.5,
        help="Lower RT bound for the main support-filtered protocol.",
    )
    parser.add_argument(
        "--target-max",
        type=float,
        default=2.5,
        help="Upper RT bound for the main support-filtered protocol.",
    )
    parser.add_argument(
        "--shifted-min",
        type=float,
        default=0.8,
        help="Lower RT bound for the shifted-crop common-inside subset.",
    )
    parser.add_argument(
        "--shifted-max",
        type=float,
        default=2.2,
        help="Upper RT bound for the shifted-crop common-inside subset.",
    )
    parser.add_argument(
        "--md-output",
        default=None,
        help="Optional Markdown output path. Defaults to <summary-dir>/protocol_summary.md.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print the summary without writing the Markdown output.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Summarize prepared split counts under the manuscript protocol."""
    args = build_parser().parse_args(argv)
    output_dir = resolve_output_dir(args.output_dir)
    summary_dir = Path(args.summary_dir).expanduser().resolve()

    rows = []
    subjects_by_split = {}
    for spec in SPLITS:
        row, subjects = split_summary(
            spec,
            output_dir,
            target_min=args.target_min,
            target_max=args.target_max,
            shifted_min=args.shifted_min,
            shifted_max=args.shifted_max,
        )
        rows.append(row)
        subjects_by_split[spec.name] = subjects

    overlaps = subject_overlaps(subjects_by_split)
    report = markdown_report(
        rows,
        overlaps,
        output_dir=output_dir,
        target_min=args.target_min,
        target_max=args.target_max,
        shifted_min=args.shifted_min,
        shifted_max=args.shifted_max,
    )

    print(report)

    if not args.no_write:
        md_output = Path(args.md_output) if args.md_output else summary_dir / "protocol_summary.md"
        md_output.parent.mkdir(parents=True, exist_ok=True)
        md_output.write_text(report)
        print(f"Wrote Markdown summary: {md_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
