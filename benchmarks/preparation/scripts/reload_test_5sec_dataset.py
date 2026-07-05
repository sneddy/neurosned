#!/usr/bin/env python
"""One-off helper to build the R11 5 s test split without touching other splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.preparation.config import DEFAULT_OUTPUT_DIR, DEFAULT_SPLIT_OUTPUT_DIR, resolve_output_dir
from benchmarks.preparation.scripts.reload_test_dataset import reload_test_dataset


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the R11 5 s one-off builder."""
    parser = argparse.ArgumentParser(description="Build only data/new_validation/r11_test_5sec.pkl.")
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory with the raw EEGChallenge cache. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_SPLIT_OUTPUT_DIR),
        help="Directory for prepared split pickle files. Defaults to %(default)s.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build only the 5 s R11 test dataset."""
    parser = build_parser()
    args = parser.parse_args(argv)

    input_dir = resolve_output_dir(args.input_dir)
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = reload_test_dataset(input_dir, output_dir, window_kind="5sec")
    print("Reloaded R11 5 s test dataset:")
    print(f"- recordings: {manifest['loaded_recordings']:,}")
    print(f"- windows: {manifest['output']['n_windows']:,}")
    print(f"- subjects: {manifest['output']['n_subjects']:,}")
    print(f"- output: {manifest['output']['output_path']}")
    print(f"- manifest: {manifest['manifest_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
