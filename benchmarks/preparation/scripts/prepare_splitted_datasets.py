#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.preparation.config import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SPLIT_OUTPUT_DIR,
    resolve_output_dir,
)
from benchmarks.preparation.preprocessing import prepare_splitted_datasets


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for split preparation."""
    parser = argparse.ArgumentParser(
        description="Prepare release-based benchmark split pickle datasets.",
    )
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
    """Run the split preparation command."""
    parser = build_parser()
    args = parser.parse_args(argv)

    input_dir = resolve_output_dir(args.input_dir)
    output_dir = resolve_output_dir(args.output_dir)
    manifest = prepare_splitted_datasets(input_dir=input_dir, output_dir=output_dir)

    manifest_path = output_dir / "prepare_manifest.json"
    print("Prepared split outputs:")
    for row in manifest["build_summary"]:
        print(f"- {row['split']} {row['window_kind']}: {row['n_windows']:,} windows -> {row['output_path']}")
    print(f"Total windows: {manifest['total_windows']:,}")
    print(f"Manifest written to: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
