#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.preparation.checks import check_release_recordings
from benchmarks.preparation.config import DEFAULT_OUTPUT_DIR, parse_releases, resolve_output_dir


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for release cache checks."""
    parser = argparse.ArgumentParser(
        description="Check that cached EEGChallenge release recordings can be opened.",
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory with the raw EEGChallenge cache. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--releases",
        nargs="+",
        help="Release labels to check, for example: R11 or R1 R2 R3. Defaults to R1..R11.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the release cache check command."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        releases = parse_releases(args.releases)
    except ValueError as exc:
        parser.exit(1, f"{exc}\n")

    input_dir = resolve_output_dir(args.input_dir)
    manifest = check_release_recordings(input_dir=input_dir, releases=releases)

    manifest_path = input_dir / "check_manifest.json"
    print(f"Checked releases: {', '.join(manifest['releases'])}")
    print(f"Total checked recordings: {manifest['total_checked_count']}")
    print(f"Failed recordings: {manifest['total_failed_count']}")
    print(f"Manifest written to: {manifest_path}")
    return int(manifest["total_failed_count"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
