#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.preparation.config import DEFAULT_OUTPUT_DIR, parse_releases, resolve_output_dir
from benchmarks.preparation.data import download_releases


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for release downloads."""
    parser = argparse.ArgumentParser(
        description="Download/cache EEGChallenge releases for benchmark preparation.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for the raw EEGChallenge cache. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--releases",
        nargs="+",
        help="Release labels to download, for example: R11 or R1 R2 R3. Defaults to R1..R11.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the release download command."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        releases = parse_releases(args.releases)
    except ValueError as exc:
        parser.exit(1, f"{exc}\n")

    output_dir = resolve_output_dir(args.output_dir)
    manifest = download_releases(output_dir=output_dir, releases=releases)

    manifest_path = output_dir / "download_manifest.json"
    print(f"Downloaded releases: {', '.join(manifest['releases'])}")
    print(f"Total dataset entries: {manifest['total_entry_count']}")
    print(f"Manifest written to: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
