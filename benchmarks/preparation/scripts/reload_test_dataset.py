#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/neurosned-matplotlib")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from braindecode.datasets.base import BaseConcatDataset

from benchmarks.preparation.config import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SPLIT_OUTPUT_DIR,
    WINDOW_CONFIGS,
    resolve_output_dir,
)
from benchmarks.preparation.data import iter_recordings, load_release_dataset, package_versions
from benchmarks.preparation.preprocessing import (
    create_release_windows,
    drop_bad_cached_bdf_headers,
    offline_preprocessors,
    preprocess_skip_bad,
    save_pickle_dataset,
    summarize_windows,
)


RELEASE = "R11"
SPLIT_NAME = "r11_test"
WINDOW_KIND = "2sec"


def read_json(path: Path) -> dict | None:
    """Read a JSON file if it exists."""
    if not path.exists():
        return None
    return json.loads(path.read_text())


def expected_recording_count(input_dir: Path) -> int | None:
    """Return the expected R11 count from check_manifest when available."""
    manifest = read_json(input_dir / "check_manifest.json")
    if manifest is None:
        return None
    for row in manifest.get("release_results", []):
        if row.get("release") == RELEASE:
            return int(row.get("ok_count") or row.get("checked_count") or row.get("entry_count"))
    return None


def load_full_test_dataset(input_dir: Path, expected_count: int | None) -> BaseConcatDataset:
    """Load R11 and wrap its recordings without re-query flattening surprises."""
    dataset = load_release_dataset(RELEASE, input_dir, download=False)
    recordings = list(iter_recordings(dataset))
    print(f"Loaded {RELEASE} recordings: {len(recordings):,}")

    if expected_count is not None and len(recordings) != expected_count:
        raise RuntimeError(
            f"{RELEASE} loaded {len(recordings):,} recordings, expected {expected_count:,} "
            "from check_manifest.json."
        )

    return BaseConcatDataset(
        recordings,
        target_transform=getattr(dataset, "target_transform", None),
    )


def reload_test_dataset(input_dir: Path, output_dir: Path) -> dict:
    """Rebuild only the R11 test pickle from the existing release cache."""
    expected_count = expected_recording_count(input_dir)
    if expected_count is not None:
        print(f"Expected {RELEASE} recordings from check_manifest: {expected_count:,}")

    dataset = load_full_test_dataset(input_dir, expected_count)
    loaded_count = len(dataset.datasets)

    dataset, skipped_bad_bdf_headers = drop_bad_cached_bdf_headers(dataset)
    dataset, skipped_preprocessing = preprocess_skip_bad(dataset, offline_preprocessors())

    windows = create_release_windows(dataset, **WINDOW_CONFIGS[WINDOW_KIND])
    output_path = output_dir / "r11_test.pkl"
    save_pickle_dataset(windows, output_path, output_dir)

    summary = summarize_windows(windows, SPLIT_NAME, WINDOW_KIND)
    summary["output_path"] = str(output_path)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "release": RELEASE,
        "split": SPLIT_NAME,
        "loaded_recordings": loaded_count,
        "preprocessed_recordings": len(dataset.datasets),
        "skipped_bad_bdf_headers_count": len(skipped_bad_bdf_headers),
        "skipped_preprocessing_count": len(skipped_preprocessing),
        "output": summary,
        "package_versions": package_versions(
            (
                "eegdash",
                "braindecode",
                "mne",
                "numpy",
                "pandas",
                "scikit-learn",
                "torch",
                "tqdm",
            )
        ),
    }

    manifest_path = output_dir / "reload_test_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for test dataset reloads."""
    parser = argparse.ArgumentParser(description="Rebuild only the R11 test split dataset.")
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
    """Run the R11 test dataset reload command."""
    parser = build_parser()
    args = parser.parse_args(argv)

    input_dir = resolve_output_dir(args.input_dir)
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = reload_test_dataset(input_dir, output_dir)
    print("Reloaded test dataset:")
    print(f"- recordings: {manifest['loaded_recordings']:,}")
    print(f"- windows: {manifest['output']['n_windows']:,}")
    print(f"- subjects: {manifest['output']['n_subjects']:,}")
    print(f"- output: {manifest['output']['output_path']}")
    print(f"- manifest: {output_dir / 'reload_test_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
