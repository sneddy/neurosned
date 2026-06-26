#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any


os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/neurosned-matplotlib")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from benchmarks.preparation.config import DEFAULT_OUTPUT_DIR, DEFAULT_SPLIT_OUTPUT_DIR
from benchmarks.preparation.data import (
    dataset_entry_count,
    iter_recordings,
    load_release_dataset,
    package_versions,
)
from benchmarks.preparation.preprocessing import prepare_full_dataset


RELEASE = "R11"
SPLIT = "r11_test"


def read_json(path: Path) -> dict | None:
    """Read a JSON file if it exists."""
    if not path.exists():
        return None
    return json.loads(path.read_text())


def release_manifest_row(manifest: dict | None, release: str) -> dict | None:
    """Find one release row in a release manifest."""
    if manifest is None:
        return None
    for row in manifest.get("release_results", []):
        if row.get("release") == release:
            return row
    return None


def split_manifest_row(manifest: dict | None, split_name: str) -> dict | None:
    """Find one split row in a prepare manifest."""
    if manifest is None:
        return None
    for row in manifest.get("split_results", []):
        if row.get("split") == split_name:
            return row
    return None


def desc_value(description: Any, key: str) -> Any:
    """Read one description field from dict-like metadata."""
    if description is None:
        return None
    if hasattr(description, "get"):
        return description.get(key)
    try:
        return description[key]
    except (KeyError, TypeError):
        return None


def normalize_value(value: Any) -> str:
    """Normalize metadata values for set comparisons."""
    if value is None or pd.isna(value):
        return ""
    return str(value)


def recording_identity(description: Any) -> tuple[str, str, str, str]:
    """Build a stable recording identity from metadata."""
    return tuple(
        normalize_value(desc_value(description, key))
        for key in ("subject", "session", "run", "task")
    )


def recording_table(recordings: tuple) -> pd.DataFrame:
    """Convert recordings to a compact metadata table."""
    rows = []
    for index, recording in enumerate(recordings):
        description = getattr(recording, "description", None)
        rows.append(
            {
                "index": index,
                "subject": normalize_value(desc_value(description, "subject")),
                "session": normalize_value(desc_value(description, "session")),
                "run": normalize_value(desc_value(description, "run")),
                "task": normalize_value(desc_value(description, "task")),
                "filecache": str(getattr(recording, "filecache", "")),
            }
        )
    return pd.DataFrame(rows)


def summarize_table(name: str, table: pd.DataFrame) -> None:
    """Print counts for a recording or window metadata table."""
    print(f"\n{name}")
    print(f"rows: {len(table):,}")
    for column in ("subject", "session", "run", "task"):
        if column in table:
            print(f"unique {column}: {table[column].nunique(dropna=True):,}")
    if "session" in table and not table.empty:
        print("sessions:")
        print(table["session"].value_counts(dropna=False).sort_index().to_string())


def summarize_cache_tree(input_dir: Path) -> None:
    """Print local R11 cache file counts without opening raw data."""
    release_dir = input_dir / "EEG2025r11"
    print("\nCache tree")
    if not release_dir.exists():
        print(f"missing: {release_dir}")
        return

    files = [path for path in release_dir.rglob("*") if path.is_file()]
    suffix_counts = Counter(path.suffix.lower() or "<no suffix>" for path in files)
    print(f"path: {release_dir}")
    print(f"files: {len(files):,}")
    print("suffixes:")
    for suffix, count in sorted(suffix_counts.items()):
        print(f"  {suffix}: {count:,}")


def load_pickle_metadata(path: Path) -> pd.DataFrame | None:
    """Load metadata from an existing prepared pickle dataset."""
    if not path.exists():
        print(f"\nPrepared pickle missing: {path}")
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        with path.open("rb") as f:
            dataset = pickle.load(f)
        metadata = dataset.get_metadata()
    return metadata


def compare_recordings_to_pickle(recordings_df: pd.DataFrame, metadata: pd.DataFrame) -> None:
    """Compare raw recording identities with prepared pickle metadata."""
    metadata_records = metadata[["subject", "session", "run", "task"]].drop_duplicates()
    raw_ids = {
        tuple(row)
        for row in recordings_df[["subject", "session", "run", "task"]].itertuples(index=False, name=None)
    }
    pickle_ids = {
        tuple(normalize_value(value) for value in row)
        for row in metadata_records.itertuples(index=False, name=None)
    }

    missing = sorted(raw_ids - pickle_ids)
    extra = sorted(pickle_ids - raw_ids)

    print("\nRaw recordings vs prepared pickle")
    print(f"raw unique recording ids: {len(raw_ids):,}")
    print(f"pickle unique recording ids: {len(pickle_ids):,}")
    print(f"missing from pickle: {len(missing):,}")
    print(f"extra in pickle: {len(extra):,}")

    if missing:
        print("first missing:")
        for row in missing[:20]:
            print(f"  subject={row[0]} session={row[1]} run={row[2]} task={row[3]}")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for R11 diagnostics."""
    parser = argparse.ArgumentParser(description="Diagnose R11 preparation count mismatch.")
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory with the raw EEGChallenge cache. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_SPLIT_OUTPUT_DIR),
        help="Directory with prepared split pickle files. Defaults to %(default)s.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run read-only R11 diagnostics."""
    args = build_parser().parse_args(argv)
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    print("Package versions")
    for package, version in package_versions(("eegdash", "braindecode", "mne", "numpy", "pandas")).items():
        print(f"{package}: {version}")

    download_manifest = read_json(input_dir / "download_manifest.json")
    check_manifest = read_json(input_dir / "check_manifest.json")
    prepare_manifest = read_json(output_dir / "prepare_manifest.json")

    print("\nManifest rows")
    print("download:", release_manifest_row(download_manifest, RELEASE))
    print("check:", release_manifest_row(check_manifest, RELEASE))
    print("prepare:", split_manifest_row(prepare_manifest, SPLIT))

    summarize_cache_tree(input_dir)

    print("\nReload EEGChallengeDataset")
    try:
        dataset = load_release_dataset(RELEASE, input_dir)
        recordings = iter_recordings(dataset)
        print(f"dataset type: {type(dataset).__name__}")
        print(f"dataset entry count: {dataset_entry_count(dataset):,}")
        print(f"recordings: {len(recordings):,}")
        recordings_df = recording_table(recordings)
        summarize_table("Reloaded R11 recordings", recordings_df)
    except Exception as exc:
        print(f"failed to reload R11 metadata: {type(exc).__name__}: {exc}")
        return 1

    print("\nReload through prepare_full_dataset")
    try:
        concat = prepare_full_dataset(input_dir, [RELEASE])
        print(f"concat recordings: {len(concat.datasets):,}")
    except Exception as exc:
        print(f"failed to reload through prepare_full_dataset: {type(exc).__name__}: {exc}")

    pickle_path = output_dir / "r11_test.pkl"
    metadata = load_pickle_metadata(pickle_path)
    if metadata is not None:
        summarize_table("Prepared r11_test.pkl metadata", metadata)
        compare_recordings_to_pickle(recordings_df, metadata)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
