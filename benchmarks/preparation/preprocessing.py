from __future__ import annotations

import contextlib
import gc
import io
import json
import pickle
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from benchmarks.preparation.config import (
    ANCHOR,
    DESCRIPTION_FIELDS,
    MINI,
    SFREQ,
    TASK,
    WINDOW_CONFIGS,
    WINDOW_METADATA_KEYS,
    split_configs,
)
from benchmarks.preparation.data import load_release_dataset, package_versions

from braindecode.datasets.base import BaseConcatDataset
from braindecode.preprocessing import Preprocessor, create_windows_from_events, preprocess
from eegdash.hbn.windows import (
    add_aux_anchors,
    add_extras_columns,
    annotate_trials_with_target,
    keep_only_recordings_with,
)

warnings.filterwarnings(
    "ignore", message="Omitted .* annotation.*outside data range", category=RuntimeWarning
)


def _read_json(path: Path) -> dict | None:
    """Read a JSON file if it exists."""
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _checked_counts(input_dir: Path) -> dict[str, int]:
    """Read per-release counts from check_manifest.json."""
    manifest = _read_json(input_dir / "check_manifest.json")
    if manifest is None:
        return {}

    counts = {}
    for row in manifest.get("release_results", []):
        release = row.get("release")
        count = row.get("ok_count") or row.get("checked_count") or row.get("entry_count")
        if release is not None and count is not None:
            counts[str(release)] = int(count)
    return counts


def prepare_full_dataset(
    data_dir: Path,
    release_list: Sequence[str],
    *,
    task: str = TASK,
    mini: bool = MINI,
    description_fields: Sequence[str] = DESCRIPTION_FIELDS,
    checked_counts: dict[str, int] | None = None,
) -> BaseConcatDataset:
    """Load selected EEGChallenge releases as one Braindecode dataset."""
    all_datasets_list = []

    for release in tqdm(release_list, desc="Loading releases", file=sys.stdout):
        ds = load_release_dataset(
            release=release,
            cache_dir=data_dir,
            task=task,
            mini=mini,
            description_fields=description_fields,
            download=False,
        )
        recordings = list(getattr(ds, "datasets", ()))
        print(f"{release} recordings: {len(recordings):,}")

        checked_count = (checked_counts or {}).get(release)
        if checked_count is not None and len(recordings) != checked_count:
            print(
                f"WARNING: {release} local cache has {len(recordings):,} recordings; "
                f"check_manifest.json has {checked_count:,}. Continuing with local cache."
            )

        all_datasets_list.extend(recordings)

    return BaseConcatDataset(all_datasets_list)


def _recording_label(ds) -> str:
    """Return a human-readable recording label."""
    desc = getattr(ds, "description", None)
    if desc is None:
        return str(getattr(ds, "filecache", "<unknown>"))

    parts = []
    for key in ("subject", "session", "run", "task"):
        if key in desc and pd.notna(desc[key]):
            parts.append(f"{key}={desc[key]}")
    return ", ".join(parts) or str(getattr(ds, "filecache", "<unknown>"))


def _bad_cached_bdf_reason(ds) -> str | None:
    """Return a reason when a cached BDF header is clearly invalid."""
    path = getattr(ds, "filecache", None)
    if path is None:
        return None

    path = Path(path)
    if path.suffix.lower() != ".bdf" or not path.exists():
        return None

    size = path.stat().st_size
    if size < 256:
        return f"cached BDF is too small ({size} bytes)"

    with path.open("rb") as f:
        header = f.read(256)

    if not header.strip(bytes([0])):
        return "first 256 BDF header bytes are all null"

    header_size = header[184:192].decode("latin-1", errors="ignore").strip()
    if not header_size.isdigit():
        return f"invalid BDF header-size field: {header_size!r}"

    if size < int(header_size):
        return f"cached BDF is smaller than its header ({size} < {header_size})"

    return None


def _print_head(df: pd.DataFrame, max_rows: int = 20) -> None:
    """Print a compact dataframe preview for CLI runs."""
    if df.empty:
        return
    print(df.head(max_rows).to_string(index=False))
    if len(df) > max_rows:
        print(f"... {len(df) - max_rows} more row(s) not shown")


def drop_bad_cached_bdf_headers(concat_ds: BaseConcatDataset) -> tuple[BaseConcatDataset, pd.DataFrame]:
    """Drop recordings with obviously broken cached BDF headers."""
    kept = []
    skipped = []

    for ds in tqdm(concat_ds.datasets, desc="Checking cached BDF headers", file=sys.stdout):
        reason = _bad_cached_bdf_reason(ds)
        if reason is None:
            kept.append(ds)
        else:
            skipped.append(
                {
                    "recording": _recording_label(ds),
                    "path": str(getattr(ds, "filecache", "")),
                    "reason": reason,
                }
            )

    skipped_df = pd.DataFrame(skipped)
    if skipped_df.empty:
        print("No invalid cached BDF headers found.")
    else:
        print(f"Dropping {len(skipped_df)} recording(s) with invalid cached BDF headers.")
        _print_head(skipped_df)

    if not kept:
        raise RuntimeError("All recordings were dropped by the cached BDF header check.")

    return BaseConcatDataset(kept, target_transform=getattr(concat_ds, "target_transform", None)), skipped_df


def _is_skippable_recording_error(exc: Exception) -> bool:
    """Return whether preprocessing can skip a recording error."""
    msg = str(exc).lower()
    skippable_fragments = (
        "bad bdf",
        "bad edf",
        "data file unreadable",
        "error reading",
        "file not found",
        "no such file",
        "could not load raw data",
    )
    return any(fragment in msg for fragment in skippable_fragments)


def preprocess_skip_bad(
    concat_ds: BaseConcatDataset,
    preprocessors: Sequence[Preprocessor],
) -> tuple[BaseConcatDataset, pd.DataFrame]:
    """Preprocess recordings while skipping known unreadable files."""
    kept = []
    skipped = []

    for ds in tqdm(concat_ds.datasets, desc="Preprocessing recordings", file=sys.stdout):
        try:
            single_ds = BaseConcatDataset([ds])
            preprocess(single_ds, preprocessors, n_jobs=1)
            kept.append(single_ds.datasets[0])
        except Exception as exc:
            if not _is_skippable_recording_error(exc):
                raise
            if hasattr(ds, "_raw"):
                ds._raw = None
            skipped.append(
                {
                    "recording": _recording_label(ds),
                    "path": str(getattr(ds, "filecache", "")),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    skipped_df = pd.DataFrame(skipped)
    if skipped_df.empty:
        print("Preprocessed all recordings successfully.")
    else:
        print(f"Skipped {len(skipped_df)} recording(s) during preprocessing.")
        _print_head(skipped_df)

    if not kept:
        raise RuntimeError("All recordings failed during preprocessing.")

    return BaseConcatDataset(kept, target_transform=getattr(concat_ds, "target_transform", None)), skipped_df


def offline_preprocessors() -> list[Preprocessor]:
    """Return the original offline preprocessing transforms."""
    return [
        Preprocessor(
            annotate_trials_with_target,
            target_field="rt_from_stimulus",
            epoch_length=2.0,
            require_stimulus=True,
            require_response=True,
            apply_on_array=False,
        ),
        Preprocessor(add_aux_anchors, apply_on_array=False),
    ]


def build_preprocessed_release_dataset(
    data_dir: Path,
    release_list: Sequence[str],
    checked_counts: dict[str, int] | None = None,
) -> tuple[BaseConcatDataset, pd.DataFrame, pd.DataFrame, int]:
    """Load releases and apply the original offline preprocessing steps."""
    print(f"Loading releases: {list(release_list)}")
    dataset_ccd = prepare_full_dataset(
        data_dir=data_dir,
        release_list=release_list,
        checked_counts=checked_counts,
    )
    loaded_count = len(dataset_ccd.datasets)
    print(f"Loaded recordings: {loaded_count:,}")

    dataset_ccd, skipped_bad_bdf_headers = drop_bad_cached_bdf_headers(dataset_ccd)
    dataset_ccd, skipped_preprocessing = preprocess_skip_bad(dataset_ccd, offline_preprocessors())

    return dataset_ccd, skipped_bad_bdf_headers, skipped_preprocessing, loaded_count


def remove_late_anchors(
    dataset: BaseConcatDataset,
    anchor: str = ANCHOR,
    shift: float = 0.5,
    winlen: float = 2.0,
) -> BaseConcatDataset:
    """Remove anchors that cannot fit inside the requested window."""
    log = []
    for idx, bd in enumerate(dataset.datasets):
        raw = bd.raw
        ann = raw.annotations
        recording_len_s = float(raw.times[-1])

        desc = np.asarray(ann.description, dtype=str)
        onset = np.asarray(ann.onset, dtype=float)

        is_anchor = desc == anchor
        n_before = int(is_anchor.sum())
        too_late = is_anchor & ((onset + shift + winlen) > recording_len_s + 1e-9)
        n_removed = int(too_late.sum())

        if n_removed > 0:
            log.append(
                {
                    "idx": idx,
                    "subject": bd.description.get("subject", "NA"),
                    "run": bd.description.get("run", "NA"),
                    "release": bd.description.get("release_number", "NA"),
                    "n_anchors_before": n_before,
                    "n_removed": n_removed,
                    "max_onset_removed": float(onset[too_late].max()),
                    "rec_len_s": recording_len_s,
                }
            )

        raw.set_annotations(ann[np.where(~too_late)[0]], verbose=False)

    if log:
        df_log = pd.DataFrame(log).sort_values(["n_removed", "n_anchors_before"], ascending=False)
        print(f"Removed anchors in total: {int(df_log['n_removed'].sum())}")
        print(f"Records with removed anchors: {len(df_log)}")
        _print_head(df_log)

    return keep_only_recordings_with(anchor, dataset)


def create_release_windows(
    preprocessed_dataset: BaseConcatDataset,
    shift_after_stim: float,
    window_len_s: float,
    epoch_len_s: float,
) -> BaseConcatDataset:
    """Create Braindecode event windows for a preprocessed release dataset."""
    dataset_with_anchors = keep_only_recordings_with(ANCHOR, preprocessed_dataset)
    dataset_with_anchors = remove_late_anchors(
        dataset_with_anchors,
        anchor=ANCHOR,
        shift=shift_after_stim,
        winlen=window_len_s,
    )

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Dropping extra columns that conflict with windowing metadata: \{'target'\}",
            category=UserWarning,
            module=r"braindecode\.preprocessing\.windowers",
        )
        windows = create_windows_from_events(
            dataset_with_anchors,
            mapping={ANCHOR: 0},
            trial_start_offset_samples=int(shift_after_stim * SFREQ),
            trial_stop_offset_samples=int((shift_after_stim + window_len_s) * SFREQ),
            window_size_samples=int(epoch_len_s * SFREQ),
            window_stride_samples=SFREQ,
            preload=True,
        )

    windows = add_extras_columns(
        windows,
        dataset_with_anchors,
        desc=ANCHOR,
        keys=WINDOW_METADATA_KEYS,
    )
    return windows


def _format_path(path: Path, root_dir: Path) -> str:
    """Format a path relative to a root when possible."""
    try:
        return str(path.relative_to(root_dir))
    except ValueError:
        return str(path)


def save_pickle_dataset(dataset: BaseConcatDataset, output_path: Path, root_dir: Path) -> None:
    """Write a prepared split dataset as a pickle file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(dataset, f)
    print(f"Saved {_format_path(output_path, root_dir)} ({len(dataset):,} windows)")


def _json_float(value) -> float | None:
    """Convert pandas/numpy scalar values to JSON-safe floats."""
    if pd.isna(value):
        return None
    return float(value)


def summarize_windows(dataset: BaseConcatDataset, split_name: str, window_kind: str) -> dict:
    """Summarize a prepared window dataset for the manifest."""
    metadata = dataset.get_metadata()
    return {
        "split": split_name,
        "window_kind": window_kind,
        "n_windows": int(len(dataset)),
        "n_subjects": int(metadata["subject"].nunique()) if "subject" in metadata else None,
        "target_min": _json_float(metadata["target"].min()) if "target" in metadata else None,
        "target_max": _json_float(metadata["target"].max()) if "target" in metadata else None,
    }


def prepare_splitted_datasets(input_dir: Path, output_dir: Path) -> dict:
    """Build release-based benchmark split pickle files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    build_summary_rows = []
    split_results = []
    checked_counts = _checked_counts(input_dir)

    for split in split_configs(output_dir):
        print("=" * 100)
        print(f"Building {split['name']} from releases: {list(split['releases'])}")

        preprocessed_dataset, skipped_bad_bdf_headers, skipped_preprocessing, loaded_count = (
            build_preprocessed_release_dataset(
                input_dir,
                split["releases"],
                checked_counts=checked_counts,
            )
        )
        output_summaries = []

        for window_kind, output_path in split["outputs"].items():
            windows = create_release_windows(
                preprocessed_dataset,
                **WINDOW_CONFIGS[window_kind],
            )
            save_pickle_dataset(windows, output_path, output_dir)

            summary = summarize_windows(windows, split["name"], window_kind)
            summary["output_path"] = str(output_path)
            build_summary_rows.append(summary)
            output_summaries.append(summary)

            del windows
            gc.collect()

        split_results.append(
            {
                "split": split["name"],
                "releases": list(split["releases"]),
                "loaded_recordings": loaded_count,
                "preprocessed_recordings": len(preprocessed_dataset.datasets),
                "skipped_bad_bdf_headers_count": len(skipped_bad_bdf_headers),
                "skipped_preprocessing_count": len(skipped_preprocessing),
                "outputs": output_summaries,
                "skipped_bad_bdf_headers": skipped_bad_bdf_headers.to_dict(orient="records"),
                "skipped_preprocessing": skipped_preprocessing.to_dict(orient="records"),
            }
        )

        del preprocessed_dataset
        gc.collect()

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "task": TASK,
        "mini": MINI,
        "description_fields": list(DESCRIPTION_FIELDS),
        "window_configs": WINDOW_CONFIGS,
        "split_results": split_results,
        "build_summary": build_summary_rows,
        "total_windows": sum(row["n_windows"] for row in build_summary_rows),
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

    manifest_path = output_dir / "prepare_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
