from __future__ import annotations

import contextlib
import importlib.metadata
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

# Keep third-party imports from writing config/cache files under a read-only home.
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/neurosned-matplotlib")

from eegdash.dataset import EEGChallengeDataset
from tqdm import tqdm

from benchmarks.preparation.config import DESCRIPTION_FIELDS, MINI, TASK


def package_versions(package_names: Sequence[str]) -> dict[str, str | None]:
    """Return installed package versions for the download manifest."""
    versions: dict[str, str | None] = {}
    for package_name in package_names:
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = None
    return versions


def dataset_entry_count(dataset) -> int:
    """Return the number of recordings exposed by an EEG dataset."""
    datasets = getattr(dataset, "datasets", None)
    if datasets is not None:
        return len(datasets)
    return len(dataset)


def iter_recordings(dataset) -> tuple:
    """Return recordings exposed by an EEG dataset."""
    return tuple(getattr(dataset, "datasets", ()))


def recording_label(recording, release: str, index: int) -> dict[str, Any]:
    """Build a compact recording identifier for manifests."""
    label = {"release": release, "index": index}
    description = getattr(recording, "description", None)
    if description is not None:
        for field in ("subject", "session", "run"):
            if field in description:
                label[field] = description[field]
    return label


def load_recording_raw(recording) -> None:
    """Load one recording raw object and release it from memory."""
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            _ = recording.raw
    finally:
        recording.raw = None


def materialize_recordings(dataset, release: str) -> int:
    """Load each recording once so its raw files are present in the cache."""
    recordings = iter_recordings(dataset)
    for recording in tqdm(
        recordings,
        desc=f"{release} runs",
        unit="run",
        leave=True,
        dynamic_ncols=True,
        file=sys.stdout,
    ):
        load_recording_raw(recording)

    return len(recordings)


def load_release_dataset(
    release: str,
    cache_dir: Path,
    *,
    task: str = TASK,
    mini: bool = MINI,
    description_fields: Sequence[str] = DESCRIPTION_FIELDS,
    download: bool = True,
):
    """Create an EEGChallengeDataset for one challenge release."""
    stderr_buffer = io.StringIO()
    stdout_buffer = io.StringIO()
    with contextlib.redirect_stderr(stderr_buffer), contextlib.redirect_stdout(stdout_buffer):
        return EEGChallengeDataset(
            release=release,
            task=task,
            mini=mini,
            description_fields=list(description_fields),
            cache_dir=cache_dir,
            download=download,
        )


def download_releases(
    output_dir: Path,
    releases: Iterable[str],
    *,
    task: str = TASK,
    mini: bool = MINI,
    description_fields: Sequence[str] = DESCRIPTION_FIELDS,
) -> dict:
    """Download selected releases and write a reproducibility manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)

    release_results = []

    for release in tqdm(
        tuple(releases),
        desc="Releases",
        unit="release",
        file=sys.stdout,
    ):
        dataset = load_release_dataset(
            release=release,
            cache_dir=output_dir,
            task=task,
            mini=mini,
            description_fields=description_fields,
        )
        entry_count = dataset_entry_count(dataset)
        materialized_count = materialize_recordings(dataset, release)
        release_results.append(
            {
                "release": release,
                "dataset_type": type(dataset).__name__,
                "entry_count": entry_count,
                "materialized_count": materialized_count,
            }
        )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "task": task,
        "mini": mini,
        "description_fields": list(description_fields),
        "releases": [result["release"] for result in release_results],
        "release_results": release_results,
        "total_entry_count": sum(result["entry_count"] for result in release_results),
        "total_materialized_count": sum(
            result["materialized_count"] for result in release_results
        ),
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

    manifest_path = output_dir / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
