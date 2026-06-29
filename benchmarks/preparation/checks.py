from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from tqdm import tqdm

from benchmarks.preparation.config import DESCRIPTION_FIELDS, MINI, TASK
from benchmarks.preparation.data import (
    dataset_entry_count,
    iter_recordings,
    load_recording_raw,
    load_release_dataset,
    package_versions,
    recording_label,
)


def check_release_recordings(
    input_dir: Path,
    releases: Iterable[str],
    *,
    task: str = TASK,
    mini: bool = MINI,
    description_fields: Sequence[str] = DESCRIPTION_FIELDS,
) -> dict:
    """Check that cached release recordings can be opened as raw data."""
    release_results = []

    for release in tqdm(
        tuple(releases),
        desc="Releases",
        unit="release",
        file=sys.stdout,
    ):
        dataset = load_release_dataset(
            release=release,
            cache_dir=input_dir,
            task=task,
            mini=mini,
            description_fields=description_fields,
            download=False,
        )
        recordings = iter_recordings(dataset)
        failed_recordings = []

        for index, recording in enumerate(
            tqdm(
                recordings,
                desc=f"{release} runs",
                unit="run",
                leave=True,
                dynamic_ncols=True,
                file=sys.stdout,
            )
        ):
            try:
                load_recording_raw(recording)
            except Exception as exc:
                failed_recordings.append(
                    {
                        **recording_label(recording, release, index),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )

        release_results.append(
            {
                "release": release,
                "dataset_type": type(dataset).__name__,
                "entry_count": dataset_entry_count(dataset),
                "checked_count": len(recordings),
                "ok_count": len(recordings) - len(failed_recordings),
                "failed_count": len(failed_recordings),
                "failed_recordings": failed_recordings,
            }
        )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "task": task,
        "mini": mini,
        "description_fields": list(description_fields),
        "releases": [result["release"] for result in release_results],
        "release_results": release_results,
        "total_entry_count": sum(result["entry_count"] for result in release_results),
        "total_checked_count": sum(result["checked_count"] for result in release_results),
        "total_ok_count": sum(result["ok_count"] for result in release_results),
        "total_failed_count": sum(result["failed_count"] for result in release_results),
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

    manifest_path = input_dir / "check_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
