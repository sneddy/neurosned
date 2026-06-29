from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "release_data"
DEFAULT_SPLIT_OUTPUT_DIR = PROJECT_ROOT / "data" / "new_validation"
DEFAULT_RELEASES = tuple(f"R{idx}" for idx in range(1, 12))

TASK = "contrastChangeDetection"
MINI = False
DESCRIPTION_FIELDS = (
    "subject",
    "session",
    "run",
    "task",
    "age",
    "gender",
    "sex",
    "p_factor",
)
ANCHOR = "stimulus_anchor"
SFREQ = 100
WINDOW_METADATA_KEYS = (
    "target",
    "rt_from_stimulus",
    "rt_from_trialstart",
    "stimulus_onset",
    "response_onset",
    "correct",
    "response_type",
)
WINDOW_CONFIGS = {
    "2sec": {
        "shift_after_stim": 0.5,
        "window_len_s": 2.0,
        "epoch_len_s": 2.0,
    },
    "5sec": {
        "shift_after_stim": 0.0,
        "window_len_s": 5.0,
        "epoch_len_s": 5.0,
    },
}

_RELEASE_RE = re.compile(r"^R([1-9]|1[0-1])$")


def parse_releases(raw_releases: Iterable[str] | None) -> tuple[str, ...]:
    """Normalize and validate release labels supplied by the CLI."""
    if raw_releases is None:
        return DEFAULT_RELEASES

    releases: list[str] = []
    for raw_value in raw_releases:
        for value in raw_value.split(","):
            release = value.strip().upper()
            if release:
                releases.append(release)

    if not releases:
        raise ValueError("At least one release must be provided.")

    invalid = [release for release in releases if _RELEASE_RE.fullmatch(release) is None]
    if invalid:
        expected = ", ".join(DEFAULT_RELEASES)
        raise ValueError(f"Invalid release label(s): {', '.join(invalid)}. Expected one of: {expected}.")

    return tuple(dict.fromkeys(releases))


def resolve_output_dir(output_dir: str | Path | None) -> Path:
    """Resolve the raw release cache directory."""
    if output_dir is None:
        return DEFAULT_OUTPUT_DIR

    path = Path(output_dir).expanduser()
    return path.resolve()


def split_configs(output_dir: Path = DEFAULT_SPLIT_OUTPUT_DIR) -> tuple[dict, ...]:
    """Return release-based split definitions and output paths."""
    return (
        {
            "name": "r1_r8_train",
            "releases": tuple(f"R{idx}" for idx in range(1, 9)),
            "outputs": {
                "2sec": output_dir / "r1_r8_train.pkl",
                "5sec": output_dir / "r1_r8_train_5sec.pkl",
            },
        },
        {
            "name": "r9_r10_val",
            "releases": ("R9", "R10"),
            "outputs": {
                "2sec": output_dir / "r9_r10_val.pkl",
                "5sec": output_dir / "r9_r10_val_5sec.pkl",
            },
        },
        {
            "name": "r11_test",
            "releases": ("R11",),
            "outputs": {
                "2sec": output_dir / "r11_test.pkl",
            },
        },
    )
