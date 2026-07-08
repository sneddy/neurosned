"""Monotonic-constraint helpers for the challenge-style HGB stacker."""

from __future__ import annotations

import re

import numpy as np


def is_time_feature_name(name: str) -> bool:
    """Return whether a meta-feature name encodes an event-time coordinate."""
    return (
        name.endswith("t_hard")
        or ("t_abs_temp" in name)
        or bool(re.search(r"q\d+_temp", name))
    )


def make_monotonic_constraints(feature_names, time_dir: int = 1) -> np.ndarray:
    """Build HGB monotonic constraints from feature names.

    This mirrors the notebook recipe: event-time features are constrained in
    the positive direction, while shape/confidence/logit features are left
    unconstrained.
    """
    constraints = []
    for name in feature_names:
        constraints.append(int(time_dir) if is_time_feature_name(str(name)) else 0)
    return np.asarray(constraints, dtype=int)
