"""Subject-disjoint folds for meta-model development scores."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_subject_balanced_folds(
    metadata: pd.DataFrame,
    *,
    subject_col: str = "subject",
    target_col: str = "target",
    n_folds: int = 5,
    n_iter: int = 10_000,
    seed: int = 42,
) -> np.ndarray:
    """Assign each subject to one fold while balancing fold mean targets.

    This mirrors the notebook procedure used during the original competition
    ensembling work: subject-level fold assignments are shuffled repeatedly and
    the assignment with the smallest standard deviation of per-fold mean RT is
    retained.
    """
    if subject_col not in metadata.columns:
        raise ValueError(f"Missing subject column: {subject_col!r}")
    if target_col not in metadata.columns:
        raise ValueError(f"Missing target column: {target_col!r}")

    subject_means = metadata.groupby(subject_col, sort=True)[target_col].mean()
    subjects = subject_means.index.to_numpy()
    n_subjects = len(subjects)
    if n_subjects < n_folds:
        raise ValueError(f"Need at least {n_folds} subjects for {n_folds} folds, got {n_subjects}.")

    base = np.arange(n_subjects) % int(n_folds)
    rng = np.random.default_rng(seed)
    best_std = float("inf")
    best_subject_to_fold: dict[object, int] | None = None

    # Include the deterministic base assignment as a valid candidate.
    for iteration in range(max(int(n_iter), 1) + 1):
        assignment = base.copy()
        if iteration > 0:
            rng.shuffle(assignment)
        subject_to_fold = dict(zip(subjects, assignment))
        folds = metadata[subject_col].map(subject_to_fold).to_numpy()
        means = metadata.assign(_fold=folds).groupby("_fold")[target_col].mean()
        fold_std = float(means.std())
        if fold_std < best_std:
            best_std = fold_std
            best_subject_to_fold = {k: int(v) for k, v in subject_to_fold.items()}

    if best_subject_to_fold is None:
        raise RuntimeError("Failed to build subject-balanced folds.")
    folds = metadata[subject_col].map(best_subject_to_fold).to_numpy()
    if np.any(pd.isna(folds)):
        raise RuntimeError("Some subjects were not assigned to folds.")
    return folds.astype(int, copy=False)
