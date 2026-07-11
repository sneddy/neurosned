"""Bootstrap confidence intervals for evaluation metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def subject_bootstrap_nrmse(predictions, metadata, *, n_samples: int, resampling_seed: int) -> dict[str, Any]:
    """Return a subject-level bootstrap interval for NRMSE."""
    frame = pd.DataFrame(
        {
            "subject": metadata["subject"].to_numpy() if "subject" in metadata else np.arange(len(metadata)),
            "target": metadata["target"].to_numpy(),
            "prediction": np.asarray(predictions),
        }
    )
    subjects = frame["subject"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(resampling_seed)
    values = []
    grouped = {subject: group for subject, group in frame.groupby("subject", sort=False)}

    for _ in range(n_samples):
        sampled_subjects = rng.choice(subjects, size=len(subjects), replace=True)
        sample = pd.concat([grouped[subject] for subject in sampled_subjects], ignore_index=True)
        target = sample["target"].to_numpy()
        prediction = sample["prediction"].to_numpy()
        rmse = float(np.sqrt(np.mean((prediction - target) ** 2)))
        denominator = float(np.std(target, ddof=1)) if len(target) > 1 else 0.0
        values.append(rmse / denominator if denominator else rmse)

    values = np.asarray(values)
    return {
        "method": "subject_bootstrap",
        "n_samples": int(n_samples),
        "resampling_seed": int(resampling_seed),
        "n_subjects": int(len(subjects)),
        "n_rows": int(len(frame)),
        "nrmse_mean": float(np.mean(values)),
        "nrmse_ci_low": float(np.quantile(values, 0.025)),
        "nrmse_ci_high": float(np.quantile(values, 0.975)),
    }
