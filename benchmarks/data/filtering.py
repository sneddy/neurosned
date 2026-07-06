"""Dataset views for filtering prepared benchmark windows."""

from __future__ import annotations

import numpy as np
from torch.utils.data import Dataset


class TargetRangeFilter(Dataset):
    """View a dataset through a target-value inclusion range.

    The wrapped dataset is expected to expose `get_metadata()` with a `target`
    column. Items are not modified; this class only remaps visible indices and
    returns filtered metadata so metrics and prediction artefacts stay aligned.
    """

    def __init__(
        self,
        base: Dataset,
        *,
        target_min: float | None = None,
        target_max: float | None = None,
    ):
        if target_min is None and target_max is None:
            raise ValueError("TargetRangeFilter needs at least one of target_min or target_max.")
        if target_min is not None and target_max is not None and float(target_min) > float(target_max):
            raise ValueError("target_min must be <= target_max.")
        if not hasattr(base, "get_metadata"):
            raise TypeError("TargetRangeFilter requires a dataset with get_metadata().")

        metadata = base.get_metadata()
        if "target" not in metadata:
            raise KeyError("TargetRangeFilter requires a 'target' column in metadata.")

        target = metadata["target"]
        mask = target.notna()
        if target_min is not None:
            mask &= target >= float(target_min)
        if target_max is not None:
            mask &= target <= float(target_max)

        self.base = base
        self.target_min = None if target_min is None else float(target_min)
        self.target_max = None if target_max is None else float(target_max)
        self.indices = np.flatnonzero(mask.to_numpy())
        self._metadata = metadata.iloc[self.indices].reset_index(drop=True)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.base[int(self.indices[idx])]

    def get_metadata(self):
        return self._metadata.copy()


def apply_target_range_filter(
    dataset: Dataset,
    *,
    target_min: float | None = None,
    target_max: float | None = None,
):
    """Return `dataset` unchanged unless a target range is configured."""
    if target_min is None and target_max is None:
        return dataset
    return TargetRangeFilter(dataset, target_min=target_min, target_max=target_max)
