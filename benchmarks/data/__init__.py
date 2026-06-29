"""Benchmark dataset helpers."""

from benchmarks.data.regression import FixedWindowDataset
from benchmarks.data.segmentation import TrainCroppingDataset

__all__ = ["FixedWindowDataset", "TrainCroppingDataset"]
