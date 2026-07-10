"""Benchmark-facing regression models."""

from benchmarks.pkg.models.regression.msa_cnn import (
    MSACNN,
    MSGCNN,
    MultiscaleSegmentAttentionHead,
    MultiscaleSegmentGatedHead,
)
from benchmarks.pkg.models.regression.msp_cnn import MSPCNN
from benchmarks.pkg.models.regression.etr_cnn import ETRCNN

__all__ = [
    "MSACNN",
    "MSGCNN",
    "MSPCNN",
    "ETRCNN",
    "MultiscaleSegmentAttentionHead",
    "MultiscaleSegmentGatedHead",
]
