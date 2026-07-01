"""Benchmark-facing segmentation models."""

from benchmarks.pkg.models.segmentation.attention_sneddy_unet import AttentionSneddyUnet
from benchmarks.pkg.models.segmentation.factorization_unet import FactorizationSneddyUnet
from benchmarks.pkg.models.segmentation.inception import EEGInceptionSeg1D
from benchmarks.pkg.models.segmentation.recurrent_unet import RecurrentSneddyUnet
from benchmarks.pkg.models.segmentation.sneddy_unet import SneddySegUNet1D

__all__ = [
    "AttentionSneddyUnet",
    "EEGInceptionSeg1D",
    "FactorizationSneddyUnet",
    "RecurrentSneddyUnet",
    "SneddySegUNet1D",
]
