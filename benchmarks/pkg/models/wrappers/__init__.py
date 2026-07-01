"""Model wrappers used by benchmark configs."""

from benchmarks.pkg.models.wrappers.normalization import WithStdPerSample, build_model

__all__ = ["WithStdPerSample", "build_model"]
