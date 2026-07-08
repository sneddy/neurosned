"""Self-contained ensembling utilities for benchmark stacking analyses."""

from benchmarks.pkg.ensembling.folds import make_subject_balanced_folds
from benchmarks.pkg.ensembling.meta_features import MetaFeatureExtractor
from benchmarks.pkg.ensembling.meta_regressor import HgbMetaRegressor, MetaRegressor, RidgeMetaRegressor
from benchmarks.pkg.ensembling.metrics import mae, nrmse, rmse, regression_metrics
from benchmarks.pkg.ensembling.monotonic import make_monotonic_constraints

__all__ = [
    "HgbMetaRegressor",
    "MetaFeatureExtractor",
    "MetaRegressor",
    "RidgeMetaRegressor",
    "mae",
    "make_subject_balanced_folds",
    "make_monotonic_constraints",
    "nrmse",
    "regression_metrics",
    "rmse",
]
