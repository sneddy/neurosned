"""Tests for raw and dual-view lagged-dynamics regressors."""

from __future__ import annotations

from pathlib import Path
import unittest

import torch

from benchmarks.pkg.config import load_experiment_config
from benchmarks.pkg.models.regression.dual_view_lagged_dynamics import (
    DualViewLaggedDynamicsRegressor,
    RawTemporalRegressor,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "benchmarks" / "configs" / "08_dual_view_lagged_dynamics"
BASELINE_CONFIG = PROJECT_ROOT / "benchmarks" / "configs" / "01_regression_baselines" / "etr_cnn.yaml"


def shared_small_params() -> dict:
    """Return a fast configuration with all temporal stages present."""
    return {
        "n_chans": 8,
        "n_times": 40,
        "sfreq": 100,
        "n_outputs": 1,
        "segment_samples": 20,
        "segment_stride": 10,
        "raw_width": 8,
        "raw_depth": 2,
        "raw_dilations": (1, 2),
        "raw_kernel": 5,
        "token_dim": 16,
        "temporal_depth": 2,
        "temporal_dilations": (1, 2),
        "temporal_kernel": 3,
        "dropout": 0.0,
        "matrix_eps": 1e-4,
    }


def small_dual(**overrides) -> DualViewLaggedDynamicsRegressor:
    """Build a compact dual-view model."""
    params = {
        **shared_small_params(),
        "projection_dim": 4,
        "lags": (2, 5),
        "cov_hidden": 12,
        "operator_hidden": 8,
        "lag_hidden": 12,
    }
    params.update(overrides)
    return DualViewLaggedDynamicsRegressor(**params)


class DualViewNumericsTest(unittest.TestCase):
    """Check view alignment, attention, scalar output, and optimization."""

    def test_dual_view_shapes_and_balanced_initial_fusion(self):
        torch.manual_seed(11)
        model = small_dual()
        result = model(torch.randn(3, 8, 40), return_dict=True)

        self.assertEqual(result["prediction"].shape, (3, 1))
        self.assertEqual(result["raw_feature_map"].shape, (3, 8, 40))
        self.assertEqual(result["raw_segment_attention"].shape, (3, 3, 20))
        self.assertEqual(result["raw_segment_tokens"].shape, (3, 3, 16))
        self.assertEqual(result["matrix_segment_tokens"].shape, (3, 3, 16))
        self.assertEqual(result["segment_tokens"].shape, (3, 3, 16))
        self.assertEqual(result["modality_attention"].shape, (3, 3, 2))
        self.assertTrue(
            torch.allclose(
                result["raw_segment_attention"].sum(dim=-1), torch.ones(3, 3), atol=1e-6
            )
        )
        self.assertTrue(
            torch.allclose(result["modality_attention"], torch.full((3, 3, 2), 0.5), atol=1e-6)
        )

    def test_dual_view_updates_both_branches(self):
        torch.manual_seed(12)
        model = small_dual()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = torch.randn(4, 8, 40)
        target = torch.ones(4, 1)

        for _ in range(2):
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(x), target)
            loss.backward()
            optimizer.step()

        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), target)
        loss.backward()
        self.assertGreater(model.spatial_projection.grad.abs().sum().item(), 0.0)
        self.assertGreater(model.raw_encoder.stem.proj.weight.grad.abs().sum().item(), 0.0)
        self.assertTrue(all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ))

    def test_covariance_only_dual_control(self):
        model = small_dual(include_cross_correlation=False, include_transition=False)
        result = model(torch.randn(2, 8, 40), return_dict=True)
        self.assertIsNotNone(result["covariance"])
        self.assertIsNone(result["lagged_correlation"])
        self.assertIsNone(result["transition"])
        self.assertEqual(result["prediction"].shape, (2, 1))

    def test_raw_only_shapes(self):
        model = RawTemporalRegressor(**shared_small_params())
        result = model(torch.randn(2, 8, 40), return_dict=True)
        self.assertEqual(result["prediction"].shape, (2, 1))
        self.assertEqual(result["raw_segment_tokens"].shape, (2, 3, 16))
        self.assertEqual(result["raw_segment_attention"].shape, (2, 3, 20))
        self.assertEqual(result["segment_attention"].shape, (2, 3))


class DualViewConfigTest(unittest.TestCase):
    """Check that group 08 is protocol-matched but model-independent."""

    def test_configs_load_build_and_match_baseline_protocol(self):
        baseline = load_experiment_config(BASELINE_CONFIG)
        expected = {
            "dual_view_full.yaml": ("dual_view_full", 3_845_962),
            "dual_view_covariance_only.yaml": ("dual_view_covariance_only", 3_335_366),
            "raw_view_only.yaml": ("raw_view_only", 2_347_523),
        }
        for filename, (name, parameter_count) in expected.items():
            with self.subTest(config=filename):
                config = load_experiment_config(CONFIG_DIR / filename)
                self.assertEqual(config.experiment, "08_dual_view_lagged_dynamics")
                self.assertEqual(config.name, name)
                self.assertEqual(config.data.model_dump(), baseline.data.model_dump())
                self.assertEqual(config.loaders.model_dump(), baseline.loaders.model_dump())
                self.assertEqual(config.optimizer.model_dump(), baseline.optimizer.model_dump())
                self.assertEqual(config.trainer.model_dump(), baseline.trainer.model_dump())
                self.assertEqual(config.evaluation.model_dump(), baseline.evaluation.model_dump())

                model = config.model.build()
                self.assertEqual(sum(p.numel() for p in model.parameters()), parameter_count)
                self.assertEqual(model(torch.randn(2, 128, 200)).shape, (2, 1))


if __name__ == "__main__":
    unittest.main()
