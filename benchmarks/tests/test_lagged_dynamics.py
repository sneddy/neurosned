"""Contract and numerical tests for lagged-dynamics regression."""

from __future__ import annotations

from pathlib import Path
import unittest

import torch

from benchmarks.pkg.config import load_experiment_config
from benchmarks.pkg.models.regression.lagged_dynamics import LaggedDynamicsRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "benchmarks" / "configs" / "07_lagged_dynamics"
BASELINE_CONFIG = PROJECT_ROOT / "benchmarks" / "configs" / "01_regression_baselines" / "etr_cnn.yaml"


def small_model(**overrides) -> LaggedDynamicsRegressor:
    """Build a fast model that preserves every architectural stage."""
    params = {
        "n_chans": 8,
        "n_times": 40,
        "sfreq": 100,
        "n_outputs": 1,
        "segment_samples": 20,
        "segment_stride": 10,
        "projection_dim": 4,
        "lags": (2, 5),
        "cov_hidden": 12,
        "operator_hidden": 8,
        "lag_hidden": 12,
        "token_dim": 16,
        "temporal_depth": 2,
        "temporal_dilations": (1, 2),
        "temporal_kernel": 3,
        "dropout": 0.0,
        "matrix_eps": 1e-4,
    }
    params.update(overrides)
    return LaggedDynamicsRegressor(**params)


class LaggedDynamicsNumericsTest(unittest.TestCase):
    """Check matrix geometry, tensor contracts, and differentiability."""

    def test_full_model_shapes_spd_and_gradients(self):
        torch.manual_seed(7)
        model = small_model()
        x = torch.randn(3, 8, 40)
        result = model(x, return_dict=True)

        self.assertEqual(result["prediction"].shape, (3, 1))
        self.assertEqual(result["projected"].shape, (3, 4, 40))
        self.assertEqual(result["segments"].shape, (3, 3, 4, 20))
        self.assertEqual(result["covariance"].shape, (3, 3, 4, 4))
        self.assertEqual(result["lagged_correlation"].shape, (3, 3, 2, 4, 4))
        self.assertEqual(result["transition"].shape, (3, 3, 2, 4, 4))
        self.assertEqual(result["segment_tokens"].shape, (3, 3, 16))
        self.assertEqual(result["operator_attention"].shape, (3, 3, 4))
        self.assertEqual(result["segment_attention"].shape, (3, 3))

        covariance = result["covariance"]
        self.assertTrue(torch.allclose(covariance, covariance.transpose(-1, -2), atol=1e-6))
        self.assertTrue(torch.all(torch.linalg.eigvalsh(covariance) > 0))
        self.assertTrue(
            torch.allclose(result["operator_attention"].sum(dim=-1), torch.ones(3, 3), atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(result["segment_attention"].sum(dim=-1), torch.ones(3), atol=1e-6)
        )

        loss = result["prediction"].square().mean()
        loss.backward()
        for parameter in (model.spatial_projection, model.raw_cov_shrinkage, model.raw_ridge):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_branch_controls_keep_scalar_contract(self):
        x = torch.randn(2, 8, 40)

        covariance_only = small_model(include_cross_correlation=False, include_transition=False)
        covariance_result = covariance_only(x, return_dict=True)
        self.assertEqual(covariance_result["prediction"].shape, (2, 1))
        self.assertIsNotNone(covariance_result["covariance"])
        self.assertIsNone(covariance_result["lagged_correlation"])
        self.assertIsNone(covariance_result["transition"])
        self.assertIsNone(covariance_result["operator_attention"])

        lagged_only = small_model(include_covariance=False)
        lagged_result = lagged_only(x, return_dict=True)
        self.assertEqual(lagged_result["prediction"].shape, (2, 1))
        self.assertIsNone(lagged_result["covariance"])
        self.assertIsNone(lagged_result["covariance_vector"])
        self.assertIsNotNone(lagged_result["lagged_correlation"])
        self.assertIsNotNone(lagged_result["transition"])

    def test_invalid_lag_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "smaller than segment_samples"):
            small_model(lags=(20,))

    def test_log_cholesky_is_stable_for_repeated_eigenvalues(self):
        model = small_model()
        covariance = torch.eye(4).view(1, 1, 4, 4).repeat(2, 3, 1, 1)
        covariance.requires_grad_(True)
        vector = model._log_covariance_vector(covariance)
        weights = torch.linspace(0.1, 1.0, vector.shape[-1])
        loss = (vector * weights).sum()
        loss.backward()

        self.assertTrue(torch.isfinite(vector).all())
        self.assertIsNotNone(covariance.grad)
        self.assertTrue(torch.isfinite(covariance.grad).all())


class LaggedDynamicsConfigTest(unittest.TestCase):
    """Ensure new configs differ from the frozen baseline only where intended."""

    def test_configs_load_build_and_match_regression_protocol(self):
        baseline = load_experiment_config(BASELINE_CONFIG)
        expected_names = {
            "lagged_dynamics_full.yaml": "lagged_dynamics_full",
            "lagged_dynamics_covariance_only.yaml": "lagged_dynamics_covariance_only",
            "lagged_dynamics_lagged_only.yaml": "lagged_dynamics_lagged_only",
        }

        for filename, expected_name in expected_names.items():
            with self.subTest(config=filename):
                config = load_experiment_config(CONFIG_DIR / filename)
                self.assertEqual(config.experiment, "07_lagged_dynamics")
                self.assertEqual(config.name, expected_name)
                self.assertEqual(config.task, "regression")
                self.assertEqual(config.data.model_dump(), baseline.data.model_dump())
                self.assertEqual(config.loaders.model_dump(), baseline.loaders.model_dump())
                self.assertEqual(config.optimizer.model_dump(), baseline.optimizer.model_dump())
                self.assertEqual(config.trainer.model_dump(), baseline.trainer.model_dump())
                self.assertEqual(config.evaluation.model_dump(), baseline.evaluation.model_dump())

                model = config.model.build()
                parameter_count = sum(parameter.numel() for parameter in model.parameters())
                self.assertGreater(parameter_count, 2_000_000)
                self.assertLess(parameter_count, 3_100_000)
                self.assertEqual(model(torch.randn(2, 128, 200)).shape, (2, 1))


if __name__ == "__main__":
    unittest.main()
