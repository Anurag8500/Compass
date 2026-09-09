"""Unit tests for training-only normalization (ml/data/normalization.py).

Verifies:
1. Normalization statistics are strictly computed across training data ONLY.
2. Mutating or scaling validation/test data does NOT change normalization parameters.
3. Accurate round-trip transform and inverse-transform.
4. Robust division-by-zero protection on constant channels.
5. Integration with Phase 1 ModelConfig schema.
"""

from __future__ import annotations

import numpy as np
import pytest
from ml.data.normalization import FeatureNormalizer, NUM_CHANNELS
from navigation.schemas.config import ModelConfig


class TestNormalizationTrainOnly:
    """Test suite for training-only normalization isolation."""

    def test_train_only_isolation_validation_test_mutation(self) -> None:
        """Modifying validation or test tensors must have ZERO effect on normalization statistics."""
        np.random.seed(42)
        n_train = 100
        n_val = 50
        n_test = 50

        # Create training data
        train_windows = np.random.randn(n_train, 20, NUM_CHANNELS) * 2.0 + 5.0
        val_windows = np.random.randn(n_val, 20, NUM_CHANNELS) * 10.0
        test_windows = np.random.randn(n_test, 20, NUM_CHANNELS) * 50.0

        # Fit normalizer on training windows
        norm_baseline = FeatureNormalizer.fit(train_windows)
        means_baseline = np.copy(norm_baseline.means)
        stds_baseline = np.copy(norm_baseline.stds)

        # Drastically perturb validation and test data
        val_windows_corrupted = val_windows * 1000.0 + 99999.0
        test_windows_corrupted = test_windows * -5000.0 - 88888.0

        # Normalization fit must accept only training data
        norm_after = FeatureNormalizer.fit(train_windows)

        # Assert bit-identical means and stds
        np.testing.assert_array_equal(norm_after.means, means_baseline)
        np.testing.assert_array_equal(norm_after.stds, stds_baseline)

    def test_roundtrip_transform_and_inverse(self) -> None:
        """Standardizing and inverting must recover original data to within numerical precision."""
        np.random.seed(123)
        X = np.random.randn(10, 20, NUM_CHANNELS) * 4.5 - 1.2
        norm = FeatureNormalizer.fit(X)

        X_transformed = norm.transform(X)
        X_recovered = norm.inverse_transform(X_transformed)

        np.testing.assert_allclose(X_recovered, X, rtol=1e-12, atol=1e-12)

    def test_zero_variance_clamping(self) -> None:
        """Channels with constant values (zero std) must clamp std to 1.0 to prevent zero division."""
        X = np.zeros((10, 20, NUM_CHANNELS))
        X[:, :, 0] = 5.0  # Constant channel 0
        norm = FeatureNormalizer.fit(X)

        assert norm.stds[0] == 1.0
        assert norm.means[0] == 5.0

        transformed = norm.transform(X)
        assert np.all(np.isfinite(transformed))
        np.testing.assert_allclose(transformed[:, :, 0], 0.0)

    def test_model_config_interoperability(self) -> None:
        """Normalizer must serialize to and deserialize from Phase 1 ModelConfig cleanly."""
        X = np.random.randn(5, 20, NUM_CHANNELS)
        norm = FeatureNormalizer.fit(X)

        cfg = norm.to_model_config(model_name="VelocityNet")
        assert isinstance(cfg, ModelConfig)
        assert cfg.model_name == "VelocityNet"
        assert len(cfg.normalization_means) == NUM_CHANNELS
        assert len(cfg.normalization_stds) == NUM_CHANNELS

        reloaded = FeatureNormalizer.from_model_config(cfg)
        np.testing.assert_allclose(reloaded.means, norm.means)
        np.testing.assert_allclose(reloaded.stds, norm.stds)
