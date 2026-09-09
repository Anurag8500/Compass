"""Unit tests for 9-channel feature computation (ml/data/features.py).

Verifies:
1. Exact 9-channel ordering and dimensionality.
2. Mathematical correctness on known synthetic analytical signals.
3. Timestamp-aware backward derivative for jerk magnitude.
4. Absence of gravity compensation (specific force preserved).
5. Non-finite input rejection.
"""

from __future__ import annotations

import numpy as np
import pytest
from ml.data.features import compute_canonical_features, NUM_CHANNELS
from navigation.schemas.config import CANONICAL_CHANNELS


class TestFeatures:
    """Test suite for canonical 9-channel feature engineering."""

    def test_canonical_channel_count_and_ordering(self) -> None:
        """Output feature matrix must have exactly 9 channels matching CANONICAL_CHANNELS."""
        assert len(CANONICAL_CHANNELS) == NUM_CHANNELS
        assert CANONICAL_CHANNELS == (
            "f_x_v",
            "f_y_v",
            "f_z_v",
            "omega_x_v",
            "omega_y_v",
            "omega_z_v",
            "norm_f_v",
            "norm_f_dot_v",
            "norm_omega_v",
        )

    def test_known_synthetic_analytical_features(self) -> None:
        """Validate exact mathematical calculation on known analytical signals."""
        n = 5
        dt_ns = 100_000_000  # 0.1 s = 10 Hz
        ts = np.arange(0, n * dt_ns, dt_ns, dtype=np.int64)

        # Constant 3-4-5 right triangle specific force: [3.0, 4.0, 0.0] -> ||f|| = 5.0
        f_m_v = np.zeros((n, 3))
        f_m_v[:, 0] = 3.0
        f_m_v[:, 1] = 4.0
        f_m_v[:, 2] = 0.0

        # Angular rate: [0.0, 0.0, 2.0] -> ||omega|| = 2.0
        w_m_v = np.zeros((n, 3))
        w_m_v[:, 2] = 2.0

        feats = compute_canonical_features(ts, f_m_v, w_m_v)

        assert feats.shape == (n, 9)

        # Kinematic channels
        np.testing.assert_allclose(feats[:, 0], 3.0)
        np.testing.assert_allclose(feats[:, 1], 4.0)
        np.testing.assert_allclose(feats[:, 2], 0.0)
        np.testing.assert_allclose(feats[:, 3], 0.0)
        np.testing.assert_allclose(feats[:, 4], 0.0)
        np.testing.assert_allclose(feats[:, 5], 2.0)

        # Derived norms
        np.testing.assert_allclose(feats[:, 6], 5.0)  # ||f|| = sqrt(3^2 + 4^2) = 5.0
        np.testing.assert_allclose(feats[:, 7], 0.0)  # Constant f -> df/dt = 0.0
        np.testing.assert_allclose(feats[:, 8], 2.0)  # ||omega|| = 2.0

    def test_causal_backward_jerk_derivative(self) -> None:
        """Linear ramp in specific force produces constant jerk via backward difference."""
        n = 10
        dt_ns = 100_000_000  # 0.1 s
        ts = np.arange(0, n * dt_ns, dt_ns, dtype=np.int64)

        # Ramp: fx increases by 0.5 m/s^2 every 0.1 s -> df/dt = 0.5 / 0.1 = 5.0 m/s^3
        f_m_v = np.zeros((n, 3))
        f_m_v[:, 0] = np.arange(n) * 0.5
        w_m_v = np.zeros((n, 3))

        feats = compute_canonical_features(ts, f_m_v, w_m_v)

        # Sample 0 has jerk 0.0 (strictly causal backward step)
        assert feats[0, 7] == 0.0
        # Samples 1 to 9 have jerk exactly 5.0 m/s^3
        np.testing.assert_allclose(feats[1:, 7], 5.0)

    def test_rejection_of_nan_and_infinite_inputs(self) -> None:
        """Features must strictly reject NaN or infinite inputs."""
        ts = np.array([0, 100_000_000], dtype=np.int64)
        f_nan = np.array([[1.0, np.nan, 0.0], [1.0, 0.0, 0.0]])
        w = np.zeros((2, 3))

        with pytest.raises(ValueError, match="non-finite"):
            compute_canonical_features(ts, f_nan, w)

        f_inf = np.array([[1.0, np.inf, 0.0], [1.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match="non-finite"):
            compute_canonical_features(ts, f_inf, w)
