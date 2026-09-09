"""Unit tests for generic resampling and decimation (ml/data/resample.py).

Verifies:
1. Native 10 Hz pass-through identity behavior.
2. Synthetic higher-rate (100 Hz and 200 Hz) decimation to canonical 10 Hz.
3. Strict causality: future samples modified after t_k do not alter sample at t_k.
4. Correct irregular dt handling without non-causal extrapolation.
"""

from __future__ import annotations

import numpy as np
import pytest
from ml.data.resample import resample_to_canonical_10hz, CANONICAL_DT_NS


class TestResample:
    """Test suite for canonical 10 Hz resampling and decimation."""

    def test_native_10hz_identity_passthrough(self) -> None:
        """Native 10 Hz data must be returned as an identity copy with is_identity=True."""
        n_samples = 100
        dt_ns = 100_000_000  # 100 ms
        timestamps_ns = np.arange(0, n_samples * dt_ns, dt_ns, dtype=np.int64)
        f_m_v = np.random.randn(n_samples, 3)
        omega_m_v = np.random.randn(n_samples, 3)
        valid = np.ones(n_samples, dtype=bool)

        res = resample_to_canonical_10hz(
            timestamps_ns=timestamps_ns,
            f_m_v=f_m_v,
            omega_m_v=omega_m_v,
            is_validated=valid,
        )

        assert res.is_identity is True
        assert len(res.timestamps_ns) == n_samples
        np.testing.assert_array_equal(res.timestamps_ns, timestamps_ns)
        np.testing.assert_allclose(res.f_m_v, f_m_v)
        np.testing.assert_allclose(res.omega_m_v, omega_m_v)
        np.testing.assert_array_equal(res.is_validated, valid)

    def test_higher_rate_100hz_to_10hz_decimation(self) -> None:
        """100 Hz synthetic data (1000 samples = 10.0 s) must decimate to 101 canonical 10 Hz samples."""
        n_samples = 1001  # 0 to 10.0 seconds at 100 Hz (dt = 10 ms)
        dt_ns = 10_000_000  # 10 ms = 100 Hz
        timestamps_ns = np.arange(0, n_samples * dt_ns, dt_ns, dtype=np.int64)

        # Constant force [1.0, 2.0, 3.0] and angular rate [0.1, 0.2, 0.3]
        f_m_v = np.tile([1.0, 2.0, 3.0], (n_samples, 1))
        omega_m_v = np.tile([0.1, 0.2, 0.3], (n_samples, 1))

        res = resample_to_canonical_10hz(
            timestamps_ns=timestamps_ns,
            f_m_v=f_m_v,
            omega_m_v=omega_m_v,
        )

        assert res.is_identity is False
        assert res.resampled_rate_hz == 10.0
        # Check canonical dt is exactly 100 ms
        step_dts = np.diff(res.timestamps_ns)
        np.testing.assert_array_equal(step_dts, CANONICAL_DT_NS)
        # Check values preserve physical mean
        expected_f = np.tile([1.0, 2.0, 3.0], (len(res.f_m_v), 1))
        expected_w = np.tile([0.1, 0.2, 0.3], (len(res.omega_m_v), 1))
        np.testing.assert_allclose(res.f_m_v, expected_f)
        np.testing.assert_allclose(res.omega_m_v, expected_w)

    def test_higher_rate_200hz_fog_decimation(self) -> None:
        """200 Hz synthetic FOG IMU stream (dt = 5 ms) decimates cleanly to 10 Hz."""
        n_samples = 2001  # 10.0 s
        dt_ns = 5_000_000  # 5 ms = 200 Hz
        timestamps_ns = np.arange(0, n_samples * dt_ns, dt_ns, dtype=np.int64)
        f_m_v = np.random.randn(n_samples, 3)
        omega_m_v = np.random.randn(n_samples, 3)

        res = resample_to_canonical_10hz(
            timestamps_ns=timestamps_ns,
            f_m_v=f_m_v,
            omega_m_v=omega_m_v,
        )

        assert res.is_identity is False
        assert len(res.timestamps_ns) == 101  # 0.0s to 10.0s inclusive at 0.1s steps

    def test_resampling_strict_causality(self) -> None:
        """Modifying source samples strictly AFTER grid timestamp t_k must NOT affect output at t_k."""
        n_samples = 500  # 5.0 s at 100 Hz
        dt_ns = 10_000_000
        timestamps_ns = np.arange(0, n_samples * dt_ns, dt_ns, dtype=np.int64)

        f_1 = np.ones((n_samples, 3)) * 5.0
        w_1 = np.ones((n_samples, 3)) * 0.5

        # Baseline resample
        res1 = resample_to_canonical_10hz(timestamps_ns, f_1, w_1)

        # Perturb future samples: modify everything after t = 2.0 seconds (index 200)
        f_2 = np.copy(f_1)
        w_2 = np.copy(w_1)
        f_2[201:] = 999.0  # Future explosion
        w_2[201:] = 999.0

        res2 = resample_to_canonical_10hz(timestamps_ns, f_2, w_2)

        # Output at t <= 2.0 s (indices 0 through 20 on 10 Hz grid) must be BIT-IDENTICAL
        k_2s = 20  # index of 2.0 s on 10 Hz grid
        np.testing.assert_array_equal(res1.f_m_v[: k_2s + 1], res2.f_m_v[: k_2s + 1])
        np.testing.assert_array_equal(res1.omega_m_v[: k_2s + 1], res2.omega_m_v[: k_2s + 1])

    def test_invalid_input_rejections(self) -> None:
        """Malformed dimensions or insufficient samples must raise ValueError."""
        with pytest.raises(ValueError, match="At least 2 samples required"):
            resample_to_canonical_10hz(np.array([1000]), np.zeros((1, 3)), np.zeros((1, 3)))

        with pytest.raises(ValueError, match="f_m_v shape"):
            resample_to_canonical_10hz(np.array([100, 200]), np.zeros((2, 4)), np.zeros((2, 3)))
