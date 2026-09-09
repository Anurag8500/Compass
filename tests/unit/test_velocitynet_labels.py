"""Unit tests for VelocityNet causal labeling pipeline (ml/data/velocitynet_labels.py).

Verifies:
1. Ground-truth target speed is associated strictly with the window-end timestamp.
2. Short median filter (W=3) is causal and does not access future samples.
3. Both raw and smoothed speeds are preserved.
4. Unvalidated samples or negative speeds are correctly marked invalid.
"""

from __future__ import annotations

import numpy as np
import pytest
from ml.data.velocitynet_labels import (
    causal_median_filter_1d,
    extract_velocitynet_labels,
)


class TestVelocityNetLabels:
    """Test suite for causal VelocityNet label assignment."""

    def test_causal_median_filter_no_future_lookahead(self) -> None:
        """Causal median filter at step k must only use samples up to k."""
        # Step response: 0, 0, 0, 10, 10, 10
        signal = np.array([0.0, 0.0, 0.0, 10.0, 10.0, 10.0])
        # W=3:
        # k=0: median([0]) = 0
        # k=1: median([0, 0]) = 0
        # k=2: median([0, 0, 0]) = 0
        # k=3: median([0, 0, 10]) = 0  (step transition delayed by 1 step causally)
        # k=4: median([0, 10, 10]) = 10
        # k=5: median([10, 10, 10]) = 10
        filtered = causal_median_filter_1d(signal, window_size=3)
        expected = np.array([0.0, 0.0, 0.0, 0.0, 10.0, 10.0])
        np.testing.assert_allclose(filtered, expected)

    def test_impulse_spike_suppression(self) -> None:
        """A single-sample outlier spike is eliminated by causal median filter."""
        signal = np.array([5.0, 5.0, 5.0, 100.0, 5.0, 5.0])
        filtered = causal_median_filter_1d(signal, window_size=3)
        # Spike at k=3:
        # k=2: median(5, 5, 5) = 5
        # k=3: median(5, 5, 100) = 5 (spike suppressed!)
        # k=4: median(5, 100, 5) = 5 (spike suppressed!)
        # k=5: median(100, 5, 5) = 5
        expected = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
        np.testing.assert_allclose(filtered, expected)

    def test_window_end_label_alignment(self) -> None:
        """Velocity label must correspond strictly to window end index."""
        n = 30
        dt_ns = 100_000_000
        ts = np.arange(0, n * dt_ns, dt_ns, dtype=np.int64)

        # Linearly increasing speed
        speed = np.arange(n, dtype=np.float64) * 0.5  # 0.0, 0.5, 1.0, ...

        # Windows ending at index 19 and index 24
        end_indices = [19, 24]
        batch = extract_velocitynet_labels(ts, speed, end_indices)

        assert len(batch.speed_raw_mps) == 2
        assert len(batch.speed_smoothed_mps) == 2
        assert len(batch.is_valid_label) == 2

        # Raw labels at ends
        assert batch.speed_raw_mps[0] == pytest.approx(speed[19])
        assert batch.speed_raw_mps[1] == pytest.approx(speed[24])

        # Smoothed labels
        # At index 19: median of speed[17], speed[18], speed[19] = speed[18]
        assert batch.speed_smoothed_mps[0] == pytest.approx(speed[18])
        # At index 24: median of speed[22], speed[23], speed[24] = speed[23]
        assert batch.speed_smoothed_mps[1] == pytest.approx(speed[23])

        assert np.all(batch.is_valid_label)

    def test_negative_or_nan_speed_flagged_invalid(self) -> None:
        """Negative speeds or NaNs in reference signal must be marked invalid."""
        ts = np.array([0, 100_000_000, 200_000_000], dtype=np.int64)
        speed = np.array([5.0, -1.0, np.nan])
        batch = extract_velocitynet_labels(ts, speed, window_end_indices=[0, 1, 2])

        assert bool(batch.is_valid_label[0]) is True
        assert bool(batch.is_valid_label[1]) is False  # Negative speed
        assert bool(batch.is_valid_label[2]) is False  # NaN speed

    def test_type_hints_evaluable(self) -> None:
        """Type hints must evaluate without NameError (e.g. typing.get_type_hints)."""
        import typing
        hints = typing.get_type_hints(extract_velocitynet_labels)
        assert "window_end_indices" in hints
