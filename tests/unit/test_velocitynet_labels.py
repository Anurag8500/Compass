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

    def test_invalid_gap_isolation_hard_boundary(self) -> None:
        """Invalid reference samples must form hard boundaries preventing cross-gap bridging.

        Indices:
        0: valid (10.0)
        1: valid (10.0)
        2: invalid (np.nan)
        3: valid (20.0)
        4: valid (20.0)
        5: valid (20.0)

        At index 3, the filtered value must depend ONLY on the segment starting at 3.
        It must be 20.0, NOT median([10.0, 20.0]) = 15.0.
        """
        signal = np.array([10.0, 10.0, np.nan, 20.0, 20.0, 20.0])
        filtered = causal_median_filter_1d(signal, window_size=3)

        assert filtered[0] == pytest.approx(10.0)
        assert filtered[1] == pytest.approx(10.0)
        assert np.isnan(filtered[2])  # Hard boundary, not filtered or fabricated

        # Hard barrier check: index 3 MUST NOT bridge to index 1 or 0
        assert filtered[3] == pytest.approx(20.0)
        assert filtered[4] == pytest.approx(20.0)
        assert filtered[5] == pytest.approx(20.0)

        # Mutate index 0 and 1: index 3 MUST remain completely unchanged
        signal_mutated = np.array([999.0, 888.0, np.nan, 20.0, 20.0, 20.0])
        filtered_mutated = causal_median_filter_1d(signal_mutated, window_size=3)
        assert filtered_mutated[3] == pytest.approx(20.0)
        assert filtered_mutated[4] == pytest.approx(20.0)

    def test_no_future_median_leakage(self) -> None:
        """Mutating reference speed samples at t > T must not alter label at T."""
        n = 30
        dt_ns = 100_000_000
        ts = np.arange(0, n * dt_ns, dt_ns, dtype=np.int64)
        speed = np.full(n, 12.5)

        end_indices = [19]
        batch_orig = extract_velocitynet_labels(ts, speed, end_indices)

        # Mutate future samples after index 19 (indices 20 to 29)
        speed_mutated = np.copy(speed)
        speed_mutated[20:] = 999.0
        batch_mutated = extract_velocitynet_labels(ts, speed_mutated, end_indices)

        assert batch_orig.speed_smoothed_mps[0] == pytest.approx(12.5)
        assert batch_mutated.speed_smoothed_mps[0] == pytest.approx(12.5)
        assert batch_orig.speed_raw_mps[0] == pytest.approx(12.5)
        assert batch_mutated.speed_raw_mps[0] == pytest.approx(12.5)

    def test_invalid_label_stores_nan_not_fabricated_zero(self) -> None:
        """Invalid reference labels must store NaN and is_valid_label=False, never 0.0."""
        ts = np.array([0, 100_000_000, 200_000_000], dtype=np.int64)
        speed = np.array([15.0, np.nan, 15.0])
        batch = extract_velocitynet_labels(ts, speed, window_end_indices=[1])

        assert bool(batch.is_valid_label[0]) is False
        assert np.isnan(batch.speed_raw_mps[0])
        assert np.isnan(batch.speed_smoothed_mps[0])

