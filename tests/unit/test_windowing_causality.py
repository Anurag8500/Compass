"""Unit tests for strictly causal 20x9 window extraction (ml/data/windowing.py).

Verifies:
1. Hard causality: mutating a future sample after window end timestamp T does NOT alter window at T.
2. Canonical window dimensions: exactly (20, 9).
3. Canonical stride: exactly 5 samples (0.5 s at 10 Hz).
4. Insufficient history handling (fewer than 20 samples yields empty list).
5. Window validity propagation from sample quality flags.
"""

from __future__ import annotations

import numpy as np
import pytest
from ml.data.windowing import extract_causal_windows, ExtractedWindow


class TestWindowingCausality:
    """Test suite for causal windowing invariants."""

    def test_windowing_dimensions_and_stride(self) -> None:
        """A stream of 60 samples (6.0 s) must yield exactly 9 windows with stride=5."""
        # 60 samples:
        # window 0 ends at 19 (indices 0..19)
        # window 1 ends at 24 (indices 5..24)
        # ...
        # window 8 ends at 59 (indices 40..59)
        # Total = 9 windows
        n = 60
        dt_ns = 100_000_000
        ts = np.arange(0, n * dt_ns, dt_ns, dtype=np.int64)
        feats = np.random.randn(n, 9)

        windows = extract_causal_windows(
            features=feats,
            timestamps_ns=ts,
            source_file_id="trip_01",
            driver_id="Driver_E",
            window_size=20,
            stride_samples=5,
        )

        assert len(windows) == 9
        for i, w in enumerate(windows):
            assert w.window.shape == (20, 9)
            assert w.source_file_id == "trip_01"
            assert w.driver_id == "Driver_E"
            expected_end_idx = 19 + i * 5
            expected_start_idx = expected_end_idx - 19
            assert w.end_idx == expected_end_idx
            assert w.start_idx == expected_start_idx
            assert w.end_timestamp_ns == ts[expected_end_idx]
            assert w.start_timestamp_ns == ts[expected_start_idx]
            assert w.is_valid is True

    def test_hard_causality_future_sample_mutation(self) -> None:
        """Mutating a future sample at index > T_end MUST NOT change the window ending at T_end."""
        n = 50
        dt_ns = 100_000_000
        ts = np.arange(0, n * dt_ns, dt_ns, dtype=np.int64)
        feats1 = np.ones((n, 9)) * 3.14

        # Baseline windowing
        windows1 = extract_causal_windows(
            features=feats1,
            timestamps_ns=ts,
            window_size=20,
            stride_samples=5,
        )

        # Inspect window 0 (covers indices 0 to 19, ending at t = 1.9s)
        win0_baseline = windows1[0].window

        # Now create mutated feature stream where future samples (index >= 20) are corrupted
        feats2 = np.copy(feats1)
        feats2[20:] = 99999.0  # Massive future explosion at t >= 2.0s

        windows2 = extract_causal_windows(
            features=feats2,
            timestamps_ns=ts,
            window_size=20,
            stride_samples=5,
        )

        win0_mutated = windows2[0].window

        # Window 0 must be 100% byte-for-byte identical despite future mutation
        np.testing.assert_array_equal(win0_baseline, win0_mutated)

    def test_insufficient_history_returns_empty(self) -> None:
        """Streams with fewer than 20 samples cannot form a valid causal window."""
        n = 19
        dt_ns = 100_000_000
        ts = np.arange(0, n * dt_ns, dt_ns, dtype=np.int64)
        feats = np.zeros((n, 9))

        windows = extract_causal_windows(feats, ts, window_size=20, stride_samples=5)
        assert len(windows) == 0

    def test_unvalidated_sample_marks_window_invalid(self) -> None:
        """A single unvalidated sample inside a 20-sample window marks the entire window invalid."""
        n = 30
        dt_ns = 100_000_000
        ts = np.arange(0, n * dt_ns, dt_ns, dtype=np.int64)
        feats = np.zeros((n, 9))
        val_mask = np.ones(n, dtype=bool)

        # Invalidate sample 10 (inside window 0 [0..19], but outside window 2 [10..29])
        val_mask[5] = False

        windows = extract_causal_windows(feats, ts, is_validated=val_mask, window_size=20, stride_samples=5)

        # Window 0 (indices 0..19) contains index 5 -> invalid
        assert windows[0].is_valid is False
        # Window 1 (indices 5..24) contains index 5 -> invalid
        assert windows[1].is_valid is False
        # Window 2 (indices 10..29) does NOT contain index 5 -> valid
        assert windows[2].is_valid is True
