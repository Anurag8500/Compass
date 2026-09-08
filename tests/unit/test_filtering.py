"""Unit tests for Phase 3 dual-stage IMU filtering."""

from __future__ import annotations

import math
import numpy as np
import pytest

from navigation.preprocessing.filtering import (
    IMUFilter,
    apply_median_filter,
    compute_butterworth_4th_coeffs,
)


class TestIMUFilter:
    """Test suite for median spike rejection, Butterworth lowpass filtering, and numerical validation."""

    def test_dc_preservation(self) -> None:
        """Constant DC signal must pass through the filter with negligible attenuation."""
        n = 100
        dc_val = 9.80665
        data = np.full((n, 3), dc_val)

        filt = IMUFilter(sampling_rate_hz=10.0, cutoff_hz=3.0, median_window_size=3)
        filtered = filt.filter_series(data)

        assert np.allclose(filtered, data, atol=1e-5)

    def test_median_filter_suppresses_isolated_spike(self) -> None:
        """Synthetic isolated spike in constant signal must be eliminated by 3-sample median filter."""
        signal = np.array([5.0, 5.0, 50.0, 5.0, 5.0, 5.0])
        filtered = apply_median_filter(signal, window_size=3)

        # The spike at index 2 (50.0) must be replaced by surrounding median (5.0)
        assert filtered[2] == pytest.approx(5.0, abs=1e-6)
        assert np.allclose(filtered, np.full(6, 5.0))

    def test_median_filter_preserves_step_edge(self) -> None:
        """Median filter preserves a real step edge transition better than linear averaging."""
        step = np.array([0.0, 0.0, 0.0, 10.0, 10.0, 10.0])
        filtered = apply_median_filter(step, window_size=3)

        assert filtered[0] == 0.0
        assert filtered[1] == 0.0
        assert filtered[4] == 10.0
        assert filtered[5] == 10.0

    def test_butterworth_lowpass_attenuates_high_frequency_vibration(self) -> None:
        """Synthetic signal with low-frequency motion (0.5 Hz) and high-frequency vibration (25 Hz).

        Sampling rate: 100 Hz. Cutoff: 5.0 Hz.
        The 25 Hz vibration must be attenuated by > 80% in amplitude, while the 0.5 Hz motion is preserved.
        """
        fs = 100.0
        t = np.arange(0.0, 3.0, 1.0 / fs)
        low_freq = 0.5
        high_freq = 25.0

        clean_motion = 2.0 * np.sin(2.0 * math.pi * low_freq * t)
        vibration = 1.0 * np.sin(2.0 * math.pi * high_freq * t)
        raw_signal = clean_motion + vibration

        filt = IMUFilter(sampling_rate_hz=fs, cutoff_hz=5.0, median_window_size=3)
        filtered = filt.filter_series(raw_signal)

        steady_slice = slice(50, 250)
        rmse_before = np.sqrt(np.mean((raw_signal[steady_slice] - clean_motion[steady_slice]) ** 2))
        rmse_after = np.sqrt(np.mean((filtered[steady_slice] - clean_motion[steady_slice]) ** 2))

        assert rmse_after < 0.20 * rmse_before, f"Expected >80% noise reduction, got {rmse_after / rmse_before:.2%}"

    def test_nyquist_and_cutoff_validation(self) -> None:
        """Filter initialization must reject cutoffs exceeding or equal to Nyquist frequency."""
        fs = 10.0
        nyquist = 5.0

        with pytest.raises(ValueError, match="Nyquist"):
            compute_butterworth_4th_coeffs(cutoff_hz=5.0, sampling_rate_hz=fs)

        with pytest.raises(ValueError, match="Nyquist"):
            IMUFilter(sampling_rate_hz=fs, cutoff_hz=5.5)

        with pytest.raises(ValueError, match="positive"):
            compute_butterworth_4th_coeffs(cutoff_hz=-1.0, sampling_rate_hz=fs)

    def test_filter_series_preserves_shape_for_nx3_array(self) -> None:
        """Nx3 specific force array filtered must maintain (N, 3) shape and remain finite."""
        n = 100
        data = np.random.normal(0, 0.1, size=(n, 3))
        data[:, 2] += 9.81

        filt = IMUFilter(sampling_rate_hz=10.0, cutoff_hz=3.0)
        out = filt.filter_series(data)

        assert out.shape == (n, 3)
        assert np.isfinite(out).all()
        assert np.mean(out[:, 2]) == pytest.approx(9.81, abs=0.1)

    def test_short_sequence_behavior_and_warning(self) -> None:
        """Verify behavior for short sequences (< 5 samples and < padlen)."""
        filt = IMUFilter(sampling_rate_hz=10.0, cutoff_hz=3.0, use_zero_phase=True)

        # Extremely short (< 5) returns copy
        short = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        out_short = filt.filter_series(short)
        assert np.array_equal(out_short, short)

        # Length 8: shorter than default padlen (15), should emit warning and fallback to causal
        medium_short = np.tile([0.0, 0.0, 9.8], (8, 1))
        with pytest.warns(RuntimeWarning, match="short for zero-phase"):
            out_med = filt.filter_series(medium_short)
        assert out_med.shape == (8, 3)
        assert np.isfinite(out_med).all()

    def test_filtering_mode_discovery(self) -> None:
        """Verify discoverable filtering_mode property reflects active configuration."""
        filt_zp = IMUFilter(sampling_rate_hz=10.0, cutoff_hz=3.0, use_zero_phase=True)
        assert filt_zp.filtering_mode == "zero_phase"

        filt_causal = IMUFilter(sampling_rate_hz=10.0, cutoff_hz=3.0, use_zero_phase=False)
        assert filt_causal.filtering_mode == "causal"

    def test_non_finite_input_raises_value_error(self) -> None:
        """Filter must reject input arrays containing NaN or Inf."""
        filt = IMUFilter(sampling_rate_hz=10.0, cutoff_hz=3.0)
        data = np.zeros((20, 3))
        data[10, 0] = np.nan

        with pytest.raises(ValueError, match="non-finite"):
            filt.filter_series(data)

    def test_butterworth_coefficient_generation(self) -> None:
        """Verify 4th-order Butterworth coefficient generation for standard 10 Hz rate."""
        b, a = compute_butterworth_4th_coeffs(cutoff_hz=3.0, sampling_rate_hz=10.0)
        assert len(b) == 5
        assert len(a) == 5
        assert a[0] == pytest.approx(1.0, abs=1e-6)
