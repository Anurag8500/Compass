"""Dual-stage IMU filtering chain for COMPASS (Phase 3).

Implements the fixed preprocessing filter pipeline:
    Raw Aligned Vehicle IMU
              |
    1. Median Spike Filter (3-5 sample window)
              |
    2. 4th-Order Butterworth Low-Pass Filter (cutoff ~3 Hz, sampling rate configurable)
              |
    Denoised Vehicle-Frame Specific Force f_m^v & Angular Rate omega_m^v

Design Requirements:
- Median window length configurable (odd integer >= 3, default 3).
- Butterworth cutoff frequency configurable (default 3.0 Hz).
- Sampling rate must be explicitly passed/configured (no hidden hardcoded rate).
- Offline validation uses zero-phase forward-backward filtering (`scipy.signal.filtfilt`),
  eliminating group delay artifacts during post-processing and training data preparation.
- Streaming/causal evaluation is also supported for live simulation.
- Note: Production Android/Kotlin runtime will deploy a causal fixed-coefficient
  direct-form II transposed IIR biquad equivalent.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple
import numpy as np

try:
    from scipy import signal as sp_signal
    SCIPY_AVAILABLE = True
except ImportError:
    sp_signal = None
    SCIPY_AVAILABLE = False


def apply_median_filter(data: np.ndarray, window_size: int = 3) -> np.ndarray:
    """Apply 1D median filter along time axis (axis 0) for spike rejection.

    Args:
        data: (N,) or (N, D) array of sensor measurements.
        window_size: Odd integer filter window length >= 3.

    Returns:
        Filtered array of same shape and dtype as data.
    """
    arr = np.asarray(data, dtype=np.float64)
    if window_size < 3:
        return arr.copy()
    if window_size % 2 == 0:
        raise ValueError(f"window_size must be an odd integer, got {window_size}")

    n_samples = arr.shape[0]
    if n_samples < window_size:
        return arr.copy()

    half_w = window_size // 2
    out = np.empty_like(arr)

    if arr.ndim == 1:
        for i in range(n_samples):
            start = max(0, i - half_w)
            end = min(n_samples, i + half_w + 1)
            out[i] = np.median(arr[start:end])
    elif arr.ndim == 2:
        for i in range(n_samples):
            start = max(0, i - half_w)
            end = min(n_samples, i + half_w + 1)
            out[i] = np.median(arr[start:end, :], axis=0)
    else:
        raise ValueError(f"data must be 1D or 2D, got {arr.ndim}D")

    return out


def compute_butterworth_4th_coeffs(cutoff_hz: float, sampling_rate_hz: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute 4th-order lowpass Butterworth digital filter coefficients (b, a).

    Uses scipy.signal.butter when available, with self-contained fallback.
    """
    if cutoff_hz <= 0.0:
        raise ValueError(f"cutoff_hz must be positive, got {cutoff_hz}")
    if sampling_rate_hz <= 0.0:
        raise ValueError(f"sampling_rate_hz must be positive, got {sampling_rate_hz}")
    nyquist = 0.5 * sampling_rate_hz
    if cutoff_hz >= nyquist:
        raise ValueError(f"cutoff_hz ({cutoff_hz} Hz) must be less than Nyquist frequency ({nyquist} Hz)")

    if SCIPY_AVAILABLE:
        b, a = sp_signal.butter(4, cutoff_hz / nyquist, btype="lowpass", analog=False)
        return b, a

    # Self-contained bilinear transform fallback for 4th-order Butterworth
    # Prewarped analog cutoff frequency
    omega_a = 2.0 * sampling_rate_hz * np.tan(np.pi * cutoff_hz / sampling_rate_hz)
    # Poles of 4th order Butterworth on s-plane: s_k = omega_a * exp(j * (2k+1) * pi / 8) for k=0,1,2,3
    # Group into two conjugate pairs:
    # Pair 1: s^2 + 2*omega_a*cos(pi/8)*s + omega_a^2
    # Pair 2: s^2 + 2*omega_a*cos(3*pi/8)*s + omega_a^2
    c1 = 2.0 * np.cos(np.pi / 8.0)
    c2 = 2.0 * np.cos(3.0 * np.pi / 8.0)
    fs2 = 2.0 * sampling_rate_hz

    # Biquad 1 via bilinear transform: s = fs2 * (1 - z^-1)/(1 + z^-1)
    a0_1 = fs2 * fs2 + c1 * omega_a * fs2 + omega_a * omega_a
    b_1 = np.array([omega_a * omega_a, 2.0 * omega_a * omega_a, omega_a * omega_a]) / a0_1
    a_1 = np.array([1.0, (2.0 * omega_a * omega_a - 2.0 * fs2 * fs2) / a0_1, (fs2 * fs2 - c1 * omega_a * fs2 + omega_a * omega_a) / a0_1])

    # Biquad 2
    a0_2 = fs2 * fs2 + c2 * omega_a * fs2 + omega_a * omega_a
    b_2 = np.array([omega_a * omega_a, 2.0 * omega_a * omega_a, omega_a * omega_a]) / a0_2
    a_2 = np.array([1.0, (2.0 * omega_a * omega_a - 2.0 * fs2 * fs2) / a0_2, (fs2 * fs2 - c2 * omega_a * fs2 + omega_a * omega_a) / a0_2])

    # Convolve polynomials
    b = np.convolve(b_1, b_2)
    a = np.convolve(a_1, a_2)
    return b, a


class IMUFilter:
    """Configurable dual-stage filter (Median Spike Filter + 4th-Order Butterworth Low-Pass)."""

    def __init__(
        self,
        sampling_rate_hz: float = 10.0,
        cutoff_hz: float = 3.0,
        median_window_size: int = 3,
        use_zero_phase: bool = True,
    ) -> None:
        """Initialize IMUFilter.

        Args:
            sampling_rate_hz: Sensor nominal sampling rate in Hz (e.g. 10.0 or 100.0).
            cutoff_hz: Low-pass filter cutoff frequency in Hz (default: 3.0 Hz).
            median_window_size: Odd integer window size for spike rejection (default: 3).
            use_zero_phase: If True, uses filtfilt for zero-phase offline processing.
        """
        self.sampling_rate_hz = float(sampling_rate_hz)
        self.cutoff_hz = float(cutoff_hz)
        self.median_window_size = int(median_window_size)
        self.use_zero_phase = use_zero_phase

        self.b, self.a = compute_butterworth_4th_coeffs(self.cutoff_hz, self.sampling_rate_hz)

    @property
    def filtering_mode(self) -> str:
        """Discoverable filtering mode: 'zero_phase' or 'causal'."""
        if self.use_zero_phase and SCIPY_AVAILABLE:
            return "zero_phase"
        return "causal"

    def filter_series(self, data: np.ndarray) -> np.ndarray:
        """Apply median spike rejection followed by 4th-order Butterworth low-pass.

        Args:
            data: (N,) or (N, D) array of sensor measurements.

        Returns:
            Filtered array of identical shape.
        """
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim not in (1, 2):
            raise ValueError(f"data must be 1D or 2D, got {arr.ndim}D")

        if not np.isfinite(arr).all():
            raise ValueError("Input data contains non-finite values (NaN/Inf)")

        if arr.shape[0] < 5:
            return arr.copy()

        # 1. Median filter stage (spike suppression)
        med_filtered = apply_median_filter(arr, window_size=self.median_window_size)

        # 2. Butterworth low-pass stage
        n_samples = med_filtered.shape[0]
        # Canonical pad length requirement for 4th-order scipy filtfilt (3 * max(len(a), len(b)) = 15)
        canonical_padlen = 3 * max(len(self.a), len(self.b))

        if self.use_zero_phase:
            if not SCIPY_AVAILABLE:
                warnings.warn(
                    "Zero-phase filtering requested but scipy.signal is unavailable. Falling back to causal IIR.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            elif n_samples <= canonical_padlen:
                warnings.warn(
                    f"Series length ({n_samples}) is too short for zero-phase filtfilt (requires > {canonical_padlen}). Falling back to causal IIR.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                # Execute zero-phase filtfilt without catching generic exceptions silently
                padlen = canonical_padlen
                if med_filtered.ndim == 1:
                    return sp_signal.filtfilt(self.b, self.a, med_filtered, padlen=padlen)
                else:
                    return sp_signal.filtfilt(self.b, self.a, med_filtered, axis=0, padlen=padlen)

        # Causal Direct-Form II / IIR filtering
        if med_filtered.ndim == 1:
            return self._apply_iir_1d(med_filtered)
        else:
            out = np.empty_like(med_filtered)
            for d in range(med_filtered.shape[1]):
                out[:, d] = self._apply_iir_1d(med_filtered[:, d])
            return out

    def _apply_iir_1d(self, x: np.ndarray) -> np.ndarray:
        """Direct-form IIR filter for 1D sequence."""
        if SCIPY_AVAILABLE:
            return sp_signal.lfilter(self.b, self.a, x)
        y = np.zeros_like(x)
        b, a = self.b, self.a
        order = len(b) - 1
        for n in range(len(x)):
            acc = 0.0
            for k in range(order + 1):
                if n - k >= 0:
                    acc += b[k] * x[n - k]
            for k in range(1, order + 1):
                if n - k >= 0:
                    acc -= a[k] * y[n - k]
            y[n] = acc / a[0]
        return y
