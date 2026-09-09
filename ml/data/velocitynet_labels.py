"""Causal VelocityNet Label Pipeline.

COMPASS Phase 6 — ML Dataset Construction.
Extracts ground-truth forward speed targets for VelocityNet.

CRITICAL CAUSALITY & TIMING INVARIANTS:
1. Window-End Association: For an input window ending at timestamp T,
   the target label is strictly the vehicle forward speed at timestamp T:
       label = v(T)
   Zero lookahead. Zero centered-window labels. Zero future speeds.
2. Causal Median Smoothing: Short median smoothing (W = 3 samples) is applied
   BEFORE assigning window-end labels to attenuate VBOX/CAN discretization jitter.
   Smoothing is STRICTLY CAUSAL:
       v_smoothed[k] = median(v[k-2], v[k-1], v[k])
   It never accesses samples after t_k.
3. Provenance: Both smoothed and raw unsmoothed reference speeds are preserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import numpy as np


@dataclass(frozen=True)
class VelocityNetLabelBatch:
    """Batch of extracted VelocityNet labels aligned with window ends."""
    speed_smoothed_mps: np.ndarray   # (M,) float32 causally smoothed targets
    speed_raw_mps: np.ndarray        # (M,) float32 raw reference speeds
    is_valid_label: np.ndarray       # (M,) bool mask (False if NaN or missing)


def causal_median_filter_1d(
    signal: np.ndarray,
    window_size: int = 3,
) -> np.ndarray:
    """Apply strictly causal 1D median filter over past samples.

    For sample k:
        filtered[k] = median(signal[max(0, k - window_size + 1) : k + 1])

    Args:
        signal: (N,) float array.
        window_size: Odd integer kernel size (default 3 samples = 300 ms at 10 Hz).

    Returns:
        (N,) float array causally smoothed.
    """
    n = len(signal)
    if n == 0:
        return np.zeros(0, dtype=signal.dtype)
    if window_size <= 1:
        return np.copy(signal)

    filtered = np.zeros(n, dtype=signal.dtype)
    for k in range(n):
        start = max(0, k - window_size + 1)
        sub = signal[start : k + 1]
        # Ignore NaNs during median if any
        valid_sub = sub[np.isfinite(sub)]
        if len(valid_sub) > 0:
            filtered[k] = float(np.median(valid_sub))
        else:
            filtered[k] = np.nan

    return filtered


def extract_velocitynet_labels(
    timestamps_ns: np.ndarray,
    v_ref_speed_mps: np.ndarray,
    window_end_indices: Sequence[int],
    is_validated: Optional[np.ndarray] = None,
    smoothing_window_size: int = 3,
) -> VelocityNetLabelBatch:
    """Extract causally aligned VelocityNet speed labels at window ends.

    Args:
        timestamps_ns: (N,) int64 timestamps in nanoseconds.
        v_ref_speed_mps: (N,) float64 reference vehicle speed in m/s (from VBOX).
        window_end_indices: Sequence of length M containing source indices of window ends.
        is_validated: Optional (N,) bool array indicating valid source samples.
        smoothing_window_size: Window size for causal median smoothing (strictly 3).

    Returns:
        VelocityNetLabelBatch containing M labels matching window ends.
    """
    n_samples = len(timestamps_ns)
    if len(v_ref_speed_mps) != n_samples:
        raise ValueError(
            f"v_ref_speed_mps length ({len(v_ref_speed_mps)}) != timestamps ({n_samples})"
        )

    val_mask = (
        np.ones(n_samples, dtype=bool)
        if is_validated is None
        else np.asarray(is_validated, dtype=bool)
    )

    # 1. Apply causal median filter over entire continuous reference stream
    smoothed_full = causal_median_filter_1d(
        v_ref_speed_mps,
        window_size=smoothing_window_size,
    )

    m_windows = len(window_end_indices)
    labels_smoothed = np.zeros(m_windows, dtype=np.float32)
    labels_raw = np.zeros(m_windows, dtype=np.float32)
    labels_valid = np.zeros(m_windows, dtype=bool)

    # 2. Extract targets at each window end index
    for m, end_idx in enumerate(window_end_indices):
        if end_idx < 0 or end_idx >= n_samples:
            raise IndexError(f"window_end_index {end_idx} out of range [0, {n_samples})")

        raw_val = float(v_ref_speed_mps[end_idx])
        sm_val = float(smoothed_full[end_idx])
        sample_valid = bool(val_mask[end_idx])

        # Validity criteria: finite, non-negative, sample validated
        is_valid = (
            sample_valid
            and np.isfinite(raw_val)
            and np.isfinite(sm_val)
            and raw_val >= 0.0
            and sm_val >= 0.0
        )

        labels_raw[m] = np.float32(raw_val if np.isfinite(raw_val) else 0.0)
        labels_smoothed[m] = np.float32(sm_val if np.isfinite(sm_val) else 0.0)
        labels_valid[m] = is_valid

    return VelocityNetLabelBatch(
        speed_smoothed_mps=labels_smoothed,
        speed_raw_mps=labels_raw,
        is_valid_label=labels_valid,
    )
