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
    valid_mask: Optional[np.ndarray] = None,
    window_size: int = 3,
) -> np.ndarray:
    """Apply strictly causal 1D median filter where invalid samples form hard boundaries.

    Contiguous valid segments are processed independently:
    - If a sample k is invalid (valid_mask is False, non-finite, or negative),
      filtered[k] is set to np.nan and the contiguous valid segment is terminated.
    - If sample k is valid:
      The filter considers only the contiguous valid run ending at k:
          eff_len = min(window_size, current_run_length)
          filtered[k] = median(signal[k - eff_len + 1 : k + 1])
      Samples preceding an invalid gap NEVER influence any sample after the gap.
      There is zero bridging, zero forward-filling, and zero lookahead.

    Args:
        signal: (N,) float array.
        valid_mask: Optional (N,) boolean array indicating valid samples.
            If None, valid samples are defined as finite and non-negative (>= 0).
        window_size: Odd integer kernel size (default 3 samples = 300 ms at 10 Hz).

    Returns:
        (N,) float array causally smoothed. Invalid sample locations contain np.nan.
    """
    n = len(signal)
    if n == 0:
        return np.zeros(0, dtype=signal.dtype)

    sig_arr = np.asarray(signal, dtype=np.float64)
    is_valid_sample = np.isfinite(sig_arr) & (sig_arr >= 0.0)
    if valid_mask is not None:
        is_valid_sample &= np.asarray(valid_mask, dtype=bool)

    if window_size <= 1:
        return np.where(is_valid_sample, sig_arr, np.nan)

    filtered = np.full(n, np.nan, dtype=np.float64)
    run_length = 0

    for k in range(n):
        if not is_valid_sample[k]:
            # Hard boundary: reset contiguous run length, do not filter invalid sample
            run_length = 0
            filtered[k] = np.nan
        else:
            run_length += 1
            eff_len = min(window_size, run_length)
            sub = sig_arr[k - eff_len + 1 : k + 1]
            filtered[k] = float(np.median(sub))

    return filtered


def extract_velocitynet_labels(
    timestamps_ns: np.ndarray,
    v_ref_speed_mps: np.ndarray,
    window_end_indices: Sequence[int],
    is_validated: Optional[np.ndarray] = None,
    smoothing_window_size: int = 3,
) -> VelocityNetLabelBatch:
    """Extract causally aligned VelocityNet speed labels at window ends.

    Invalid reference samples form hard boundaries:
    - They are not smoothed or bridged across.
    - Windows ending on an invalid sample have is_valid_label = False,
      and their labels contain np.nan (never silently fabricated 0.0).

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

    # Sample-level validity: validated by upstream pipeline, finite, and non-negative
    sample_valid = (
        val_mask
        & np.isfinite(v_ref_speed_mps)
        & (v_ref_speed_mps >= 0.0)
    )

    # 1. Apply causal median filter with hard invalid boundaries
    smoothed_full = causal_median_filter_1d(
        v_ref_speed_mps,
        valid_mask=sample_valid,
        window_size=smoothing_window_size,
    )

    m_windows = len(window_end_indices)
    labels_smoothed = np.full(m_windows, np.nan, dtype=np.float32)
    labels_raw = np.full(m_windows, np.nan, dtype=np.float32)
    labels_valid = np.zeros(m_windows, dtype=bool)

    # 2. Extract targets at each window end index
    for m, end_idx in enumerate(window_end_indices):
        if end_idx < 0 or end_idx >= n_samples:
            raise IndexError(f"window_end_index {end_idx} out of range [0, {n_samples})")

        end_is_valid = bool(sample_valid[end_idx])
        raw_val = float(v_ref_speed_mps[end_idx])
        sm_val = float(smoothed_full[end_idx])

        # Validity criteria:
        # The end sample must be strictly valid (sample_valid == True),
        # both raw and smoothed speeds must be finite and non-negative.
        is_valid = (
            end_is_valid
            and np.isfinite(raw_val)
            and np.isfinite(sm_val)
            and raw_val >= 0.0
            and sm_val >= 0.0
        )

        if is_valid:
            labels_raw[m] = np.float32(raw_val)
            labels_smoothed[m] = np.float32(sm_val)
            labels_valid[m] = True
        else:
            # Explicitly do NOT fabricate 0.0 or bridge across invalid samples
            labels_raw[m] = np.float32(np.nan)
            labels_smoothed[m] = np.float32(np.nan)
            labels_valid[m] = False

    return VelocityNetLabelBatch(
        speed_smoothed_mps=labels_smoothed,
        speed_raw_mps=labels_raw,
        is_valid_label=labels_valid,
    )
