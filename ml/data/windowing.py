"""Strictly Causal 20x9 Feature Windowing.

COMPASS Phase 6 — ML Dataset Construction.
Shared windowing engine for BOTH VelocityNet and BiasNet.

Canonical Parameters:
    - Sampling Frequency: 10.0 Hz
    - Window Duration: 2.0 seconds
    - Window Size: 20 samples
    - Sliding Stride: 0.5 seconds = 5 samples

CRITICAL CAUSALITY INVARIANT:
The window ending at timestamp T uses data ONLY up to and including T:
    window[t] in [T - 1.9s, ..., T]
Zero future padding. Zero centered windows. Zero lookahead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import numpy as np

WINDOW_SIZE_SAMPLES: int = 20
DEFAULT_STRIDE_SAMPLES: int = 5  # 0.5 s at 10 Hz
NUM_CHANNELS: int = 9


@dataclass(frozen=True)
class ExtractedWindow:
    """Represents a single extracted 20x9 ML input window and its provenance."""
    window: np.ndarray             # (20, 9) float64 feature matrix
    source_file_id: str            # Original recording / trip ID
    driver_id: str                 # Driver identifier
    start_timestamp_ns: int        # Timestamp of earliest sample in window (t_{i-19})
    end_timestamp_ns: int          # Timestamp of latest sample in window (t_i)
    start_idx: int                 # Source index of earliest sample
    end_idx: int                   # Source index of latest sample (window end)
    is_valid: bool                 # True if all 20 samples pass validation criteria


def extract_causal_windows(
    features: np.ndarray,
    timestamps_ns: np.ndarray,
    is_validated: Optional[np.ndarray] = None,
    source_file_id: str = "",
    driver_id: str = "",
    window_size: int = WINDOW_SIZE_SAMPLES,
    stride_samples: int = DEFAULT_STRIDE_SAMPLES,
) -> List[ExtractedWindow]:
    """Extract strictly causal sliding feature windows.

    Args:
        features: (N, 9) feature matrix.
        timestamps_ns: (N,) int64 timestamps in nanoseconds.
        is_validated: Optional (N,) bool array indicating valid samples.
        source_file_id: Traceability identifier for source recording.
        driver_id: Driver identifier for split isolation.
        window_size: Number of samples per window (strictly 20).
        stride_samples: Step between consecutive window ends (5 = 0.5 s at 10 Hz).

    Returns:
        List of ExtractedWindow objects.
    """
    n_samples = len(timestamps_ns)
    if features.shape[0] != n_samples or features.shape[1] != NUM_CHANNELS:
        raise ValueError(
            f"features shape ({features.shape}) must match ({n_samples}, {NUM_CHANNELS})"
        )

    if window_size <= 0 or stride_samples <= 0:
        raise ValueError("window_size and stride_samples must be strictly positive")

    if is_validated is None:
        val_mask = np.ones(n_samples, dtype=bool)
    else:
        if len(is_validated) != n_samples:
            raise ValueError(f"is_validated length ({len(is_validated)}) != timestamps ({n_samples})")
        val_mask = np.asarray(is_validated, dtype=bool)

    windows: List[ExtractedWindow] = []

    # Insufficient history for even one window
    if n_samples < window_size:
        return windows

    # Window end index i starts at window_size - 1 (index 19 for window_size=20)
    # advances by stride_samples up to n_samples - 1
    for end_idx in range(window_size - 1, n_samples, stride_samples):
        start_idx = end_idx - window_size + 1

        sub_features = features[start_idx : end_idx + 1]  # Shape: (window_size, 9)
        sub_valid = val_mask[start_idx : end_idx + 1]
        sub_ts = timestamps_ns[start_idx : end_idx + 1]

        # Invariants & Validity checks:
        # 1. All samples in window must be validated
        # 2. All features must be finite
        # 3. Timestamps within window must be strictly non-decreasing
        all_validated = bool(np.all(sub_valid))
        all_finite = bool(np.all(np.isfinite(sub_features)))
        monotonic_ts = bool(np.all(np.diff(sub_ts) >= 0))

        is_window_valid = all_validated and all_finite and monotonic_ts

        extracted = ExtractedWindow(
            window=np.copy(sub_features),
            source_file_id=source_file_id,
            driver_id=driver_id,
            start_timestamp_ns=int(sub_ts[0]),
            end_timestamp_ns=int(sub_ts[-1]),
            start_idx=start_idx,
            end_idx=end_idx,
            is_valid=is_window_valid,
        )
        windows.append(extracted)

    return windows
