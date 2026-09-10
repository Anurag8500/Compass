"""Strict Causal Window Buffer for ML Features (Phase 9).

Ensures that neural models receive strictly causal (20, 9) canonical feature windows
composed entirely of past and present samples (t_sample <= t_update).
No look-ahead or future information leakage is permitted.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple
import numpy as np

from ml.data.features import compute_canonical_features

WINDOW_LENGTH: int = 20


@dataclass(frozen=True)
class IMUSample:
    """Individual vehicle-frame motion sample."""
    timestamp_ns: int
    f_m_v: np.ndarray       # (3,) specific force in vehicle frame (m/s^2)
    omega_m_v: np.ndarray   # (3,) angular velocity in vehicle frame (rad/s)


class CausalWindowBuffer:
    """Rolling causal buffer maintaining the last 20 motion samples at 10 Hz."""

    def __init__(self, max_length: int = WINDOW_LENGTH) -> None:
        self.max_length = int(max_length)
        self._buffer: Deque[IMUSample] = deque(maxlen=self.max_length)

    def reset(self) -> None:
        """Clear all buffered history."""
        self._buffer.clear()

    @property
    def count(self) -> int:
        """Number of samples currently buffered."""
        return len(self._buffer)

    @property
    def is_full(self) -> bool:
        """True if exactly max_length samples are available."""
        return len(self._buffer) >= self.max_length

    def push(
        self,
        timestamp_ns: int,
        f_m_v: np.ndarray,
        omega_m_v: np.ndarray,
    ) -> None:
        """Push a fresh vehicle-frame motion sample into the causal buffer.

        Args:
            timestamp_ns: Sample timestamp in nanoseconds.
            f_m_v: (3,) vehicle-frame specific force.
            omega_m_v: (3,) vehicle-frame angular velocity.
        """
        f_arr = np.asarray(f_m_v, dtype=np.float64).reshape(3)
        w_arr = np.asarray(omega_m_v, dtype=np.float64).reshape(3)
        t_ns = int(timestamp_ns)

        if len(self._buffer) > 0 and t_ns <= self._buffer[-1].timestamp_ns:
            # Monotonicity enforcement: ignore duplicate or out-of-order timestamps
            return

        self._buffer.append(IMUSample(timestamp_ns=t_ns, f_m_v=f_arr, omega_m_v=w_arr))

    def get_causal_window(
        self,
        current_timestamp_ns: Optional[int] = None,
    ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
        """Retrieve the canonical (20, 9) feature array strictly causal with respect to current_timestamp_ns.

        Args:
            current_timestamp_ns: Optional reference timestamp. All samples must satisfy
                                  sample_ts <= current_timestamp_ns.

        Returns:
            Tuple of (success, features_20x9, timestamps_20, reason_code).
        """
        if len(self._buffer) < self.max_length:
            return False, None, None, "INSUFFICIENT_HISTORY"

        samples = list(self._buffer)
        if len(samples) != self.max_length:
            return False, None, None, "INSUFFICIENT_HISTORY"

        ts_arr = np.array([s.timestamp_ns for s in samples], dtype=np.int64)

        # Strict causality check
        if current_timestamp_ns is not None:
            if np.any(ts_arr > int(current_timestamp_ns)):
                return False, None, None, "LOOKAHEAD_VIOLATION"

        f_mat = np.array([s.f_m_v for s in samples], dtype=np.float64)
        w_mat = np.array([s.omega_m_v for s in samples], dtype=np.float64)

        try:
            feats = compute_canonical_features(ts_arr, f_mat, w_mat)
        except Exception as e:
            return False, None, None, f"FEATURE_COMPUTATION_ERROR_{type(e).__name__}"

        return True, feats, ts_arr, None
