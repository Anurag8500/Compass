"""GNSS Outage Detection and Persistent Rejection Tracking (Phase 10).

In accordance with Master Plan Section 17 and Trace Part 19:
- Outage detection is strictly time- and timestamp-based (no sample-count assumptions).
- Incorporates an outage-confirmation grace period to prevent overreacting to single
  missed fixes or transient jitter.
- Separates physical absence of fixes (missing fix / prolonged gap) from persistent
  innovation-gate rejection of received fixes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Optional
import numpy as np


class OutageCondition(str, Enum):
    """Classification of the GNSS availability and outage condition."""
    NOMINAL = "NOMINAL"
    GRACE_PERIOD = "GRACE_PERIOD"
    CONFIRMED_OUTAGE = "CONFIRMED_OUTAGE"
    PERSISTENT_REJECTION = "PERSISTENT_REJECTION"


@dataclass(frozen=True)
class OutageDetectorConfig:
    """Configuration parameters for time-based outage detection.

    Attributes:
        expected_interval_s: Expected nominal period between GNSS fixes [s] (default 1.0s for 1 Hz).
        grace_period_s: Grace period after expected fix before confirming outage [s] (default 2.0s).
        persistent_rejection_threshold: Number of consecutive ESKF innovation gate rejections
            required to declare a persistent rejection condition (default 3).
    """
    expected_interval_s: float = 1.0
    grace_period_s: float = 2.0
    persistent_rejection_threshold: int = 3

    @property
    def outage_timeout_s(self) -> float:
        """Total elapsed time without a fix required to confirm an outage."""
        return self.expected_interval_s + self.grace_period_s


@dataclass(frozen=True)
class OutageStatus:
    """Snapshot of current GNSS outage evaluation.

    Attributes:
        condition: Current outage classification (NOMINAL, GRACE_PERIOD, CONFIRMED_OUTAGE, PERSISTENT_REJECTION).
        is_outage: True if GNSS is confirmed unavailable or unusable (CONFIRMED_OUTAGE or PERSISTENT_REJECTION).
        time_since_last_fix_s: Elapsed time in seconds since the last received GNSS fix.
        consecutive_rejections: Number of consecutive received fixes rejected by the ESKF gate.
        last_fix_timestamp_ns: Timestamp of the last received fix in nanoseconds, or None.
        outage_start_timestamp_ns: Timestamp when the current outage condition was entered, or None.
    """
    condition: OutageCondition
    is_outage: bool
    time_since_last_fix_s: float
    consecutive_rejections: int
    last_fix_timestamp_ns: Optional[int]
    outage_start_timestamp_ns: Optional[int]


class GNSSOutageDetector:
    """Time-aware detector tracking GNSS signal absence and innovation gate rejections."""

    def __init__(self, config: Optional[OutageDetectorConfig] = None) -> None:
        self.config = config or OutageDetectorConfig()
        self._last_fix_timestamp_ns: Optional[int] = None
        self._session_start_timestamp_ns: Optional[int] = None
        self._consecutive_rejections: int = 0
        self._outage_start_timestamp_ns: Optional[int] = None
        self._total_fixes_received: int = 0
        self._total_fixes_accepted: int = 0
        self._total_fixes_rejected: int = 0

    def reset(self, session_start_timestamp_ns: Optional[int] = None) -> None:
        """Reset all internal state and counters."""
        self._last_fix_timestamp_ns = None
        self._session_start_timestamp_ns = int(session_start_timestamp_ns) if session_start_timestamp_ns is not None else None
        self._consecutive_rejections = 0
        self._outage_start_timestamp_ns = None
        self._total_fixes_received = 0
        self._total_fixes_accepted = 0
        self._total_fixes_rejected = 0

    def record_fix_received(self, timestamp_ns: int) -> None:
        """Record the arrival of a GNSS fix sample."""
        self._last_fix_timestamp_ns = int(timestamp_ns)
        self._total_fixes_received += 1

    def record_update_result(self, timestamp_ns: int, applied: bool) -> None:
        """Record whether the ESKF update for this fix was accepted or rejected."""
        if applied:
            self._total_fixes_accepted += 1
            self._consecutive_rejections = 0
            # If we were previously in outage due to persistent rejection, clearing rejections ends it
            if self._outage_start_timestamp_ns is not None and self._consecutive_rejections == 0:
                self._outage_start_timestamp_ns = None
        else:
            self._total_fixes_rejected += 1
            self._consecutive_rejections += 1
            if (
                self._consecutive_rejections >= self.config.persistent_rejection_threshold
                and self._outage_start_timestamp_ns is None
            ):
                self._outage_start_timestamp_ns = int(timestamp_ns)

    def evaluate_outage(self, current_timestamp_ns: int) -> OutageStatus:
        """Evaluate the current outage condition at the given evaluation timestamp.

        Args:
            current_timestamp_ns: Current IMU or system epoch in nanoseconds.

        Returns:
            OutageStatus structure with exact outage classification and telemetry.
        """
        t_ns = int(current_timestamp_ns)

        if self._last_fix_timestamp_ns is None:
            if self._session_start_timestamp_ns is not None:
                elapsed_init_s = max(0.0, (t_ns - self._session_start_timestamp_ns) * 1e-9)
                if elapsed_init_s <= self.config.outage_timeout_s:
                    cond = (
                        OutageCondition.NOMINAL
                        if elapsed_init_s <= self.config.expected_interval_s
                        else OutageCondition.GRACE_PERIOD
                    )
                    return OutageStatus(
                        condition=cond,
                        is_outage=False,
                        time_since_last_fix_s=elapsed_init_s,
                        consecutive_rejections=0,
                        last_fix_timestamp_ns=None,
                        outage_start_timestamp_ns=None,
                    )

            # Timeout elapsed without any fix
            if self._outage_start_timestamp_ns is None:
                self._outage_start_timestamp_ns = t_ns
            return OutageStatus(
                condition=OutageCondition.CONFIRMED_OUTAGE,
                is_outage=True,
                time_since_last_fix_s=float("inf"),
                consecutive_rejections=self._consecutive_rejections,
                last_fix_timestamp_ns=None,
                outage_start_timestamp_ns=self._outage_start_timestamp_ns,
            )

        elapsed_s = max(0.0, (t_ns - self._last_fix_timestamp_ns) * 1e-9)

        # 1. Check persistent rejection
        if self._consecutive_rejections >= self.config.persistent_rejection_threshold:
            if self._outage_start_timestamp_ns is None:
                self._outage_start_timestamp_ns = t_ns
            return OutageStatus(
                condition=OutageCondition.PERSISTENT_REJECTION,
                is_outage=True,
                time_since_last_fix_s=elapsed_s,
                consecutive_rejections=self._consecutive_rejections,
                last_fix_timestamp_ns=self._last_fix_timestamp_ns,
                outage_start_timestamp_ns=self._outage_start_timestamp_ns,
            )

        # 2. Check time-based signal absence
        if elapsed_s > self.config.outage_timeout_s:
            if self._outage_start_timestamp_ns is None:
                # Outage officially confirmed at the expiration of the timeout
                timeout_ns = int(self.config.outage_timeout_s * 1e9)
                self._outage_start_timestamp_ns = self._last_fix_timestamp_ns + timeout_ns
            return OutageStatus(
                condition=OutageCondition.CONFIRMED_OUTAGE,
                is_outage=True,
                time_since_last_fix_s=elapsed_s,
                consecutive_rejections=self._consecutive_rejections,
                last_fix_timestamp_ns=self._last_fix_timestamp_ns,
                outage_start_timestamp_ns=self._outage_start_timestamp_ns,
            )

        if elapsed_s > self.config.expected_interval_s:
            # Within grace period: fix delayed, but outage NOT yet confirmed
            return OutageStatus(
                condition=OutageCondition.GRACE_PERIOD,
                is_outage=False,
                time_since_last_fix_s=elapsed_s,
                consecutive_rejections=self._consecutive_rejections,
                last_fix_timestamp_ns=self._last_fix_timestamp_ns,
                outage_start_timestamp_ns=None,
            )

        # Signal arriving nominally
        self._outage_start_timestamp_ns = None
        return OutageStatus(
            condition=OutageCondition.NOMINAL,
            is_outage=False,
            time_since_last_fix_s=elapsed_s,
            consecutive_rejections=self._consecutive_rejections,
            last_fix_timestamp_ns=self._last_fix_timestamp_ns,
            outage_start_timestamp_ns=None,
        )

    @property
    def total_fixes_received(self) -> int:
        return self._total_fixes_received

    @property
    def total_fixes_accepted(self) -> int:
        return self._total_fixes_accepted

    @property
    def total_fixes_rejected(self) -> int:
        return self._total_fixes_rejected


def generate_synthetic_outage_mask(
    timestamps_ns: list[int] | np.ndarray,
    outage_start_offset_s: float,
    outage_duration_s: float,
    session_start_ns: Optional[int] = None,
) -> np.ndarray:
    """Generate a boolean mask indicating GNSS availability.

    Args:
        timestamps_ns: Sequence of sample timestamps in nanoseconds.
        outage_start_offset_s: Seconds from session start when GNSS becomes unavailable.
        outage_duration_s: Total seconds of GNSS blackout.
        session_start_ns: Optional anchor timestamp (defaults to timestamps_ns[0]).

    Returns:
        bool ndarray of same length: True = GNSS AVAILABLE; False = GNSS MASKED (OUTAGE).
    """
    ts = np.asarray(timestamps_ns, dtype=np.int64)
    if len(ts) == 0:
        return np.array([], dtype=bool)

    t0 = ts[0] if session_start_ns is None else int(session_start_ns)
    elapsed_s = (ts - t0) * 1e-9

    outage_start_s = float(outage_start_offset_s)
    outage_end_s = outage_start_s + float(outage_duration_s)

    in_outage = (elapsed_s >= outage_start_s) & (elapsed_s < outage_end_s)
    return ~in_outage
