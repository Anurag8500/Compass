"""GNSS outage detection: trust scoring + FSM driving + fix-gap and persistent-rejection handling.

Bridges raw GNSS rows to the OutageFSM. Three responsibilities:

  1. Calls compute_trust_score() (now NIS-aware) per arriving fix and feeds
     the score into OutageFSM.update().

  2. Silent dropout path — if no fix arrives for > EXPECTED_FIX_INTERVAL_MS,
     a synthetic zero-trust tick is injected so the FSM grace-period timer
     advances during tunnel / underpass scenarios where rows simply stop.

  3. Persistent-rejection path (from Anurag's design) — if the Kalman filter
     has rejected PERSISTENT_REJECTION_THRESHOLD consecutive fixes via its
     innovation gate, the detector signals an outage even though GPS rows are
     still arriving. This catches urban-canyon multipath where the signal
     is physically present but consistently wrong.

     How to use:
         After each ESKF GNSS update, call record_eskf_result(accepted).
         The detector tracks consecutive rejections internally and drives
         the FSM trust score down accordingly.

Architecture
------------
OutageDetector owns no navigation state. It is a pure classification layer
that answers "which operational mode are we in?" given fix quality signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from navigation.gnss.fsm import GNSSState, OutageFSM
from navigation.gnss.trust_score import compute_trust_score, MIN_TRUST_SCORE


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Gap threshold before injecting a zero-trust tick (silent dropout detection).
# 1.5× the nominal 10 Hz GPS fix interval with margin for jitter.
EXPECTED_FIX_INTERVAL_MS: float = 1500.0

# Number of consecutive ESKF innovation-gate rejections before the detector
# forces zero trust on the next FSM update (persistent-rejection detection).
PERSISTENT_REJECTION_THRESHOLD: int = 3


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OutageDetectionResult:
    """Output of OutageDetector.process_fix().

    Attributes:
        state:                FSM state after processing this fix.
        trust_score:          Composite quality score in [0, 1].
        fix_gap_ms:           Ms since previous fix (0.0 on first call).
        consecutive_rejections: Running count of ESKF rejections since last
                              accepted fix. Useful for diagnostics.
    """
    state: GNSSState
    trust_score: float
    fix_gap_ms: float
    consecutive_rejections: int


# ---------------------------------------------------------------------------
# OutageDetector
# ---------------------------------------------------------------------------

class OutageDetector:
    """Drives OutageFSM from raw GNSS rows with gap-filling and rejection tracking.

    Args:
        fsm:                      Optional pre-built OutageFSM (useful in tests).
        expected_fix_interval_ms: Override for EXPECTED_FIX_INTERVAL_MS.
        persistent_rejection_threshold: Override for PERSISTENT_REJECTION_THRESHOLD.
    """

    def __init__(
        self,
        fsm: Optional[OutageFSM] = None,
        expected_fix_interval_ms: float = EXPECTED_FIX_INTERVAL_MS,
        persistent_rejection_threshold: int = PERSISTENT_REJECTION_THRESHOLD,
    ) -> None:
        self._fsm = fsm if fsm is not None else OutageFSM()
        self._expected_interval_ms = expected_fix_interval_ms
        self._rejection_threshold  = persistent_rejection_threshold

        self._last_fix_timestamp_ms: Optional[float] = None
        self._last_nis: Optional[float] = None          # most recent ESKF NIS
        self._consecutive_rejections: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def state(self) -> GNSSState:
        """Current FSM state without advancing."""
        return self._fsm.state

    @property
    def consecutive_rejections(self) -> int:
        """Number of consecutive ESKF gate rejections since last accepted fix."""
        return self._consecutive_rejections

    def record_eskf_result(self, accepted: bool, nis: Optional[float] = None) -> None:
        """Report the outcome of an ESKF GNSS measurement update.

        Call this once per GNSS fix, immediately after the ESKF update step.

        Args:
            accepted: True if the Kalman innovation gate accepted the fix;
                      False if it was rejected (Mahalanobis distance too large).
            nis:      Normalized Innovation Squared from the rejected/accepted
                      update. Used to sharpen the trust score on the next fix.
        """
        if nis is not None:
            self._last_nis = float(nis)

        if accepted:
            self._consecutive_rejections = 0
        else:
            self._consecutive_rejections += 1

    def process_fix(
        self,
        row: Dict[str, Any],
        previous_row: Optional[Dict[str, Any]],
        timestamp_ms: float,
    ) -> OutageDetectionResult:
        """Process one incoming GNSS fix row.

        Computes trust (now NIS-aware), checks for persistent rejection,
        advances the FSM, and records the fix timestamp.

        Args:
            row:          Current GPS data row (keys per trust_score.py).
            previous_row: Immediately preceding GPS row, or None.
            timestamp_ms: Timestamp of this fix in milliseconds.

        Returns:
            OutageDetectionResult with state, score, gap, and rejection count.
        """
        fix_gap_ms = (
            timestamp_ms - self._last_fix_timestamp_ms
            if self._last_fix_timestamp_ms is not None
            else 0.0
        )

        # If consecutive rejections hit the threshold, override trust to zero
        # regardless of what the raw fix quality looks like.
        if self._consecutive_rejections >= self._rejection_threshold:
            trust = MIN_TRUST_SCORE
        else:
            trust = compute_trust_score(row, previous_row, nis=self._last_nis)

        state = self._fsm.update(trust, timestamp_ms)
        self._last_fix_timestamp_ms = timestamp_ms

        return OutageDetectionResult(
            state=state,
            trust_score=trust,
            fix_gap_ms=fix_gap_ms,
            consecutive_rejections=self._consecutive_rejections,
        )

    def tick_no_fix(self, current_timestamp_ms: float) -> GNSSState:
        """Advance the FSM during IMU cycles where no GPS row arrives.

        Injects a zero-trust tick only when the gap exceeds
        EXPECTED_FIX_INTERVAL_MS so the grace-period timer progresses during
        a silent dropout. Short gaps (normal 10 Hz jitter) are ignored.

        Args:
            current_timestamp_ms: Current IMU epoch timestamp in ms.

        Returns:
            Current FSM state.
        """
        if self._last_fix_timestamp_ms is None:
            self._fsm.update(0.0, current_timestamp_ms)
            return self._fsm.state

        gap_ms = current_timestamp_ms - self._last_fix_timestamp_ms
        if gap_ms > self._expected_interval_ms:
            self._fsm.update(0.0, current_timestamp_ms)

        return self._fsm.state

    def reset(self) -> None:
        """Reset all state. Call when starting a new navigation session."""
        self._fsm = OutageFSM()
        self._last_fix_timestamp_ms  = None
        self._last_nis               = None
        self._consecutive_rejections = 0
