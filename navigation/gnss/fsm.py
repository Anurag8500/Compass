"""Outage-detection finite state machine for GNSS/INS navigation.

Transitions between three states based on per-row trust scores produced by
compute_trust_score(). Designed to be robust against single noisy readings by
requiring conditions to hold continuously for a minimum duration before any
state change is confirmed.

States
------
GNSS_AIDED   : GPS is healthy; navigation uses GNSS measurements normally.
DR_ONLY      : GPS outage confirmed; navigation falls back to dead-reckoning only.
REACQUIRING  : Trust has risen above the low threshold but has not yet held
               stable long enough to confirm full recovery.

Transitions
-----------
GNSS_AIDED   → DR_ONLY      : trust stays < LOW_TRUST_THRESHOLD for >= GRACE_PERIOD_MS
DR_ONLY      → REACQUIRING  : trust rises >= LOW_TRUST_THRESHOLD (immediate)
REACQUIRING  → GNSS_AIDED   : trust stays >= HIGH_TRUST_THRESHOLD for >= CONVERGENCE_MS
REACQUIRING  → DR_ONLY      : trust drops < LOW_TRUST_THRESHOLD before convergence completes
"""

from enum import Enum
from typing import List, Tuple


# Score below which a GPS fix is considered untrustworthy
LOW_TRUST_THRESHOLD: float = 0.3

# Score above which a GPS fix is considered reliably good
HIGH_TRUST_THRESHOLD: float = 0.7

# How long trust must stay below LOW_TRUST_THRESHOLD before declaring an outage (ms)
GRACE_PERIOD_MS: float = 2000.0

# How long trust must stay above HIGH_TRUST_THRESHOLD before confirming recovery (ms)
CONVERGENCE_MS: float = 1000.0


# ---------------------------------------------------------------------------
# State enum
# ---------------------------------------------------------------------------

class GNSSState(str, Enum):
    GNSS_AIDED   = "GNSS_AIDED"
    DR_ONLY      = "DR_ONLY"
    REACQUIRING  = "REACQUIRING"


# ---------------------------------------------------------------------------
# FSM class
# ---------------------------------------------------------------------------

class OutageFSM:
    """Finite state machine that tracks GNSS outage and recovery phases.

    Call update(trust_score, timestamp_ms) once per GPS row.  The method
    returns the state that is active *after* processing that row.

    Args:
        low_threshold:   Override for LOW_TRUST_THRESHOLD.
        high_threshold:  Override for HIGH_TRUST_THRESHOLD.
        grace_period_ms: Override for GRACE_PERIOD_MS.
        convergence_ms:  Override for CONVERGENCE_MS.
    """

    def __init__(
        self,
        low_threshold:   float = LOW_TRUST_THRESHOLD,
        high_threshold:  float = HIGH_TRUST_THRESHOLD,
        grace_period_ms: float = GRACE_PERIOD_MS,
        convergence_ms:  float = CONVERGENCE_MS,
    ) -> None:
        self._low_thresh   = low_threshold
        self._high_thresh  = high_threshold
        self._grace_ms     = grace_period_ms
        self._conv_ms      = convergence_ms

        self._state: GNSSState = GNSSState.GNSS_AIDED

        # Timestamp at which the current pending-transition timer started.
        # None means no timer is running.
        self._timer_start_ms: float | None = None

        # Full history of (timestamp_ms, state) after every update() call.
        self._history: List[Tuple[float, GNSSState]] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, trust_score: float, timestamp_ms: float) -> GNSSState:
        """Process one GPS fix and return the current state.

        Args:
            trust_score:   Float in [0, 1] from compute_trust_score().
            timestamp_ms:  Absolute timestamp for this row (TIME SINCE START ms).

        Returns:
            The active GNSSState after evaluating this row.
        """
        if self._state == GNSSState.GNSS_AIDED:
            self._state = self._update_gnss_aided(trust_score, timestamp_ms)

        elif self._state == GNSSState.DR_ONLY:
            self._state = self._update_dr_only(trust_score, timestamp_ms)

        elif self._state == GNSSState.REACQUIRING:
            self._state = self._update_reacquiring(trust_score, timestamp_ms)

        self._history.append((timestamp_ms, self._state))
        return self._state

    def get_state_history(self) -> List[Tuple[float, GNSSState]]:
        """Return the full list of (timestamp_ms, state) pairs recorded so far."""
        return list(self._history)

    @property
    def state(self) -> GNSSState:
        """Current state without advancing the machine."""
        return self._state

    # ------------------------------------------------------------------
    # Per-state logic (internal)
    # ------------------------------------------------------------------

    def _update_gnss_aided(self, trust: float, ts: float) -> GNSSState:
        """GNSS_AIDED: wait for trust to stay low for GRACE_PERIOD_MS."""
        if trust <= self._low_thresh:
            # Start or continue the grace-period timer
            if self._timer_start_ms is None:
                self._timer_start_ms = ts
            elif (ts - self._timer_start_ms) >= self._grace_ms:
                # Outage confirmed — transition
                self._timer_start_ms = None
                return GNSSState.DR_ONLY
        else:
            # Trust recovered; reset any pending timer
            self._timer_start_ms = None

        return GNSSState.GNSS_AIDED

    def _update_dr_only(self, trust: float, ts: float) -> GNSSState:
        """DR_ONLY: any rise above LOW_TRUST_THRESHOLD triggers REACQUIRING immediately."""
        if trust > self._low_thresh:
            self._timer_start_ms = None   # convergence timer starts fresh in next state
            return GNSSState.REACQUIRING

        return GNSSState.DR_ONLY

    def _update_reacquiring(self, trust: float, ts: float) -> GNSSState:
        """REACQUIRING: wait for trust to stay high for CONVERGENCE_MS; bail if it drops."""
        if trust <= self._low_thresh:
            # Trust collapsed before convergence — back to DR_ONLY
            self._timer_start_ms = None
            return GNSSState.DR_ONLY

        if trust >= self._high_thresh:
            # Start or continue convergence timer
            if self._timer_start_ms is None:
                self._timer_start_ms = ts
            elif (ts - self._timer_start_ms) >= self._conv_ms:
                # Stable recovery confirmed — transition
                self._timer_start_ms = None
                return GNSSState.GNSS_AIDED
        else:
            # Trust is between thresholds — not high enough to converge,
            # not low enough to abort; hold in REACQUIRING, reset convergence timer
            self._timer_start_ms = None

        return GNSSState.REACQUIRING
