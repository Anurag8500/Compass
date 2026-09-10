"""Authoritative Three-State GNSS Finite State Machine (Phase 10).

In accordance with Master Plan Section 17 and Trace Part 25:
- The FSM strictly contains exactly THREE states:
    GNSS_AIDED ⇄ DR_ONLY ⇄ REACQUIRING
- "Degraded GNSS" is NOT a fourth FSM state; it lives entirely inside GNSS_AIDED
  via continuous trust score weighting.
- A fourth state for "long vs. short outage" is rejected: the ESKF error-state
  covariance P natively and continuously tracks elapsed outage uncertainty.
- Hysteresis and minimum dwell times prevent rapid oscillation (anti-flapping).
- Explicit transition logging records exact timestamps, reasons, and trust telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from navigation.schemas.state import GNSSMode
from navigation.gnss.outage_detection import OutageCondition


@dataclass(frozen=True)
class FSMConfig:
    """Hysteresis, dwell time, and threshold parameters for the GNSS Mode FSM.

    Attributes:
        min_dwell_time_s: Minimum duration required to remain in a state before allowing
            normal exit transitions [s] (default 2.0s).
        reacq_timeout_s: Maximum duration permitted in REACQUIRING without achieving
            convergence before falling back to DR_ONLY [s] (default 10.0s).
    """
    min_dwell_time_s: float = 2.0
    reacq_timeout_s: float = 10.0


@dataclass(frozen=True)
class GNSSModeTransition:
    """Immutable audit record of a discrete state machine mode transition.

    Attributes:
        timestamp_ns: Nanosecond epoch when transition occurred.
        previous_mode: State prior to transition.
        new_mode: State entered after transition.
        reason: Explanatory transition rationale.
        trust_score: Evaluated GNSS trust score at transition epoch.
        dwell_duration_s: Elapsed time spent in previous state prior to exit.
    """
    timestamp_ns: int
    previous_mode: GNSSMode
    new_mode: GNSSMode
    reason: str
    trust_score: float
    dwell_duration_s: float

    def to_dict(self) -> dict:
        return {
            "timestamp_ns": self.timestamp_ns,
            "previous_mode": self.previous_mode.value,
            "new_mode": self.new_mode.value,
            "reason": self.reason,
            "trust_score": self.trust_score,
            "dwell_duration_s": self.dwell_duration_s,
        }


class GNSSModeFSM:
    """Authoritative 3-state Finite State Machine orchestrating GNSS operational modes."""

    def __init__(
        self,
        config: Optional[FSMConfig] = None,
        initial_mode: GNSSMode = GNSSMode.GNSS_AIDED,
        initial_timestamp_ns: int = 0,
    ) -> None:
        self.config = config or FSMConfig()
        self._current_mode: GNSSMode = initial_mode
        self._mode_entry_timestamp_ns: int = int(initial_timestamp_ns)
        self._transition_history: List[GNSSModeTransition] = []

    def reset(self, initial_mode: GNSSMode = GNSSMode.GNSS_AIDED, timestamp_ns: int = 0) -> None:
        """Reset state machine to initial mode."""
        self._current_mode = initial_mode
        self._mode_entry_timestamp_ns = int(timestamp_ns)
        self._transition_history.clear()

    def force_mode(
        self,
        mode: GNSSMode,
        timestamp_ns: int,
        reason: str = "FORCED_TRANSITION",
        trust_score: float = 1.0,
    ) -> GNSSModeTransition:
        """Forcibly set state machine mode (e.g. for safety abort or supervisory override)."""
        t_ns = int(timestamp_ns)
        dwell_s = self.time_in_current_mode_s(t_ns)
        record = GNSSModeTransition(
            timestamp_ns=t_ns,
            previous_mode=self._current_mode,
            new_mode=mode,
            reason=reason,
            trust_score=float(trust_score),
            dwell_duration_s=dwell_s,
        )
        self._transition_history.append(record)
        self._current_mode = mode
        self._mode_entry_timestamp_ns = t_ns
        return record

    @property
    def current_mode(self) -> GNSSMode:
        return self._current_mode

    @property
    def mode_entry_timestamp_ns(self) -> int:
        return self._mode_entry_timestamp_ns

    @property
    def transition_count(self) -> int:
        return len(self._transition_history)

    @property
    def transition_history(self) -> List[GNSSModeTransition]:
        return list(self._transition_history)

    def time_in_current_mode_s(self, current_timestamp_ns: int) -> float:
        """Calculate elapsed dwell time in the current state."""
        return max(0.0, (int(current_timestamp_ns) - self._mode_entry_timestamp_ns) * 1e-9)

    def evaluate_transition(
        self,
        current_timestamp_ns: int,
        is_outage: bool,
        outage_reason: str,
        is_returning_fix_valid: bool = False,
        is_recovery_converged: bool = False,
        trust_score: float = 1.0,
    ) -> Optional[GNSSModeTransition]:
        """Evaluate FSM state transitions adhering to Master Plan Section 17.

        State transition matrix:
        1. GNSS_AIDED:
           - If outage confirmed or persistent rejection: transition to DR_ONLY.
             (Grace period is enforced upstream in GNSSOutageDetector).
        2. DR_ONLY:
           - If plausible returning fix validated: transition to REACQUIRING.
             (Enforces min_dwell_time_s to prevent rapid flapping).
        3. REACQUIRING:
           - If recovery converged over required fixes: transition to GNSS_AIDED.
           - If fix timeout or implausible fix received: fallback to DR_ONLY.

        Args:
            current_timestamp_ns: Current evaluation epoch in nanoseconds.
            is_outage: True if GNSS outage is confirmed by GNSSOutageDetector.
            outage_reason: Diagnostic string explaining outage condition.
            is_returning_fix_valid: True if a returning fix passed plausibility validation.
            is_recovery_converged: True if bounded-rate recovery has achieved convergence.
            trust_score: Current evaluated continuous trust score.

        Returns:
            GNSSModeTransition if a discrete state change occurred, else None.
        """
        t_ns = int(current_timestamp_ns)
        dwell_s = self.time_in_current_mode_s(t_ns)

        new_mode: Optional[GNSSMode] = None
        transition_reason: Optional[str] = None

        if self._current_mode == GNSSMode.GNSS_AIDED:
            # Entry to DR_ONLY on confirmed outage
            if is_outage:
                # Emergency safety condition: outage confirmed after grace period
                new_mode = GNSSMode.DR_ONLY
                transition_reason = f"OUTAGE_DETECTED_{outage_reason}"

        elif self._current_mode == GNSSMode.DR_ONLY:
            # Entry to REACQUIRING when a valid returning fix is received
            if is_returning_fix_valid:
                if dwell_s >= self.config.min_dwell_time_s:
                    new_mode = GNSSMode.REACQUIRING
                    transition_reason = "VALID_RETURNING_FIX_ACQUIRED"

        elif self._current_mode == GNSSMode.REACQUIRING:
            # Check authoritative reacquisition timeout, confirmed signal loss, or convergence
            if dwell_s >= self.config.reacq_timeout_s:
                # FSM-authoritative maximum duration in REACQUIRING exceeded without convergence
                new_mode = GNSSMode.DR_ONLY
                transition_reason = f"REACQUISITION_TIMEOUT_EXCEEDED_{dwell_s:.2f}S"
            elif is_outage and outage_reason == OutageCondition.CONFIRMED_OUTAGE.value:
                new_mode = GNSSMode.DR_ONLY
                transition_reason = f"REACQUISITION_FAILED_{outage_reason}"
            elif is_recovery_converged:
                if dwell_s >= self.config.min_dwell_time_s:
                    new_mode = GNSSMode.GNSS_AIDED
                    transition_reason = "REACQUISITION_CONVERGENCE_ACHIEVED"

        if new_mode is not None and new_mode != self._current_mode:
            record = GNSSModeTransition(
                timestamp_ns=t_ns,
                previous_mode=self._current_mode,
                new_mode=new_mode,
                reason=transition_reason or "UNKNOWN",
                trust_score=float(trust_score),
                dwell_duration_s=dwell_s,
            )
            self._transition_history.append(record)
            self._current_mode = new_mode
            self._mode_entry_timestamp_ns = t_ns
            return record

        return None
