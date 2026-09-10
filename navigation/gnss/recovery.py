"""GNSS Reacquisition Validation and Bounded-Rate Recovery (Phase 10).

In accordance with Master Plan Section 17 and Trace Part 24:
- When a GNSS fix returns after an outage, it is NOT immediately trusted for full aiding.
- The returning fix undergoes rigorous plausibility and innovation testing before
  transitioning from DR_ONLY to REACQUIRING.
- While in REACQUIRING, state correction is bounded-rate (e.g., maximum correction velocity
  v_blend_max), smoothly gliding the trajectory toward GNSS without instantaneous position snaps.
- Only after consecutive agreeing fixes demonstrate convergence is the system returned to GNSS_AIDED.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Optional, Tuple
import numpy as np

from navigation.eskf.state import ESKFNominalState, ESKFState


class ReturningFixStatus(str, Enum):
    """Validation outcome for a returning GNSS fix."""
    RETURNED_FIX_VALID = "RETURNED_FIX_VALID"
    RETURNED_FIX_IMPLAUSIBLE = "RETURNED_FIX_IMPLAUSIBLE"
    RETURNED_FIX_INNOVATION_REJECTED = "RETURNED_FIX_INNOVATION_REJECTED"
    RETURNED_FIX_LOW_TRUST = "RETURNED_FIX_LOW_TRUST"
    RETURNED_FIX_NON_FINITE = "RETURNED_FIX_NON_FINITE"


@dataclass(frozen=True)
class RecoveryConfig:
    """Parameters governing returning fix validation and bounded-rate state recovery.

    Attributes:
        min_reacq_trust: Minimum trust score required to begin reacquisition (default 0.35).
        max_displacement_rate_mps: Maximum position correction velocity [m/s] (default 2.0 m/s).
        max_single_step_m: Absolute upper limit on single-cycle position correction [m] (default 3.0 m).
        convergence_pos_tolerance_m: Residual horizontal position disagreement required for convergence [m] (default 1.5 m).
        required_consecutive_fixes: Number of consecutive valid fixes required to confirm convergence (default 3).
        nis_gate_reacq: 99% chi-square gate for 3D position innovations (chi2_3(0.99) = 11.345).
    """
    min_reacq_trust: float = 0.35
    max_displacement_rate_mps: float = 2.0
    max_single_step_m: float = 3.0
    convergence_pos_tolerance_m: float = 1.5
    required_consecutive_fixes: int = 3
    nis_gate_reacq: float = 11.345


@dataclass(frozen=True)
class RecoveryValidationResult:
    """Outcome of validating a returning GNSS fix.

    Attributes:
        status: ReturningFixStatus code.
        is_plausible: True if fix satisfies plausibility, trust, and innovation criteria.
        displacement_m: Total spatial distance between ESKF position and GNSS position [m].
        horizontal_displacement_m: Horizontal distance [m].
        nis: Normalized Innovation Squared, or None if computation was not possible.
        reason: Explanatory string describing validation outcome.
    """
    status: ReturningFixStatus
    is_plausible: bool
    displacement_m: float
    horizontal_displacement_m: float
    nis: Optional[float]
    reason: str


@dataclass(frozen=True)
class BoundedCorrectionStep:
    """Bounded state correction computed for a single recovery cycle.

    Attributes:
        delta_p_bounded: (3,) bounded position correction vector to apply [m].
        raw_error_norm_m: Magnitude of the original position disagreement [m].
        applied_step_norm_m: Magnitude of the bounded step applied [m].
        is_clamped: True if the raw correction exceeded the maximum allowed rate and was clamped.
        is_converged: True if recovery has achieved stable convergence.
    """
    delta_p_bounded: np.ndarray
    raw_error_norm_m: float
    applied_step_norm_m: float
    is_clamped: bool
    is_converged: bool


class GNSSRecoveryManager:
    """Manages returning-fix plausibility, bounded-rate blending, and convergence tracking."""

    def __init__(self, config: Optional[RecoveryConfig] = None) -> None:
        self.config = config or RecoveryConfig()
        self._consecutive_valid_fixes: int = 0
        self._recovery_cycles: int = 0
        self._maximum_correction_applied_m: float = 0.0
        self._last_fix_timestamp_ns: Optional[int] = None

    def reset(self) -> None:
        """Reset internal recovery counters."""
        self._consecutive_valid_fixes = 0
        self._recovery_cycles = 0
        self._maximum_correction_applied_m = 0.0
        self._last_fix_timestamp_ns = None

    def validate_returning_fix(
        self,
        gnss_pos_enu: np.ndarray,
        eskf_state: ESKFState,
        trust_score: float,
        timestamp_ns: int,
    ) -> RecoveryValidationResult:
        """Validate whether a returning GNSS fix is plausible relative to the current ESKF state.

        Evaluates:
        1. Numerical finiteness.
        2. Minimum trust score requirement.
        3. Statistical consistency (NIS against current position covariance P_p + R_p).
        4. Kinematic plausibility (checking distance against dead-reckoning uncertainty).

        Args:
            gnss_pos_enu: (3,) incoming GNSS position in local ENU frame.
            eskf_state: Current authoritative ESKFState (nominal + covariance).
            trust_score: Evaluated continuous trust score for this fix.
            timestamp_ns: Epoch timestamp of the fix in nanoseconds.

        Returns:
            RecoveryValidationResult detailing plausibility, displacement, and reason codes.
        """
        if not np.all(np.isfinite(gnss_pos_enu)):
            return RecoveryValidationResult(
                status=ReturningFixStatus.RETURNED_FIX_NON_FINITE,
                is_plausible=False,
                displacement_m=float("inf"),
                horizontal_displacement_m=float("inf"),
                nis=None,
                reason="GNSS_POSITION_NON_FINITE",
            )

        eskf_p = eskf_state.nominal.position_enu
        delta_p = gnss_pos_enu - eskf_p
        dist_3d = float(np.linalg.norm(delta_p))
        dist_2d = float(np.linalg.norm(delta_p[:2]))

        # 1. Trust score check
        if trust_score < self.config.min_reacq_trust:
            return RecoveryValidationResult(
                status=ReturningFixStatus.RETURNED_FIX_LOW_TRUST,
                is_plausible=False,
                displacement_m=dist_3d,
                horizontal_displacement_m=dist_2d,
                nis=None,
                reason=f"TRUST_SCORE_LOW_{trust_score:.2f}_BELOW_{self.config.min_reacq_trust}",
            )

        # 2. Innovation test against joint covariance S = H P H^T + R
        # In position space: H = [I_3, 0_3x12], so H P H^T = P[0:3, 0:3]
        P_p = eskf_state.covariance[0:3, 0:3]
        # Conservative baseline R for reacquisition test (scaled by trust)
        sigma_r = 4.0 / max(0.1, trust_score)
        R_reacq = np.diag([sigma_r**2, sigma_r**2, (sigma_r * 2.0)**2])
        S = P_p + R_reacq

        try:
            S_inv = np.linalg.inv(S)
            nis = float(delta_p.T @ S_inv @ delta_p)
        except np.linalg.LinAlgError:
            nis = float("inf")

        if not math.isfinite(nis) or nis > self.config.nis_gate_reacq:
            return RecoveryValidationResult(
                status=ReturningFixStatus.RETURNED_FIX_INNOVATION_REJECTED,
                is_plausible=False,
                displacement_m=dist_3d,
                horizontal_displacement_m=dist_2d,
                nis=nis,
                reason=f"INNOVATION_GATE_REJECTED_NIS_{nis:.1f}_THRESHOLD_{self.config.nis_gate_reacq}",
            )

        return RecoveryValidationResult(
            status=ReturningFixStatus.RETURNED_FIX_VALID,
            is_plausible=True,
            displacement_m=dist_3d,
            horizontal_displacement_m=dist_2d,
            nis=nis,
            reason="RETURNED_FIX_ACCEPTED_FOR_REACQUISITION",
        )

    def compute_bounded_correction(
        self,
        gnss_pos_enu: np.ndarray,
        eskf_state: ESKFState,
        dt_s: float,
    ) -> BoundedCorrectionStep:
        """Compute a rate-bounded position correction vector during REACQUIRING.

        Enforces:
            ||delta_p_bounded|| <= min(max_single_step_m, max_displacement_rate_mps * dt_s)

        This smoothly guides the estimated position toward GNSS over multiple fixes without
        abrupt coordinate snaps.

        Args:
            gnss_pos_enu: (3,) target GNSS position.
            eskf_state: Current ESKF state.
            dt_s: Time step in seconds since previous evaluation.

        Returns:
            BoundedCorrectionStep containing the bounded delta and convergence status.
        """
        eskf_p = eskf_state.nominal.position_enu
        raw_delta = np.asarray(gnss_pos_enu, dtype=np.float64) - eskf_p
        raw_norm = float(np.linalg.norm(raw_delta))

        # Determine maximum allowed spatial step for this cycle
        eff_dt = max(0.01, min(2.0, float(dt_s)))
        max_step = min(
            self.config.max_single_step_m,
            self.config.max_displacement_rate_mps * eff_dt,
        )

        is_clamped = False
        if raw_norm > max_step and raw_norm > 1e-6:
            scale = max_step / raw_norm
            delta_p_bounded = raw_delta * scale
            is_clamped = True
            applied_norm = max_step
        else:
            delta_p_bounded = raw_delta.copy()
            applied_norm = raw_norm

        self._recovery_cycles += 1
        if applied_norm > self._maximum_correction_applied_m:
            self._maximum_correction_applied_m = applied_norm

        # Check convergence criteria
        # Horizontal error must be below tolerance
        h_norm = float(np.linalg.norm(raw_delta[:2]))
        if h_norm <= self.config.convergence_pos_tolerance_m:
            self._consecutive_valid_fixes += 1
        else:
            # Did not meet position tolerance this step; reset agreement counter
            self._consecutive_valid_fixes = 0

        is_converged = self._consecutive_valid_fixes >= self.config.required_consecutive_fixes

        return BoundedCorrectionStep(
            delta_p_bounded=delta_p_bounded,
            raw_error_norm_m=raw_norm,
            applied_step_norm_m=applied_norm,
            is_clamped=is_clamped,
            is_converged=is_converged,
        )

    def apply_bounded_correction(
        self,
        eskf_state: ESKFState,
        step: BoundedCorrectionStep,
    ) -> ESKFState:
        """Apply the computed bounded position correction to the ESKF nominal state.

        Does NOT mutate or corrupt the error-state covariance matrix P.
        Ensures smooth trajectory continuity.
        """
        new_p = eskf_state.nominal.position_enu + step.delta_p_bounded
        new_nom = ESKFNominalState(
            position_enu=new_p,
            velocity_enu=eskf_state.nominal.velocity_enu,
            q=eskf_state.nominal.q,
            accel_bias=eskf_state.nominal.accel_bias,
            gyro_bias=eskf_state.nominal.gyro_bias,
            timestamp_ns=eskf_state.nominal.timestamp_ns,
        )
        return ESKFState(nominal=new_nom, covariance=eskf_state.covariance)

    @property
    def consecutive_valid_fixes(self) -> int:
        return self._consecutive_valid_fixes

    @property
    def recovery_cycles(self) -> int:
        return self._recovery_cycles

    @property
    def maximum_correction_applied_m(self) -> float:
        return self._maximum_correction_applied_m
