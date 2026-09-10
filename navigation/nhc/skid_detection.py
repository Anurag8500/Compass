"""Conservative Skid / Slip and Innovation Inconsistency Detection for NHC (Phase 11).

In accordance with Phase 11 safety rules:
1. NHC must NEVER force a false zero lateral/vertical velocity constraint during
   a genuine skid, sharp turn, abnormal motion, or large attitude error.
2. The primary line of defense is statistical innovation consistency:
       d^2 = y^T S^-1 y
   - Low innovation (d^2 <= gate_threshold) -> NORMAL: apply with base R_nhc.
   - Elevated innovation (gate_threshold < d^2 <= severe_threshold) -> RELAXED: adaptively inflate R_nhc.
   - Severe inconsistency (d^2 > severe_threshold) -> SKIPPED: omit update entirely.
3. Kinematic dynamic monitors (yaw rate |w_z|, lateral specific force |f_y|) provide
   secondary conservative alerts during aggressive cornering.
4. Distinguishes statistical innovation inconsistency from confirmed physical skid in telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Optional, Tuple
import numpy as np


class NHCStatus(str, Enum):
    """Execution status and relaxation level for an NHC cycle."""
    NORMAL = "NORMAL"
    RELAXED = "RELAXED"
    SKIPPED = "SKIPPED"
    SKIPPED_STATIONARY = "SKIPPED_STATIONARY"
    SKIPPED_LOW_SPEED = "SKIPPED_LOW_SPEED"
    SKIPPED_UNALIGNED_FRAME = "SKIPPED_UNALIGNED_FRAME"
    NOT_ATTEMPTED = "NOT_ATTEMPTED"


@dataclass(frozen=True)
class SkidDetectorConfig:
    """Configuration for conservative NHC consistency and dynamic relaxation.

    Attributes:
        chi2_gate_threshold: Normalized innovation squared threshold for normal acceptance
            (default 9.210 = chi2_2(0.99)).
        severe_gate_threshold: Threshold above which NHC is completely skipped to avoid
            estimator corruption (default 16.0).
        max_inflation_factor: Upper bound on covariance scale factor when relaxed (default 25.0).
        max_yaw_rate_rads: Yaw rate magnitude above which cornering slip angle becomes
            significant (default 0.70 rad/s ~ 40 deg/s).
        max_lateral_accel_mps2: Lateral acceleration magnitude above which centripetal
            cornering induces noticeable tire slip (default 3.5 m/s^2 ~ 0.35g).
    """
    chi2_gate_threshold: float = 9.210
    severe_gate_threshold: float = 16.0
    max_inflation_factor: float = 25.0
    max_yaw_rate_rads: float = 0.70
    max_lateral_accel_mps2: float = 3.5


@dataclass(frozen=True)
class SkidEvaluationResult:
    """Detailed evaluation from the NHC consistency and relaxation detector.

    Attributes:
        status: Actionable decision (NORMAL, RELAXED, SKIPPED).
        applied: Whether an ESKF update should be executed.
        inflation_factor: Multiplier for measurement covariance R_nhc (>= 1.0).
        nis: Evaluated normalized innovation squared (Mahalanobis distance squared).
        reason: Explanatory rationale code.
        is_kinematically_dynamic: Whether high angular velocity or lateral accel was detected.
    """
    status: NHCStatus
    applied: bool
    inflation_factor: float
    nis: float
    reason: str
    is_kinematically_dynamic: bool = False


class SkidSlipDetector:
    """Conservative consistency and relaxation evaluator for NHC pseudo-measurements."""

    def __init__(self, config: Optional[SkidDetectorConfig] = None) -> None:
        self.config = config or SkidDetectorConfig()

    def evaluate(
        self,
        nis: float,
        omega_v: Optional[np.ndarray] = None,
        f_v: Optional[np.ndarray] = None,
    ) -> SkidEvaluationResult:
        """Evaluate NHC innovation consistency and dynamic conditions.

        Args:
            nis: Normalized innovation squared d^2 = y^T S^-1 y for the 2D measurement.
            omega_v: Optional (3,) vehicle-frame angular velocity [rad/s].
            f_v: Optional (3,) vehicle-frame specific force [m/s^2].

        Returns:
            SkidEvaluationResult detailing status, inflation factor, and reason code.
        """
        if not math.isfinite(nis) or nis < 0.0:
            return SkidEvaluationResult(
                status=NHCStatus.SKIPPED,
                applied=False,
                inflation_factor=1.0,
                nis=float("nan") if not math.isfinite(nis) else nis,
                reason="NON_FINITE_INNOVATION",
                is_kinematically_dynamic=False,
            )

        # 1. Check kinematic dynamic indicators (yaw rate and lateral acceleration)
        yaw_rate = 0.0
        lat_accel = 0.0
        high_yaw = False
        high_lat_accel = False

        if omega_v is not None:
            w_arr = np.asarray(omega_v, dtype=np.float64).reshape(3)
            yaw_rate = abs(float(w_arr[2]))
            high_yaw = yaw_rate > self.config.max_yaw_rate_rads

        if f_v is not None:
            f_arr = np.asarray(f_v, dtype=np.float64).reshape(3)
            lat_accel = abs(float(f_arr[1]))
            high_lat_accel = lat_accel > self.config.max_lateral_accel_mps2

        is_dynamic = bool(high_yaw or high_lat_accel)

        # 2. Severe innovation inconsistency check -> SKIP
        if nis > self.config.severe_gate_threshold:
            return SkidEvaluationResult(
                status=NHCStatus.SKIPPED,
                applied=False,
                inflation_factor=1.0,
                nis=nis,
                reason="SKIPPED_SEVERE_INNOVATION",
                is_kinematically_dynamic=is_dynamic,
            )

        # 3. Dynamic cornering check with high NIS -> SKIP
        if is_dynamic and nis > self.config.chi2_gate_threshold:
            return SkidEvaluationResult(
                status=NHCStatus.SKIPPED,
                applied=False,
                inflation_factor=1.0,
                nis=nis,
                reason="SKIPPED_DYNAMIC_CORNERING_INCONSISTENCY",
                is_kinematically_dynamic=True,
            )

        # 4. Elevated NIS or moderate dynamic motion -> RELAX (adaptive covariance inflation)
        if nis > self.config.chi2_gate_threshold or is_dynamic:
            # Scale covariance smoothly with NIS ratio
            nis_ratio = nis / max(self.config.chi2_gate_threshold, 1e-3)
            inflation = min(self.config.max_inflation_factor, max(1.0, float(nis_ratio)))
            
            # If dynamic motion alone triggered relaxation
            if is_dynamic and nis <= self.config.chi2_gate_threshold:
                inflation = max(2.0, inflation)
                reason = "ACCEPTED_RELAXED_KINEMATIC"
            else:
                reason = "ACCEPTED_RELAXED_INNOVATION"

            return SkidEvaluationResult(
                status=NHCStatus.RELAXED,
                applied=True,
                inflation_factor=inflation,
                nis=nis,
                reason=reason,
                is_kinematically_dynamic=is_dynamic,
            )

        # 5. Normal statistically consistent measurement -> ACCEPT with base covariance
        return SkidEvaluationResult(
            status=NHCStatus.NORMAL,
            applied=True,
            inflation_factor=1.0,
            nis=nis,
            reason="ACCEPTED_NORMAL",
            is_kinematically_dynamic=False,
        )
