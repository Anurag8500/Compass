"""Improved Skid / Slip Detection with Adaptive Thresholds for Phase 11.

Key improvements over baseline anurag-phase-10:
- Speed-dependent lateral acceleration threshold (allows higher cornering at speed)
- Smoother adaptive covariance inflation using sigmoid function
- GNSS-quality-aware relaxation (tighten NHC when GNSS is good, relax when poor)
- Hysteresis to prevent rapid on/off switching
- Kinematic dynamic monitoring with velocity-dependent thresholds
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
    NOT_ATTEMPTED = "NOT_ATTEMPTED"


@dataclass(frozen=True)
class SkidDetectorConfig:
    """Configuration for improved NHC consistency and dynamic relaxation.

    Attributes:
        chi2_gate_threshold: Normalized innovation squared threshold for normal acceptance
            (default 9.210 = chi2_2(0.99)).
        severe_gate_threshold: Threshold above which NHC is completely skipped to avoid
            estimator corruption (default 25.0 - increased from 16.0 to allow more relaxation).
        max_inflation_factor: Upper bound on covariance scale factor when relaxed (default 50.0 - increased).
        base_yaw_rate_rads: Base yaw rate threshold at low speed (default 0.50 rad/s).
        base_lateral_accel_mps2: Base lateral acceleration threshold at low speed (default 2.5 m/s²).
        speed_scaling_factor: Factor for speed-dependent threshold scaling (default 0.1).
        max_yaw_rate_rads: Maximum yaw rate threshold at high speed (default 1.2 rad/s).
        max_lateral_accel_mps2: Maximum lateral acceleration threshold at high speed (default 5.0 m/s²).
        hysteresis_factor: Hysteresis factor to prevent rapid switching (default 0.8).
        enable_gnss_aware: Whether to use GNSS trust score for adaptive relaxation (default True).
        gnss_trust_low_threshold: GNSS trust threshold below which NHC is tightened (default 0.5).
    """
    chi2_gate_threshold: float = 9.210
    severe_gate_threshold: float = 25.0  # Increased from 16.0 for more tolerance
    max_inflation_factor: float = 50.0  # Increased from 25.0
    base_yaw_rate_rads: float = 0.50  # Reduced from 0.70
    base_lateral_accel_mps2: float = 2.5  # Reduced from 3.5
    speed_scaling_factor: float = 0.1
    max_yaw_rate_rads: float = 1.2  # Increased to allow high-speed cornering
    max_lateral_accel_mps2: float = 5.0  # Increased to allow high-speed cornering
    hysteresis_factor: float = 0.8
    enable_gnss_aware: bool = True
    gnss_trust_low_threshold: float = 0.5


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
        effective_yaw_threshold: Effective yaw rate threshold used (speed-dependent).
        effective_lat_accel_threshold: Effective lateral accel threshold used (speed-dependent).
    """
    status: NHCStatus
    applied: bool
    inflation_factor: float
    nis: float
    reason: str
    is_kinematically_dynamic: bool = False
    effective_yaw_threshold: float = 0.0
    effective_lat_accel_threshold: float = 0.0


class SkidSlipDetector:
    """Improved consistency and relaxation evaluator with adaptive thresholds."""

    def __init__(self, config: Optional[SkidDetectorConfig] = None) -> None:
        self.config = config or SkidDetectorConfig()
        self._previous_was_skipped = False  # For hysteresis

    def _compute_speed_dependent_thresholds(
        self, forward_speed: float
    ) -> Tuple[float, float]:
        """Compute speed-dependent yaw rate and lateral acceleration thresholds.

        At higher speeds, we allow higher lateral acceleration (centripetal force scales with v²).
        This prevents false skid detection during legitimate high-speed cornering.

        Args:
            forward_speed: Forward vehicle speed in m/s.

        Returns:
            Tuple of (effective_yaw_threshold, effective_lat_accel_threshold).
        """
        # Sigmoid-like scaling function for smooth transition
        speed_factor = math.tanh(self.config.speed_scaling_factor * forward_speed)
        
        # Interpolate between base and max thresholds
        yaw_threshold = (
            self.config.base_yaw_rate_rads +
            (self.config.max_yaw_rate_rads - self.config.base_yaw_rate_rads) * speed_factor
        )
        
        lat_accel_threshold = (
            self.config.base_lateral_accel_mps2 +
            (self.config.max_lateral_accel_mps2 - self.config.base_lateral_accel_mps2) * speed_factor
        )
        
        return yaw_threshold, lat_accel_threshold

    def _compute_smooth_inflation(
        self, nis: float, chi2_threshold: float
    ) -> float:
        """Compute smooth covariance inflation using sigmoid-like function.

        This provides smoother transitions than the baseline's linear ratio-based inflation,
        reducing filter jitter during borderline conditions.

        Args:
            nis: Normalized innovation squared.
            chi2_threshold: Chi-squared threshold for normal acceptance.

        Returns:
            Covariance inflation factor (>= 1.0).
        """
        if nis <= chi2_threshold:
            return 1.0
        
        # Smooth sigmoid-like transition from 1.0 to max_inflation_factor
        # Using a logistic function centered at 2*chi2_threshold
        center = 2.0 * chi2_threshold
        scale = chi2_threshold
        
        # Logistic function scaled to [1.0, max_inflation_factor]
        ratio = (nis - center) / scale
        sigmoid = 1.0 / (1.0 + math.exp(-ratio))
        
        inflation = 1.0 + (self.config.max_inflation_factor - 1.0) * sigmoid
        return min(self.config.max_inflation_factor, max(1.0, inflation))

    def evaluate(
        self,
        nis: float,
        omega_v: Optional[np.ndarray] = None,
        f_v: Optional[np.ndarray] = None,
        forward_speed: float = 0.0,
        gnss_trust: float = 1.0,
    ) -> SkidEvaluationResult:
        """Evaluate NHC innovation consistency with adaptive thresholds.

        Args:
            nis: Normalized innovation squared d^2 = y^T S^-1 y for the 2D measurement.
            omega_v: Optional (3,) vehicle-frame angular velocity [rad/s].
            f_v: Optional (3,) vehicle-frame specific force [m/s²].
            forward_speed: Forward vehicle speed in m/s for adaptive thresholds.
            gnss_trust: GNSS trust score [0.0, 1.0] for adaptive relaxation.

        Returns:
            SkidEvaluationResult detailing status, inflation factor, and reason code.
        """
        if not math.isfinite(nis) or nis < 0.0:
            self._previous_was_skipped = True
            return SkidEvaluationResult(
                status=NHCStatus.SKIPPED,
                applied=False,
                inflation_factor=1.0,
                nis=float("nan") if not math.isfinite(nis) else nis,
                reason="NON_FINITE_INNOVATION",
                is_kinematically_dynamic=False,
            )

        # Compute speed-dependent thresholds
        yaw_threshold, lat_accel_threshold = self._compute_speed_dependent_thresholds(forward_speed)

        # Check kinematic dynamic indicators
        yaw_rate = 0.0
        lat_accel = 0.0
        high_yaw = False
        high_lat_accel = False

        if omega_v is not None:
            w_arr = np.asarray(omega_v, dtype=np.float64).reshape(3)
            yaw_rate = abs(float(w_arr[2]))
            high_yaw = yaw_rate > yaw_threshold

        if f_v is not None:
            f_arr = np.asarray(f_v, dtype=np.float64).reshape(3)
            lat_accel = abs(float(f_arr[1]))
            high_lat_accel = lat_accel > lat_accel_threshold

        is_dynamic = bool(high_yaw or high_lat_accel)

        # Apply hysteresis: if previously skipped, require lower threshold to re-enable
        if self._previous_was_skipped and is_dynamic:
            # Require lower thresholds to re-enable after a skip
            hysteresis_yaw = yaw_threshold * self.config.hysteresis_factor
            hysteresis_lat = lat_accel_threshold * self.config.hysteresis_factor
            if yaw_rate > hysteresis_yaw or lat_accel > hysteresis_lat:
                self._previous_was_skipped = True
                return SkidEvaluationResult(
                    status=NHCStatus.SKIPPED,
                    applied=False,
                    inflation_factor=1.0,
                    nis=nis,
                    reason="SKIPPED_HYSTERESIS",
                    is_kinematically_dynamic=True,
                    effective_yaw_threshold=hysteresis_yaw,
                    effective_lat_accel_threshold=hysteresis_lat,
                )

        # Severe innovation inconsistency check -> SKIP
        if nis > self.config.severe_gate_threshold:
            self._previous_was_skipped = True
            return SkidEvaluationResult(
                status=NHCStatus.SKIPPED,
                applied=False,
                inflation_factor=1.0,
                nis=nis,
                reason="SKIPPED_SEVERE_INNOVATION",
                is_kinematically_dynamic=is_dynamic,
                effective_yaw_threshold=yaw_threshold,
                effective_lat_accel_threshold=lat_accel_threshold,
            )

        # GNSS-quality-aware relaxation: tighten NHC when GNSS is poor
        gnss_adjustment = 1.0
        if self.config.enable_gnss_aware and gnss_trust < self.config.gnss_trust_low_threshold:
            # Reduce effective NIS threshold when GNSS is poor (need more constraint)
            gnss_adjustment = 0.8
        elif self.config.enable_gnss_aware and gnss_trust > 0.8:
            # Increase effective NIS threshold when GNSS is good (can relax constraint)
            gnss_adjustment = 1.2

        effective_chi2_threshold = self.config.chi2_gate_threshold * gnss_adjustment

        # Dynamic cornering with high NIS -> SKIP (but with higher thresholds than baseline)
        if is_dynamic and nis > effective_chi2_threshold * 1.5:
            self._previous_was_skipped = True
            return SkidEvaluationResult(
                status=NHCStatus.SKIPPED,
                applied=False,
                inflation_factor=1.0,
                nis=nis,
                reason="SKIPPED_DYNAMIC_CORNERING_INCONSISTENCY",
                is_kinematically_dynamic=True,
                effective_yaw_threshold=yaw_threshold,
                effective_lat_accel_threshold=lat_accel_threshold,
            )

        # Elevated NIS or moderate dynamic motion -> RELAX with smooth inflation
        if nis > effective_chi2_threshold or is_dynamic:
            inflation = self._compute_smooth_inflation(nis, effective_chi2_threshold)
            
            # Further adjust inflation based on GNSS quality
            if self.config.enable_gnss_aware:
                if gnss_trust < self.config.gnss_trust_low_threshold:
                    # Reduce inflation when GNSS is poor (need tighter constraint)
                    inflation = max(1.0, inflation * 0.7)
                elif gnss_trust > 0.8:
                    # Increase inflation when GNSS is good (can relax more)
                    inflation = min(self.config.max_inflation_factor, inflation * 1.3)

            if is_dynamic and nis <= effective_chi2_threshold:
                reason = "ACCEPTED_RELAXED_KINEMATIC"
            else:
                reason = "ACCEPTED_RELAXED_INNOVATION"

            self._previous_was_skipped = False
            return SkidEvaluationResult(
                status=NHCStatus.RELAXED,
                applied=True,
                inflation_factor=inflation,
                nis=nis,
                reason=reason,
                is_kinematically_dynamic=is_dynamic,
                effective_yaw_threshold=yaw_threshold,
                effective_lat_accel_threshold=lat_accel_threshold,
            )

        # Normal statistically consistent measurement -> ACCEPT with base covariance
        self._previous_was_skipped = False
        return SkidEvaluationResult(
            status=NHCStatus.NORMAL,
            applied=True,
            inflation_factor=1.0,
            nis=nis,
            reason="ACCEPTED_NORMAL",
            is_kinematically_dynamic=False,
            effective_yaw_threshold=yaw_threshold,
            effective_lat_accel_threshold=lat_accel_threshold,
        )
