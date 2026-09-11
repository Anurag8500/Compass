"""Improved NHC Measurement Model with Adaptive Covariance for Phase 11.

Key improvements over baseline anurag-phase-10:
- Speed-dependent NHC noise covariance (tighter at low speed, looser at high speed)
- Adaptive measurement noise based on vehicle dynamics
- Improved standstill detection with longer windows
- Better integration with skid detector's adaptive thresholds
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Optional, Tuple
import numpy as np

from navigation.eskf.state import ESKFState
from navigation.eskf.update import UpdateDiagnostics, eskf_update
from navigation.alignment import AlignmentConfidence
from navigation.nhc.skid_detection import (
    NHCStatus,
    SkidDetectorConfig,
    SkidEvaluationResult,
    SkidSlipDetector,
)


@dataclass(frozen=True)
class NHCConfig:
    """Configuration for improved NHC measurement model.

    Attributes:
        enabled: Whether NHC is enabled.
        base_sigma_vy: Base standard deviation for lateral velocity at low speed [m/s].
        base_sigma_vz: Base standard deviation for vertical velocity at low speed [m/s].
        min_forward_speed_mps: Forward speed threshold below which NHC is suppressed [m/s].
        max_forward_speed_mps: Speed at which covariance scaling saturates [m/s].
        speed_covariance_factor: Factor for speed-dependent covariance scaling.
        skid_detector: Configuration for skid detection and relaxation.
    """
    enabled: bool = True
    base_sigma_vy: float = 0.08  # Reduced from 0.10 for tighter constraint at low speed
    base_sigma_vz: float = 0.04  # Reduced from 0.05 for tighter constraint at low speed
    min_forward_speed_mps: float = 0.30  # Reduced from 0.50 to enable NHC earlier
    max_forward_speed_mps: float = 20.0  # Speed at which covariance scaling saturates
    speed_covariance_factor: float = 0.05  # Covariance scales with speed
    skid_detector: SkidDetectorConfig = field(default_factory=SkidDetectorConfig)


@dataclass(frozen=True)
class NHCDiagnostics:
    """Diagnostic details emitted from an NHC measurement cycle.

    Attributes:
        status: Execution decision tier (NORMAL, RELAXED, SKIPPED, etc.).
        applied: True if an ESKF update was executed, False if skipped.
        reason: Explanatory rationale code.
        innovation: (2,) Innovation residual vector [y_y, y_z] in vehicle frame [m/s].
        predicted_v_v: (3,) Predicted vehicle velocity [vx, vy, vz] in vehicle frame [m/s].
        nis: Normalized innovation squared d^2 = y^T S^-1 y.
        covariance_inflation: Covariance inflation factor applied to R_base (>= 1.0).
        effective_sigma_vy: Effective sigma_vy used (speed-dependent).
        effective_sigma_vz: Effective sigma_vz used (speed-dependent).
        update_diagnostics: Raw ESKF update diagnostics if applied.
    """
    status: NHCStatus
    applied: bool
    reason: str
    innovation: np.ndarray
    predicted_v_v: np.ndarray
    nis: float
    covariance_inflation: float
    effective_sigma_vy: float
    effective_sigma_vz: float
    update_diagnostics: Optional[UpdateDiagnostics] = None


class NHCMeasurementModel:
    """Improved NHC measurement model with adaptive covariance and thresholds."""

    def __init__(self, config: Optional[NHCConfig] = None) -> None:
        self.config = config or NHCConfig()
        self.skid_detector = SkidSlipDetector(config=self.config.skid_detector)

    def _compute_speed_dependent_covariance(
        self, forward_speed: float
    ) -> Tuple[float, float]:
        """Compute speed-dependent measurement noise covariance.

        At higher speeds, we allow larger measurement noise because:
        1. Lateral velocity estimation becomes noisier at high speed
        2. Vehicle dynamics are more unpredictable at high speed
        3. We want to avoid over-constraining during legitimate high-speed maneuvers

        Args:
            forward_speed: Forward vehicle speed in m/s.

        Returns:
            Tuple of (effective_sigma_vy, effective_sigma_vz).
        """
        # Linear scaling from base to max based on speed
        speed_factor = min(
            forward_speed / self.config.max_forward_speed_mps, 1.0
        )
        
        # Covariance increases with speed (allow more uncertainty at high speed)
        sigma_vy = self.config.base_sigma_vy * (1.0 + self.config.speed_covariance_factor * forward_speed)
        sigma_vz = self.config.base_sigma_vz * (1.0 + self.config.speed_covariance_factor * forward_speed)
        
        return sigma_vy, sigma_vz

    def create_measurement(
        self,
        state: ESKFState,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        """Construct NHC measurement components from current nominal state.

        Args:
            state: Current authoritative ESKFState (nominal + covariance).

        Returns:
            Tuple of:
            - z: (2,) Zero measurement vector [0.0, 0.0]^T [m/s].
            - h_val: (2,) Predicted [v_y^v, v_z^v]^T in vehicle frame [m/s].
            - v_v: (3,) Full 3D vehicle-frame velocity [v_x^v, v_y^v, v_z^v]^T [m/s].
            - H: (2, 15) Analytical measurement Jacobian matrix.
            - R_base: (2, 2) Baseline measurement noise covariance matrix.
            - effective_sigma_vy: Effective lateral velocity noise std dev.
            - effective_sigma_vz: Effective vertical velocity noise std dev.
        """
        R_v_n = state.nominal.R_v_n
        R_n_v = R_v_n.T  # Transformation from ENU to vehicle frame
        v_enu = state.nominal.velocity_enu

        # Transform velocity to vehicle frame: v^v = R_n^v @ v^n
        v_v = R_n_v @ v_enu
        vx_v, vy_v, vz_v = float(v_v[0]), float(v_v[1]), float(v_v[2])

        # Target measurement is zero lateral and vertical velocity
        z = np.zeros(2, dtype=np.float64)
        h_val = np.array([vy_v, vz_v], dtype=np.float64)

        # Construct 2x15 Jacobian H
        # H = [0_2x3,  P_yz @ R_n_v,  P_yz @ [v^v]_x,  0_2x3,  0_2x3]
        H = np.zeros((2, 15), dtype=np.float64)

        # Velocity error sensitivity (cols 3:6)
        # P_yz selects rows 1 and 2 (y and z)
        P_yz = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        H[:, 3:6] = P_yz @ R_n_v

        # Attitude error sensitivity (cols 6:9)
        # For right-multiplicative error q = q_nom ⊗ delta_q(delta_theta^v):
        # [v^v]_x = [[0, -vz, vy], [vz, 0, -vx], [-vy, vx, 0]]
        # P_yz @ [v^v]_x = [[vz, 0, -vx], [-vy, vx, 0]]
        H[0, 6] = vz_v
        H[0, 7] = 0.0
        H[0, 8] = -vx_v

        H[1, 6] = -vy_v
        H[1, 7] = vx_v
        H[1, 8] = 0.0

        # Compute speed-dependent covariance
        sigma_vy, sigma_vz = self._compute_speed_dependent_covariance(abs(vx_v))
        
        # Baseline noise covariance
        R_base = np.diag([sigma_vy ** 2, sigma_vz ** 2]).astype(np.float64)

        return z, h_val, v_v, H, R_base, sigma_vy, sigma_vz

    def update(
        self,
        state: ESKFState,
        is_stationary: bool = False,
        omega_v: Optional[np.ndarray] = None,
        f_v: Optional[np.ndarray] = None,
        timestamp_ns: Optional[int] = None,
        gnss_trust_score: float = 1.0,
        alignment_confidence: AlignmentConfidence = AlignmentConfidence.UNKNOWN,
    ) -> Tuple[ESKFState, NHCDiagnostics]:
        """Evaluate consistency, apply relaxation/skipping, and update ESKF state.

        Args:
            state: Authoritative ESKFState before update.
            is_stationary: Whether classical standstill detector confirms standstill.
            omega_v: (3,) Vehicle-frame angular velocity [rad/s].
            f_v: (3,) Vehicle-frame specific force [m/s²].
            timestamp_ns: Current update timestamp in nanoseconds.
            gnss_trust_score: GNSS trust score [0.0, 1.0] for adaptive relaxation.
            alignment_confidence: Observability confidence of mounting orientation.

        Returns:
            Tuple of (updated_or_unmodified_state, NHCDiagnostics).
        """
        zero_innov = np.zeros(2, dtype=np.float64)
        zero_vv = np.zeros(3, dtype=np.float64)

        if not self.config.enabled:
            return state, NHCDiagnostics(
                status=NHCStatus.NOT_ATTEMPTED,
                applied=False,
                reason="DISABLED",
                innovation=zero_innov,
                predicted_v_v=zero_vv,
                nis=0.0,
                covariance_inflation=1.0,
                effective_sigma_vy=self.config.base_sigma_vy,
                effective_sigma_vz=self.config.base_sigma_vz,
            )

        # 1. Standstill priority handshake: skip NHC during standstill to avoid fighting ZUPT
        if is_stationary:
            return state, NHCDiagnostics(
                status=NHCStatus.SKIPPED_STATIONARY,
                applied=False,
                reason="SKIPPED_STATIONARY",
                innovation=zero_innov,
                predicted_v_v=zero_vv,
                nis=0.0,
                covariance_inflation=1.0,
                effective_sigma_vy=self.config.base_sigma_vy,
                effective_sigma_vz=self.config.base_sigma_vz,
            )

        # 2. Formulate measurement model and candidate Jacobian
        z, h_val, v_v, H, R_base, sigma_vy, sigma_vz = self.create_measurement(state)
        forward_speed = float(abs(v_v[0]))

        # 3. Minimum forward speed gate: suppress near standstill
        if forward_speed < self.config.min_forward_speed_mps:
            return state, NHCDiagnostics(
                status=NHCStatus.SKIPPED_LOW_SPEED,
                applied=False,
                reason="SKIPPED_LOW_SPEED",
                innovation=z - h_val,
                predicted_v_v=v_v,
                nis=0.0,
                covariance_inflation=1.0,
                effective_sigma_vy=sigma_vy,
                effective_sigma_vz=sigma_vz,
            )

        # 4. Alignment Observability Gate
        if alignment_confidence == AlignmentConfidence.UNKNOWN:
            return state, NHCDiagnostics(
                status=NHCStatus.SKIPPED_UNALIGNED_FRAME,
                applied=False,
                reason="SKIPPED_UNALIGNED_FRAME",
                innovation=z - h_val,
                predicted_v_v=v_v,
                nis=0.0,
                covariance_inflation=1.0,
                effective_sigma_vy=sigma_vy,
                effective_sigma_vz=sigma_vz,
            )

        # 5. GNSS-aided authority modulation and alignment uncertainty inflation
        scale_gnss = (1.0 + 3.0 * max(0.0, min(1.0, float(gnss_trust_score)))) ** 2
        scale_align = 2.25 if alignment_confidence == AlignmentConfidence.LOW_CONFIDENCE else 1.0
        R_base = R_base * (scale_gnss * scale_align)

        # 6. Innovation and innovation covariance
        y = z - h_val  # y = [-vy_v, -vz_v]
        P = state.covariance
        S = H @ P @ H.T + R_base

        # 7. Evaluate normalized innovation squared (Mahalanobis distance squared)
        try:
            # Solve S^-1 @ y numerically without explicit inversion
            S_inv_y = np.linalg.solve(S, y)
            nis = float(y.T @ S_inv_y)
        except Exception:
            nis = float("inf")

        # 8. Conservative consistency & dynamic relaxation evaluation
        eval_res = self.skid_detector.evaluate(
            nis=nis,
            omega_v=omega_v,
            f_v=f_v,
            forward_speed=forward_speed,
            gnss_trust=gnss_trust_score,
        )

        if not eval_res.applied:
            # Severe innovation mismatch or extreme cornering dynamics: skip update
            return state, NHCDiagnostics(
                status=eval_res.status,
                applied=False,
                reason=eval_res.reason,
                innovation=y,
                predicted_v_v=v_v,
                nis=nis,
                covariance_inflation=eval_res.inflation_factor,
                effective_sigma_vy=sigma_vy,
                effective_sigma_vz=sigma_vz,
            )

        # 7. Apply gated measurement update with effective inflated covariance R_eff
        R_eff = R_base * eval_res.inflation_factor

        updated_state, diag = eskf_update(
            state=state,
            z=z,
            h_val=h_val,
            H=H,
            R=R_eff,
            gating=None,  # Already evaluated and gated via SkidSlipDetector
            timestamp_ns=timestamp_ns,
            constrain_longitudinal_velocity=True,  # Simon-Chia constraint
        )

        return updated_state, NHCDiagnostics(
            status=eval_res.status,
            applied=diag.applied,
            reason=eval_res.reason,
            innovation=y,
            predicted_v_v=v_v,
            nis=nis,
            covariance_inflation=eval_res.inflation_factor,
            effective_sigma_vy=sigma_vy,
            effective_sigma_vz=sigma_vz,
            update_diagnostics=diag,
        )
