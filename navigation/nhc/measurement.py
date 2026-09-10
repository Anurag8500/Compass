"""Non-Holonomic Constraints (NHC) ESKF Measurement Model (Phase 11).

Formulates the 2D pseudo-measurement assuming zero lateral and vertical velocity
in the vehicle coordinate frame:
    z_nhc = [0, 0]^T,   h_nhc(x) = [v_y^v, v_z^v]^T

Authoritative Conventions:
1. Vehicle Frame: Forward-Lateral-Up (FLU) right-handed convention.
2. Navigation Frame: Local East-North-Up (ENU) tangent-plane frame.
3. Velocity Frame Mapping:
       v^v = (R_v^n)^T @ v^n
4. Right-Multiplicative Body-Frame Error State:
       q = normalize(q_nom ⊗ delta_q(delta_theta^v))
5. Exact 2x15 Error-State Jacobian:
       H_nhc = [0_2x3,  P_yz @ (R_v^n)^T,  P_yz @ [v^v]_x,  0_2x3,  0_2x3]
   where:
       P_yz = [[0, 1, 0], [0, 0, 1]]
       [v^v]_x is the skew-symmetric cross-product matrix of nominal vehicle velocity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional, Tuple
import numpy as np

from navigation.eskf.state import ESKFState
from navigation.eskf.update import UpdateDiagnostics, eskf_update
from navigation.ins.attitude import quaternion_to_rotation_matrix
from navigation.nhc.skid_detection import NHCStatus, SkidDetectorConfig, SkidSlipDetector


@dataclass(frozen=True)
class NHCConfig:
    """Configuration for Non-Holonomic Constraints (NHC) measurement adapter.

    Attributes:
        enabled: Master toggle for NHC integration.
        sigma_vy: Standard deviation of lateral velocity pseudo-measurement noise [m/s] (default 0.10).
        sigma_vz: Standard deviation of vertical velocity pseudo-measurement noise [m/s] (default 0.05).
        min_forward_speed_mps: Forward speed threshold below which NHC is suppressed to avoid
            near-standstill singularity or conflict with ZUPT [m/s] (default 0.50).
        enable_attitude_coupling: Whether to include attitude error sensitivity block in H
            (default True). If False, operates in decoupled velocity-only subspace mode.
        preserve_forward_speed: Whether to enforce the kinematic subspace constraint that NHC
            strictly updates lateral and vertical velocity, preventing artificial deceleration
            via cross-covariance bleeding into forward velocity (default True).
        skid_detector: Configuration for conservative consistency evaluation and relaxation.
    """
    enabled: bool = True
    sigma_vy: float = 0.10
    sigma_vz: float = 0.05
    min_forward_speed_mps: float = 0.50
    enable_attitude_coupling: bool = True
    preserve_forward_speed: bool = True
    skid_detector: SkidDetectorConfig = field(default_factory=SkidDetectorConfig)


@dataclass(frozen=True)
class NHCDiagnostics:
    """Diagnostic details emitted from an NHC measurement cycle.

    Attributes:
        status: Execution decision tier (NORMAL, RELAXED, SKIPPED, SKIPPED_STATIONARY, etc.).
        applied: True if an ESKF update was executed, False if skipped.
        reason: Explanatory rationale code.
        innovation: (2,) Innovation residual vector [y_y, y_z] in vehicle frame [m/s].
        predicted_v_v: (3,) Predicted vehicle velocity [vx, vy, vz] in vehicle frame [m/s].
        nis: Normalized innovation squared d^2 = y^T S^-1 y.
        covariance_inflation: Covariance inflation factor applied to R_base (>= 1.0).
        update_diagnostics: Raw ESKF update diagnostics if applied.
    """
    status: NHCStatus
    applied: bool
    reason: str
    innovation: np.ndarray
    predicted_v_v: np.ndarray
    nis: float
    covariance_inflation: float
    update_diagnostics: Optional[UpdateDiagnostics] = None


class NHCMeasurementModel:
    """Integrates Non-Holonomic Constraints as gated, relaxation-aware 2D velocity updates."""

    def __init__(self, config: Optional[NHCConfig] = None) -> None:
        self.config = config or NHCConfig()
        self.skid_detector = SkidSlipDetector(config=self.config.skid_detector)

    def create_measurement(
        self,
        state: ESKFState,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        if self.config.enable_attitude_coupling:
            H[0, 6] = vz_v
            H[0, 7] = 0.0
            H[0, 8] = -vx_v

            H[1, 6] = -vy_v
            H[1, 7] = vx_v
            H[1, 8] = 0.0

        # Baseline noise covariance
        R_base = np.diag([self.config.sigma_vy ** 2, self.config.sigma_vz ** 2]).astype(np.float64)

        return z, h_val, v_v, H, R_base

    def update(
        self,
        state: ESKFState,
        is_stationary: bool = False,
        omega_v: Optional[np.ndarray] = None,
        f_v: Optional[np.ndarray] = None,
        timestamp_ns: Optional[int] = None,
    ) -> Tuple[ESKFState, NHCDiagnostics]:
        """Evaluate consistency, apply relaxation/skipping, and update ESKF state.

        Args:
            state: Authoritative ESKFState before update.
            is_stationary: Whether classical standstill detector confirms standstill.
            omega_v: (3,) Vehicle-frame angular velocity [rad/s].
            f_v: (3,) Vehicle-frame specific force [m/s^2].
            timestamp_ns: Current update timestamp in nanoseconds.

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
            )

        # 2. Formulate measurement model and candidate Jacobian
        z, h_val, v_v, H, R_base = self.create_measurement(state)
        forward_speed = float(v_v[0])

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
            )

        # 4. Innovation and innovation covariance
        y = z - h_val  # y = [-vy_v, -vz_v]
        P = state.covariance
        S = H @ P @ H.T + R_base

        # 5. Evaluate normalized innovation squared (Mahalanobis distance squared)
        try:
            # Solve S^-1 @ y numerically without explicit inversion
            S_inv_y = np.linalg.solve(S, y)
            nis = float(y.T @ S_inv_y)
        except Exception:
            nis = float("inf")

        # 6. Conservative consistency & dynamic relaxation evaluation
        eval_res = self.skid_detector.evaluate(nis=nis, omega_v=omega_v, f_v=f_v)

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
        )

        # 8. Kinematic Subspace Constraint:
        # NHC is strictly a 2D virtual measurement constraining lateral (v_y^v = 0) and
        # vertical (v_z^v = 0) motion. Forward velocity (v_x^v) is unobserved by NHC and must
        # not suffer artificial deceleration due to cross-covariance coupling (P_v_theta, P_vx_vy)
        # during high-speed cruising.
        if diag.applied and self.config.preserve_forward_speed:
            R_v_n_pre = state.nominal.R_v_n
            v_v_pre = R_v_n_pre.T @ state.nominal.velocity_enu

            R_v_n_post = updated_state.nominal.R_v_n
            v_v_post = R_v_n_post.T @ updated_state.nominal.velocity_enu

            # Preserve forward velocity v_x^v from nominal state prior to NHC update
            v_v_clean = np.array([v_v_pre[0], v_v_post[1], v_v_post[2]], dtype=np.float64)
            v_enu_clean = R_v_n_post @ v_v_clean

            from navigation.eskf.state import ESKFNominalState
            clean_nom = ESKFNominalState(
                position_enu=updated_state.nominal.position_enu,
                velocity_enu=v_enu_clean,
                q=updated_state.nominal.q,
                accel_bias=updated_state.nominal.accel_bias,
                gyro_bias=updated_state.nominal.gyro_bias,
                timestamp_ns=updated_state.nominal.timestamp_ns,
            )
            updated_state = ESKFState(nominal=clean_nom, covariance=updated_state.covariance)

        return updated_state, NHCDiagnostics(
            status=eval_res.status,
            applied=diag.applied,
            reason=eval_res.reason,
            innovation=y,
            predicted_v_v=v_v,
            nis=nis,
            covariance_inflation=eval_res.inflation_factor,
            update_diagnostics=diag,
        )
