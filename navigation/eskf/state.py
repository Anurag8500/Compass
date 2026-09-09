"""ESKF State Representations and Manifold Mechanics (Phase 5).

Implements:
1. 16-element Nominal State (position, velocity, attitude quaternion, accel bias, gyro bias).
2. 15-element Error State representation on the SO(3) manifold.
3. 15x15 Error-State Covariance matrix.
4. Consistent right-multiplicative body-frame attitude error injection:
       q_true = q_nom ⊗ delta_q(delta_theta)
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence, Tuple, Union
import numpy as np


from navigation.ins.attitude import (
    delta_quaternion,
    quaternion_multiply,
    quaternion_normalize,
    quaternion_to_euler_deg,
    quaternion_to_rotation_matrix,
)
from navigation.schemas.state import NavigationState, OrientationState, GNSSMode


def skew(v: Union[np.ndarray, Tuple[float, float, float]]) -> np.ndarray:
    """Construct 3x3 skew-symmetric cross-product matrix [v]_x.

    Satisfies: skew(v) @ u == np.cross(v, u).
    """
    v_arr = np.asarray(v, dtype=np.float64).reshape(3)
    return np.array([
        [0.0, -v_arr[2], v_arr[1]],
        [v_arr[2], 0.0, -v_arr[0]],
        [-v_arr[1], v_arr[0], 0.0],
    ], dtype=np.float64)


@dataclass(frozen=True)
class ESKFNominalState:
    """16-element nominal kinematic navigation state for ESKF.

    Attributes:
        position_enu: (3,) float64 [East, North, Up] in meters.
        velocity_enu: (3,) float64 [v_East, v_North, v_Up] in m/s.
        q: (4,) float64 Hamilton unit quaternion [w, x, y, z] mapping vehicle to ENU (R_v^n).
        accel_bias: (3,) float64 accelerometer bias [b_ax, b_ay, b_az] in vehicle frame (m/s^2).
        gyro_bias: (3,) float64 gyroscope bias [b_gx, b_gy, b_gz] in vehicle frame (rad/s).
        timestamp_ns: Nanosecond epoch timestamp of current estimate.
    """
    position_enu: np.ndarray
    velocity_enu: np.ndarray
    q: np.ndarray
    accel_bias: np.ndarray
    gyro_bias: np.ndarray
    timestamp_ns: int = 0

    def __post_init__(self) -> None:
        pos = np.asarray(self.position_enu, dtype=np.float64)
        vel = np.asarray(self.velocity_enu, dtype=np.float64)
        q = np.asarray(self.q, dtype=np.float64)
        ba = np.asarray(self.accel_bias, dtype=np.float64)
        bg = np.asarray(self.gyro_bias, dtype=np.float64)

        if pos.shape != (3,):
            raise ValueError(f"position_enu must have shape (3,), got {pos.shape}")
        if vel.shape != (3,):
            raise ValueError(f"velocity_enu must have shape (3,), got {vel.shape}")
        if q.shape != (4,):
            raise ValueError(f"q must have shape (4,), got {q.shape}")
        if ba.shape != (3,):
            raise ValueError(f"accel_bias must have shape (3,), got {ba.shape}")
        if bg.shape != (3,):
            raise ValueError(f"gyro_bias must have shape (3,), got {bg.shape}")

        if not (np.isfinite(pos).all() and np.isfinite(vel).all() and np.isfinite(q).all() and
                np.isfinite(ba).all() and np.isfinite(bg).all()):
            raise ValueError("Nominal state components must be finite")

        q_norm = float(np.linalg.norm(q))
        if abs(q_norm - 1.0) > 1e-4:
            raise ValueError(f"Quaternion must be approximately normalized, got norm {q_norm:.6f}")

    @property
    def R_v_n(self) -> np.ndarray:
        """3x3 rotation matrix mapping vehicle vectors to local ENU (v^n = R_v^n @ v^v)."""
        return quaternion_to_rotation_matrix(self.q)

    @property
    def euler_deg(self) -> Tuple[float, float, float]:
        """Diagnostic Euler angles (roll, pitch, yaw) in degrees."""
        return quaternion_to_euler_deg(self.q)

    def to_vector(self) -> np.ndarray:
        """Flatten nominal state into 16-element vector."""
        return np.concatenate([
            self.position_enu,
            self.velocity_enu,
            self.q,
            self.accel_bias,
            self.gyro_bias,
        ])

    @property
    def nominal_vector(self) -> np.ndarray:
        """16-element nominal state vector."""
        return self.to_vector()

    @property
    def error_state_dim(self) -> int:
        """Dimension of error state manifold (15)."""
        return 15

    @classmethod
    def from_components(
        cls,
        position_enu: Sequence[float] | np.ndarray,
        velocity_enu: Sequence[float] | np.ndarray,
        q: Optional[Sequence[float] | np.ndarray] = None,
        accel_bias: Optional[Sequence[float] | np.ndarray] = None,
        gyro_bias: Optional[Sequence[float] | np.ndarray] = None,
        timestamp_ns: int = 0,
    ) -> ESKFNominalState:
        """Construct nominal state with convenient optional defaults."""
        pos = np.asarray(position_enu, dtype=np.float64).reshape(3)
        vel = np.asarray(velocity_enu, dtype=np.float64).reshape(3)
        q_arr = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64) if q is None else np.asarray(q, dtype=np.float64).reshape(4)
        ba = np.zeros(3, dtype=np.float64) if accel_bias is None else np.asarray(accel_bias, dtype=np.float64).reshape(3)
        bg = np.zeros(3, dtype=np.float64) if gyro_bias is None else np.asarray(gyro_bias, dtype=np.float64).reshape(3)

        return cls(
            position_enu=pos,
            velocity_enu=vel,
            q=quaternion_normalize(q_arr),
            accel_bias=ba,
            gyro_bias=bg,
            timestamp_ns=timestamp_ns,
        )

    @classmethod
    def from_vector(cls, vec: np.ndarray, timestamp_ns: int = 0) -> ESKFNominalState:
        """Reconstruct nominal state from 16-element vector."""
        vec = np.asarray(vec, dtype=np.float64)
        if vec.shape != (16,):
            raise ValueError(f"Vector must have 16 elements, got shape {vec.shape}")
        return cls(
            position_enu=vec[0:3].copy(),
            velocity_enu=vec[3:6].copy(),
            q=quaternion_normalize(vec[6:10].copy()),
            accel_bias=vec[10:13].copy(),
            gyro_bias=vec[13:16].copy(),
            timestamp_ns=timestamp_ns,
        )



    def inject_error(self, delta_x: np.ndarray, timestamp_ns: Optional[int] = None) -> ESKFNominalState:
        """Inject 15-dimensional error state into 16-dimensional nominal state.

        Convention: Right-multiplicative body-frame attitude error:
            q_true = normalize(q_nom ⊗ delta_q(delta_theta))

        Args:
            delta_x: (15,) float64 error vector [delta_p(3), delta_v(3), delta_theta(3), delta_ba(3), delta_bg(3)].
            timestamp_ns: Optional new timestamp (defaults to current).

        Returns:
            ESKFNominalState: Updated, normalized nominal state.
        """
        dx = np.asarray(delta_x, dtype=np.float64)
        if dx.shape != (15,):
            raise ValueError(f"delta_x must have shape (15,), got {dx.shape}")

        delta_p = dx[0:3]
        delta_v = dx[3:6]
        delta_theta = dx[6:9]
        delta_ba = dx[9:12]
        delta_bg = dx[12:15]

        new_pos = self.position_enu + delta_p
        new_vel = self.velocity_enu + delta_v
        new_ba = self.accel_bias + delta_ba
        new_bg = self.gyro_bias + delta_bg

        # Right-multiplicative body-frame injection
        dq = delta_quaternion(delta_theta, dt=1.0)
        new_q = quaternion_normalize(quaternion_multiply(self.q, dq))

        t = self.timestamp_ns if timestamp_ns is None else int(timestamp_ns)

        return ESKFNominalState(
            position_enu=new_pos,
            velocity_enu=new_vel,
            q=new_q,
            accel_bias=new_ba,
            gyro_bias=new_bg,
            timestamp_ns=t,
        )

    def to_navigation_state(
        self,
        covariance: np.ndarray,
        reference_point: Tuple[float, float],
        mode: GNSSMode = GNSSMode.GNSS_AIDED,
    ) -> NavigationState:
        """Convert to authoritative Phase 1 NavigationState schema."""
        cov_list = covariance.tolist() if isinstance(covariance, np.ndarray) else covariance
        return NavigationState(
            position_local=(float(self.position_enu[0]), float(self.position_enu[1]), float(self.position_enu[2])),
            velocity_local=(float(self.velocity_enu[0]), float(self.velocity_enu[1]), float(self.velocity_enu[2])),
            orientation=OrientationState(
                q=(float(self.q[0]), float(self.q[1]), float(self.q[2]), float(self.q[3])),
                gyro_bias=(float(self.gyro_bias[0]), float(self.gyro_bias[1]), float(self.gyro_bias[2])),
            ),
            accel_bias=(float(self.accel_bias[0]), float(self.accel_bias[1]), float(self.accel_bias[2])),
            covariance=cov_list,
            reference_point=reference_point,
            mode=mode,
            timestamp_ns=self.timestamp_ns,
        )


@dataclass(frozen=True)
class ESKFState:
    """Joint ESKF nominal state and 15x15 error-state covariance.

    Attributes:
        nominal: 16-element ESKFNominalState.
        covariance: (15, 15) float64 error-state covariance matrix P.
    """
    nominal: ESKFNominalState
    covariance: np.ndarray

    def __post_init__(self) -> None:
        cov = np.asarray(self.covariance, dtype=np.float64)
        if cov.shape != (15, 15):
            raise ValueError(f"covariance must have shape (15, 15), got {cov.shape}")
        if not np.isfinite(cov).all():
            raise ValueError("Covariance matrix contains non-finite elements (NaN/Inf)")
        # Numerical symmetry verification (tolerance 1e-4)
        asym = float(np.max(np.abs(cov - cov.T)))
        if asym > 1e-4:
            raise ValueError(f"Covariance matrix is materially asymmetric (max asymmetry: {asym:.2e})")
        # Positive-semidefiniteness verification (tolerance -1e-4)
        min_eig = float(np.min(np.linalg.eigvalsh(cov)))
        if min_eig < -1e-4:
            raise ValueError(f"Covariance matrix is materially indefinite (min eigenvalue: {min_eig:.2e})")

    @property
    def position_enu(self) -> np.ndarray:
        return self.nominal.position_enu

    @property
    def velocity_enu(self) -> np.ndarray:
        return self.nominal.velocity_enu

    @property
    def q(self) -> np.ndarray:
        return self.nominal.q

    @property
    def R_v_n(self) -> np.ndarray:
        return self.nominal.R_v_n

    @property
    def accel_bias(self) -> np.ndarray:
        return self.nominal.accel_bias

    @property
    def gyro_bias(self) -> np.ndarray:
        return self.nominal.gyro_bias

    @property
    def timestamp_ns(self) -> int:
        return self.nominal.timestamp_ns

    @property
    def pos_cov(self) -> np.ndarray:
        """3x3 position error covariance."""
        return self.covariance[0:3, 0:3]

    @property
    def vel_cov(self) -> np.ndarray:
        """3x3 velocity error covariance."""
        return self.covariance[3:6, 3:6]

    @property
    def att_cov(self) -> np.ndarray:
        """3x3 attitude error covariance (so(3))."""
        return self.covariance[6:9, 6:9]

    @property
    def accel_bias_cov(self) -> np.ndarray:
        """3x3 accelerometer bias error covariance."""
        return self.covariance[9:12, 9:12]

    @property
    def gyro_bias_cov(self) -> np.ndarray:
        """3x3 gyroscope bias error covariance."""
        return self.covariance[12:15, 12:15]

    def inject_error(
        self,
        delta_x: np.ndarray,
        new_covariance: Optional[np.ndarray] = None,
        timestamp_ns: Optional[int] = None,
    ) -> ESKFState:
        """Inject error state into nominal state and apply covariance reset transformation.

        This is the single authoritative entry point for error injection and covariance reset.

        Args:
            delta_x: (15,) float64 error state correction vector [dp, dv, dtheta, dba, dbg].
            new_covariance: Optional (15, 15) float64 updated covariance matrix (e.g. from
                Joseph update). If None, self.covariance is transformed.
            timestamp_ns: Optional nanosecond timestamp for the updated state.

        Returns:
            ESKFState with updated nominal state and transformed covariance matrix.
        """
        dx = np.asarray(delta_x, dtype=np.float64)
        if dx.shape != (15,):
            raise ValueError(f"delta_x must have shape (15,), got {dx.shape}")
        t = self.timestamp_ns if timestamp_ns is None else int(timestamp_ns)
        new_nom = self.nominal.inject_error(dx, timestamp_ns=t)
        cov = self.covariance if new_covariance is None else new_covariance
        # Apply error-state covariance reset transformation
        cov_reset = reset_covariance(cov, dx[6:9])
        return ESKFState(nominal=new_nom, covariance=cov_reset)


def compute_reset_jacobian(delta_theta: np.ndarray) -> np.ndarray:
    """Compute 15x15 error-state reset Jacobian J_reset for right-multiplicative attitude error.

    Under right-multiplicative body-frame attitude error convention:
        q_true = q_nom ⊗ delta_q(delta_theta)
    Following correction injection with attitude update delta_theta_hat:
        q_nom_new = normalize(q_nom ⊗ delta_q(delta_theta_hat))
    The new post-reset attitude error delta_theta_plus satisfies:
        delta_q(delta_theta_plus) = delta_q(-delta_theta_hat) ⊗ delta_q(delta_theta)
    To first order:
        delta_theta_plus ≈ (I - 0.5 * [delta_theta_hat]_x) delta_theta - delta_theta_hat

    The sensitivity G_theta = d(delta_theta_plus) / d(delta_theta) is:
        G_theta = I_3 - 0.5 * [delta_theta_hat]_x
    All other error states (delta_p, delta_v, delta_ba, delta_bg) have identity sensitivity.

    Args:
        delta_theta: (3,) float64 estimated angular correction vector delta_theta_hat [rad].

    Returns:
        (15, 15) float64 block-diagonal reset Jacobian J_reset.
    """
    dtheta = np.asarray(delta_theta, dtype=np.float64).reshape(3)
    J_reset = np.eye(15, dtype=np.float64)
    J_reset[6:9, 6:9] = np.eye(3, dtype=np.float64) - 0.5 * skew(dtheta)
    return J_reset


def reset_covariance(covariance: np.ndarray, delta_theta: np.ndarray) -> np.ndarray:
    """Apply error-state reset transformation to 15x15 covariance matrix.

    Transforms covariance P_reset = J_reset @ P @ J_reset.T and enforces numerical symmetry.

    Args:
        covariance: (15, 15) float64 prior error covariance matrix.
        delta_theta: (3,) float64 estimated angular correction vector delta_theta_hat [rad].

    Returns:
        (15, 15) float64 transformed, symmetric, positive-semidefinite covariance matrix.
    """
    P = np.asarray(covariance, dtype=np.float64)
    if P.shape != (15, 15):
        raise ValueError(f"Covariance must have shape (15, 15), got {P.shape}")
    J_reset = compute_reset_jacobian(delta_theta)
    P_reset = J_reset @ P @ J_reset.T
    return 0.5 * (P_reset + P_reset.T)


def inject_error(
    state: Union[ESKFState, ESKFNominalState],
    delta_x: np.ndarray,
    new_covariance: Optional[np.ndarray] = None,
    timestamp_ns: Optional[int] = None,
) -> Union[ESKFState, ESKFNominalState]:
    """Module-level convenience function to inject 15D error vector into ESKF state."""
    if isinstance(state, ESKFState):
        return state.inject_error(delta_x, new_covariance=new_covariance, timestamp_ns=timestamp_ns)
    return state.inject_error(delta_x, timestamp_ns=timestamp_ns)


