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

    def inject_error(self, delta_x: np.ndarray, new_covariance: Optional[np.ndarray] = None) -> ESKFState:
        """Inject error state into nominal state and update covariance."""
        new_nom = self.nominal.inject_error(delta_x)
        cov = self.covariance if new_covariance is None else new_covariance
        # Enforce exact numerical symmetry
        cov_sym = 0.5 * (cov + cov.T)
        return ESKFState(nominal=new_nom, covariance=cov_sym)


def inject_error(
    state: Union[ESKFState, ESKFNominalState],
    delta_x: np.ndarray,
    new_covariance: Optional[np.ndarray] = None,
) -> Union[ESKFState, ESKFNominalState]:
    """Module-level convenience function to inject 15D error vector into ESKF state."""
    if isinstance(state, ESKFState):
        return state.inject_error(delta_x, new_covariance=new_covariance)
    return state.inject_error(delta_x)

