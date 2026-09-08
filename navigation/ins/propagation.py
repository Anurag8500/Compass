"""Deterministic Strapdown Inertial Navigation System (INS) Propagation (Phase 4).

Implements the classical dead-reckoning equations in the local East-North-Up (ENU)
Cartesian frame:
    q[k+1] = normalize(q[k] ⊗ delta_q(omega_v[k] * dt))
    g^n = [0, 0, -g]^T,   g = 9.80665 m/s^2
    a_true^n = R_v^n[k] @ (f_m^v[k] - b_a^v) + g^n
    v[k+1] = v[k] + a_true^n * dt
    p[k+1] = p[k] + v[k] * dt + 0.5 * a_true^n * dt^2

Architectural Invariants:
1. Frame Semantics:
   - Position & Velocity: Represented strictly in meters and m/s in the local ENU
     tangent-plane frame relative to a fixed session origin.
   - Specific Force f_m^v: Provided directly by Phase 3 in the vehicle frame (FLU).
   - Gravity: NOT stripped in preprocessing; added as physical downward acceleration
     along local -Z_n inside strapdown mechanization.
2. Accelerometer Bias:
   - Phase 4 open-loop propagation maintains nominal prior b_a^v = [0, 0, 0]^T by default.
   - Accepts configurable constant bias vector for controlled ablation experiments.
   - Does NOT implement ESKF bias estimation or BiasNet ML inference.
3. Deterministic & Independently Testable:
   - Reusable by downstream Phase 5 ESKF prediction step.
   - No hidden global state; rigorous input validation; skips non-computable samples cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Optional, Tuple
import numpy as np

from navigation.ins.attitude import (
    delta_quaternion,
    propagate_attitude,
    quaternion_normalize,
    quaternion_to_euler_deg,
    quaternion_to_rotation_matrix,
)

STANDARD_GRAVITY: float = 9.80665
MAX_ALLOWABLE_DT_S: float = 1.0


@dataclass(frozen=True)
class INSState:
    """Propagated kinematic navigation state in local ENU frame.

    Attributes:
        timestamp_ns: Nanosecond timestamp of the state estimate.
        position_enu: (3,) float64 [East, North, Up] in meters.
        velocity_enu: (3,) float64 [v_East, v_North, v_Up] in m/s.
        q: (4,) float64 unit attitude quaternion [w, x, y, z] mapping vehicle to ENU (R_v^n).
        accel_bias: (3,) float64 accelerometer bias [b_ax, b_ay, b_az] in vehicle frame (m/s^2).
        coordinate_accel_enu: (3,) float64 coordinate acceleration in local ENU frame (m/s^2).
    """
    timestamp_ns: int
    position_enu: np.ndarray
    velocity_enu: np.ndarray
    q: np.ndarray
    accel_bias: np.ndarray
    coordinate_accel_enu: np.ndarray

    def __post_init__(self) -> None:
        if self.position_enu.shape != (3,):
            raise ValueError(f"position_enu must have shape (3,), got {self.position_enu.shape}")
        if self.velocity_enu.shape != (3,):
            raise ValueError(f"velocity_enu must have shape (3,), got {self.velocity_enu.shape}")
        if self.q.shape != (4,):
            raise ValueError(f"q must have shape (4,), got {self.q.shape}")
        if self.accel_bias.shape != (3,):
            raise ValueError(f"accel_bias must have shape (3,), got {self.accel_bias.shape}")
        if self.coordinate_accel_enu.shape != (3,):
            raise ValueError(f"coordinate_accel_enu must have shape (3,), got {self.coordinate_accel_enu.shape}")

    @property
    def R_v_n(self) -> np.ndarray:
        """3x3 rotation matrix mapping vehicle vectors to local ENU."""
        return quaternion_to_rotation_matrix(self.q)

    @property
    def euler_deg(self) -> Tuple[float, float, float]:
        """Diagnostic (roll, pitch, yaw) Euler angles in degrees."""
        return quaternion_to_euler_deg(self.q)


@dataclass(frozen=True)
class INSTrajectory:
    """Full time-history of propagated INS states over a batch trajectory."""
    timestamps_ns: np.ndarray       # (M,) int64
    positions_enu: np.ndarray       # (M, 3) float64
    velocities_enu: np.ndarray      # (M, 3) float64
    quaternions: np.ndarray         # (M, 4) float64
    accelerations_enu: np.ndarray   # (M, 3) float64
    final_state: INSState
    steps_integrated: int
    steps_skipped: int


class StrapdownINS:
    """Deterministic strapdown INS propagator for open-loop dead reckoning.

    Integrates specific force and angular velocity in local Cartesian ENU coordinates.
    """

    def __init__(
        self,
        initial_position: Optional[np.ndarray | Tuple[float, float, float]] = None,
        initial_velocity: Optional[np.ndarray | Tuple[float, float, float]] = None,
        initial_q: Optional[np.ndarray | Tuple[float, float, float, float]] = None,
        initial_accel_bias: Optional[np.ndarray | Tuple[float, float, float]] = None,
        initial_timestamp_ns: int = 0,
        gravity_magnitude: float = STANDARD_GRAVITY,
    ) -> None:
        """Initialize strapdown INS with explicit state conditions.

        Args:
            initial_position: [East, North, Up] in meters (default: [0, 0, 0]).
            initial_velocity: [v_East, v_North, v_Up] in m/s (default: [0, 0, 0]).
            initial_q: Initial attitude quaternion [w, x, y, z] R_v^n (default: [1, 0, 0, 0] identity).
            initial_accel_bias: Fixed accelerometer bias prior (default: [0, 0, 0]).
            initial_timestamp_ns: Initial epoch timestamp in nanoseconds.
            gravity_magnitude: Local gravity scalar (default: 9.80665 m/s^2).
        """
        pos = np.zeros(3, dtype=np.float64) if initial_position is None else np.asarray(initial_position, dtype=np.float64)
        vel = np.zeros(3, dtype=np.float64) if initial_velocity is None else np.asarray(initial_velocity, dtype=np.float64)
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64) if initial_q is None else quaternion_normalize(np.asarray(initial_q, dtype=np.float64))
        bias = np.zeros(3, dtype=np.float64) if initial_accel_bias is None else np.asarray(initial_accel_bias, dtype=np.float64)

        self.gravity_magnitude = float(gravity_magnitude)
        self.g_n = np.array([0.0, 0.0, -self.gravity_magnitude], dtype=np.float64)

        # Compute initial coordinate acceleration assuming level stationary start
        R_v_n_init = quaternion_to_rotation_matrix(q)
        f_init = np.array([0.0, 0.0, self.gravity_magnitude], dtype=np.float64)
        a_init = R_v_n_init @ (f_init - bias) + self.g_n

        self._state = INSState(
            timestamp_ns=initial_timestamp_ns,
            position_enu=pos,
            velocity_enu=vel,
            q=q,
            accel_bias=bias,
            coordinate_accel_enu=a_init,
        )

    @property
    def current_state(self) -> INSState:
        """Current propagated state."""
        return self._state

    def step(
        self,
        f_m_v: np.ndarray | Tuple[float, float, float],
        omega_m_v: np.ndarray | Tuple[float, float, float],
        dt: float,
        timestamp_ns: int,
    ) -> INSState:
        """Execute one canonical discrete strapdown propagation step.

        Equations:
            q[k+1] = normalize(q[k] ⊗ delta_q(omega_v[k] * dt))
            R_v^n[k] = quaternion_to_rotation_matrix(q[k])
            a_true^n = R_v^n[k] @ (f_m^v - b_a^v) + g^n
            v[k+1] = v[k] + a_true^n * dt
            p[k+1] = p[k] + v[k] * dt + 0.5 * a_true^n * dt^2

        Args:
            f_m_v: Vehicle-frame specific force measurement [m/s^2].
            omega_m_v: Vehicle-frame angular velocity measurement [rad/s].
            dt: Timestep duration in seconds. Must satisfy 0 < dt <= MAX_ALLOWABLE_DT_S.
            timestamp_ns: Target epoch timestamp in nanoseconds.

        Returns:
            INSState: New propagated state at timestamp_ns.
        """
        if dt <= 0.0:
            raise ValueError(f"Integration timestep dt must be strictly positive, got {dt}")
        if dt > MAX_ALLOWABLE_DT_S:
            raise ValueError(f"Integration timestep dt={dt:.3f}s exceeds maximum allowable limit ({MAX_ALLOWABLE_DT_S}s)")

        f_v = np.asarray(f_m_v, dtype=np.float64)
        w_v = np.asarray(omega_m_v, dtype=np.float64)

        if not np.isfinite(f_v).all():
            raise ValueError("Specific force f_m_v contains non-finite values (NaN/Inf)")
        if not np.isfinite(w_v).all():
            raise ValueError("Angular velocity omega_m_v contains non-finite values (NaN/Inf)")

        p_k = self._state.position_enu
        v_k = self._state.velocity_enu
        q_k = self._state.q
        bias = self._state.accel_bias

        # 1. Kinematic coordinate acceleration using current attitude R_v^n[k]
        R_v_n_k = quaternion_to_rotation_matrix(q_k)
        f_debiased = f_v - bias
        a_true_n = R_v_n_k @ f_debiased + self.g_n

        # 2. Position and velocity propagation (canonical discrete integral)
        p_next = p_k + v_k * dt + 0.5 * a_true_n * (dt * dt)
        v_next = v_k + a_true_n * dt

        # 3. Attitude propagation to k+1
        q_next = propagate_attitude(q_k, w_v, dt)

        self._state = INSState(
            timestamp_ns=timestamp_ns,
            position_enu=p_next,
            velocity_enu=v_next,
            q=q_next,
            accel_bias=bias,
            coordinate_accel_enu=a_true_n,
        )
        return self._state

    def propagate_trajectory(
        self,
        timestamps_ns: np.ndarray,
        f_m_v: np.ndarray,
        omega_m_v: np.ndarray,
        is_validated: Optional[np.ndarray] = None,
        max_dt_s: float = MAX_ALLOWABLE_DT_S,
    ) -> INSTrajectory:
        """Batch propagate strapdown INS over continuous vehicle-frame streams.

        Args:
            timestamps_ns: (N,) int64 monotonic non-decreasing timestamp sequence.
            f_m_v: (N, 3) float64 vehicle-frame specific force [m/s^2].
            omega_m_v: (N, 3) float64 vehicle-frame angular velocity [rad/s].
            is_validated: Optional (N,) bool mask indicating validated samples.
            max_dt_s: Maximum allowable timestep gap before sample is skipped.

        Returns:
            INSTrajectory: Full record of positions, velocities, attitudes, and accelerations.
        """
        n_samples = len(timestamps_ns)
        if len(f_m_v) != n_samples or len(omega_m_v) != n_samples:
            raise ValueError(
                f"Array length mismatch: timestamps ({n_samples}), "
                f"f_m_v ({len(f_m_v)}), omega_m_v ({len(omega_m_v)})"
            )
        if n_samples == 0:
            raise ValueError("Cannot propagate empty trajectory (0 samples)")

        valid_mask = np.ones(n_samples, dtype=bool) if is_validated is None else is_validated

        pos_hist = np.zeros((n_samples, 3), dtype=np.float64)
        vel_hist = np.zeros((n_samples, 3), dtype=np.float64)
        q_hist = np.zeros((n_samples, 4), dtype=np.float64)
        accel_hist = np.zeros((n_samples, 3), dtype=np.float64)

        pos_hist[0] = self._state.position_enu
        vel_hist[0] = self._state.velocity_enu
        q_hist[0] = self._state.q
        accel_hist[0] = self._state.coordinate_accel_enu

        integrated_count = 1
        skipped_count = 0

        for k in range(n_samples - 1):
            t_curr = int(timestamps_ns[k])
            t_next = int(timestamps_ns[k + 1])

            # Check validity mask
            if not valid_mask[k + 1]:
                skipped_count += 1
                pos_hist[k + 1] = pos_hist[k]
                vel_hist[k + 1] = vel_hist[k]
                q_hist[k + 1] = q_hist[k]
                accel_hist[k + 1] = accel_hist[k]
                continue

            dt = (t_next - t_curr) * 1e-9

            # Reject non-positive or excessive time gaps
            if dt <= 0.0 or dt > max_dt_s or not math.isfinite(dt):
                skipped_count += 1
                pos_hist[k + 1] = pos_hist[k]
                vel_hist[k + 1] = vel_hist[k]
                q_hist[k + 1] = q_hist[k]
                accel_hist[k + 1] = accel_hist[k]
                continue

            # Step propagation
            state = self.step(
                f_m_v=f_m_v[k],
                omega_m_v=omega_m_v[k],
                dt=dt,
                timestamp_ns=t_next,
            )

            pos_hist[k + 1] = state.position_enu
            vel_hist[k + 1] = state.velocity_enu
            q_hist[k + 1] = state.q
            accel_hist[k + 1] = state.coordinate_accel_enu
            integrated_count += 1

        return INSTrajectory(
            timestamps_ns=timestamps_ns.copy(),
            positions_enu=pos_hist,
            velocities_enu=vel_hist,
            quaternions=q_hist,
            accelerations_enu=accel_hist,
            final_state=self._state,
            steps_integrated=integrated_count,
            steps_skipped=skipped_count,
        )
