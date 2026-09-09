"""ESKF Prediction and Covariance Propagation (Phase 5).

Implements:
1. Nominal-state kinematic propagation matching Phase 4 canonical strapdown INS equations.
2. 15-state linearized error dynamics matrix F_d consistent with right-multiplicative
   body-frame attitude error:
       q_true = q_nom ⊗ delta_q(delta_theta)
3. Physically parameterized discrete process noise covariance Q_d.
4. Covariance propagation P_next = F_d P F_d^T + Q_d with numerical symmetrization.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Union
import numpy as np

from navigation.ins.attitude import (
    delta_quaternion,
    quaternion_multiply,
    quaternion_normalize,
)
from navigation.ins.propagation import STANDARD_GRAVITY
from navigation.eskf.state import ESKFNominalState, ESKFState, skew

MAX_ALLOWABLE_DT_S: float = 1.0


@dataclass(frozen=True)
class ProcessNoiseConfig:
    """Configurable continuous-time noise parameters for ESKF covariance propagation.

    Attributes:
        accel_noise_std: Accelerometer white noise density [m/s^2 / sqrt(Hz)].
        gyro_noise_std: Gyroscope white noise density [rad/s / sqrt(Hz)].
        accel_bias_rw_std: Accelerometer bias random-walk density [m/s^3 / sqrt(Hz)].
        gyro_bias_rw_std: Gyroscope bias random-walk density [rad/s^2 / sqrt(Hz)].
    """
    accel_noise_std: float = 0.15          # ~0.15 m/s^2 (consumer phone IMU)
    gyro_noise_std: float = 0.015          # ~0.015 rad/s (~0.85 deg/s)
    accel_bias_rw_std: float = 0.001       # 1e-3 m/s^3
    gyro_bias_rw_std: float = 0.0001       # 1e-4 rad/s^2

    def __post_init__(self) -> None:
        if self.accel_noise_std <= 0 or not math.isfinite(self.accel_noise_std):
            raise ValueError(f"accel_noise_std must be positive finite, got {self.accel_noise_std}")
        if self.gyro_noise_std <= 0 or not math.isfinite(self.gyro_noise_std):
            raise ValueError(f"gyro_noise_std must be positive finite, got {self.gyro_noise_std}")
        if self.accel_bias_rw_std <= 0 or not math.isfinite(self.accel_bias_rw_std):
            raise ValueError(f"accel_bias_rw_std must be positive finite, got {self.accel_bias_rw_std}")
        if self.gyro_bias_rw_std <= 0 or not math.isfinite(self.gyro_bias_rw_std):
            raise ValueError(f"gyro_bias_rw_std must be positive finite, got {self.gyro_bias_rw_std}")


def compute_discrete_F(
    nominal: ESKFNominalState,
    f_m_v: np.ndarray,
    omega_m_v: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Compute 15x15 discrete error-state transition matrix F_d.

    Error State Ordering:
        [delta_p(3), delta_v(3), delta_theta(3), delta_ba(3), delta_bg(3)]

    Under right-multiplicative body-frame attitude error:
        q_true = q_nom ⊗ delta_q(delta_theta)
        R_v_true^n = R_v^n (I + [delta_theta]_x)

    The linearized continuous dynamics are:
        delta_p_dot = delta_v
        delta_v_dot = - R_v^n [f_unbiased]_x delta_theta - R_v^n delta_ba
        delta_theta_dot = - [omega_unbiased]_x delta_theta - delta_bg
        delta_ba_dot = 0
        delta_bg_dot = 0

    Second-order position and first-order velocity discrete integration yields:
        delta_p[k+1] = delta_p[k] + delta_v[k]*dt - 0.5*dt^2 * R_v^n [f_unbiased]_x delta_theta[k] - 0.5*dt^2 * R_v^n delta_ba[k]
        delta_v[k+1] = delta_v[k] - dt * R_v^n [f_unbiased]_x delta_theta[k] - dt * R_v^n delta_ba[k]
        delta_theta[k+1] = (I - dt * [omega_unbiased]_x) delta_theta[k] - dt * delta_bg[k]
        delta_ba[k+1] = delta_ba[k]
        delta_bg[k+1] = delta_bg[k]
    """
    f_arr = np.asarray(f_m_v, dtype=np.float64).reshape(3)
    w_arr = np.asarray(omega_m_v, dtype=np.float64).reshape(3)

    f_unbiased = f_arr - nominal.accel_bias
    w_unbiased = w_arr - nominal.gyro_bias

    R_v_n = nominal.R_v_n

    F_d = np.eye(15, dtype=np.float64)

    # Position row (0:3)
    F_d[0:3, 3:6] = np.eye(3, dtype=np.float64) * dt
    F_d[0:3, 6:9] = -0.5 * (dt * dt) * (R_v_n @ skew(f_unbiased))
    F_d[0:3, 9:12] = -0.5 * (dt * dt) * R_v_n

    # Velocity row (3:6)
    F_d[3:6, 6:9] = -dt * (R_v_n @ skew(f_unbiased))
    F_d[3:6, 9:12] = -dt * R_v_n

    # Attitude row (6:9)
    F_d[6:9, 6:9] = np.eye(3, dtype=np.float64) - dt * skew(w_unbiased)
    F_d[6:9, 12:15] = -dt * np.eye(3, dtype=np.float64)

    # Bias rows (9:12, 12:15) are identity (already on diagonal)

    return F_d


def compute_discrete_Q(
    config: ProcessNoiseConfig,
    dt: float,
    R_v_n: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute 15x15 discrete-time process noise covariance Q_d.

    Uses canonical closed-form polynomial continuous-to-discrete block integration approximations scaled by dt:
        Position:  1/3 * dt^3 * sigma_a^2
        Velocity:  dt * sigma_a^2
        Attitude:  dt * sigma_g^2
        AccelBias: dt * sigma_ba^2
        GyroBias:  dt * sigma_bg^2
    """
    Q_d = np.zeros((15, 15), dtype=np.float64)

    var_a = config.accel_noise_std ** 2
    var_g = config.gyro_noise_std ** 2
    var_ba = config.accel_bias_rw_std ** 2
    var_bg = config.gyro_bias_rw_std ** 2

    # Position variance
    pos_var = (1.0 / 3.0) * (dt ** 3) * var_a
    # Position-velocity cross-correlation
    pos_vel_cov = 0.5 * (dt ** 2) * var_a
    # Velocity variance
    vel_var = dt * var_a

    Q_d[0:3, 0:3] = np.eye(3, dtype=np.float64) * pos_var
    Q_d[0:3, 3:6] = np.eye(3, dtype=np.float64) * pos_vel_cov
    Q_d[3:6, 0:3] = np.eye(3, dtype=np.float64) * pos_vel_cov
    Q_d[3:6, 3:6] = np.eye(3, dtype=np.float64) * vel_var

    Q_d[6:9, 6:9] = np.eye(3, dtype=np.float64) * (dt * var_g)
    Q_d[9:12, 9:12] = np.eye(3, dtype=np.float64) * (dt * var_ba)
    Q_d[12:15, 12:15] = np.eye(3, dtype=np.float64) * (dt * var_bg)

    return Q_d


def predict_eskf(
    state: ESKFState,
    f_m_v: np.ndarray,
    omega_m_v: np.ndarray,
    dt: float,
    timestamp_ns: int,
    process_noise: Optional[ProcessNoiseConfig] = None,
    gravity_magnitude: float = STANDARD_GRAVITY,
) -> ESKFState:
    """Propagate ESKF nominal state and error covariance over timestep dt.

    Nominal strapdown mechanics:
        f_unbiased = f_m_v - b_a^v
        omega_unbiased = omega_m_v - b_g^v
        a^n = R_v^n @ f_unbiased + g^n
        v_{k+1} = v_k + a^n * dt
        p_{k+1} = p_k + v_k * dt + 0.5 * a^n * dt^2
        q_{k+1} = normalize(q_k ⊗ delta_q(omega_unbiased * dt))
        b_a_{k+1} = b_a_k
        b_g_{k+1} = b_g_k

    Covariance mechanics:
        P_{k+1} = F_d P_k F_d^T + Q_d
        P_{k+1} = 0.5 * (P_{k+1} + P_{k+1}^T)

    Args:
        state: Current joint ESKF state (nominal + covariance).
        f_m_v: (3,) specific force in vehicle frame [m/s^2].
        omega_m_v: (3,) angular velocity in vehicle frame [rad/s].
        dt: Timestep duration in seconds.
        timestamp_ns: Target nanosecond epoch timestamp.
        process_noise: Optional noise density configuration.
        gravity_magnitude: Standard gravity constant (default: 9.80665).

    Returns:
        ESKFState: Propagated state and covariance.
    """
    if not math.isfinite(dt):
        raise ValueError(f"Timestep dt must be finite, got {dt}")
    if dt <= 0.0:
        raise ValueError(f"Timestep dt must be strictly positive, got {dt}")
    if dt > MAX_ALLOWABLE_DT_S:
        raise ValueError(f"Timestep dt ({dt:.4f}s) exceeds maximum allowable ({MAX_ALLOWABLE_DT_S}s)")

    noise_cfg = ProcessNoiseConfig() if process_noise is None else process_noise

    nom = state.nominal
    f_arr = np.asarray(f_m_v, dtype=np.float64).reshape(3)
    w_arr = np.asarray(omega_m_v, dtype=np.float64).reshape(3)

    # 1. Unbiased specific force and rotation rate in vehicle frame
    f_unbiased = f_arr - nom.accel_bias
    w_unbiased = w_arr - nom.gyro_bias

    # 2. Coordinate acceleration in local ENU
    g_n = np.array([0.0, 0.0, -float(gravity_magnitude)], dtype=np.float64)
    R_v_n = nom.R_v_n
    a_true_n = R_v_n @ f_unbiased + g_n

    # 3. Canonical position & velocity integration
    p_next = nom.position_enu + nom.velocity_enu * dt + 0.5 * a_true_n * (dt * dt)
    v_next = nom.velocity_enu + a_true_n * dt

    # 4. Attitude propagation
    dq = delta_quaternion(w_unbiased, dt)
    q_next = quaternion_normalize(quaternion_multiply(nom.q, dq))

    new_nominal = ESKFNominalState(
        position_enu=p_next,
        velocity_enu=v_next,
        q=q_next,
        accel_bias=nom.accel_bias.copy(),
        gyro_bias=nom.gyro_bias.copy(),
        timestamp_ns=timestamp_ns,
    )

    # 5. Covariance propagation
    F_d = compute_discrete_F(nom, f_arr, w_arr, dt)
    Q_d = compute_discrete_Q(noise_cfg, dt)

    P_next = F_d @ state.covariance @ F_d.T + Q_d
    # Controlled numerical symmetrization
    P_next = 0.5 * (P_next + P_next.T)

    return ESKFState(nominal=new_nominal, covariance=P_next)
