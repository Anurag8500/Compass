"""Bootstrap Attitude Estimator for COMPASS (Phase 3).

CRITICAL ARCHITECTURAL CONTRACT:
================================
THIS MODULE IS AN OFFLINE TEST / VALIDATION UTILITY ONLY.

It exists strictly to validate coordinate transformations, rotation math,
and gravity resolution in Phase 3 unit and integration tests BEFORE the
central ESKF filter is implemented in Phase 5.

DO NOT IMPORT OR USE THIS MODULE AS A DEPENDENCY IN LIVE NAVIGATION LOGIC.
In the production runtime, attitude estimation is performed jointly inside
the ESKF state estimator via error-state quaternion propagation and
covariance updates.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple
import numpy as np


def quaternion_multiply(
    q1: Tuple[float, float, float, float] | np.ndarray,
    q2: Tuple[float, float, float, float] | np.ndarray,
) -> np.ndarray:
    """Hamilton quaternion product q = q1 ⊗ q2 with scalar first convention [w, x, y, z]."""
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)


def quaternion_to_rotation_matrix(q: Tuple[float, float, float, float] | np.ndarray) -> np.ndarray:
    """Convert Hamilton quaternion [w, x, y, z] to 3x3 direction cosine matrix R(q).

    Transforms vectors as v_n = R(q) @ v_v.
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    norm_sq = w * w + x * x + y * y + z * z
    if norm_sq < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / norm_sq
    return np.array([
        [1.0 - s * (y * y + z * z), s * (x * y - w * z), s * (x * z + w * y)],
        [s * (x * y + w * z), 1.0 - s * (x * x + z * z), s * (y * z - w * x)],
        [s * (x * z - w * y), s * (y * z + w * x), 1.0 - s * (x * x + y * y)],
    ], dtype=np.float64)


def quaternion_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Create a unit quaternion from unit axis and angle in radians."""
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-12 or abs(angle_rad) < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    u = axis / axis_norm
    half = angle_rad * 0.5
    sin_half = math.sin(half)
    return np.array([math.cos(half), u[0] * sin_half, u[1] * sin_half, u[2] * sin_half], dtype=np.float64)


def quaternion_to_euler_rad(q: Tuple[float, float, float, float] | np.ndarray) -> Tuple[float, float, float]:
    """Extract Euler angles (roll, pitch, yaw) in radians from quaternion [w, x, y, z].

    Sequence: Z (yaw) -> Y (pitch) -> X (roll).
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


class BootstrapAttitudeEstimator:
    """Offline test-only complementary filter for quaternion attitude estimation.

    WARNING: FOR OFFLINE VALIDATION AND UNIT TESTING ONLY.
    NOT FOR PRODUCTION NAVIGATION.
    """

    def __init__(
        self,
        kp_tilt: float = 0.5,
        accel_gate_min_mps2: float = 7.0,
        accel_gate_max_mps2: float = 12.0,
        initial_q: Optional[Tuple[float, float, float, float] | np.ndarray] = None,
    ) -> None:
        """Initialize bootstrap attitude estimator.

        Args:
            kp_tilt: Proportional gain for gravity-tilt correction (default 0.5).
            accel_gate_min_mps2: Minimum specific force magnitude to accept tilt update.
            accel_gate_max_mps2: Maximum specific force magnitude to accept tilt update.
            initial_q: Optional initial quaternion [w, x, y, z]. Defaults to identity [1, 0, 0, 0].
        """
        self.kp_tilt = kp_tilt
        self.accel_gate_min = accel_gate_min_mps2
        self.accel_gate_max = accel_gate_max_mps2

        if initial_q is not None:
            q_arr = np.asarray(initial_q, dtype=np.float64)
            self.q = q_arr / np.linalg.norm(q_arr)
        else:
            self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def initialize_from_gravity(self, roll_rad: float, pitch_rad: float, yaw_rad: float = 0.0) -> None:
        """Initialize quaternion from Euler angles in radians."""
        # Standard Euler to quaternion conversion
        cr = math.cos(roll_rad * 0.5)
        sr = math.sin(roll_rad * 0.5)
        cp = math.cos(pitch_rad * 0.5)
        sp = math.sin(pitch_rad * 0.5)
        cy = math.cos(yaw_rad * 0.5)
        sy = math.sin(yaw_rad * 0.5)

        self.q = np.array([
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ], dtype=np.float64)
        self.q /= np.linalg.norm(self.q)

    def update(
        self,
        accel_mps2: Tuple[float, float, float] | np.ndarray,
        gyro_rads: Tuple[float, float, float] | np.ndarray,
        dt_s: float,
    ) -> np.ndarray:
        """Propagate gyro integration and apply gravity tilt correction.

        Args:
            accel_mps2: Measured specific force in vehicle frame (m/s^2).
            gyro_rads: Debiased angular velocity in vehicle frame (rad/s).
            dt_s: Time step in seconds.

        Returns:
            Updated unit quaternion [w, x, y, z].
        """
        if dt_s <= 0.0 or not np.isfinite(dt_s):
            return self.q.copy()

        gyro = np.asarray(gyro_rads, dtype=np.float64)
        accel = np.asarray(accel_mps2, dtype=np.float64)

        # 1. Gyro integration step: q_{k+1} = q_k ⊗ Δq(ω Δt)
        angle = float(np.linalg.norm(gyro)) * dt_s
        if angle > 1e-12:
            delta_q = quaternion_from_axis_angle(gyro, angle)
            self.q = quaternion_multiply(self.q, delta_q)
            self.q /= np.linalg.norm(self.q)

        # 2. Gravity tilt correction (only when vehicle is not experiencing severe linear dynamics)
        a_norm = float(np.linalg.norm(accel))
        if self.accel_gate_min <= a_norm <= self.accel_gate_max:
            # Measured support reaction unit vector in vehicle frame
            a_unit = accel / a_norm
            # Expected support reaction in vehicle frame from current attitude R_v^n:
            # In ENU: support reaction is +Z = [0, 0, 1]. In vehicle frame: v_g = (R_v^n)^T @ [0, 0, 1]
            R_v_n = quaternion_to_rotation_matrix(self.q)
            v_g = R_v_n[2, :]  # 3rd row of R_v_n is (R_v_n)^T @ [0, 0, 1]

            # Error cross product in vehicle frame (negative feedback rotates v_g onto a_unit)
            tilt_error = np.cross(a_unit, v_g)
            err_norm = float(np.linalg.norm(tilt_error))
            if err_norm > 1e-8:
                corr_angle = self.kp_tilt * err_norm * dt_s
                corr_q = quaternion_from_axis_angle(tilt_error, corr_angle)
                self.q = quaternion_multiply(self.q, corr_q)
                self.q /= np.linalg.norm(self.q)

        return self.q.copy()

    def get_quaternion(self) -> Tuple[float, float, float, float]:
        """Get current attitude quaternion (w, x, y, z)."""
        return (float(self.q[0]), float(self.q[1]), float(self.q[2]), float(self.q[3]))

    def get_rotation_matrix(self) -> np.ndarray:
        """Get current 3x3 rotation matrix R_v^n."""
        return quaternion_to_rotation_matrix(self.q)

    def get_euler_deg(self) -> Tuple[float, float, float]:
        """Get current Euler angles in degrees (roll, pitch, yaw)."""
        r, p, y = quaternion_to_euler_rad(self.q)
        return (math.degrees(r), math.degrees(p), math.degrees(y))
