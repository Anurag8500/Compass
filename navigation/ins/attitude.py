"""Attitude representation and quaternion propagation for COMPASS (Phase 4).

This module implements quaternion mathematics and attitude propagation for the
classical strapdown Inertial Navigation System (INS).

Authoritative Conventions:
1. Quaternion Format:
   - Hamilton convention, scalar-first:
         q = [w, x, y, z]^T = [q_w, q_x, q_y, q_z]^T
   - Unity constraint: ||q|| = sqrt(w^2 + x^2 + y^2 + z^2) = 1.0.

2. Coordinate Frame Mapping:
   - The attitude quaternion q represents the rotation from the vehicle frame (v)
     to the local East-North-Up navigation frame (n):
         R_v^n @ v_vehicle = v_navigation
   - Vehicle Frame (v): Forward-Lateral-Up (FLU) right-handed convention.
   - Navigation Frame (n): East-North-Up (ENU) right-handed tangent plane.

3. Propagation & Composition Order:
   - Gyroscope measurements omega_v are resolved in the vehicle frame (body-fixed).
   - The finite angular increment over timestep dt is theta = omega_v * dt.
   - For a body-fixed angular rate, the right-multiplication Hamilton product applies:
         q[k+1] = normalize(q[k] ⊗ delta_q(omega_v * dt))
   - After propagation, q is explicitly normalized to eliminate numerical drift.

4. Numerical Stability:
   - delta_q utilizes a Taylor series expansion for small angles (||theta|| < 1e-8 rad)
     to prevent division by zero or loss of numerical precision.
"""

from __future__ import annotations

import math
from typing import Tuple
import numpy as np


def quaternion_norm(q: np.ndarray | Tuple[float, float, float, float]) -> float:
    """Compute Euclidean L2 norm of quaternion q = [w, x, y, z]."""
    q_arr = np.asarray(q, dtype=np.float64)
    return float(np.linalg.norm(q_arr))


def quaternion_normalize(q: np.ndarray | Tuple[float, float, float, float]) -> np.ndarray:
    """Normalize quaternion q = [w, x, y, z] to unit length.

    Raises:
        ValueError: If quaternion has zero or near-zero norm (non-invertible/degenerate).
    """
    q_arr = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q_arr)
    if norm < 1e-12:
        raise ValueError(f"Cannot normalize degenerate quaternion with near-zero norm: {norm:.2e}")
    return q_arr / norm


def quaternion_conjugate(q: np.ndarray | Tuple[float, float, float, float]) -> np.ndarray:
    """Compute conjugate of quaternion q = [w, x, y, z] -> [w, -x, -y, -z]."""
    q_arr = np.asarray(q, dtype=np.float64)
    return np.array([q_arr[0], -q_arr[1], -q_arr[2], -q_arr[3]], dtype=np.float64)


def quaternion_inverse(q: np.ndarray | Tuple[float, float, float, float]) -> np.ndarray:
    """Compute inverse of quaternion q: q^(-1) = conjugate(q) / ||q||^2."""
    q_arr = np.asarray(q, dtype=np.float64)
    norm_sq = float(np.dot(q_arr, q_arr))
    if norm_sq < 1e-12:
        raise ValueError(f"Cannot invert degenerate quaternion with near-zero norm squared: {norm_sq:.2e}")
    return np.array([q_arr[0], -q_arr[1], -q_arr[2], -q_arr[3]], dtype=np.float64) / norm_sq


def quaternion_multiply(
    p: np.ndarray | Tuple[float, float, float, float],
    q: np.ndarray | Tuple[float, float, float, float],
) -> np.ndarray:
    """Compute Hamilton quaternion product r = p ⊗ q.

    Convention: Scalar first [w, x, y, z].
    For p = [p_w, p_v], q = [q_w, q_v]:
        r_w = p_w * q_w - p_v · q_v
        r_v = p_w * q_v + q_w * p_v + p_v × q_v
    """
    p_arr = np.asarray(p, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)

    pw, px, py, pz = p_arr[0], p_arr[1], p_arr[2], p_arr[3]
    qw, qx, qy, qz = q_arr[0], q_arr[1], q_arr[2], q_arr[3]

    w = pw * qw - px * qx - py * qy - pz * qz
    x = pw * qx + px * qw + py * qz - pz * qy
    y = pw * qy - px * qz + py * qw + pz * qx
    z = pw * qz + px * qy - py * qx + pz * qw

    return np.array([w, x, y, z], dtype=np.float64)


def quaternion_to_rotation_matrix(q: np.ndarray | Tuple[float, float, float, float]) -> np.ndarray:
    """Convert unit attitude quaternion q = [w, x, y, z] to 3x3 rotation matrix R_v^n.

    The resulting matrix satisfies:
        v_n = R_v^n @ v_v
    where v_v is a vector in the vehicle frame and v_n is the vector in the local ENU frame.

    Properties:
        - Orthogonal: R @ R^T = I
        - Proper rotation: det(R) = +1.0
    """
    q_norm = quaternion_normalize(q)
    w, x, y, z = q_norm[0], q_norm[1], q_norm[2], q_norm[3]

    # Direction cosine matrix (Hamilton scalar-first)
    R = np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)

    return R


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 proper rotation matrix R to unit quaternion q = [w, x, y, z].

    Uses Shepperd's numerically robust algorithm to avoid catastrophic cancellation.
    """
    R_arr = np.asarray(R, dtype=np.float64)
    if R_arr.shape != (3, 3):
        raise ValueError(f"Rotation matrix must have shape (3, 3), got {R_arr.shape}")

    tr = float(np.trace(R_arr))

    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0  # s = 4 * w
        w = 0.25 * s
        x = (R_arr[2, 1] - R_arr[1, 2]) / s
        y = (R_arr[0, 2] - R_arr[2, 0]) / s
        z = (R_arr[1, 0] - R_arr[0, 1]) / s
    elif (R_arr[0, 0] > R_arr[1, 1]) and (R_arr[0, 0] > R_arr[2, 2]):
        s = math.sqrt(1.0 + R_arr[0, 0] - R_arr[1, 1] - R_arr[2, 2]) * 2.0  # s = 4 * x
        w = (R_arr[2, 1] - R_arr[1, 2]) / s
        x = 0.25 * s
        y = (R_arr[0, 1] + R_arr[1, 0]) / s
        z = (R_arr[0, 2] + R_arr[2, 0]) / s
    elif R_arr[1, 1] > R_arr[2, 2]:
        s = math.sqrt(1.0 + R_arr[1, 1] - R_arr[0, 0] - R_arr[2, 2]) * 2.0  # s = 4 * y
        w = (R_arr[0, 2] - R_arr[2, 0]) / s
        x = (R_arr[0, 1] + R_arr[1, 0]) / s
        y = 0.25 * s
        z = (R_arr[1, 2] + R_arr[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R_arr[2, 2] - R_arr[0, 0] - R_arr[1, 1]) * 2.0  # s = 4 * z
        w = (R_arr[1, 0] - R_arr[0, 1]) / s
        x = (R_arr[0, 2] + R_arr[2, 0]) / s
        y = (R_arr[1, 2] + R_arr[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z], dtype=np.float64)
    if q[0] < 0.0:
        q = -q  # Canonical positive scalar part
    return quaternion_normalize(q)


def delta_quaternion(
    omega_v: np.ndarray | Tuple[float, float, float],
    dt: float,
) -> np.ndarray:
    """Compute finite rotation quaternion delta_q from vehicle angular velocity over timestep dt.

    The vehicle-frame rotation vector is theta = omega_v * dt.
    The exact closed-form quaternion representation is:
        delta_q = [cos(||theta|| / 2), (theta / ||theta||) * sin(||theta|| / 2)]^T

    For small angles (||theta|| < 1e-8), Taylor series expansion prevents division by zero:
        cos(theta / 2) ≈ 1 - theta^2 / 8
        sin(theta / 2) / theta ≈ 1/2 - theta^2 / 48

    Args:
        omega_v: 3-axis angular velocity in vehicle frame [rad/s].
        dt: Integration timestep [seconds]. Must be > 0.

    Returns:
        delta_q: Unit quaternion [w, x, y, z].
    """
    if dt <= 0.0:
        raise ValueError(f"Timestep dt must be strictly positive, got {dt}")

    w_arr = np.asarray(omega_v, dtype=np.float64)
    if w_arr.shape != (3,):
        raise ValueError(f"omega_v must be a 3-element vector, got shape {w_arr.shape}")

    theta = w_arr * dt
    angle_sq = float(np.dot(theta, theta))
    angle = math.sqrt(angle_sq)

    if angle < 1e-8:
        # 4th-order Taylor series expansions around 0
        w = 1.0 - 0.125 * angle_sq
        s = 0.5 - angle_sq / 48.0
        x = theta[0] * s
        y = theta[1] * s
        z = theta[2] * s
    else:
        half_angle = 0.5 * angle
        w = math.cos(half_angle)
        s = math.sin(half_angle) / angle
        x = theta[0] * s
        y = theta[1] * s
        z = theta[2] * s

    return quaternion_normalize(np.array([w, x, y, z], dtype=np.float64))


def propagate_attitude(
    q_current: np.ndarray | Tuple[float, float, float, float],
    omega_v: np.ndarray | Tuple[float, float, float],
    dt: float,
) -> np.ndarray:
    """Propagate vehicle attitude quaternion q_current over timestep dt under angular rate omega_v.

    Equation:
        q_next = normalize(q_current ⊗ delta_q(omega_v * dt))

    Args:
        q_current: Current unit quaternion [w, x, y, z] mapping vehicle to navigation frame (R_v^n).
        omega_v: 3-axis vehicle-frame angular velocity [rad/s].
        dt: Elapsed timestep [seconds].

    Returns:
        q_next: Propagated unit quaternion [w, x, y, z].
    """
    dq = delta_quaternion(omega_v, dt)
    q_next_unnorm = quaternion_multiply(q_current, dq)
    return quaternion_normalize(q_next_unnorm)


def quaternion_to_euler_deg(q: np.ndarray | Tuple[float, float, float, float]) -> Tuple[float, float, float]:
    """Extract diagnostic Z-Y-X Euler angles (roll, pitch, yaw) in degrees from quaternion q.

    Convention: R_v^n = R_z(yaw) @ R_y(pitch) @ R_x(roll)
    - Roll: Rotation about vehicle forward axis X_v [deg].
    - Pitch: Rotation about vehicle lateral axis Y_v [deg].
    - Yaw: Rotation about vertical axis Z_v [deg]. Track heading relative to North in ENU.

    NOTE: Used for human inspection and logging only. Quaternions are the sole internal representation.
    """
    R = quaternion_to_rotation_matrix(q)
    sin_p = -float(R[2, 0])
    sin_p = max(-1.0, min(1.0, sin_p))
    pitch_rad = math.asin(sin_p)

    if abs(R[2, 0]) < 0.999999:
        roll_rad = math.atan2(float(R[2, 1]), float(R[2, 2]))
        yaw_rad = math.atan2(float(R[1, 0]), float(R[0, 0]))
    else:
        # Gimbal lock singularity
        roll_rad = 0.0
        yaw_rad = math.atan2(-float(R[0, 1]), float(R[1, 1]))

    return math.degrees(roll_rad), math.degrees(pitch_rad), math.degrees(yaw_rad)
