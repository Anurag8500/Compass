"""Shared 9-Channel Feature Engineering for COMPASS ML Models.

COMPASS Phase 6 — ML Dataset Construction.
Governs the feature representation for BOTH VelocityNet and BiasNet.

Channel Ordering (Strict Invariant):
    0: f_x_v       (Forward specific force, m/s^2)
    1: f_y_v       (Lateral/right specific force, m/s^2)
    2: f_z_v       (Vertical/down specific force, m/s^2)
    3: omega_x_v   (Roll angular velocity, rad/s)
    4: omega_y_v   (Pitch angular velocity, rad/s)
    5: omega_z_v   (Yaw angular velocity, rad/s)
    6: norm_f_v    (||f^v|| specific force vector magnitude, m/s^2)
    7: norm_f_dot_v(||f_dot^v|| specific force jerk magnitude, m/s^3)
    8: norm_omega_v(||omega^v|| angular velocity vector magnitude, rad/s)

CRITICAL INVARIANTS:
1. Input motion signals are in the VEHICLE FRAME.
2. DO NOT gravity-compensate: specific force includes the physical gravity reaction.
   Gravity resolution is the sole responsibility of the ESKF strapdown mechanization.
3. Jerk magnitude is computed using a strictly causal timestamp-aware backward difference:
   f_dot[k] = (f[k] - f[k-1]) / delta_t[k].
"""

from __future__ import annotations

from typing import Tuple
import numpy as np

from navigation.schemas.config import CANONICAL_CHANNELS

NUM_CHANNELS: int = 9


def compute_canonical_features(
    timestamps_ns: np.ndarray,
    f_m_v: np.ndarray,
    omega_m_v: np.ndarray,
) -> np.ndarray:
    """Compute the canonical 9-channel feature matrix from vehicle-frame motion streams.

    Args:
        timestamps_ns: (N,) int64 timestamps in nanoseconds.
        f_m_v: (N, 3) float64 vehicle-frame specific force in m/s^2.
        omega_m_v: (N, 3) float64 vehicle-frame angular velocity in rad/s.

    Returns:
        (N, 9) float64 array with features arranged in CANONICAL_CHANNELS order.
    """
    n_samples = len(timestamps_ns)
    if n_samples == 0:
        return np.zeros((0, NUM_CHANNELS), dtype=np.float64)

    if f_m_v.shape != (n_samples, 3):
        raise ValueError(f"f_m_v shape must be ({n_samples}, 3), got {f_m_v.shape}")
    if omega_m_v.shape != (n_samples, 3):
        raise ValueError(f"omega_m_v shape must be ({n_samples}, 3), got {omega_m_v.shape}")

    # Validate finite inputs
    if not np.all(np.isfinite(f_m_v)):
        raise ValueError("f_m_v contains non-finite (NaN or Inf) values")
    if not np.all(np.isfinite(omega_m_v)):
        raise ValueError("omega_m_v contains non-finite (NaN or Inf) values")

    features = np.zeros((n_samples, NUM_CHANNELS), dtype=np.float64)

    # Channels 0-2: Vehicle-frame specific force components
    features[:, 0:3] = f_m_v

    # Channels 3-5: Vehicle-frame angular velocity components
    features[:, 3:6] = omega_m_v

    # Channel 6: Specific force Euclidean norm ||f^v||
    norm_f = np.linalg.norm(f_m_v, axis=1)
    features[:, 6] = norm_f

    # Channel 7: Causal timestamp-aware backward jerk magnitude ||f_dot^v||
    # f_dot[k] = (f[k] - f[k-1]) / dt[k]
    f_dot = np.zeros_like(f_m_v)
    if n_samples > 1:
        dts_s = np.diff(timestamps_ns) * 1e-9  # (N-1,)
        # Guard against zero or negative delta_t
        safe_dts = np.where(dts_s > 1e-6, dts_s, 0.1)  # Fallback to nominal 10 Hz dt if degenerate
        delta_f = np.diff(f_m_v, axis=0)  # (N-1, 3)
        f_dot[1:] = delta_f / safe_dts[:, np.newaxis]
        # For k=0, causal backward jerk is 0.0 (no prior sample exists in recording)
        f_dot[0] = 0.0

    norm_f_dot = np.linalg.norm(f_dot, axis=1)
    features[:, 7] = norm_f_dot

    # Channel 8: Angular velocity Euclidean norm ||omega^v||
    norm_omega = np.linalg.norm(omega_m_v, axis=1)
    features[:, 8] = norm_omega

    # Final finite validation
    if not np.all(np.isfinite(features)):
        raise ValueError("Generated feature matrix contains non-finite values")

    return features
