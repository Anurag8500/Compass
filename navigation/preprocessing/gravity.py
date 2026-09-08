"""Gravity resolution and validation utility for COMPASS (Phase 3).

CRITICAL ARCHITECTURAL CONTRACT:
================================
THIS MODULE IS A MATHEMATICAL TEST AND VALIDATION UTILITY.

DO NOT USE THIS MODULE TO PRODUCE A PERMANENT GRAVITY-COMPENSATED STREAM
FOR LIVE NAVIGATION.

In the live COMPASS architecture, the preprocessing chain outputs vehicle-frame
specific force f_m^v directly. Physical gravity addition is performed internally
inside the central ESKF strapdown mechanization using its continuously estimated
vehicle attitude R_v^n:
    a^n = R_v^n (f_m^v - b_a^v) + g^n

with:
    g^n = [0, 0, -g]^T in local East-North-Up (ENU) navigation frame.
    g = 9.80665 m/s^2 (standard gravity).

Canonical Stationary Sanity Check:
For a stationary level vehicle facing north:
    R_v^n = I
    b_a^v = [0, 0, 0]^T
    f_m^v = [0, 0, +g]^T  (support reaction force pushing upward)
    a^n = I [0, 0, +g]^T + [0, 0, -g]^T = [0, 0, 0]^T  (identically zero coordinate acceleration)
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

# Standard Earth gravitational acceleration (WGS84 / SI standard)
STANDARD_GRAVITY_MPS2: float = 9.80665


def resolve_gravity(
    f_m_v: np.ndarray | Tuple[float, float, float],
    R_v_n: np.ndarray,
    b_a_v: Optional[np.ndarray | Tuple[float, float, float]] = None,
    g_val: float = STANDARD_GRAVITY_MPS2,
) -> np.ndarray:
    """Resolve coordinate acceleration in the local ENU navigation frame.

    Calculates:
        a^n = R_v^n (f_m^v - b_a^v) + g^n
    where:
        g^n = [0, 0, -g_val]^T

    Args:
        f_m_v: (3,) or (N, 3) measured vehicle-frame specific force (m/s^2).
        R_v_n: (3, 3) rotation matrix from vehicle frame to ENU navigation frame,
               OR (N, 3, 3) for time series.
        b_a_v: Optional (3,) accelerometer bias in vehicle frame (m/s^2). Defaults to 0.
        g_val: Magnitude of standard gravity (m/s^2, default 9.80665).

    Returns:
        (3,) or (N, 3) coordinate acceleration a^n in local ENU navigation frame.
    """
    f_arr = np.asarray(f_m_v, dtype=np.float64)
    R_arr = np.asarray(R_v_n, dtype=np.float64)

    # Physical gravity vector in local ENU navigation frame (points down along -Z)
    g_n = np.array([0.0, 0.0, -g_val], dtype=np.float64)

    if b_a_v is not None:
        b_arr = np.asarray(b_a_v, dtype=np.float64)
        f_debiased = f_arr - b_arr
    else:
        f_debiased = f_arr

    # Single vector evaluation
    if f_debiased.ndim == 1:
        if R_arr.shape != (3, 3):
            raise ValueError(f"For single vector, R_v_n must be (3, 3), got {R_arr.shape}")
        return (R_arr @ f_debiased) + g_n

    # Batch (N, 3) evaluation
    if f_debiased.ndim == 2:
        if R_arr.ndim == 2 and R_arr.shape == (3, 3):
            # Constant attitude across all samples
            return (R_arr @ f_debiased.T).T + g_n
        elif R_arr.ndim == 3 and R_arr.shape == (f_debiased.shape[0], 3, 3):
            # Time-varying attitude per sample
            res = np.zeros_like(f_debiased)
            for i in range(f_debiased.shape[0]):
                res[i] = (R_arr[i] @ f_debiased[i]) + g_n
            return res
        else:
            raise ValueError(f"Shape mismatch: f_m_v={f_debiased.shape} vs R_v_n={R_arr.shape}")

    raise ValueError(f"f_m_v must be 1D or 2D, got {f_debiased.ndim}D")


def verify_stationary_gravity_resolution(
    f_m_v: np.ndarray | Tuple[float, float, float],
    tol_mps2: float = 1e-4,
    g_val: float = STANDARD_GRAVITY_MPS2,
) -> bool:
    """Verify that a stationary level vehicle specific force resolves to zero coordinate acceleration."""
    R_level = np.eye(3, dtype=np.float64)
    b_zero = np.zeros(3, dtype=np.float64)
    a_n = resolve_gravity(f_m_v, R_level, b_a_v=b_zero, g_val=g_val)
    norm_a = float(np.linalg.norm(a_n))
    return norm_a <= tol_mps2
