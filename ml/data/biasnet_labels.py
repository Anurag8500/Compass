"""Inverse-Problem BiasNet Target Generation and Identifiability Pipeline.

COMPASS Phase 8 — Learned IMU Bias Correction.
Generates pseudo-ground-truth bias corrections (delta_ba, delta_bg) across a
configurable short optimization horizon by solving an inverse estimation problem
against independent high-precision reference telemetry (VBOX RTK GNSS).

Mathematical Formulation:
    For a window ending at t_end with horizon H in [0.5, 2.0] s:
    1. Initialize local ENU coordinate frame and reference state at t_0 = t_end - H:
           p_0 = [0, 0, 0]^T
           v_0 = v_ref_spd(t_0) * [sin(psi(t_0)), cos(psi(t_0)), 0]^T
           q_0 = R_to_quaternion(R(psi(t_0)))
    2. Given unknown parameter vector delta_b = [delta_ba(3), delta_bg(3)]^T in R^6:
           ba_eff = ba_prior + delta_ba
           bg_eff = bg_prior + delta_bg
    3. Propagate state forward from t_0 to t_end using canonical Phase 4/5 strapdown mechanization:
           a^n[k] = R[k] @ (f_m^v[k] - ba_eff) + g^n
           p[k+1] = p[k] + v[k]*dt + 0.5*a^n[k]*dt^2
           v[k+1] = v[k] + a^n[k]*dt
           q[k+1] = normalize(q[k] (x) delta_q((omega_m^v[k] - bg_eff)*dt))
    4. Residual Vector r(delta_b):
           e_p[k] = p[k] - p_ref[k]
           e_v[k] = v[k] - v_ref[k]
           e_theta[k] = 2 * (q[k]^(-1) (x) q_ref[k])_xyz
           r(delta_b) = [ W_p * e_p, W_v * e_v, W_theta * e_theta ]
    5. Solve via Damped Gauss-Newton / Levenberg-Marquardt:
           delta_b* = argmin ||r(delta_b)||^2
           (J^T J + lambda I) delta = -J^T r
    6. Identifiability & Conditioning Audit:
           - SVD of Jacobian: J = U S V^T
           - Condition number: kappa = sigma_max / sigma_min
           - Residual reduction ratio: ||r_before|| / ||r_after||
           - Bound activity: |delta_ba| <= B_ba, |delta_bg| <= B_bg
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from navigation.frames.local_geo import GeoReference
from navigation.ins.attitude import (
    delta_quaternion,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_normalize,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)
from navigation.ins.propagation import STANDARD_GRAVITY


@dataclass(frozen=True)
class BiasNetOptimizationConfig:
    """Configuration parameters for inverse-problem bias label optimization and gating."""
    horizon_s: float = 1.0                     # Target optimization horizon duration [s]
    dt_s: float = 0.1                          # Canonical 10 Hz integration timestep [s]
    weight_position: float = 1.0               # W_p: position residual scale [m^-1]
    weight_velocity: float = 1.0               # W_v: velocity residual scale [(m/s)^-1]
    weight_orientation: float = 10.0           # W_theta: orientation residual scale [rad^-1]
    bound_accel_mps2: float = 0.3              # Physical clamp limit for delta_ba [m/s^2]
    bound_gyro_rads: float = 0.05              # Physical clamp limit for delta_bg [rad/s]
    max_condition_number: float = 1000.0       # Threshold for Jacobian ill-conditioning rejection
    min_effective_rank: int = 6                # Minimum required Jacobian effective rank
    min_residual_reduction: float = 1.05       # Minimum required residual reduction (||r0|| / ||r*||)
    max_iterations: int = 10                   # Maximum Levenberg-Marquardt iterations
    convergence_step_tol: float = 1e-6         # Step norm tolerance ||delta||
    damping_init: float = 1e-4                 # Initial LM diagonal damping lambda
    fd_step_accel: float = 1e-5                # Finite-difference step for accel bias [m/s^2]
    fd_step_gyro: float = 1e-6                 # Finite-difference step for gyro bias [rad/s]
    residual_mode: str = "multi_point"         # "multi_point" (all steps) or "endpoint" (final only)


@dataclass
class BiasOptimizationResult:
    """Detailed result of bias optimization and identifiability audit for a single window."""
    delta_b_unconstrained: np.ndarray   # (6,) [dba_x, dba_y, dba_z, dbg_x, dbg_y, dbg_z]
    delta_b_constrained: np.ndarray     # (6,) clamped to configured physical limits
    bounds_active: bool                 # True if unconstrained solution violated physical bounds
    residual_before: float              # ||r(0)||_2
    residual_after: float               # ||r(delta_b*)||_2
    residual_reduction_ratio: float     # ||r_before|| / max(||r_after||, 1e-12)
    converged: bool                     # True if solver converged within tolerances
    iterations: int                     # Number of LM iterations executed
    singular_values: np.ndarray         # (6,) Singular values of Jacobian sigma_1 >= ... >= sigma_6
    condition_number: float             # sigma_max / sigma_min
    effective_rank: int                 # Count of singular values > 1e-3 * sigma_max
    sensitivities: np.ndarray           # (6,) L2 norm of Jacobian columns
    is_eligible: bool                   # True if window passes ALL identifiability and quality gates
    reason_code: str                    # "VALID", "ILL_CONDITIONED", "BOUNDS_ACTIVE", etc.


def heading_to_quaternion(heading_deg: float) -> np.ndarray:
    """Convert azimuth heading (clockwise from North in ENU) to body-to-ENU unit quaternion.

    In vehicle FLU frame:
        +X is Forward, +Y is Left, +Z is Up.
    In local ENU tangent plane:
        +X is East, +Y is North, +Z is Up.
    For heading psi (radians):
        Forward unit vector in ENU = [sin(psi), cos(psi), 0]^T
        Left unit vector in ENU    = [-cos(psi), sin(psi), 0]^T
        Up unit vector in ENU      = [0, 0, 1]^T
    """
    psi = math.radians(float(heading_deg))
    R = np.array([
        [math.sin(psi), -math.cos(psi), 0.0],
        [math.cos(psi),  math.sin(psi), 0.0],
        [0.0,            0.0,           1.0],
    ], dtype=np.float64)
    return rotation_matrix_to_quaternion(R)


def compute_quaternion_angle_error(q_est: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    """Compute 3D small-angle rotation error vector delta_theta in vehicle frame.

    Under right-multiplicative error convention:
        q_ref = normalize(q_est (x) delta_q(delta_theta))
    Therefore:
        delta_q = q_est^(-1) (x) q_ref
    """
    q_inv = quaternion_inverse(q_est)
    q_err = quaternion_multiply(q_inv, q_ref)
    if q_err[0] < 0.0:
        q_err = -q_err  # Shortest rotation

    w = np.clip(float(q_err[0]), -1.0, 1.0)
    v = q_err[1:4]
    v_norm = float(np.linalg.norm(v))
    if v_norm < 1e-8:
        return 2.0 * v
    angle = 2.0 * math.atan2(v_norm, w)
    return (angle / v_norm) * v


def forward_propagate_window(
    p0: np.ndarray,
    v0: np.ndarray,
    q0: np.ndarray,
    f_m_v: np.ndarray,
    omega_m_v: np.ndarray,
    dt_arr: np.ndarray,
    ba: np.ndarray,
    bg: np.ndarray,
    gravity: float = STANDARD_GRAVITY,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward propagate strapdown kinematics across K steps using explicit bias vectors.

    Args:
        p0: (3,) initial ENU position [m].
        v0: (3,) initial ENU velocity [m/s].
        q0: (4,) initial body-to-ENU unit quaternion.
        f_m_v: (K, 3) vehicle-frame specific force measurements.
        omega_m_v: (K, 3) vehicle-frame angular rate measurements.
        dt_arr: (K,) integration timesteps in seconds.
        ba: (3,) effective accelerometer bias [m/s^2].
        bg: (3,) effective gyroscope bias [rad/s].
        gravity: scalar gravity magnitude (default 9.80665 m/s^2).

    Returns:
        (p_traj, v_traj, q_traj) arrays of shape (K+1, 3), (K+1, 3), (K+1, 4).
    """
    K = len(f_m_v)
    p_traj = np.zeros((K + 1, 3), dtype=np.float64)
    v_traj = np.zeros((K + 1, 3), dtype=np.float64)
    q_traj = np.zeros((K + 1, 4), dtype=np.float64)

    p_traj[0] = p0
    v_traj[0] = v0
    q_traj[0] = quaternion_normalize(q0)

    g_n = np.array([0.0, 0.0, -gravity], dtype=np.float64)

    for k in range(K):
        dt = float(dt_arr[k])
        f_debias = f_m_v[k] - ba
        w_debias = omega_m_v[k] - bg

        R_k = quaternion_to_rotation_matrix(q_traj[k])
        a_n = R_k @ f_debias + g_n

        # Discrete position and velocity integration
        p_traj[k + 1] = p_traj[k] + v_traj[k] * dt + 0.5 * a_n * (dt * dt)
        v_traj[k + 1] = v_traj[k] + a_n * dt

        # Attitude propagation
        dq = delta_quaternion(w_debias, dt)
        q_next = quaternion_multiply(q_traj[k], dq)
        q_traj[k + 1] = quaternion_normalize(q_next)

    return p_traj, v_traj, q_traj


def evaluate_residual(
    delta_b: np.ndarray,
    p0: np.ndarray,
    v0: np.ndarray,
    q0: np.ndarray,
    f_m_v: np.ndarray,
    omega_m_v: np.ndarray,
    dt_arr: np.ndarray,
    p_ref_traj: np.ndarray,
    v_ref_traj: np.ndarray,
    q_ref_traj: np.ndarray,
    ba_prior: np.ndarray,
    bg_prior: np.ndarray,
    cfg: BiasNetOptimizationConfig,
) -> np.ndarray:
    """Compute scaled residual vector r(delta_b) between propagated state and reference."""
    ba_eff = ba_prior + delta_b[0:3]
    bg_eff = bg_prior + delta_b[3:6]

    p_prop, v_prop, q_prop = forward_propagate_window(
        p0=p0,
        v0=v0,
        q0=q0,
        f_m_v=f_m_v,
        omega_m_v=omega_m_v,
        dt_arr=dt_arr,
        ba=ba_eff,
        bg=bg_eff,
    )

    K = len(f_m_v)
    if cfg.residual_mode == "endpoint":
        # Final step comparison only
        e_p = (p_prop[-1] - p_ref_traj[-1]) * cfg.weight_position
        e_v = (v_prop[-1] - v_ref_traj[-1]) * cfg.weight_velocity
        e_theta = compute_quaternion_angle_error(q_prop[-1], q_ref_traj[-1]) * cfg.weight_orientation
        return np.concatenate([e_p, e_v, e_theta])

    # Multi-point residual across all K integration endpoints (k = 1 ... K)
    norm_factor = 1.0 / math.sqrt(K)
    res_list: List[np.ndarray] = []
    for k in range(1, K + 1):
        e_p = (p_prop[k] - p_ref_traj[k]) * (cfg.weight_position * norm_factor)
        e_v = (v_prop[k] - v_ref_traj[k]) * (cfg.weight_velocity * norm_factor)
        e_th = compute_quaternion_angle_error(q_prop[k], q_ref_traj[k]) * (cfg.weight_orientation * norm_factor)
        res_list.extend([e_p, e_v, e_th])

    return np.concatenate(res_list)


def compute_numerical_jacobian(
    delta_b: np.ndarray,
    p0: np.ndarray,
    v0: np.ndarray,
    q0: np.ndarray,
    f_m_v: np.ndarray,
    omega_m_v: np.ndarray,
    dt_arr: np.ndarray,
    p_ref_traj: np.ndarray,
    v_ref_traj: np.ndarray,
    q_ref_traj: np.ndarray,
    ba_prior: np.ndarray,
    bg_prior: np.ndarray,
    cfg: BiasNetOptimizationConfig,
) -> np.ndarray:
    """Compute central finite-difference Jacobian J = dr / d(delta_b) in R^(M x 6)."""
    r_dim = 9 if cfg.residual_mode == "endpoint" else 9 * len(f_m_v)
    J = np.zeros((r_dim, 6), dtype=np.float64)

    steps = [
        cfg.fd_step_accel, cfg.fd_step_accel, cfg.fd_step_accel,
        cfg.fd_step_gyro, cfg.fd_step_gyro, cfg.fd_step_gyro,
    ]

    for j in range(6):
        h = steps[j]
        db_plus = delta_b.copy()
        db_plus[j] += h

        db_minus = delta_b.copy()
        db_minus[j] -= h

        r_plus = evaluate_residual(
            db_plus, p0, v0, q0, f_m_v, omega_m_v, dt_arr,
            p_ref_traj, v_ref_traj, q_ref_traj, ba_prior, bg_prior, cfg,
        )
        r_minus = evaluate_residual(
            db_minus, p0, v0, q0, f_m_v, omega_m_v, dt_arr,
            p_ref_traj, v_ref_traj, q_ref_traj, ba_prior, bg_prior, cfg,
        )

        J[:, j] = (r_plus - r_minus) / (2.0 * h)

    return J


def solve_window_bias_correction(
    f_m_v_win: np.ndarray,
    omega_m_v_win: np.ndarray,
    timestamps_ns_win: np.ndarray,
    ref_lat_win: np.ndarray,
    ref_lon_win: np.ndarray,
    ref_alt_m_win: np.ndarray,
    ref_speed_mps_win: np.ndarray,
    ref_heading_deg_win: np.ndarray,
    ba_prior: Optional[np.ndarray] = None,
    bg_prior: Optional[np.ndarray] = None,
    config: Optional[BiasNetOptimizationConfig] = None,
) -> BiasOptimizationResult:
    """Solve the short-horizon inverse problem for a single IMU/reference window.

    Args:
        f_m_v_win: (W, 3) specific force in vehicle frame [m/s^2].
        omega_m_v_win: (W, 3) angular velocity in vehicle frame [rad/s].
        timestamps_ns_win: (W,) nanosecond timestamps.
        ref_lat_win: (W,) reference WGS84 latitude [deg].
        ref_lon_win: (W,) reference WGS84 longitude [deg].
        ref_alt_m_win: (W,) reference true elevation [m].
        ref_speed_mps_win: (W,) reference speed [m/s].
        ref_heading_deg_win: (W,) reference heading [deg].
        ba_prior: Optional (3,) prior accel bias (default 0).
        bg_prior: Optional (3,) prior gyro bias (default 0).
        config: Optional BiasNetOptimizationConfig.

    Returns:
        BiasOptimizationResult containing solutions, diagnostics, and eligibility gate.
    """
    cfg = config or BiasNetOptimizationConfig()
    W = len(timestamps_ns_win)
    if W < 3:
        return _make_failure_result("WINDOW_TOO_SHORT")

    # 1. Validate full window arrays are finite upfront
    for arr in [f_m_v_win, omega_m_v_win, ref_lat_win, ref_lon_win, ref_alt_m_win, ref_speed_mps_win, ref_heading_deg_win]:
        if not np.isfinite(arr).all():
            return _make_failure_result("NON_FINITE_INPUT")

    # 2. Determine sub-horizon integration slice ending at window end
    target_dt_ns = int(round(cfg.horizon_s * 1e9))
    t_end = int(timestamps_ns_win[-1])
    t_target_start = t_end - target_dt_ns

    # Find closest start index
    idx_start = int(np.searchsorted(timestamps_ns_win, t_target_start))
    idx_start = max(0, min(idx_start, W - 2))

    f_sub = f_m_v_win[idx_start:-1]          # (K, 3) measurements during integration intervals
    w_sub = omega_m_v_win[idx_start:-1]      # (K, 3)
    ts_sub = timestamps_ns_win[idx_start:]   # (K+1,) state node timestamps
    K = len(f_sub)

    if K < 2:
        return _make_failure_result("HORIZON_TOO_SHORT")

    # Timestep durations
    dt_arr = (ts_sub[1:] - ts_sub[:-1]) * 1e-9
    if (dt_arr <= 0.0).any() or (dt_arr > 0.5).any():
        return _make_failure_result("TIMESTEP_ANOMALY")

    # 3. Setup Ground Truth Reference Trajectory in Local ENU
    lat_sub = ref_lat_win[idx_start:]
    lon_sub = ref_lon_win[idx_start:]
    alt_sub = ref_alt_m_win[idx_start:]
    spd_sub = ref_speed_mps_win[idx_start:]
    hdg_sub = ref_heading_deg_win[idx_start:]

    geo_ref = GeoReference(lat_ref=float(lat_sub[0]), lon_ref=float(lon_sub[0]), alt_ref=float(alt_sub[0]))
    e_ref, n_ref, u_ref = geo_ref.geodetic_to_enu(lat_sub, lon_sub, alt_sub)
    p_ref_traj = np.column_stack([e_ref, n_ref, u_ref])

    v_ref_traj = np.zeros((K + 1, 3), dtype=np.float64)
    q_ref_traj = np.zeros((K + 1, 4), dtype=np.float64)

    for k in range(K + 1):
        psi = math.radians(float(hdg_sub[k]))
        spd = float(spd_sub[k])
        v_ref_traj[k] = [spd * math.sin(psi), spd * math.cos(psi), 0.0]
        q_ref_traj[k] = heading_to_quaternion(hdg_sub[k])

    # Initial state conditions at t_0
    p0 = p_ref_traj[0]
    v0 = v_ref_traj[0]
    q0 = q_ref_traj[0]

    ba0 = np.zeros(3, dtype=np.float64) if ba_prior is None else np.asarray(ba_prior, dtype=np.float64)
    bg0 = np.zeros(3, dtype=np.float64) if bg_prior is None else np.asarray(bg_prior, dtype=np.float64)

    # 4. Initial Residual & Jacobian Evaluation at delta_b = 0
    delta_b = np.zeros(6, dtype=np.float64)
    r_curr = evaluate_residual(delta_b, p0, v0, q0, f_sub, w_sub, dt_arr, p_ref_traj, v_ref_traj, q_ref_traj, ba0, bg0, cfg)
    res_before = float(np.linalg.norm(r_curr))

    damping = cfg.damping_init
    converged = False
    iter_count = 0

    J = compute_numerical_jacobian(delta_b, p0, v0, q0, f_sub, w_sub, dt_arr, p_ref_traj, v_ref_traj, q_ref_traj, ba0, bg0, cfg)

    # 5. Levenberg-Marquardt Optimization Loop
    for it in range(cfg.max_iterations):
        iter_count += 1
        J = compute_numerical_jacobian(delta_b, p0, v0, q0, f_sub, w_sub, dt_arr, p_ref_traj, v_ref_traj, q_ref_traj, ba0, bg0, cfg)
        JtJ = J.T @ J
        g = J.T @ r_curr

        # Damped normal equations: (J^T J + lambda I) step = -g
        A = JtJ + damping * np.eye(6, dtype=np.float64)
        try:
            step = np.linalg.solve(A, -g)
        except np.linalg.LinAlgError:
            step = -np.linalg.pinv(A) @ g

        step_norm = float(np.linalg.norm(step))
        cand_db = delta_b + step

        r_cand = evaluate_residual(cand_db, p0, v0, q0, f_sub, w_sub, dt_arr, p_ref_traj, v_ref_traj, q_ref_traj, ba0, bg0, cfg)
        res_cand = float(np.linalg.norm(r_cand))

        if res_cand < np.linalg.norm(r_curr):
            delta_b = cand_db
            r_curr = r_cand
            damping = max(damping / 5.0, 1e-7)
            if step_norm < cfg.convergence_step_tol:
                converged = True
                break
        else:
            damping = min(damping * 10.0, 1e4)

    res_after = float(np.linalg.norm(r_curr))
    reduction_ratio = res_before / max(res_after, 1e-12)

    # 6. SVD & Conditioning Audit on Final Jacobian
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    sigma_max = float(S[0])
    sigma_min = float(S[-1])
    cond_num = sigma_max / max(sigma_min, 1e-15)

    eff_rank = int(np.sum(S > (1e-3 * sigma_max)))
    sensitivities = np.linalg.norm(J, axis=0)

    # 7. Clamping & Physical Bound Check
    unconstrained = delta_b.copy()
    b_acc = cfg.bound_accel_mps2
    b_gyr = cfg.bound_gyro_rads

    bounds_violated = bool(
        (np.abs(unconstrained[0:3]) > b_acc).any() or
        (np.abs(unconstrained[3:6]) > b_gyr).any()
    )

    constrained = unconstrained.copy()
    constrained[0:3] = np.clip(constrained[0:3], -b_acc, b_acc)
    constrained[3:6] = np.clip(constrained[3:6], -b_gyr, b_gyr)

    # 8. Identifiability Eligibility Decision Gate
    is_eligible = True
    reason = "VALID"

    if cond_num > cfg.max_condition_number:
        is_eligible = False
        reason = "ILL_CONDITIONED"
    elif eff_rank < cfg.min_effective_rank:
        is_eligible = False
        reason = "DEFICIENT_RANK"
    elif bounds_violated:
        is_eligible = False
        reason = "BOUNDS_ACTIVE"
    elif reduction_ratio < cfg.min_residual_reduction:
        is_eligible = False
        reason = "POOR_RESIDUAL_REDUCTION"
    elif not np.isfinite(unconstrained).all():
        is_eligible = False
        reason = "NON_FINITE_SOLUTION"

    return BiasOptimizationResult(
        delta_b_unconstrained=unconstrained,
        delta_b_constrained=constrained,
        bounds_active=bounds_violated,
        residual_before=res_before,
        residual_after=res_after,
        residual_reduction_ratio=reduction_ratio,
        converged=converged,
        iterations=iter_count,
        singular_values=S,
        condition_number=cond_num,
        effective_rank=eff_rank,
        sensitivities=sensitivities,
        is_eligible=is_eligible,
        reason_code=reason,
    )


def _make_failure_result(reason: str) -> BiasOptimizationResult:
    """Construct default rejected optimization result."""
    return BiasOptimizationResult(
        delta_b_unconstrained=np.full(6, np.nan, dtype=np.float64),
        delta_b_constrained=np.full(6, np.nan, dtype=np.float64),
        bounds_active=False,
        residual_before=np.nan,
        residual_after=np.nan,
        residual_reduction_ratio=0.0,
        converged=False,
        iterations=0,
        singular_values=np.zeros(6, dtype=np.float64),
        condition_number=np.inf,
        effective_rank=0,
        sensitivities=np.zeros(6, dtype=np.float64),
        is_eligible=False,
        reason_code=reason,
    )
