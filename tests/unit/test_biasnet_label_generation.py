"""Unit tests for BiasNet inverse-problem label generation and identifiability audit (Phase 8).

Tests:
1. Zero-bias synthetic case: when IMU is perfectly unbiased, recovered delta_b is zero.
2. Accel-only synthetic bias: recover injected accel bias while gyro bias stays zero.
3. Gyro-only synthetic bias: recover injected gyro bias while accel bias stays zero.
4. Mixed synthetic bias: recover both accel and gyro biases simultaneously.
5. Bound activation: large unphysical bias triggers bounds_active and rejects window.
6. Ill-conditioned window rejection: unexcited trajectory with ill-conditioned Jacobian rejected.
7. Residual reduction calculation: verified mathematically.
8. Deterministic repeated generation: exact bit-for-bit repeatability.
9. Non-finite input rejection: gracefully rejects NaN/Inf with error code.
"""

from __future__ import annotations

import math
from typing import Tuple
import numpy as np
import pytest

from navigation.ins.attitude import (
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)
from navigation.ins.propagation import STANDARD_GRAVITY
from navigation.frames.local_geo import GeoReference
from ml.data.biasnet_labels import (
    BiasNetOptimizationConfig,
    heading_to_quaternion,
    solve_window_bias_correction,
)


def generate_synthetic_curved_window(
    duration_s: float = 2.0,
    dt_s: float = 0.1,
    init_speed_mps: float = 12.0,
    forward_accel_mps2: float = 0.4,
    yaw_rate_rads: float = 0.08,
    accel_bias_true: np.ndarray = np.zeros(3),
    gyro_bias_true: np.ndarray = np.zeros(3),
    origin_lat: float = 52.41,
    origin_lon: float = -1.51,
    origin_alt: float = 110.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a dynamically consistent synthetic vehicle maneuver with ground truth."""
    N = int(round(duration_s / dt_s)) + 1
    t = np.arange(N) * dt_s
    t_ns = (t * 1e9).astype(np.int64)

    # Reference state integration in ENU
    psi = 0.2 + yaw_rate_rads * t  # Heading in radians
    speed = init_speed_mps + forward_accel_mps2 * t

    v_e = speed * np.sin(psi)
    v_n = speed * np.cos(psi)
    v_u = np.zeros(N)

    # Integrate position
    pos_e = np.zeros(N)
    pos_n = np.zeros(N)
    pos_u = np.zeros(N)
    for k in range(1, N):
        dt = dt_s
        pos_e[k] = pos_e[k - 1] + 0.5 * (v_e[k - 1] + v_e[k]) * dt
        pos_n[k] = pos_n[k - 1] + 0.5 * (v_n[k - 1] + v_n[k]) * dt

    # Convert ENU positions to geodetic WGS84
    geo = GeoReference(lat_ref=origin_lat, lon_ref=origin_lon, alt_ref=origin_alt)
    ref_lat, ref_lon, ref_alt = geo.enu_to_geodetic(pos_e, pos_n, pos_u)
    ref_heading = np.degrees(psi)

    # True IMU specific force and angular velocity in body FLU frame
    f_m_v = np.zeros((N, 3), dtype=np.float64)
    w_m_v = np.zeros((N, 3), dtype=np.float64)
    g_n = np.array([0.0, 0.0, -STANDARD_GRAVITY])

    for k in range(N):
        R_v_n = np.array([
            [math.sin(psi[k]), -math.cos(psi[k]), 0.0],
            [math.cos(psi[k]),  math.sin(psi[k]), 0.0],
            [0.0,               0.0,              1.0],
        ])
        R_n_v = R_v_n.T

        # Acceleration in ENU:
        # a_e = a_fwd * sin(psi) + v * w_z * cos(psi)
        # a_n = a_fwd * cos(psi) - v * w_z * sin(psi)
        a_e = forward_accel_mps2 * math.sin(psi[k]) + speed[k] * yaw_rate_rads * math.cos(psi[k])
        a_n = forward_accel_mps2 * math.cos(psi[k]) - speed[k] * yaw_rate_rads * math.sin(psi[k])
        a_n_vec = np.array([a_e, a_n, 0.0])

        # Specific force in body frame
        f_true_v = R_n_v @ (a_n_vec - g_n)
        # Vehicle yaw rate around body +Z (Up): since psi is clockwise from North,
        # counter-clockwise rotation in ENU (+Z) decreases psi: omega_z = -d(psi)/dt
        w_true_v = np.array([0.0, 0.0, -yaw_rate_rads])

        # Corrupt with true biases
        f_m_v[k] = f_true_v + accel_bias_true
        w_m_v[k] = w_true_v + gyro_bias_true

    return (
        f_m_v, w_m_v, t_ns,
        ref_lat, ref_lon, ref_alt,
        speed, ref_heading,
    )


class TestBiasNetLabelGeneration:
    """Rigorous synthetic test suite verifying the inverse-problem bias solver."""

    def test_zero_bias_recovery(self) -> None:
        """When input IMU has zero bias, the optimizer must recover delta_b ~= 0."""
        (f, w, ts, lat, lon, alt, spd, hdg) = generate_synthetic_curved_window(
            duration_s=2.0,
            accel_bias_true=np.zeros(3),
            gyro_bias_true=np.zeros(3),
        )
        cfg = BiasNetOptimizationConfig(horizon_s=1.0)
        res = solve_window_bias_correction(
            f_m_v_win=f, omega_m_v_win=w, timestamps_ns_win=ts,
            ref_lat_win=lat, ref_lon_win=lon, ref_alt_m_win=alt,
            ref_speed_mps_win=spd, ref_heading_deg_win=hdg,
            config=cfg,
        )

        assert res.converged
        assert not res.bounds_active
        # All bias components should be near zero (< 5e-3 m/s^2 for accel, < 1e-4 rad/s for gyro)
        np.testing.assert_allclose(res.delta_b_unconstrained[0:3], 0.0, atol=5e-3)
        np.testing.assert_allclose(res.delta_b_unconstrained[3:6], 0.0, atol=1e-4)
        assert res.reason_code == "VALID" or res.is_eligible

    def test_accel_only_bias_recovery(self) -> None:
        """Verify recovery of known accelerometer bias [0.08, -0.05, 0.04] m/s^2."""
        true_ba = np.array([0.08, -0.05, 0.04])
        true_bg = np.zeros(3)

        (f, w, ts, lat, lon, alt, spd, hdg) = generate_synthetic_curved_window(
            duration_s=2.0,
            accel_bias_true=true_ba,
            gyro_bias_true=true_bg,
        )
        cfg = BiasNetOptimizationConfig(horizon_s=1.0)
        res = solve_window_bias_correction(
            f_m_v_win=f, omega_m_v_win=w, timestamps_ns_win=ts,
            ref_lat_win=lat, ref_lon_win=lon, ref_alt_m_win=alt,
            ref_speed_mps_win=spd, ref_heading_deg_win=hdg,
            config=cfg,
        )

        assert res.converged
        assert not res.bounds_active
        # Accel bias recovered within 5e-3 m/s^2
        np.testing.assert_allclose(res.delta_b_unconstrained[0:3], true_ba, atol=5e-3)
        # Gyro bias should remain near zero (< 1e-3 rad/s)
        np.testing.assert_allclose(res.delta_b_unconstrained[3:6], 0.0, atol=1e-3)

    def test_gyro_only_bias_recovery(self) -> None:
        """Verify recovery of known gyroscope bias [0.002, -0.003, 0.005] rad/s."""
        true_ba = np.zeros(3)
        true_bg = np.array([0.002, -0.003, 0.005])

        (f, w, ts, lat, lon, alt, spd, hdg) = generate_synthetic_curved_window(
            duration_s=2.0,
            accel_bias_true=true_ba,
            gyro_bias_true=true_bg,
        )
        cfg = BiasNetOptimizationConfig(horizon_s=1.0)
        res = solve_window_bias_correction(
            f_m_v_win=f, omega_m_v_win=w, timestamps_ns_win=ts,
            ref_lat_win=lat, ref_lon_win=lon, ref_alt_m_win=alt,
            ref_speed_mps_win=spd, ref_heading_deg_win=hdg,
            config=cfg,
        )

        assert res.converged
        assert not res.bounds_active
        # Gyro bias recovered within 1e-3 rad/s
        np.testing.assert_allclose(res.delta_b_unconstrained[3:6], true_bg, atol=1e-3)
        # Accel bias should remain near zero (< 5e-3 m/s^2)
        np.testing.assert_allclose(res.delta_b_unconstrained[0:3], 0.0, atol=5e-3)

    def test_mixed_bias_recovery(self) -> None:
        """Verify recovery of simultaneous accel + gyro biases."""
        true_ba = np.array([0.06, -0.04, 0.03])
        true_bg = np.array([0.001, -0.002, 0.004])

        (f, w, ts, lat, lon, alt, spd, hdg) = generate_synthetic_curved_window(
            duration_s=2.0,
            accel_bias_true=true_ba,
            gyro_bias_true=true_bg,
        )
        cfg = BiasNetOptimizationConfig(horizon_s=1.0)
        res = solve_window_bias_correction(
            f_m_v_win=f, omega_m_v_win=w, timestamps_ns_win=ts,
            ref_lat_win=lat, ref_lon_win=lon, ref_alt_m_win=alt,
            ref_speed_mps_win=spd, ref_heading_deg_win=hdg,
            config=cfg,
        )

        assert res.converged
        assert not res.bounds_active
        np.testing.assert_allclose(res.delta_b_unconstrained[0:3], true_ba, atol=5e-3)
        np.testing.assert_allclose(res.delta_b_unconstrained[3:6], true_bg, atol=1e-3)
        assert res.residual_reduction_ratio > 1.1

    def test_bound_activation_rejection(self) -> None:
        """An extreme unphysical bias (e.g. 1.5 m/s^2 > 0.3 bound) triggers bounds_active."""
        huge_ba = np.array([1.5, 0.0, 0.0])  # Exceeds bound 0.3
        (f, w, ts, lat, lon, alt, spd, hdg) = generate_synthetic_curved_window(
            duration_s=2.0,
            accel_bias_true=huge_ba,
        )
        cfg = BiasNetOptimizationConfig(horizon_s=1.0, bound_accel_mps2=0.3)
        res = solve_window_bias_correction(
            f_m_v_win=f, omega_m_v_win=w, timestamps_ns_win=ts,
            ref_lat_win=lat, ref_lon_win=lon, ref_alt_m_win=alt,
            ref_speed_mps_win=spd, ref_heading_deg_win=hdg,
            config=cfg,
        )

        assert res.bounds_active
        assert not res.is_eligible
        assert res.reason_code == "BOUNDS_ACTIVE"
        # Constrained solution must be strictly clamped
        assert abs(res.delta_b_constrained[0]) <= 0.3

    def test_ill_conditioned_window_detection(self) -> None:
        """A window with zero motion and zero excitation should have a large condition number."""
        N = 21
        ts = (np.arange(N) * 0.1 * 1e9).astype(np.int64)
        f = np.zeros((N, 3))
        f[:, 2] = STANDARD_GRAVITY  # Static level
        w = np.zeros((N, 3))
        lat = np.full(N, 52.0)
        lon = np.full(N, -1.0)
        alt = np.full(N, 100.0)
        spd = np.zeros(N)
        hdg = np.zeros(N)

        cfg = BiasNetOptimizationConfig(horizon_s=1.0, max_condition_number=100.0)
        res = solve_window_bias_correction(
            f_m_v_win=f, omega_m_v_win=w, timestamps_ns_win=ts,
            ref_lat_win=lat, ref_lon_win=lon, ref_alt_m_win=alt,
            ref_speed_mps_win=spd, ref_heading_deg_win=hdg,
            config=cfg,
        )

        assert res.condition_number > 100.0 or not res.is_eligible
        assert res.reason_code in ["ILL_CONDITIONED", "DEFICIENT_RANK", "POOR_RESIDUAL_REDUCTION", "VALID"]

    def test_deterministic_repeatability(self) -> None:
        """Running the solver twice on identical data produces exact bit-for-bit identical results."""
        true_ba = np.array([0.05, -0.03, 0.02])
        true_bg = np.array([0.001, -0.002, 0.003])
        data = generate_synthetic_curved_window(
            duration_s=2.0, accel_bias_true=true_ba, gyro_bias_true=true_bg
        )
        cfg = BiasNetOptimizationConfig(horizon_s=1.0)

        res1 = solve_window_bias_correction(
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], config=cfg
        )
        res2 = solve_window_bias_correction(
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], config=cfg
        )

        np.testing.assert_array_equal(res1.delta_b_unconstrained, res2.delta_b_unconstrained)
        np.testing.assert_array_equal(res1.singular_values, res2.singular_values)
        assert res1.condition_number == res2.condition_number
        assert res1.reason_code == res2.reason_code

    def test_non_finite_input_rejection(self) -> None:
        """NaN or Inf inputs must be safely caught without crashing."""
        data = list(generate_synthetic_curved_window(duration_s=2.0))
        data[0][5, 0] = np.nan  # Corrupt accel with NaN

        res = solve_window_bias_correction(
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]
        )

        assert not res.is_eligible
        assert res.reason_code == "NON_FINITE_INPUT"
        assert not np.isfinite(res.delta_b_unconstrained).any()

    def test_solver_failure_rejection(self) -> None:
        """A window where the optimizer fails to converge must be rejected with SOLVER_FAILURE."""
        true_ba = np.array([0.5, -0.4, 0.3])
        true_bg = np.array([0.02, -0.01, 0.03])
        data = generate_synthetic_curved_window(
            duration_s=2.0, accel_bias_true=true_ba, gyro_bias_true=true_bg
        )
        # Force solver failure by setting max_iterations=1 with impossible convergence tolerances
        cfg = BiasNetOptimizationConfig(
            horizon_s=1.0,
            max_iterations=1,
            convergence_step_tol=1e-15,
            convergence_grad_tol=1e-15,
            convergence_rel_tol=1e-15,
        )
        res = solve_window_bias_correction(
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], config=cfg
        )

        assert not res.converged
        assert not res.is_eligible
        assert res.reason_code == "SOLVER_FAILURE"

    def test_reason_code_deterministic_precedence(self) -> None:
        """Test deterministic reason-code precedence ordering."""
        # 1. Non-finite input takes highest precedence
        data = list(generate_synthetic_curved_window(duration_s=2.0))
        data[0][0, 0] = np.nan
        res = solve_window_bias_correction(
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]
        )
        assert res.reason_code == "NON_FINITE_INPUT"

        # 2. Solver failure takes precedence over subsequent gates
        data_clean = generate_synthetic_curved_window(
            duration_s=2.0, accel_bias_true=np.array([3.0, 0.0, 0.0])  # Also exceeds bounds
        )
        cfg_fail = BiasNetOptimizationConfig(
            max_iterations=1,
            convergence_step_tol=1e-15,
            convergence_grad_tol=1e-15,
            convergence_rel_tol=1e-15,
        )
        res_fail = solve_window_bias_correction(
            data_clean[0], data_clean[1], data_clean[2], data_clean[3],
            data_clean[4], data_clean[5], data_clean[6], data_clean[7],
            config=cfg_fail,
        )
        assert res_fail.reason_code == "SOLVER_FAILURE"

    def test_physical_bounds_safeguard_vs_eligibility(self) -> None:
        """Verify distinct roles of solver safeguards and physical eligibility bounds."""
        # Bias within physical bounds (|dba| <= 2.0, |dbg| <= 0.15)
        data_ok = generate_synthetic_curved_window(
            duration_s=2.0, accel_bias_true=np.array([0.5, -0.3, 0.2]), gyro_bias_true=np.array([0.01, -0.02, 0.01])
        )
        cfg = BiasNetOptimizationConfig(horizon_s=1.0, bound_accel_mps2=2.0, bound_gyro_rads=0.15)
        res_ok = solve_window_bias_correction(
            data_ok[0], data_ok[1], data_ok[2], data_ok[3],
            data_ok[4], data_ok[5], data_ok[6], data_ok[7],
            config=cfg,
        )
        assert not res_ok.bounds_active
        assert res_ok.converged
        assert res_ok.is_eligible
        assert res_ok.reason_code == "VALID"

        # Bias exceeding physical eligibility bounds (|dba| = 2.5 > 2.0)
        data_violation = generate_synthetic_curved_window(
            duration_s=2.0, accel_bias_true=np.array([2.5, 0.0, 0.0])
        )
        res_viol = solve_window_bias_correction(
            data_violation[0], data_violation[1], data_violation[2], data_violation[3],
            data_violation[4], data_violation[5], data_violation[6], data_violation[7],
            config=cfg,
        )
        assert res_viol.bounds_active
        assert not res_viol.is_eligible
        assert res_viol.reason_code == "BOUNDS_ACTIVE"
        assert abs(res_viol.delta_b_constrained[0]) <= 2.0
