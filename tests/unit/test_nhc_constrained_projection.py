"""Comprehensive Mathematical and Algorithmic Tests for Simon-Chia Constrained NHC Projection (Phase 11).

Verifies:
1. Mathematical consistency of forward-speed constraint C @ delta_x = 0.
2. Idempotence of projection operator M^2 = M and nullspace orthogonality C @ M = 0.
3. Strict symmetry and positive-definiteness (PSD) of posterior covariance P_new.
4. Absence of artificial forward deceleration during lateral/vertical velocity corrections.
5. Invariant preservation when NHC is disabled.
6. Deliberate mounting frame error detection and gating response.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from navigation.eskf.state import ESKFNominalState, ESKFState
from navigation.ins.attitude import quaternion_to_rotation_matrix, rotation_matrix_to_quaternion
from navigation.nhc.measurement import NHCConfig, NHCMeasurementModel, NHCStatus


def euler_to_quaternion(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Helper to construct unit quaternion from Z-Y-X Euler angles in degrees."""
    r, p, y = math.radians(roll_deg), math.radians(pitch_deg), math.radians(yaw_deg)
    Rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]], dtype=np.float64)
    Ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]], dtype=np.float64)
    Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]], dtype=np.float64)
    R = Rz @ Ry @ Rx
    return rotation_matrix_to_quaternion(R)


def make_test_state(
    v_body: np.ndarray,
    attitude_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    pos_enu: np.ndarray | None = None,
) -> ESKFState:
    """Create controlled ESKF state with specified vehicle-frame velocity."""
    q = euler_to_quaternion(*attitude_deg)
    R_v_n = quaternion_to_rotation_matrix(q)
    v_enu = R_v_n @ np.asarray(v_body, dtype=np.float64)
    p_enu = np.zeros(3) if pos_enu is None else np.asarray(pos_enu, dtype=np.float64)

    # Well-conditioned positive-definite covariance
    np.random.seed(123)
    A = np.random.randn(15, 15) * 0.1
    P = A @ A.T + np.diag([
        1.0, 1.0, 2.0,       # pos
        0.5, 0.5, 0.2,       # vel
        0.01, 0.01, 0.05,    # att
        1e-4, 1e-4, 1e-4,    # ba
        1e-6, 1e-6, 1e-6,    # bg
    ])
    P = 0.5 * (P + P.T)

    nom = ESKFNominalState(
        position_enu=p_enu,
        velocity_enu=v_enu,
        q=q,
        accel_bias=np.zeros(3),
        gyro_bias=np.zeros(3),
        timestamp_ns=1_000_000_000,
    )
    return ESKFState(nominal=nom, covariance=P)


def test_forward_speed_preservation_mathematical_consistency():
    """Verify that C @ delta_x = 0 identically for arbitrary 3D attitudes and velocities."""
    for att in [(0.0, 0.0, 0.0), (15.0, -10.0, 45.0), (-5.0, 20.0, -120.0)]:
        v_body = np.array([14.5, 0.25, -0.15])
        state = make_test_state(v_body, attitude_deg=att)
        nhc = NHCMeasurementModel(NHCConfig(enabled=True, preserve_forward_speed=True))

        up_state, diag = nhc.update(state)
        assert diag.applied is True

        # Pre-update vehicle forward unit vector in ENU
        ex_n = state.nominal.R_v_n[:, 0]
        delta_v_enu = diag.update_diagnostics.delta_x[3:6]

        # The mathematical along-track velocity correction MUST be exactly zero
        delta_vx_body = float(ex_n @ delta_v_enu)
        assert abs(delta_vx_body) < 1e-12, f"Along-track correction non-zero: {delta_vx_body}"


def test_projection_operator_properties():
    """Verify idempotence M^2 = M and nullspace C @ M = 0 directly."""
    att = (10.0, 5.0, -30.0)
    state = make_test_state(np.array([12.0, 0.1, 0.0]), attitude_deg=att)
    P = state.covariance
    ex_n = state.nominal.R_v_n[:, 0]

    C = np.zeros((1, 15), dtype=np.float64)
    C[0, 3:6] = ex_n

    u = P @ C.T
    Cu = float((C @ u)[0, 0])
    M = np.eye(15, dtype=np.float64) - (u @ C) / Cu

    # 1. Nullspace constraint
    assert np.allclose(C @ M, 0.0, atol=1e-14), "C @ M != 0"
    # 2. Idempotence: M @ M = M
    assert np.allclose(M @ M, M, atol=1e-14), "M is not idempotent"
    # 3. Orthogonal projection of u: M @ u = 0
    assert np.allclose(M @ u, 0.0, atol=1e-14), "M @ u != 0"


def test_covariance_consistency_after_constrained_update():
    """Verify posterior covariance symmetry, PSD property, and conditioning."""
    v_body = np.array([15.0, 0.2, 0.1])
    state = make_test_state(v_body, attitude_deg=(5.0, -3.0, 60.0))
    nhc = NHCMeasurementModel(NHCConfig(enabled=True, preserve_forward_speed=True))

    up_state, diag = nhc.update(state)
    assert diag.applied is True

    P_post = up_state.covariance
    # Symmetry
    assert np.allclose(P_post, P_post.T, atol=1e-12), "Posterior covariance not symmetric"

    # Strictly positive eigenvalues
    eigvals = np.linalg.eigvalsh(P_post)
    min_eig = float(np.min(eigvals))
    assert min_eig > 0.0, f"Covariance not positive definite, min eigenvalue = {min_eig}"


def test_nhc_disabled_regression():
    """Verify that disabled NHC leaves state and covariance completely unmodified."""
    v_body = np.array([10.0, 0.5, -0.3])
    state = make_test_state(v_body)
    nhc = NHCMeasurementModel(NHCConfig(enabled=False))

    up_state, diag = nhc.update(state)
    assert diag.applied is False
    assert diag.status == NHCStatus.NOT_ATTEMPTED
    assert np.array_equal(up_state.nominal.position_enu, state.nominal.position_enu)
    assert np.array_equal(up_state.nominal.velocity_enu, state.nominal.velocity_enu)
    assert np.array_equal(up_state.covariance, state.covariance)


def test_deliberate_mounting_yaw_error_detection():
    """Verify that severe mounting yaw error induces elevated NIS and is safely gated."""
    # 30 degree mounting yaw error on 15 m/s motion -> vy^v = -7.5 m/s
    v_body = np.array([15.0 * math.cos(math.radians(30)), -15.0 * math.sin(math.radians(30)), 0.0])
    state = make_test_state(v_body)
    # Realistic operational attitude uncertainty (1 deg heading std)
    state.covariance[6:9, 6:9] = np.diag([1e-4, 1e-4, (math.radians(1.0)) ** 2])
    nhc = NHCMeasurementModel(NHCConfig(enabled=True))

    up_state, diag = nhc.update(state)
    assert diag.applied is False
    assert diag.status == NHCStatus.SKIPPED
    assert diag.nis > 16.0
    # State remains unmodified
    assert np.array_equal(up_state.nominal.velocity_enu, state.nominal.velocity_enu)
