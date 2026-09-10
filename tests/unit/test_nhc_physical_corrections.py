"""Explicit Physical Correction Direction Tests for NHC (Phase 11 Step 4).

Tests Cases A through F:
- Case A: Zero lateral velocity -> zero correction.
- Case B: Positive lateral velocity (v_y^v > 0) -> update reduces v_y^v towards 0.
- Case C: Negative lateral velocity (v_y^v < 0) -> update increases v_y^v towards 0.
- Case D: Positive vertical velocity (v_z^v > 0) -> update reduces v_z^v towards 0.
- Case E: Non-identity 3D attitude -> update operates consistently in vehicle FLU frame.
- Case F: Known yaw mounting error -> surfaces inconsistency rather than silent failure.
"""

import math
import numpy as np
import pytest

from navigation.eskf.state import ESKFNominalState, ESKFState
from navigation.ins.attitude import rotation_matrix_to_quaternion
from navigation.nhc.measurement import NHCConfig, NHCMeasurementModel
from navigation.nhc.skid_detection import NHCStatus


def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Z-Y-X Euler angles to 3x3 rotation matrix: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return Rz @ Ry @ Rx


def make_state_with_body_velocity(
    v_body: np.ndarray,
    attitude_euler_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    sigma_v: float = 0.5,
    sigma_theta: float = 0.05,
) -> ESKFState:
    """Helper to construct ESKF state with specified vehicle-frame velocity."""
    roll, pitch, yaw = [math.radians(a) for a in attitude_euler_deg]
    R_v_n = euler_to_rotation_matrix(roll, pitch, yaw)
    v_v = np.asarray(v_body, dtype=np.float64).reshape(3)
    v_enu = R_v_n @ v_v

    q = rotation_matrix_to_quaternion(R_v_n)
    nom = ESKFNominalState(
        position_enu=np.zeros(3, dtype=np.float64),
        velocity_enu=v_enu,
        q=q,
        accel_bias=np.zeros(3, dtype=np.float64),
        gyro_bias=np.zeros(3, dtype=np.float64),
        timestamp_ns=1_000_000_000,
    )

    P = np.eye(15, dtype=np.float64) * 1e-4
    P[3:6, 3:6] = np.diag([sigma_v**2, sigma_v**2, sigma_v**2])
    P[6:9, 6:9] = np.diag([sigma_theta**2, sigma_theta**2, sigma_theta**2])

    return ESKFState(nominal=nom, covariance=P)


def test_case_a_zero_lateral_velocity():
    """Case A: Zero lateral and vertical velocity -> zero correction, forward speed preserved."""
    state = make_state_with_body_velocity([12.0, 0.0, 0.0])
    nhc = NHCMeasurementModel(NHCConfig(enabled=True))

    up_state, diag = nhc.update(state)
    assert diag.status == NHCStatus.NORMAL
    assert diag.applied is True

    R_v_n = up_state.nominal.R_v_n
    v_v_after = R_v_n.T @ up_state.nominal.velocity_enu

    assert np.isclose(v_v_after[0], 12.0, atol=1e-5)
    assert np.isclose(v_v_after[1], 0.0, atol=1e-5)
    assert np.isclose(v_v_after[2], 0.0, atol=1e-5)


def test_case_b_positive_lateral_velocity():
    """Case B: Positive lateral velocity (v_y^v = +0.20 m/s) -> correction pushes v_y^v negative towards 0."""
    v_pre = np.array([12.0, 0.20, 0.0])
    state = make_state_with_body_velocity(v_pre)
    nhc = NHCMeasurementModel(NHCConfig(enabled=True))

    up_state, diag = nhc.update(state)
    assert diag.applied is True

    R_v_n = up_state.nominal.R_v_n
    v_v_after = R_v_n.T @ up_state.nominal.velocity_enu

    # v_y^v must have moved towards zero (decreased)
    assert v_v_after[1] < v_pre[1], f"Expected vy < {v_pre[1]}, got {v_v_after[1]}"
    assert abs(v_v_after[1]) < abs(v_pre[1]), f"Expected |vy| < {v_pre[1]}, got {abs(v_v_after[1])}"
    # Forward velocity must be preserved
    assert np.isclose(v_v_after[0], v_pre[0], atol=1e-4)


def test_case_c_negative_lateral_velocity():
    """Case C: Negative lateral velocity (v_y^v = -0.20 m/s) -> correction pushes v_y^v positive towards 0."""
    v_pre = np.array([12.0, -0.20, 0.0])
    state = make_state_with_body_velocity(v_pre)
    nhc = NHCMeasurementModel(NHCConfig(enabled=True))

    up_state, diag = nhc.update(state)
    assert diag.applied is True

    R_v_n = up_state.nominal.R_v_n
    v_v_after = R_v_n.T @ up_state.nominal.velocity_enu

    # v_y^v must have moved towards zero (increased)
    assert v_v_after[1] > v_pre[1], f"Expected vy > {v_pre[1]}, got {v_v_after[1]}"
    assert abs(v_v_after[1]) < abs(v_pre[1]), f"Expected |vy| < {v_pre[1]}, got {abs(v_v_after[1])}"
    # Forward velocity must be preserved
    assert np.isclose(v_v_after[0], v_pre[0], atol=1e-4)


def test_case_d_positive_vertical_velocity():
    """Case D: Positive vertical velocity (v_z^v = +0.15 m/s) -> correction pushes v_z^v negative towards 0."""
    v_pre = np.array([10.0, 0.0, 0.15])
    state = make_state_with_body_velocity(v_pre)
    nhc = NHCMeasurementModel(NHCConfig(enabled=True))

    up_state, diag = nhc.update(state)
    assert diag.applied is True

    R_v_n = up_state.nominal.R_v_n
    v_v_after = R_v_n.T @ up_state.nominal.velocity_enu

    # v_z^v must have moved towards zero (decreased)
    assert v_v_after[2] < v_pre[2], f"Expected vz < {v_pre[2]}, got {v_v_after[2]}"
    assert abs(v_v_after[2]) < abs(v_pre[2]), f"Expected |vz| < {v_pre[2]}, got {abs(v_v_after[2])}"
    # Forward velocity must be preserved
    assert np.isclose(v_v_after[0], v_pre[0], atol=1e-4)


def test_case_e_non_identity_attitude():
    """Case E: Arbitrary 3D attitude -> update operates correctly in vehicle frame regardless of orientation."""
    v_pre = np.array([14.0, 0.18, -0.10])
    attitude_deg = (12.0, -8.0, 135.0)  # Arbitrary roll, pitch, yaw
    state = make_state_with_body_velocity(v_pre, attitude_euler_deg=attitude_deg)
    nhc = NHCMeasurementModel(NHCConfig(enabled=True))

    up_state, diag = nhc.update(state)
    assert diag.applied is True

    R_v_n = up_state.nominal.R_v_n
    v_v_after = R_v_n.T @ up_state.nominal.velocity_enu

    # Both lateral and vertical errors must be reduced towards 0 in vehicle frame
    assert abs(v_v_after[1]) < abs(v_pre[1])
    assert abs(v_v_after[2]) < abs(v_pre[2])
    assert np.isclose(v_v_after[0], v_pre[0], atol=1e-4)


def test_case_f_known_yaw_mounting_error():
    """Case F: Known yaw mounting error -> surfaces inconsistency and skips rather than corrupting."""
    # When mounting yaw is rotated by 30 degrees, forward motion at 15 m/s
    # produces vy^v = -15 * sin(30 deg) = -7.5 m/s in the misaligned frame
    v_pre = np.array([15.0 * math.cos(math.radians(30)), -15.0 * math.sin(math.radians(30)), 0.0])
    state = make_state_with_body_velocity(v_pre)
    nhc = NHCMeasurementModel(NHCConfig(enabled=True))

    up_state, diag = nhc.update(state)
    # Must detect severe innovation inconsistency and skip
    assert diag.status == NHCStatus.SKIPPED
    assert diag.applied is False
    assert diag.nis > 16.0
