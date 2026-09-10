"""Explicit NHC Frame Consistency & Mounting Verification Tests (Phase 11 Step 3).

Verifies:
1. When true vehicle frame is known and device mounting rotation R_b^v is known,
   a vehicle moving purely forward produces:
       v^v = [forward_speed, approx 0, approx 0]
   before NHC is applied.
2. When a known yaw mounting error is deliberately injected, the resulting v^v
   contains substantial false lateral velocity v_y^v != 0.
3. NHC diagnostics detect this frame inconsistency (via elevated NIS, RELAXED or
   SKIPPED status) rather than falsely asserting a clean zero-lateral-velocity update.
4. Fails if the code silently assumes the wrong body frame.
"""

import math
import numpy as np
import pytest

from navigation.eskf.state import ESKFNominalState, ESKFState
from navigation.ins.attitude import rotation_matrix_to_quaternion
from navigation.nhc.measurement import NHCConfig, NHCMeasurementModel
from navigation.nhc.skid_detection import NHCStatus, SkidDetectorConfig


def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Z-Y-X Euler angles to 3x3 rotation matrix: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return Rz @ Ry @ Rx


def create_synthetic_state(
    v_forward_mps: float,
    attitude_euler_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    mounting_yaw_error_deg: float = 0.0,
) -> tuple[ESKFState, float]:
    """Construct an authoritative ESKF state with pure forward vehicle motion and optional mounting error.

    Args:
        v_forward_mps: True vehicle forward speed [m/s].
        attitude_euler_deg: (roll, pitch, yaw) of vehicle frame in ENU [deg].
        mounting_yaw_error_deg: Mounting yaw error between device and true vehicle frame [deg].

    Returns:
        Tuple of (ESKFState, true_speed).
    """
    roll, pitch, yaw = [math.radians(a) for a in attitude_euler_deg]
    R_v_n = euler_to_rotation_matrix(roll, pitch, yaw)

    # True vehicle-frame velocity is purely forward
    v_vehicle_true = np.array([v_forward_mps, 0.0, 0.0], dtype=np.float64)

    # Velocity in navigation frame ENU
    v_enu = R_v_n @ v_vehicle_true

    # If mounting yaw error exists, the estimated vehicle frame R_est differs by mounting yaw
    psi_err = math.radians(mounting_yaw_error_deg)
    R_mount_err = np.array([
        [math.cos(psi_err), -math.sin(psi_err), 0.0],
        [math.sin(psi_err),  math.cos(psi_err), 0.0],
        [0.0,                0.0,               1.0],
    ], dtype=np.float64)

    # Estimated R_v_n in the presence of mounting error
    R_v_n_est = R_v_n @ R_mount_err
    q_est = rotation_matrix_to_quaternion(R_v_n_est)

    nom = ESKFNominalState(
        position_enu=np.zeros(3, dtype=np.float64),
        velocity_enu=v_enu,
        q=q_est,
        accel_bias=np.zeros(3, dtype=np.float64),
        gyro_bias=np.zeros(3, dtype=np.float64),
        timestamp_ns=1_000_000_000,
    )

    # Realistic covariance
    P = np.eye(15, dtype=np.float64) * 1e-4
    P[3:6, 3:6] = np.diag([0.05, 0.05, 0.01])
    P[6:9, 6:9] = np.diag([1e-4, 1e-4, 1e-3])

    return ESKFState(nominal=nom, covariance=P), v_forward_mps


def test_consistent_frame_pure_forward_motion():
    """Verify v^v = [forward_speed, approx 0, approx 0] when frame is consistent."""
    speed = 18.5  # m/s
    # Arbitrary 3D attitude in ENU
    state, true_spd = create_synthetic_state(
        v_forward_mps=speed,
        attitude_euler_deg=(5.0, -3.0, 42.0),
        mounting_yaw_error_deg=0.0,
    )

    nhc = NHCMeasurementModel(NHCConfig(enabled=True))
    z, h_val, v_v, H, R_base = nhc.create_measurement(state)

    # 1. Forward velocity must match true speed
    assert np.isclose(v_v[0], true_spd, atol=1e-5), f"Expected vx={true_spd}, got {v_v[0]}"
    # 2. Lateral and vertical velocity must be identically zero
    assert np.isclose(v_v[1], 0.0, atol=1e-5), f"Expected vy=0, got {v_v[1]}"
    assert np.isclose(v_v[2], 0.0, atol=1e-5), f"Expected vz=0, got {v_v[2]}"

    # 3. NHC innovation must be zero
    innov = z - h_val
    assert np.allclose(innov, 0.0, atol=1e-5)

    # 4. Update must report ACCEPTED_NORMAL with nis ~ 0
    updated_state, diag = nhc.update(state)
    assert diag.status == NHCStatus.NORMAL
    assert diag.applied is True
    assert diag.nis < 1e-5


def test_frame_inconsistency_mounting_yaw_error_detected():
    """Verify deliberate yaw mounting error induces lateral velocity and triggers inconsistency."""
    speed = 15.0  # m/s
    mounting_yaw_err_deg = 15.0  # 15 degrees mounting misalignment

    state, true_spd = create_synthetic_state(
        v_forward_mps=speed,
        attitude_euler_deg=(0.0, 0.0, 30.0),
        mounting_yaw_error_deg=mounting_yaw_err_deg,
    )

    nhc = NHCMeasurementModel(NHCConfig(enabled=True))
    z, h_val, v_v, H, R_base = nhc.create_measurement(state)

    # Because of the 15 deg mounting error:
    # v_x^v = speed * cos(15 deg) ≈ 14.49 m/s
    # v_y^v = -speed * sin(15 deg) ≈ -3.88 m/s
    expected_vy = -speed * math.sin(math.radians(mounting_yaw_err_deg))
    assert np.isclose(v_v[1], expected_vy, atol=1e-3), f"Expected vy={expected_vy}, got {v_v[1]}"

    # The resulting lateral innovation is large: |y_y| ≈ 3.88 m/s
    innov = z - h_val
    assert abs(innov[0]) > 3.0

    # The Mahalanobis NIS must reflect severe inconsistency (well above severe_gate_threshold 16.0)
    updated_state, diag = nhc.update(state)
    assert diag.nis > 16.0, f"Expected NIS > 16.0 for 15-deg mounting error, got {diag.nis}"

    # Must be SKIPPED with severe innovation inconsistency reason
    assert diag.status == NHCStatus.SKIPPED
    assert diag.applied is False
    assert "INNOVATION" in diag.reason or "SKIPPED" in diag.reason


def test_frame_inconsistency_moderate_mounting_yaw_error_relaxes():
    """Verify moderate mounting error (e.g. 8 degrees) triggers RELAXED covariance rather than blind confidence."""
    speed = 12.0  # m/s
    mounting_yaw_err_deg = 8.0  # 8 degrees mounting error (between normal gate and severe skip)

    state, true_spd = create_synthetic_state(
        v_forward_mps=speed,
        attitude_euler_deg=(0.0, 0.0, 15.0),
        mounting_yaw_error_deg=mounting_yaw_err_deg,
    )

    nhc = NHCMeasurementModel(NHCConfig(enabled=True))
    updated_state, diag = nhc.update(state)

    # Lateral velocity is ~12 * sin(8 deg) ≈ 1.67 m/s
    # NIS d^2 is elevated into the relaxed tier (9.210 < d^2 <= 16.0) or skipped
    assert diag.nis > 9.210  # Exceeds normal 99% chi2 acceptance gate
    if diag.applied:
        assert diag.status == NHCStatus.RELAXED
        assert diag.covariance_inflation > 1.0
        assert "RELAXED" in diag.reason
    else:
        assert diag.status == NHCStatus.SKIPPED
