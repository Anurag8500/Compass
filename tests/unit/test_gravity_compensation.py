"""Unit tests for Phase 3 gravity resolution and offline bootstrap attitude estimation."""

from __future__ import annotations

import math
import numpy as np
import pytest

from navigation.preprocessing.bootstrap_attitude import (
    BootstrapAttitudeEstimator,
    quaternion_from_axis_angle,
    quaternion_multiply,
    quaternion_to_euler_rad,
    quaternion_to_rotation_matrix,
)
from navigation.preprocessing.gravity import (
    STANDARD_GRAVITY_MPS2,
    resolve_gravity,
    verify_stationary_gravity_resolution,
)


class TestGravityResolution:
    """Test suite for mathematical gravity-resolution in ENU navigation frame."""

    def test_canonical_stationary_level_test_zero_acceleration(self) -> None:
        """Canonical stationary test: R_v^n = I, b_a^v = 0, f_m^v = [0, 0, +g]^T yields a^n = [0, 0, 0]^T identically."""
        g = STANDARD_GRAVITY_MPS2
        R_v_n = np.eye(3)
        b_a_v = np.zeros(3)
        f_m_v = np.array([0.0, 0.0, g])

        a_n = resolve_gravity(f_m_v, R_v_n, b_a_v=b_a_v, g_val=g)

        assert np.allclose(a_n, np.zeros(3), atol=1e-12), f"Expected [0, 0, 0], got {a_n}"
        assert verify_stationary_gravity_resolution(f_m_v) is True

    def test_rotated_stationary_vehicle_yields_zero_acceleration(self) -> None:
        """Stationary vehicle on an incline (30 deg pitch, 15 deg roll).
        
        True kinematic acceleration is zero. Measured specific force in vehicle frame is
        f_m^v = (R_v^n)^T * (-g^n) = (R_v^n)^T * [0, 0, +g]^T.
        Resolving gravity must yield a^n = [0, 0, 0]^T.
        """
        g = STANDARD_GRAVITY_MPS2
        # Arbitrary rotation matrix
        axis = np.array([1.0, 2.0, -1.0])
        axis /= np.linalg.norm(axis)
        angle = math.radians(35.0)
        q = quaternion_from_axis_angle(axis, angle)
        R_v_n = quaternion_to_rotation_matrix(q)

        # Support reaction in vehicle frame
        f_m_v = R_v_n.T @ np.array([0.0, 0.0, g])

        a_n = resolve_gravity(f_m_v, R_v_n, b_a_v=None, g_val=g)
        assert np.allclose(a_n, np.zeros(3), atol=1e-12)

    def test_dynamic_acceleration_recovery(self) -> None:
        """Inject known true vehicle acceleration a_true^n, synthesize f_m^v, and verify recovery."""
        g = STANDARD_GRAVITY_MPS2
        a_true_n = np.array([2.5, -1.2, 0.4])  # True vehicle coordinate acceleration in ENU

        # Known vehicle orientation
        q = quaternion_from_axis_angle(np.array([0.0, 0.0, 1.0]), math.radians(60.0))
        R_v_n = quaternion_to_rotation_matrix(q)
        b_a_v = np.array([0.05, -0.02, 0.01])  # Accelerometer bias

        # Physical measurement in vehicle frame: f_m^v = (R_v^n)^T (a_true^n - g^n) + b_a^v
        g_n = np.array([0.0, 0.0, -g])
        f_m_v = (R_v_n.T @ (a_true_n - g_n)) + b_a_v

        # Resolve coordinate acceleration
        a_recovered_n = resolve_gravity(f_m_v, R_v_n, b_a_v=b_a_v, g_val=g)

        assert np.allclose(a_recovered_n, a_true_n, atol=1e-12)

    def test_batch_time_series_resolution(self) -> None:
        """Verify batch evaluation over N samples with constant and time-varying orientation."""
        g = STANDARD_GRAVITY_MPS2
        n = 50
        R_v_n = np.eye(3)
        # 50 stationary samples
        f_m_v = np.tile([0.0, 0.0, g], (n, 1))

        a_n_batch = resolve_gravity(f_m_v, R_v_n, g_val=g)
        assert a_n_batch.shape == (n, 3)
        assert np.allclose(a_n_batch, np.zeros((n, 3)), atol=1e-12)


class TestBootstrapAttitudeEstimator:
    """Test suite for the offline validation complementary attitude estimator."""

    def test_identity_quaternion_stationary_hold(self) -> None:
        """Stationary sensor with zero angular rate should maintain identity attitude."""
        est = BootstrapAttitudeEstimator(initial_q=(1.0, 0.0, 0.0, 0.0))
        accel = np.array([0.0, 0.0, 9.80665])
        gyro = np.array([0.0, 0.0, 0.0])

        for _ in range(50):
            est.update(accel, gyro, dt_s=0.1)

        q = est.get_quaternion()
        assert q[0] == pytest.approx(1.0, abs=1e-6)
        assert q[1] == pytest.approx(0.0, abs=1e-6)
        assert q[2] == pytest.approx(0.0, abs=1e-6)
        assert q[3] == pytest.approx(0.0, abs=1e-6)

    def test_pure_gyro_integration_90_deg_yaw_turn(self) -> None:
        """Turn at constant 90 deg/s for 1.0 second around Z axis: final yaw must be 90 deg."""
        est = BootstrapAttitudeEstimator(kp_tilt=0.0)  # Disable tilt correction to isolate gyro math
        gyro = np.array([0.0, 0.0, math.radians(90.0)])
        accel = np.array([0.0, 0.0, 9.80665])
        dt = 0.01  # 100 Hz integration

        for _ in range(100):
            est.update(accel, gyro, dt_s=dt)

        roll_deg, pitch_deg, yaw_deg = est.get_euler_deg()
        assert yaw_deg == pytest.approx(90.0, abs=0.5)
        assert roll_deg == pytest.approx(0.0, abs=0.1)
        assert pitch_deg == pytest.approx(0.0, abs=0.1)

    def test_quaternion_math_properties(self) -> None:
        """Verify quaternion multiplication, normalization, and axis-angle generation."""
        q_id = np.array([1.0, 0.0, 0.0, 0.0])
        q_rot = quaternion_from_axis_angle(np.array([1.0, 0.0, 0.0]), math.pi / 2.0)

        # Multiplying by identity leaves quaternion unchanged
        res = quaternion_multiply(q_id, q_rot)
        assert np.allclose(res, q_rot, atol=1e-12)

        # Inverse rotation: q * q_inv = identity
        q_inv = np.array([q_rot[0], -q_rot[1], -q_rot[2], -q_rot[3]])
        prod = quaternion_multiply(q_rot, q_inv)
        assert np.allclose(prod, q_id, atol=1e-12)
