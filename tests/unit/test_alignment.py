"""Unit tests for Phase 3 device-to-vehicle mounting alignment."""

from __future__ import annotations

import math
import numpy as np
import pytest

from navigation.preprocessing.alignment import (
    MountingAlignment,
    estimate_mounting_alignment,
    rotation_matrix_from_vectors,
    rotation_matrix_to_euler_deg,
)
from navigation.preprocessing.calibration import CalibrationProfile
from navigation.schemas.imu import RawIMUSample


class TestMountingAlignment:
    """Test suite for mounting orientation estimation, rotation math, and frame conversions."""

    def test_identity_alignment_for_level_facing_up_phone(self) -> None:
        """Level phone with screen facing up has support reaction along +Z; R_b^v must be Identity."""
        n = 50
        # Phone at rest facing up: f = [0, 0, 9.80665]
        accel_stat = np.tile([0.0, 0.0, 9.80665], (n, 1))

        alignment = estimate_mounting_alignment(stationary_accel=accel_stat)

        assert np.allclose(alignment.R_b_v, np.eye(3), atol=1e-6)
        assert alignment.pitch_deg == pytest.approx(0.0, abs=1e-4)
        assert alignment.roll_deg == pytest.approx(0.0, abs=1e-4)
        assert alignment.yaw_deg == pytest.approx(0.0, abs=1e-4)
        assert alignment.is_yaw_aligned is False

    def test_90_degree_pitch_rotation(self) -> None:
        """Phone mounted vertically on dashboard (screen facing backward, +Y pointing up).

        Gravity pulls down, so support reaction acts along +Y in body frame: f = [0, 9.81, 0].
        R_b^v must rotate +Y onto +Z.
        """
        accel_stat = np.tile([0.0, 9.80665, 0.0], (40, 1))
        alignment = estimate_mounting_alignment(stationary_accel=accel_stat)

        # Transform the measured specific force to vehicle frame: must point along +Z
        f_v = alignment.transform_specific_force(np.array([0.0, 9.80665, 0.0]))
        assert f_v[0] == pytest.approx(0.0, abs=1e-4)
        assert f_v[1] == pytest.approx(0.0, abs=1e-4)
        assert f_v[2] == pytest.approx(9.80665, abs=1e-4)

    def test_180_degree_inversion(self) -> None:
        """Phone mounted upside down (+Z pointing down): f = [0, 0, -9.81].

        R_b^v must invert the Z axis so that support reaction in vehicle frame is +Z = +9.81.
        """
        accel_stat = np.tile([0.0, 0.0, -9.80665], (40, 1))
        alignment = estimate_mounting_alignment(stationary_accel=accel_stat)

        f_v = alignment.transform_specific_force(np.array([0.0, 0.0, -9.80665]))
        assert f_v[2] == pytest.approx(9.80665, abs=1e-4)

    def test_rotation_matrix_from_vectors_properties(self) -> None:
        """Test mathematical properties of minimum-angle rotation matrix: orthogonality and determinant +1."""
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([-2.0, 1.0, 2.0])

        R = rotation_matrix_from_vectors(v1, v2)

        # Must be orthogonal: R @ R.T = I
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
        # Determinant must be +1 (proper rotation, right-handed)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-10)
        # Rotates v1 direction to v2 direction
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)
        assert np.allclose(R @ v1_norm, v2_norm, atol=1e-10)

    def test_vector_round_trip_and_orthogonality(self) -> None:
        """Verify vector transformation round trip: v_b -> v_v -> v_b recovers original vector within 1e-12."""
        # Arbitrary rotation with roll=15 deg, pitch=-10 deg, yaw=35 deg
        r, p, y = math.radians(15.0), math.radians(-10.0), math.radians(35.0)
        Rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
        Ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
        Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx

        alignment = MountingAlignment(R_b_v=R, roll_deg=15.0, pitch_deg=-10.0, yaw_deg=35.0, is_yaw_aligned=True)

        v_b = np.array([1.5, -2.3, 9.8])
        v_v = alignment.transform_specific_force(v_b)
        v_b_recovered = alignment.R_b_v.T @ v_v

        assert np.allclose(v_b_recovered, v_b, atol=1e-12)
        assert np.allclose(alignment.R_b_v @ alignment.R_b_v.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(alignment.R_b_v) == pytest.approx(1.0, abs=1e-12)

    def test_euler_angle_extraction_consistency(self) -> None:
        """Verify that rotation_matrix_to_euler_deg recovers known Euler angles."""
        for r_in, p_in, y_in in [(0.0, 0.0, 0.0), (10.0, -15.0, 30.0), (-25.0, 20.0, -45.0)]:
            r, p, y = math.radians(r_in), math.radians(p_in), math.radians(y_in)
            Rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
            Ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
            Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
            R = Rz @ Ry @ Rx

            r_out, p_out, y_out = rotation_matrix_to_euler_deg(R)
            assert r_out == pytest.approx(r_in, abs=1e-4)
            assert p_out == pytest.approx(p_in, abs=1e-4)
            assert y_out == pytest.approx(y_in, abs=1e-4)

    def test_known_mounting_yaw_recovery_via_acceleration_correlation(self) -> None:
        """Synthesize known mounting yaw (+30 deg and -45 deg) and verify recovery within 0.5 deg."""
        for known_yaw_deg in [30.0, -45.0]:
            accel_stat = np.tile([0.0, 0.0, 9.80665], (50, 1))

            n_mov = 60
            ts = np.arange(0, n_mov * 100_000_000, 100_000_000, dtype=np.int64)

            # Vehicle accelerates straight forward at a_x = 1.2 m/s^2 from 5 m/s to ~12 m/s
            a_x_true = 1.2
            speeds = 5.0 + a_x_true * (np.arange(n_mov) * 0.1)
            bearings = np.full(n_mov, 90.0)  # Constant course east
            gyros_veh = np.zeros((n_mov, 3))  # Straight line: zero angular rate

            # True vehicle frame specific force: [a_x, 0, +g]
            f_veh = np.tile([a_x_true, 0.0, 9.80665], (n_mov, 1))

            # Phone is mounted with yaw = known_yaw_deg (counter-clockwise about Z)
            # R_b^v = R_z(known_yaw)
            yaw_rad = math.radians(known_yaw_deg)
            R_true = np.array([
                [math.cos(yaw_rad), -math.sin(yaw_rad), 0.0],
                [math.sin(yaw_rad), math.cos(yaw_rad), 0.0],
                [0.0, 0.0, 1.0],
            ])

            # Measured in body frame: f_b = (R_b^v)^T f_v
            accel_b = (R_true.T @ f_veh.T).T
            gyro_b = (R_true.T @ gyros_veh.T).T

            alignment = estimate_mounting_alignment(
                stationary_accel=accel_stat,
                moving_gnss_speed_mps=speeds,
                moving_gnss_bearing_deg=bearings,
                moving_gyro=gyro_b,
                timestamps_ns=ts,
                moving_accel=accel_b,
                min_speed_mps=3.0,
            )

            assert alignment.is_yaw_aligned is True
            assert alignment.yaw_deg == pytest.approx(known_yaw_deg, abs=0.5)
            assert np.allclose(alignment.R_b_v, R_true, atol=1e-2)

            # Transformed specific force in vehicle frame must recover forward acceleration
            f_recovered = alignment.transform_specific_force(accel_b[10])
            assert f_recovered[0] == pytest.approx(a_x_true, abs=0.05)
            assert f_recovered[1] == pytest.approx(0.0, abs=0.05)
            assert f_recovered[2] == pytest.approx(9.80665, abs=0.05)

    def test_unobservable_yaw_conditions_return_unresolved(self) -> None:
        """Verify that when vehicle motion does not provide longitudinal acceleration observability,
        yaw is NOT fabricated and alignment.is_yaw_aligned is False."""
        accel_stat = np.tile([0.0, 0.0, 9.80665], (50, 1))
        n_mov = 50
        ts = np.arange(0, n_mov * 100_000_000, 100_000_000, dtype=np.int64)

        # Condition 1: Constant cruising speed (dv/dt = 0)
        speeds_cruise = np.full(n_mov, 15.0)
        f_b_cruise = np.tile([0.0, 0.0, 9.80665], (n_mov, 1))
        gyros_zero = np.zeros((n_mov, 3))

        align_cruise = estimate_mounting_alignment(
            stationary_accel=accel_stat,
            moving_gnss_speed_mps=speeds_cruise,
            moving_gyro=gyros_zero,
            timestamps_ns=ts,
            moving_accel=f_b_cruise,
            min_speed_mps=3.0,
        )
        assert align_cruise.is_yaw_aligned is False
        assert align_cruise.yaw_deg == pytest.approx(0.0, abs=1e-6)
        assert "UNRESOLVED" in align_cruise.alignment_status

        # Condition 2: Low speed below threshold (v = 1.0 m/s < 3.0 m/s)
        speeds_slow = np.full(n_mov, 1.0)
        align_slow = estimate_mounting_alignment(
            stationary_accel=accel_stat,
            moving_gnss_speed_mps=speeds_slow,
            moving_gyro=gyros_zero,
            timestamps_ns=ts,
            moving_accel=f_b_cruise,
            min_speed_mps=3.0,
        )
        assert align_slow.is_yaw_aligned is False

        # Condition 3: High turn rate during acceleration (cornering: |omega_z| > 0.05 rad/s)
        gyros_turn = np.tile([0.0, 0.0, 0.20], (n_mov, 1))
        speeds_accel = 5.0 + 1.0 * (np.arange(n_mov) * 0.1)
        f_b_accel = np.tile([1.0, 0.0, 9.80665], (n_mov, 1))
        align_turn = estimate_mounting_alignment(
            stationary_accel=accel_stat,
            moving_gnss_speed_mps=speeds_accel,
            moving_gyro=gyros_turn,
            timestamps_ns=ts,
            moving_accel=f_b_accel,
            min_speed_mps=3.0,
        )
        assert align_turn.is_yaw_aligned is False

    def test_explicit_reference_yaw_assignment(self) -> None:
        """When an external reference yaw is provided, verify it is applied directly."""
        accel_stat = np.tile([0.0, 0.0, 9.80665], (30, 1))
        alignment = estimate_mounting_alignment(
            stationary_accel=accel_stat,
            reference_yaw_deg=45.0,
        )
        assert alignment.is_yaw_aligned is True
        assert alignment.yaw_deg == pytest.approx(45.0, abs=1e-4)
        assert "RESOLVED_REFERENCE_AZIMUTH" in alignment.alignment_status

    def test_align_raw_imu_sample_converts_to_aligned_schema(self) -> None:
        """Verify that align_sample converts RawIMUSample into AlignedIMUSample with bias removal."""
        R = np.eye(3)
        alignment = MountingAlignment(R_b_v=R, roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0, is_yaw_aligned=True)

        cal = CalibrationProfile(
            gyro_bias=(0.01, 0.02, -0.01),
            initial_roll_rad=0.0,
            initial_pitch_rad=0.0,
            sample_count=50,
            duration_s=5.0,
            is_valid=True,
            accel_bias_prior=(0.05, -0.05, 0.0),
        )

        raw = RawIMUSample(
            timestamp_ns=1_000_000_000,
            accel=(0.05, 0.95, 9.81),
            gyro=(0.01, 0.12, -0.01),
        )

        aligned = alignment.align_sample(raw, calibration=cal)

        # Accel bias [0.05, -0.05, 0.0] subtracted: [0.0, 1.0, 9.81]
        assert aligned.accel_vehicle[0] == pytest.approx(0.0, abs=1e-6)
        assert aligned.accel_vehicle[1] == pytest.approx(1.0, abs=1e-6)
        assert aligned.accel_vehicle[2] == pytest.approx(9.81, abs=1e-6)

        # Gyro bias [0.01, 0.02, -0.01] subtracted: [0.0, 0.10, 0.0]
        assert aligned.gyro_vehicle[0] == pytest.approx(0.0, abs=1e-6)
        assert aligned.gyro_vehicle[1] == pytest.approx(0.10, abs=1e-6)
        assert aligned.gyro_vehicle[2] == pytest.approx(0.0, abs=1e-6)

        assert aligned.timestamp_ns == 1_000_000_000
        assert aligned.is_usable_for_integration is True
