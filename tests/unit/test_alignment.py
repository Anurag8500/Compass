"""Unit tests for Phase 3 device-to-vehicle mounting alignment."""

from __future__ import annotations

import math
import numpy as np
import pytest

from navigation.preprocessing.alignment import (
    MountingAlignment,
    estimate_mounting_alignment,
    rotation_matrix_from_vectors,
)
from navigation.preprocessing.calibration import CalibrationProfile
from navigation.schemas.imu import RawIMUSample


class TestMountingAlignment:
    """Test suite for mounting orientation estimation and coordinate transformation."""

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
        # Determinant must be +1 (proper rotation, not reflection)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-10)
        # Rotates v1 direction to v2 direction
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)
        assert np.allclose(R @ v1_norm, v2_norm, atol=1e-10)

    def test_yaw_alignment_from_gnss_course_and_gyro(self) -> None:
        """Synthetic moving segment: phone mounted at 30 deg yaw relative to vehicle forward."""
        # Level phone facing up
        accel_stat = np.tile([0.0, 0.0, 9.80665], (50, 1))

        # Vehicle driving straight with speed 15 m/s at GNSS course 45 deg (North-East)
        n_mov = 60
        speeds = np.full(n_mov, 15.0)
        bearings = np.full(n_mov, 45.0)  # Constant straight course
        ts = np.arange(0, n_mov * 100_000_000, 100_000_000, dtype=np.int64)
        gyros = np.zeros((n_mov, 3))     # Zero turn rate on straight road

        alignment = estimate_mounting_alignment(
            stationary_accel=accel_stat,
            moving_gnss_speed_mps=speeds,
            moving_gnss_bearing_deg=bearings,
            moving_gyro=gyros,
            timestamps_ns=ts,
            min_speed_mps=3.0,
        )

        assert alignment.is_yaw_aligned is True
        # R_b_v must be valid orthogonal rotation matrix
        assert np.allclose(alignment.R_b_v @ alignment.R_b_v.T, np.eye(3), atol=1e-6)

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
