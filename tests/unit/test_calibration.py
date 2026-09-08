"""Unit tests for Phase 3 stationary IMU calibration."""

from __future__ import annotations

import math
import numpy as np
import pytest

from navigation.preprocessing.calibration import CalibrationProfile, calibrate_stationary_window


class TestStationaryCalibration:
    """Test suite for stationary gyro bias and tilt calibration."""

    def test_recovers_known_gyro_bias(self) -> None:
        """Inject known constant gyro bias and small random noise, verify recovery within tolerance."""
        np.random.seed(42)
        n = 100
        true_bias = np.array([0.015, -0.022, 0.008])
        noise = np.random.normal(0, 0.0005, size=(n, 3))
        gyro_samples = true_bias + noise

        # Flat level stationary accelerometer: [0, 0, 9.80665] + noise
        accel_samples = np.tile([0.0, 0.0, 9.80665], (n, 1)) + np.random.normal(0, 0.01, size=(n, 3))
        ts = np.arange(0, n * 100_000_000, 100_000_000, dtype=np.int64)

        cal = calibrate_stationary_window(accel_samples, gyro_samples, timestamps_ns=ts)

        assert cal.is_valid is True
        assert cal.sample_count == n
        assert cal.duration_s == pytest.approx(9.9, rel=1e-2)

        # Recovered gyro bias should match true bias within 3-sigma noise limit (~0.0005 rad/s)
        assert cal.gyro_bias[0] == pytest.approx(true_bias[0], abs=5e-4)
        assert cal.gyro_bias[1] == pytest.approx(true_bias[1], abs=5e-4)
        assert cal.gyro_bias[2] == pytest.approx(true_bias[2], abs=5e-4)

    def test_recovers_known_roll_and_pitch(self) -> None:
        """Inject known pitch and roll angles, verify tilt angle recovery from gravity reaction vector."""
        g = 9.80665
        # Pitch: 15 degrees nose-up, Roll: -10 degrees bank left
        pitch_true_deg = 15.0
        roll_true_deg = -10.0
        pitch_rad = math.radians(pitch_true_deg)
        roll_rad = math.radians(roll_true_deg)

        # Expected unit gravity reaction vector in body frame:
        # pitch = atan2(ux, sqrt(uy^2 + uz^2))
        # roll = atan2(-uy, uz)
        # Therefore:
        # ux = sin(pitch)
        # uy = -cos(pitch) * sin(roll)
        # uz = cos(pitch) * cos(roll)
        ux = math.sin(pitch_rad)
        uy = -math.cos(pitch_rad) * math.sin(roll_rad)
        uz = math.cos(pitch_rad) * math.cos(roll_rad)
        u_expected = np.array([ux, uy, uz])

        n = 50
        accel_samples = np.tile(g * u_expected, (n, 1))
        gyro_samples = np.zeros((n, 3))

        cal = calibrate_stationary_window(accel_samples, gyro_samples)

        assert cal.is_valid is True
        assert math.degrees(cal.initial_pitch_rad) == pytest.approx(pitch_true_deg, abs=1e-4)
        assert math.degrees(cal.initial_roll_rad) == pytest.approx(roll_true_deg, abs=1e-4)

    def test_nominal_zero_accel_bias_prior_documented_and_enforced(self) -> None:
        """Verify that single-pose calibration assigns strictly nominal zero prior to accel bias."""
        n = 30
        accel = np.tile([0.0, 0.0, 9.81], (n, 1))
        gyro = np.zeros((n, 3))

        cal = calibrate_stationary_window(accel, gyro)
        assert cal.accel_bias_prior == (0.0, 0.0, 0.0)

    def test_insufficient_samples_marks_invalid(self) -> None:
        """Verify that a window with fewer samples than min_samples returns is_valid=False."""
        n = 5
        accel = np.tile([0.0, 0.0, 9.81], (n, 1))
        gyro = np.zeros((n, 3))

        cal = calibrate_stationary_window(accel, gyro, min_samples=20)
        assert cal.is_valid is False
        assert cal.sample_count == 5

    def test_prior_profile_fallback_when_insufficient_samples(self) -> None:
        """Verify that a prior profile is utilized as initial estimate with appropriate flags."""
        prior = CalibrationProfile(
            gyro_bias=(0.005, -0.005, 0.001),
            initial_roll_rad=0.02,
            initial_pitch_rad=-0.01,
            sample_count=100,
            duration_s=10.0,
            is_valid=True,
            accel_bias_prior=(0.01, -0.01, 0.02),
        )
        n = 3
        accel = np.tile([0.0, 0.0, 9.81], (n, 1))
        gyro = np.zeros((n, 3))

        cal = calibrate_stationary_window(accel, gyro, min_samples=20, prior_profile=prior)
        # Preserves prior estimates but marks current calibration as non-authoritative
        assert cal.is_valid is False
        assert cal.gyro_bias == prior.gyro_bias
        assert cal.accel_bias_prior == prior.accel_bias_prior

    def test_calibration_profile_json_roundtrip(self) -> None:
        """Verify serialization and deserialization of CalibrationProfile."""
        profile = CalibrationProfile(
            gyro_bias=(0.0123, -0.0456, 0.0078),
            initial_roll_rad=0.0345,
            initial_pitch_rad=-0.0678,
            sample_count=150,
            duration_s=15.0,
            is_valid=True,
            accel_bias_prior=(0.0, 0.0, 0.0),
            gyro_std=(0.001, 0.001, 0.001),
            accel_norm=9.807,
        )
        d = profile.to_dict()
        loaded = CalibrationProfile.from_dict(d)
        assert loaded == profile
