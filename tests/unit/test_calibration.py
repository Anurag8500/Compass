"""Unit tests for Phase 3 stationary IMU calibration."""

from __future__ import annotations

import math
import numpy as np
import pytest

from navigation.preprocessing.calibration import CalibrationProfile, calibrate_stationary_window


class TestStationaryCalibration:
    """Test suite for stationary gyro bias, tilt calibration, and input validation."""

    def test_exact_zero_gyro(self) -> None:
        """When gyro is identically zero, recovered bias must be exactly (0.0, 0.0, 0.0)."""
        n = 30
        accel = np.tile([0.0, 0.0, 9.80665], (n, 1))
        gyro = np.zeros((n, 3))
        cal = calibrate_stationary_window(accel, gyro)

        assert cal.is_valid is True
        assert cal.gyro_bias == (0.0, 0.0, 0.0)
        assert cal.accel_bias_prior == (0.0, 0.0, 0.0)

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
        pitch_true_deg = 15.0
        roll_true_deg = -10.0
        pitch_rad = math.radians(pitch_true_deg)
        roll_rad = math.radians(roll_true_deg)

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

    def test_nan_inf_inputs_rejected(self) -> None:
        """Verify that NaN or Inf in accel or gyro returns is_valid=False."""
        n = 30
        accel_nan = np.tile([0.0, 0.0, 9.81], (n, 1))
        accel_nan[10, 0] = np.nan
        gyro = np.zeros((n, 3))

        cal = calibrate_stationary_window(accel_nan, gyro)
        assert cal.is_valid is False

        accel_inf = np.tile([0.0, 0.0, 9.81], (n, 1))
        gyro_inf = np.zeros((n, 3))
        gyro_inf[5, 1] = np.inf

        cal_inf = calibrate_stationary_window(accel_inf, gyro_inf)
        assert cal_inf.is_valid is False

    def test_invalid_timestamps_raise(self) -> None:
        """Verify that non-monotonic or negative timestamps raise ValueError."""
        n = 30
        accel = np.tile([0.0, 0.0, 9.81], (n, 1))
        gyro = np.zeros((n, 3))

        # Negative timestamp
        ts_neg = np.arange(-10, n - 10, dtype=np.int64) * 100_000_000
        with pytest.raises(ValueError, match="negative"):
            calibrate_stationary_window(accel, gyro, timestamps_ns=ts_neg)

        # Non-monotonic timestamp
        ts_non_mono = np.arange(0, n * 100_000_000, 100_000_000, dtype=np.int64)
        ts_non_mono[15] = ts_non_mono[10]  # backward jump
        with pytest.raises(ValueError, match="monotonic"):
            calibrate_stationary_window(accel, gyro, timestamps_ns=ts_non_mono)

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
        assert cal.is_valid is False
        assert cal.gyro_bias == prior.gyro_bias
        assert cal.accel_bias_prior == prior.accel_bias_prior

    def test_immutability_of_inputs(self) -> None:
        """Verify that calibrate_stationary_window does not mutate input arrays."""
        n = 40
        accel = np.tile([0.1, -0.2, 9.81], (n, 1))
        gyro = np.tile([0.01, 0.02, -0.01], (n, 1))
        accel_copy = accel.copy()
        gyro_copy = gyro.copy()

        calibrate_stationary_window(accel, gyro)
        assert np.array_equal(accel, accel_copy)
        assert np.array_equal(gyro, gyro_copy)

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
