"""Unit tests for Phase 3 conservative recalibration trigger detector."""

from __future__ import annotations

import math
import numpy as np
import pytest

from navigation.preprocessing.recalibration_trigger import RecalibrationDetector, RecalibrationEvent


class TestRecalibrationDetector:
    """Test suite covering all 7 critical recalibration detection scenarios."""

    def test_1_nominal_driving_produces_no_triggers(self) -> None:
        """Nominal smooth driving with normal turns and accelerations must not trigger events."""
        detector = RecalibrationDetector(
            gravity_shift_threshold_deg=35.0,
            angular_rate_step_threshold_rads=5.0,
        )
        detector.set_baseline_gravity([0.0, 0.0, 9.80665])

        n = 100
        ts = np.arange(0, n * 100_000_000, 100_000_000, dtype=np.int64)
        # Normal vehicle acceleration (small pitch/roll variation, moderate speed)
        accel = np.tile([0.2, -0.1, 9.80], (n, 1))
        gyro = np.tile([0.01, -0.01, 0.05], (n, 1))  # Normal slow turn

        events = detector.scan_series(accel, gyro, ts)
        assert len(events) == 0, f"Expected 0 triggers on nominal driving, got {len(events)}"

    def test_2_isolated_gyro_shock_step(self) -> None:
        """Sudden instantaneous angular velocity shock step (> 5.0 rad/s) triggers an event immediately."""
        detector = RecalibrationDetector(
            gravity_shift_threshold_deg=35.0,
            angular_rate_step_threshold_rads=5.0,
        )
        detector.set_baseline_gravity([0.0, 0.0, 9.80665])

        n = 30
        ts = np.arange(0, n * 100_000_000, 100_000_000, dtype=np.int64)
        accel = np.tile([0.0, 0.0, 9.80665], (n, 1))
        gyro = np.zeros((n, 3))
        # Inject sudden angular velocity step at index 10
        gyro[10] = [0.0, 6.5, 0.0]

        events = detector.scan_series(accel, gyro, ts)
        assert len(events) >= 1
        assert events[0].timestamp_ns == ts[10]
        assert "Angular velocity shock step" in events[0].reason

    def test_3_sustained_gravity_shift_in_motion(self) -> None:
        """Phone mounting dislodges (shifts by 45 deg) and persists across low-dynamics motion."""
        detector = RecalibrationDetector(
            gravity_shift_threshold_deg=35.0,
            angular_rate_step_threshold_rads=5.0,
            persistence_samples=15,
        )
        detector.set_baseline_gravity([0.0, 0.0, 9.80665])

        n = 50
        ts = np.arange(0, n * 100_000_000, 100_000_000, dtype=np.int64)
        accel = np.tile([0.0, 0.0, 9.80665], (n, 1))
        gyro = np.zeros((n, 3))

        # Rotate gravity by 45 deg around X after index 10
        angle = math.radians(45.0)
        Rx = np.array([[1, 0, 0], [0, math.cos(angle), -math.sin(angle)], [0, math.sin(angle), math.cos(angle)]])
        accel[10:] = (Rx @ accel[10:].T).T

        events = detector.scan_series(accel, gyro, ts)
        assert len(events) >= 1
        assert "Persistent gravity direction shift" in events[0].reason

    def test_4_simulated_mount_drop_composite(self) -> None:
        """Simulate realistic mount drop: violent gyro spike followed by permanently tilted resting orientation."""
        detector = RecalibrationDetector(
            gravity_shift_threshold_deg=35.0,
            angular_rate_step_threshold_rads=5.0,
            cooldown_samples=10,
        )
        detector.set_baseline_gravity([0.0, 0.0, 9.80665])

        n = 60
        ts = np.arange(0, n * 100_000_000, 100_000_000, dtype=np.int64)
        accel = np.tile([0.0, 0.0, 9.80665], (n, 1))
        gyro = np.zeros((n, 3))

        # At index 15, phone knocked: angular velocity spike 8.0 rad/s
        gyro[15] = [0.0, 8.0, 0.0]
        # Rest of trip is tilted at 50 degrees
        angle = math.radians(50.0)
        Ry = np.array([[math.cos(angle), 0, math.sin(angle)], [0, 1, 0], [-math.sin(angle), 0, math.cos(angle)]])
        accel[16:] = (Ry @ accel[16:].T).T

        events = detector.scan_series(accel, gyro, ts)
        assert len(events) >= 1
        assert any("Angular velocity shock step" in e.reason for e in events)

    def test_5_stationary_confirmed_gravity_shift(self) -> None:
        """When vehicle is confirmed stationary, gravity shift triggers with rapid confirmation (5 samples)."""
        detector = RecalibrationDetector(
            gravity_shift_threshold_deg=30.0,
            persistence_samples=20,
            stationary_persistence_samples=5,
        )
        detector.set_baseline_gravity([0.0, 0.0, 9.80665])

        n = 20
        ts = np.arange(0, n * 100_000_000, 100_000_000, dtype=np.int64)
        # Vehicle is resting, but phone was adjusted by 40 degrees
        angle = math.radians(40.0)
        Rx = np.array([[1, 0, 0], [0, math.cos(angle), -math.sin(angle)], [0, math.sin(angle), math.cos(angle)]])
        accel_tilted = (Rx @ np.tile([0.0, 0.0, 9.80665], (n, 1)).T).T
        gyro = np.zeros((n, 3))
        stat_mask = np.ones(n, dtype=bool)

        events = detector.scan_series(accel_tilted, gyro, ts, stationary_mask=stat_mask)
        assert len(events) >= 1
        assert "Stationary-confirmed gravity direction shift" in events[0].reason
        # Triggered in under 10 samples due to stationary fast-confirmation
        assert events[0].timestamp_ns < ts[8]

    def test_6_high_dynamics_does_not_trigger_false_gravity_alarm(self) -> None:
        """High dynamic acceleration or sharp braking (|a| != 1g or |omega| high) must not trigger gravity shift."""
        detector = RecalibrationDetector(
            gravity_shift_threshold_deg=35.0,
            angular_rate_step_threshold_rads=10.0,  # high gyro threshold to isolate accel check
            persistence_samples=10,
        )
        detector.set_baseline_gravity([0.0, 0.0, 9.80665])

        n = 30
        ts = np.arange(0, n * 100_000_000, 100_000_000, dtype=np.int64)
        # Strong braking: total acceleration norm is 14 m/s^2 (well away from 1g)
        accel_braking = np.tile([8.0, 0.0, 9.80665], (n, 1))
        gyro = np.tile([0.0, 0.0, 0.0], (n, 1))

        events = detector.scan_series(accel_braking, gyro, ts, stationary_mask=np.zeros(n, dtype=bool))
        assert len(events) == 0, f"Expected 0 false triggers during braking dynamics, got {len(events)}"

    def test_7_cooldown_behavior(self) -> None:
        """After an event triggers, subsequent triggers are suppressed during the cooldown window."""
        detector = RecalibrationDetector(
            gravity_shift_threshold_deg=35.0,
            angular_rate_step_threshold_rads=5.0,
            cooldown_samples=10,
        )
        detector.set_baseline_gravity([0.0, 0.0, 9.80665])

        n = 30
        ts = np.arange(0, n * 100_000_000, 100_000_000, dtype=np.int64)
        accel = np.tile([0.0, 0.0, 9.80665], (n, 1))
        gyro = np.zeros((n, 3))
        # Shock at index 5 and another shock at index 8 (within 10-sample cooldown)
        gyro[5] = [0.0, 7.0, 0.0]
        gyro[8] = [0.0, 7.0, 0.0]
        # Another shock at index 22 (outside cooldown)
        gyro[22] = [0.0, 7.0, 0.0]

        events = detector.scan_series(accel, gyro, ts)
        # Only index 5 and index 22 should trigger; index 8 is suppressed by cooldown
        assert len(events) == 2
        assert events[0].timestamp_ns == ts[5]
        assert events[1].timestamp_ns == ts[22]
