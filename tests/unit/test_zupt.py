"""Unit tests for Classical Gated Zero Velocity Update (ZUPT) detector and model (Phase 5).

Validates:
- Standstill detection on synthetic quiescent data.
- Rejection of motion under angular velocity, acceleration deviation, variance, and speed criteria.
- Sliding window buffer fill and reset mechanics.
- ZUPT measurement update: velocity reduction and variance contraction.
- Outlier / dynamic motion gating rejection and state immutability.
- Validation on real IO-VNBD Trip S1 data (stationary start vs. dynamic driving).
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pytest

from navigation.eskf.gating import MahalanobisGating
from navigation.eskf.measurements.zupt import (
    ClassicalZUPTDetector,
    ZUPTDetectorConfig,
    ZUPTMeasurementConfig,
    ZUPTMeasurementModel,
)
from navigation.eskf.state import ESKFNominalState, ESKFState


class TestClassicalZUPTDetector:
    """Validates the zero-ML classical standstill detector physics invariants."""

    def test_standstill_detected_on_synthetic_quiescent_data(self) -> None:
        """Stationary sensor reading 1g along Z and zero rotation must be flagged stationary."""
        detector = ClassicalZUPTDetector()
        f_stat = np.array([0.0, 0.0, 9.80665])
        w_stat = np.array([0.0, 0.0, 0.0])

        f_window = np.tile(f_stat, (8, 1))
        w_window = np.tile(w_stat, (8, 1))

        diag = detector.evaluate_window(omega_window=w_window, accel_window=f_window)
        assert diag.is_stationary is True
        assert diag.accel_norm_dev == pytest.approx(0.0, abs=1e-6)
        assert diag.accel_norm_var == pytest.approx(0.0, abs=1e-6)
        assert diag.omega_norm_mean == pytest.approx(0.0, abs=1e-6)

    def test_motion_rejection_high_angular_velocity(self) -> None:
        """Angular velocity exceeding 0.05 rad/s must reject standstill."""
        detector = ClassicalZUPTDetector()
        f_stat = np.tile([0.0, 0.0, 9.80665], (8, 1))
        w_moving = np.tile([0.06, 0.0, 0.0], (8, 1))  # > 0.05 rad/s

        diag = detector.evaluate_window(omega_window=w_moving, accel_window=f_stat)
        assert diag.is_stationary is False

    def test_motion_rejection_accel_deviation_from_gravity(self) -> None:
        """Specific force norm deviating by > 0.25 m/s^2 from g must reject standstill."""
        detector = ClassicalZUPTDetector()
        # Magnitude = 9.80665 + 0.35 = 10.15665
        f_accelerating = np.tile([0.0, 0.0, 9.80665 + 0.35], (8, 1))
        w_stat = np.tile([0.0, 0.0, 0.0], (8, 1))

        diag = detector.evaluate_window(omega_window=w_stat, accel_window=f_accelerating)
        assert diag.is_stationary is False
        assert diag.accel_norm_dev > 0.25

    def test_motion_rejection_high_accel_variance(self) -> None:
        """Vibration / rough motion with norm variance > 0.015 (m/s^2)^2 must reject standstill."""
        detector = ClassicalZUPTDetector()
        # Alternating +/- 0.2 around g has zero mean deviation but high variance (0.04 > 0.015)
        f_vibrating = np.array([
            [0.0, 0.0, 9.80665 + 0.2],
            [0.0, 0.0, 9.80665 - 0.2],
        ] * 4)
        w_stat = np.tile([0.0, 0.0, 0.0], (8, 1))

        diag = detector.evaluate_window(omega_window=w_stat, accel_window=f_vibrating)
        assert diag.is_stationary is False
        assert diag.accel_norm_var > 0.015

    def test_motion_rejection_auxiliary_speed(self) -> None:
        """Auxiliary GNSS speed exceeding 0.1 m/s must reject standstill."""
        detector = ClassicalZUPTDetector()
        f_stat = np.tile([0.0, 0.0, 9.80665], (8, 1))
        w_stat = np.tile([0.0, 0.0, 0.0], (8, 1))

        diag = detector.evaluate_window(omega_window=w_stat, accel_window=f_stat, speed_mps=0.25)
        assert diag.is_stationary is False

    def test_streaming_buffer_fill_and_reset(self) -> None:
        """Detector requires full window (8 samples) before declaring standstill."""
        detector = ClassicalZUPTDetector(ZUPTDetectorConfig(window_size=8))
        f_stat = np.array([0.0, 0.0, 9.80665])
        w_stat = np.array([0.0, 0.0, 0.0])

        for i in range(7):
            diag = detector.push(omega_v=w_stat, f_v=f_stat)
            assert diag.is_stationary is False
            assert diag.window_len == i + 1

        # 8th sample fills window
        diag8 = detector.push(omega_v=w_stat, f_v=f_stat)
        assert diag8.is_stationary is True
        assert diag8.window_len == 8

        # Reset clears window
        detector.reset()
        diag_after_reset = detector.push(omega_v=w_stat, f_v=f_stat)
        assert diag_after_reset.is_stationary is False
        assert diag_after_reset.window_len == 1


class TestZUPTMeasurementUpdate:
    """Validates ESKF update with zero-velocity pseudo-measurement."""

    def test_zupt_reduces_velocity_and_covariance(self) -> None:
        """Applying ZUPT to a drifting velocity state drives velocity to ~0 and contracts covariance."""
        nom = ESKFNominalState.from_components(
            position_enu=[0.0, 0.0, 0.0],
            velocity_enu=[0.05, -0.04, 0.02],  # slight drift
            q=[1.0, 0.0, 0.0, 0.0],
            timestamp_ns=1_000_000,
        )
        P0 = np.eye(15, dtype=np.float64) * 0.5
        state = ESKFState(nominal=nom, covariance=P0)

        zupt_model = ZUPTMeasurementModel(ZUPTMeasurementConfig(velocity_noise_sigma=0.03))
        updated_state, diag = zupt_model.update(state)

        assert diag.applied is True
        assert diag.gating is not None
        assert diag.gating.accepted is True

        # Velocity magnitude must decrease
        v_prior_norm = np.linalg.norm(state.velocity_enu)
        v_post_norm = np.linalg.norm(updated_state.velocity_enu)
        assert v_post_norm < v_prior_norm
        assert v_post_norm < 0.01  # significantly suppressed

        # Velocity covariance must decrease
        for i in range(3, 6):
            assert updated_state.covariance[i, i] < P0[i, i]

    def test_zupt_gating_rejects_spurious_update_during_motion(self) -> None:
        """If ZUPT is triggered during actual motion, innovation gating rejects it and preserves state."""
        nom = ESKFNominalState.from_components(
            position_enu=[0.0, 0.0, 0.0],
            velocity_enu=[15.0, 0.0, 0.0],  # highway driving 15 m/s
            q=[1.0, 0.0, 0.0, 0.0],
            timestamp_ns=1_000_000,
        )
        P0 = np.eye(15, dtype=np.float64) * 0.1
        state = ESKFState(nominal=nom, covariance=P0)

        zupt_model = ZUPTMeasurementModel(gating=MahalanobisGating(confidence_level=0.99))
        updated_state, diag = zupt_model.update(state)

        assert diag.applied is False
        assert diag.gating is not None
        assert diag.gating.accepted is False

        # State velocity and covariance remain strictly unchanged
        assert np.allclose(updated_state.velocity_enu, state.velocity_enu)
        assert np.array_equal(updated_state.covariance, P0)


class TestRealDataZUPTValidation:
    """Validates ZUPT detector on real IO-VNBD Trip S1 inertial data."""

    def test_trip_s1_standstill_vs_driving(self) -> None:
        """Verify detector classifies known initial standstill as stationary and dynamic driving as moving."""
        from data.pipeline.sync import SynchronizedTrip
        from data.pipeline.stationary_detect import StationaryDetector
        from navigation.preprocessing.pipeline import PreprocessingPipeline

        npz_path = Path("data/cache/iovnbd/Uncategorised_S1.npz")
        if not npz_path.exists():
            pytest.skip(f"Trip S1 cached data not found at {npz_path}")

        trip = SynchronizedTrip.load_npz(npz_path)

        # Preprocess with Phase 3 calibrated pipeline
        stat_detector = StationaryDetector()
        segs, stat_mask = stat_detector.detect(trip.timestamps_ns, trip.accel_raw, trip.gyro_raw)

        pipeline = PreprocessingPipeline(sampling_rate_hz=10.0, filter_cutoff_hz=3.0, median_window_size=3)
        preprocessed = pipeline.process_trip(trip, stationary_mask=stat_mask)

        f_v = preprocessed.f_m_v
        w_v = preprocessed.omega_m_v

        detector = ClassicalZUPTDetector()

        # Known standstill at start of Trip S1: samples 60 to 68 (quiescent parking lot)
        diag_stationary = detector.evaluate_window(
            omega_window=w_v[60:68],
            accel_window=f_v[60:68],
        )
        assert diag_stationary.is_stationary is True
        assert diag_stationary.accel_norm_dev < 0.25
        assert diag_stationary.omega_norm_mean < 0.05

        # Dynamic driving at sample 19600 (Ablation Stage 1 window)
        diag_moving = detector.evaluate_window(
            omega_window=w_v[19600:19608],
            accel_window=f_v[19600:19608],
        )
        assert diag_moving.is_stationary is False

