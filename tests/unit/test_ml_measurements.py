"""Unit Tests for Phase 9 ML Measurement Models, ModelRunner, and Scheduling.

Validates:
1. ModelRunner input contracts, normalization, and bounds.
2. CausalWindowBuffer strict causality and history requirements.
3. MLCadenceScheduler time-aware execution rate (~2 Hz and ~1 Hz).
4. VelocityNet measurement model: forward projection, 15D Jacobian, uncertainty bounding.
5. BiasNet measurement model: pseudo-measurement formulation, 15D error-state indexing, covariance.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from navigation.eskf.gating import MahalanobisGating
from navigation.eskf.predict import predict_eskf
from navigation.eskf.scheduling import CadenceConfig, MLCadenceScheduler
from navigation.eskf.state import ESKFNominalState, ESKFState
from navigation.eskf.measurements.velocitynet import (
    CausalEMA,
    VelocityNetConfig,
    VelocityNetMeasurementModel,
)
from navigation.eskf.measurements.biasnet import (
    PHASE8_R_BIAS_DIAG,
    BiasNetConfig,
    BiasNetMeasurementModel,
)
from navigation.ml.model_runner import (
    BiasNetOutput,
    MLWindowValidator,
    ModelRunnerConfig,
    ONNXModelRunner,
    VelocityNetOutput,
)
from navigation.ml.window_buffer import CausalWindowBuffer


# ==============================================================================
# 1. ModelRunner and Window Validation Tests
# ==============================================================================

class TestMLWindowValidator:
    """Test suite for feature array validation and out-of-distribution detection."""

    def test_valid_window(self) -> None:
        raw = np.random.randn(20, 9)
        raw[:, 2] += 9.81  # Normal gravity
        is_val, reason = MLWindowValidator.validate_features(raw)
        assert is_val is True
        assert reason is None

    def test_nan_rejection(self) -> None:
        raw = np.zeros((20, 9))
        raw[5, 3] = np.nan
        is_val, reason = MLWindowValidator.validate_features(raw)
        assert is_val is False
        assert reason == "NON_FINITE_INPUT"

    def test_inf_rejection(self) -> None:
        raw = np.zeros((20, 9))
        raw[10, 0] = np.inf
        is_val, reason = MLWindowValidator.validate_features(raw)
        assert is_val is False
        assert reason == "NON_FINITE_INPUT"

    def test_invalid_shape_rejection(self) -> None:
        raw = np.zeros((19, 9))
        is_val, reason = MLWindowValidator.validate_features(raw)
        assert is_val is False
        assert "INVALID_SHAPE" in reason

    def test_ood_accel_rejection(self) -> None:
        raw = np.zeros((20, 9))
        raw[0, 0] = 150.0  # > 100 m/s^2 threshold
        is_val, reason = MLWindowValidator.validate_features(raw)
        assert is_val is False
        assert reason == "OOD_ACCEL_FEATURES"

    def test_ood_gyro_rejection(self) -> None:
        raw = np.zeros((20, 9))
        raw[0, 4] = 45.0  # > 30 rad/s threshold
        is_val, reason = MLWindowValidator.validate_features(raw)
        assert is_val is False
        assert reason == "OOD_GYRO_FEATURES"

    def test_timestamp_continuity_check(self) -> None:
        raw = np.zeros((20, 9))
        ts = np.arange(20) * int(1e8)  # 10 Hz: 0.1s steps
        is_val, reason = MLWindowValidator.validate_features(raw, timestamps_ns=ts)
        assert is_val is True

        # Non-monotonic
        ts_bad = ts.copy()
        ts_bad[10] = ts_bad[9] - 1
        is_val, reason = MLWindowValidator.validate_features(raw, timestamps_ns=ts_bad)
        assert is_val is False
        assert reason == "NON_MONOTONIC_TIMESTAMPS"

        # Large gap (> 0.5s)
        ts_gap = ts.copy()
        ts_gap[15:] += int(1e9)  # 1s jump
        is_val, reason = MLWindowValidator.validate_features(raw, timestamps_ns=ts_gap)
        assert is_val is False
        assert reason == "TIMESTAMP_GAP"


class TestONNXModelRunner:
    """Test suite for frozen ONNX model execution and output contracts."""

    @pytest.fixture
    def runner(self) -> ONNXModelRunner:
        return ONNXModelRunner()

    def test_velocitynet_inference_contract(self, runner: ONNXModelRunner) -> None:
        raw = np.zeros((20, 9), dtype=np.float64)
        raw[:, 2] = 9.81
        ts = np.arange(20) * int(1e8)
        out = runner.run_velocitynet(raw, ts)

        assert isinstance(out, VelocityNetOutput)
        assert out.valid is True
        assert math.isfinite(out.speed_mps)
        assert math.isfinite(out.log_variance)
        assert -10.0 <= out.log_variance <= 10.0
        # Check variance bounds [1.0, 25.0]
        assert 1.0 <= out.variance <= 25.0

    def test_biasnet_inference_contract(self, runner: ONNXModelRunner) -> None:
        raw = np.zeros((20, 9), dtype=np.float64)
        raw[:, 2] = 9.81
        ts = np.arange(20) * int(1e8)
        out = runner.run_biasnet(raw, ts)

        assert isinstance(out, BiasNetOutput)
        assert out.valid is True
        assert out.delta_accel_bias.shape == (3,)
        assert out.delta_gyro_bias.shape == (3,)
        assert out.delta_bias_vector.shape == (6,)
        # Check physical clamp bounds
        assert np.all(np.abs(out.delta_accel_bias) <= 2.0)
        assert np.all(np.abs(out.delta_gyro_bias) <= 0.15)


# ==============================================================================
# 2. Causal Window Buffer Tests
# ==============================================================================

class TestCausalWindowBuffer:
    """Test suite for strict causal buffering."""

    def test_insufficient_history(self) -> None:
        buf = CausalWindowBuffer(max_length=20)
        for i in range(10):
            buf.push(i * int(1e8), np.zeros(3), np.zeros(3))
        has_win, feats, ts, reason = buf.get_causal_window()
        assert has_win is False
        assert reason == "INSUFFICIENT_HISTORY"

    def test_full_causal_window(self) -> None:
        buf = CausalWindowBuffer(max_length=20)
        for i in range(25):
            buf.push(i * int(1e8), np.array([0.0, 0.0, 9.81]), np.zeros(3))
        has_win, feats, ts, reason = buf.get_causal_window(current_timestamp_ns=24 * int(1e8))
        assert has_win is True
        assert feats.shape == (20, 9)
        assert len(ts) == 20
        # Verify samples correspond to timestamps 5 to 24 (all <= 24)
        assert ts[0] == 5 * int(1e8)
        assert ts[-1] == 24 * int(1e8)

    def test_strict_causality_lookahead_violation(self) -> None:
        buf = CausalWindowBuffer(max_length=20)
        for i in range(20):
            buf.push(i * int(1e8), np.zeros(3), np.zeros(3))
        # Query with timestamp before the latest buffered sample
        has_win, feats, ts, reason = buf.get_causal_window(current_timestamp_ns=15 * int(1e8))
        assert has_win is False
        assert reason == "LOOKAHEAD_VIOLATION"


# ==============================================================================
# 3. Cadence Scheduler Tests
# ==============================================================================

class TestMLCadenceScheduler:
    """Test suite for time-aware update scheduling."""

    def test_scheduling_rates(self) -> None:
        sched = MLCadenceScheduler(CadenceConfig(velocitynet_interval_s=0.5, biasnet_interval_s=1.0))
        
        # t = 0.0s: first sample triggers both
        t0 = 0
        v0, b0 = sched.evaluate_cycle(t0)
        assert v0 is True
        assert b0 is True
        sched.record_velocitynet_execution(t0)
        sched.record_biasnet_execution(t0)

        # t = 0.1s to 0.4s: neither should run
        for step in range(1, 5):
            t = int(step * 1e8)
            v, b = sched.evaluate_cycle(t)
            assert v is False
            assert b is False

        # t = 0.5s: VelocityNet runs (~2 Hz), BiasNet does not
        t5 = int(5e8)
        v5, b5 = sched.evaluate_cycle(t5)
        assert v5 is True
        assert b5 is False
        sched.record_velocitynet_execution(t5)

        # t = 1.0s: Both run
        t10 = int(1e9)
        v10, b10 = sched.evaluate_cycle(t10)
        assert v10 is True
        assert b10 is True

    def test_duplicate_timestamp_rejection(self) -> None:
        sched = MLCadenceScheduler()
        t = int(1e9)
        sched.evaluate_cycle(t)
        # Immediate identical or sub-millisecond timestamp
        v_dup, b_dup = sched.evaluate_cycle(t + int(1e5))  # +0.1 ms
        assert v_dup is False
        assert b_dup is False

    def test_no_repeated_warmup_execution_attempts(self) -> None:
        """Proves that when buffer is not ready, schedule advances rather than repeating every 10 Hz sample."""
        sched = MLCadenceScheduler(CadenceConfig(velocitynet_interval_s=0.5, biasnet_interval_s=1.0))
        due_vnet_count = 0
        due_bnet_count = 0

        # Simulate 20 samples at 10 Hz (0.0s to 1.9s)
        for step in range(20):
            t_ns = int(step * 1e8)
            v_due, b_due = sched.evaluate_cycle(t_ns)
            if v_due:
                due_vnet_count += 1
                # Mark scheduled because buffer was warming up
                sched.mark_velocitynet_scheduled(t_ns)
            if b_due:
                due_bnet_count += 1
                # Mark scheduled because buffer was warming up
                sched.mark_biasnet_scheduled(t_ns)

        # Over 2.0s:
        # VNet due at 0.0s, 0.5s, 1.0s, 1.5s -> exactly 4 times, NOT 20 times!
        assert due_vnet_count == 4, f"Expected 4 VNet due checks over 2.0s, got {due_vnet_count}"
        # BNet due at 0.0s, 1.0s -> exactly 2 times, NOT 20 times!
        assert due_bnet_count == 2, f"Expected 2 BNet due checks over 2.0s, got {due_bnet_count}"

    def test_cadence_intervals_strictly_spaced(self) -> None:
        """Proves that successful executions enforce minimum 0.5s and 1.0s spacing."""
        sched = MLCadenceScheduler(CadenceConfig(velocitynet_interval_s=0.5, biasnet_interval_s=1.0))
        vnet_exec_times_s: list[float] = []
        bnet_exec_times_s: list[float] = []

        # Run 50 samples at 10 Hz (5.0s)
        for step in range(50):
            t_ns = int(step * 1e8)
            v_due, b_due = sched.evaluate_cycle(t_ns)
            if v_due:
                sched.mark_velocitynet_executed(t_ns)
                vnet_exec_times_s.append(step * 0.1)
            if b_due:
                sched.mark_biasnet_executed(t_ns)
                bnet_exec_times_s.append(step * 0.1)

        # Check all intervals between consecutive VNet executions >= 0.5s
        v_diffs = np.diff(vnet_exec_times_s)
        assert len(v_diffs) > 0
        assert np.all(v_diffs >= 0.499), f"VNet cadence violated: diffs={v_diffs}"

        # Check all intervals between consecutive BNet executions >= 1.0s
        b_diffs = np.diff(bnet_exec_times_s)
        assert len(b_diffs) > 0
        assert np.all(b_diffs >= 0.999), f"BNet cadence violated: diffs={b_diffs}"


# ==============================================================================
# 4. VelocityNet Measurement Adapter Tests
# ==============================================================================

class TestVelocityNetMeasurementModel:
    """Test suite for VelocityNet ESKF adapter."""

    @pytest.fixture
    def initial_state(self) -> ESKFState:
        nom = ESKFNominalState.from_components(
            position_enu=[0.0, 0.0, 0.0],
            velocity_enu=[10.0, 0.0, 0.0],  # Moving East
            q=[1.0, 0.0, 0.0, 0.0],         # Identity: Vehicle +X is East, +Y is North, +Z is Up
            accel_bias=[0.0, 0.0, 0.0],
            gyro_bias=[0.0, 0.0, 0.0],
        )
        P = np.eye(15, dtype=np.float64) * 0.1
        return ESKFState(nominal=nom, covariance=P)

    def test_jacobian_and_measurement_projection(self, initial_state: ESKFState) -> None:
        model = VelocityNetMeasurementModel(VelocityNetConfig(use_causal_ema=False, min_speed_threshold_mps=0.0))
        pred = VelocityNetOutput(speed_mps=12.0, log_variance=1.0, variance=2.718, valid=True)

        updated_state, diag = model.update(initial_state, pred, timestamp_ns=100)
        assert diag.applied is True
        assert diag.z_speed == 12.0
        assert diag.predicted_speed == pytest.approx(10.0)

        # Velocity in East should be corrected upwards towards 12.0
        assert updated_state.nominal.velocity_enu[0] > 10.0
        # North and Up velocities should remain near zero
        assert abs(updated_state.nominal.velocity_enu[1]) < 1e-4
        assert abs(updated_state.nominal.velocity_enu[2]) < 1e-4

    def test_attitude_dependent_forward_projection(self) -> None:
        # Facing North: vehicle +X points North (ENU [0, 1, 0]), vehicle +Y points West (ENU [-1, 0, 0])
        # R_v^n = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        from navigation.ins.attitude import rotation_matrix_to_quaternion
        R_north = np.array([
            [0.0, -1.0, 0.0],
            [1.0,  0.0, 0.0],
            [0.0,  0.0, 1.0],
        ], dtype=np.float64)
        q_north = rotation_matrix_to_quaternion(R_north)
        
        nom = ESKFNominalState.from_components(
            position_enu=[0.0, 0.0, 0.0],
            velocity_enu=[0.0, 8.0, 0.0],  # Moving North at 8 m/s
            q=q_north,
        )
        state = ESKFState(nominal=nom, covariance=np.eye(15, dtype=np.float64) * 0.1)

        model = VelocityNetMeasurementModel(VelocityNetConfig(use_causal_ema=False, min_speed_threshold_mps=0.0))
        pred = VelocityNetOutput(speed_mps=10.0, log_variance=0.0, variance=1.0, valid=True)

        updated_state, diag = model.update(state, pred)
        assert diag.applied is True
        # Forward speed projected along North: predicted was 8.0 m/s
        assert diag.predicted_speed == pytest.approx(8.0, abs=1e-3)
        # North velocity should increase towards 10 m/s
        assert updated_state.nominal.velocity_enu[1] > 8.0

    def test_standstill_motion_gating(self, initial_state: ESKFState) -> None:
        model = VelocityNetMeasurementModel(VelocityNetConfig(min_speed_threshold_mps=0.5, use_causal_ema=False))
        # Speed below threshold
        pred = VelocityNetOutput(speed_mps=0.2, log_variance=0.0, variance=1.0, valid=True)

        updated_state, diag = model.update(initial_state, pred)
        assert diag.applied is False
        assert diag.reason == "STANDSTILL_SUPPRESSED"
        # State strictly unmodified
        np.testing.assert_array_equal(updated_state.nominal.velocity_enu, initial_state.nominal.velocity_enu)


# ==============================================================================
# 5. BiasNet Measurement Adapter Tests
# ==============================================================================

class TestBiasNetMeasurementModel:
    """Test suite for BiasNet ESKF adapter."""

    @pytest.fixture
    def initial_state(self) -> ESKFState:
        nom = ESKFNominalState.from_components(
            position_enu=[0.0, 0.0, 0.0],
            velocity_enu=[5.0, 0.0, 0.0],
            accel_bias=[0.05, -0.02, 0.10],
            gyro_bias=[0.001, -0.002, 0.003],
        )
        P = np.eye(15, dtype=np.float64) * 0.01
        return ESKFState(nominal=nom, covariance=P)

    def test_bias_error_state_indexing_and_update(self, initial_state: ESKFState) -> None:
        """Verify H_b strictly maps to error-state indices [9:12] and [12:15]."""
        model = BiasNetMeasurementModel(BiasNetConfig())
        delta_ba = np.array([0.1, 0.0, -0.05], dtype=np.float64)
        delta_bg = np.array([0.005, -0.001, 0.0], dtype=np.float64)
        pred = BiasNetOutput(
            delta_accel_bias=delta_ba,
            delta_gyro_bias=delta_bg,
            delta_bias_vector=np.concatenate([delta_ba, delta_bg]),
            valid=True,
        )

        updated_state, diag = model.update(initial_state, pred)
        assert diag.applied is True

        # Accel bias should be nudged in direction of delta_ba
        assert updated_state.nominal.accel_bias[0] > initial_state.nominal.accel_bias[0]
        # Gyro bias should be nudged in direction of delta_bg
        assert updated_state.nominal.gyro_bias[0] > initial_state.nominal.gyro_bias[0]

        # Position, velocity, and orientation must not be corrupted by bias Jacobians
        # (cross-covariance in P is zero in initial state)
        np.testing.assert_allclose(updated_state.nominal.position_enu, initial_state.nominal.position_enu, atol=1e-6)
        np.testing.assert_allclose(updated_state.nominal.velocity_enu, initial_state.nominal.velocity_enu, atol=1e-6)
        np.testing.assert_allclose(updated_state.nominal.q, initial_state.nominal.q, atol=1e-6)

    def test_frozen_phase8_covariance_preserved(self) -> None:
        model = BiasNetMeasurementModel(BiasNetConfig(covariance_mode="fixed_phase8"))
        np.testing.assert_array_equal(model.R_diag, np.array(PHASE8_R_BIAS_DIAG))


# ==============================================================================
# 6. ML Telemetry Accounting Tests
# ==============================================================================

class TestMLTelemetryAccounting:
    """Test suite ensuring strict mathematical consistency of ML telemetry fields."""

    def test_velocitynet_telemetry_accounting_standstill(self) -> None:
        """Standstill suppression counts as update_rejected with reason STANDSTILL_SUPPRESSED."""
        from navigation.core import NavigationCore, NavigationCoreConfig
        from navigation.ins.attitude import rotation_matrix_to_quaternion

        core = NavigationCore(NavigationCoreConfig(
            velocitynet_enabled=True,
            velocitynet=VelocityNetConfig(min_speed_threshold_mps=15.0),
            biasnet_enabled=False,
            zupt_enabled=False,
            gnss_enabled=False,
        ))

        core.initialize(
            lat0=0.0,
            lon0=0.0,
            alt0=0.0,
            p0_enu=np.zeros(3),
            v0_enu=np.zeros(3),
            q0=np.array([1.0, 0.0, 0.0, 0.0]),
            gyro_bias0=np.zeros(3),
            timestamp_ns=0,
        )

        # Feed 40 stationary samples (0 m/s) at 10 Hz
        for step in range(40):
            t_ns = int((step + 1) * 1e8)
            core.step_imu(
                f_m_v=np.array([0.0, 0.0, 9.81]),
                omega_m_v=np.zeros(3),
                dt_s=0.1,
                timestamp_ns=t_ns,
            )


        telem = core.get_ml_telemetry()["velocitynet"]

        # Telemetry invariants:
        assert telem["scheduler_due"] >= telem["inference_executed"]
        assert telem["inference_executed"] >= telem["update_accepted"]
        assert telem["inference_executed"] == telem["update_accepted"] + telem["update_rejected"]
        assert telem["buffer_not_ready"] > 0
        # In stationary data, updates should be suppressed by standstill logic
        assert telem["update_rejected"] >= 1
        assert "STANDSTILL_SUPPRESSED" in telem["rejection_reasons"]

