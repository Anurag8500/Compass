"""Mandatory Filter Authority and Innovation Gating Safety Tests (Phase 9).

Per SIH 26168 Section 15:
The ESKF remains the ONLY authoritative navigation state estimator.
Neural models must NEVER directly overwrite:
    position, velocity, quaternion, accelerometer bias, or gyroscope bias.

This integration test proves this authority invariant executably:
1. Feeds deliberately absurd VelocityNet predictions (e.g. 1000 m/s, negative extreme).
2. Feeds deliberately absurd BiasNet predictions (e.g. 50 m/s^2, 10 rad/s).
3. Verifies that the generic Mahalanobis innovation gate strictly REJECTS the measurement.
4. Verifies that the nominal state vector is completely UNCHANGED.
5. Verifies that the error-state covariance matrix P remains finite, symmetric, and PSD.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from navigation.eskf.gating import MahalanobisGating
from navigation.eskf.state import ESKFNominalState, ESKFState
from navigation.eskf.measurements.velocitynet import (
    VelocityNetConfig,
    VelocityNetMeasurementModel,
)
from navigation.eskf.measurements.biasnet import (
    BiasNetConfig,
    BiasNetMeasurementModel,
)
from navigation.ml.model_runner import BiasNetOutput, VelocityNetOutput


@pytest.fixture
def baseline_state() -> ESKFState:
    """A realistic, healthy ESKF state cruising at 15 m/s."""
    nom = ESKFNominalState.from_components(
        position_enu=[125.0, 340.0, 15.0],
        velocity_enu=[15.0, 0.0, 0.0],
        q=[1.0, 0.0, 0.0, 0.0],
        accel_bias=[0.02, -0.01, 0.05],
        gyro_bias=[0.0005, -0.0002, 0.0001],
        timestamp_ns=1_000_000_000,
    )
    # Realistic covariance
    P = np.diag([
        2.0, 2.0, 5.0,           # Pos (m^2)
        0.2, 0.2, 0.5,           # Vel (m/s)^2
        0.001, 0.001, 0.005,     # Att (rad^2)
        0.01, 0.01, 0.01,        # Accel bias (m/s^2)^2
        0.0001, 0.0001, 0.0001,  # Gyro bias (rad/s)^2
    ])
    return ESKFState(nominal=nom, covariance=P)


def test_velocitynet_absurd_speed_rejection(baseline_state: ESKFState) -> None:
    """Proves an extreme VelocityNet speed is gated out and does not corrupt state."""
    model = VelocityNetMeasurementModel(VelocityNetConfig(gate_threshold=16.0, use_causal_ema=False))
    
    # Absurd forward speed: 1000.0 m/s (Mach ~3)
    absurd_speed = 1000.0
    pred = VelocityNetOutput(
        speed_mps=absurd_speed,
        log_variance=0.0,
        variance=1.0,
        valid=True,
    )

    updated_state, diag = model.update(baseline_state, pred, timestamp_ns=1_000_000_000)

    # 1. Update must be rejected
    assert diag.applied is False
    assert diag.reason == "GATE_REJECTED"
    assert diag.update_diagnostics is not None
    assert diag.update_diagnostics.applied is False
    assert diag.update_diagnostics.gating is not None
    assert diag.update_diagnostics.gating.accepted is False
    assert diag.update_diagnostics.gating.mahalanobis_sq > 16.0

    # 2. State must be strictly identical to baseline state
    np.testing.assert_array_equal(updated_state.nominal.position_enu, baseline_state.nominal.position_enu)
    np.testing.assert_array_equal(updated_state.nominal.velocity_enu, baseline_state.nominal.velocity_enu)
    np.testing.assert_array_equal(updated_state.nominal.q, baseline_state.nominal.q)
    np.testing.assert_array_equal(updated_state.nominal.accel_bias, baseline_state.nominal.accel_bias)
    np.testing.assert_array_equal(updated_state.nominal.gyro_bias, baseline_state.nominal.gyro_bias)

    # 3. Covariance must be strictly unmodified
    np.testing.assert_array_equal(updated_state.covariance, baseline_state.covariance)


def test_biasnet_absurd_bias_rejection(baseline_state: ESKFState) -> None:
    """Proves an extreme BiasNet correction is gated out and does not corrupt state."""
    model = BiasNetMeasurementModel(BiasNetConfig(gate_threshold=25.0))

    # Absurd bias correction: 50 m/s^2 accel bias and 5 rad/s gyro bias
    absurd_ba = np.array([50.0, -40.0, 30.0], dtype=np.float64)
    absurd_bg = np.array([5.0, -3.0, 4.0], dtype=np.float64)
    pred = BiasNetOutput(
        delta_accel_bias=absurd_ba,
        delta_gyro_bias=absurd_bg,
        delta_bias_vector=np.concatenate([absurd_ba, absurd_bg]),
        valid=True,
    )

    updated_state, diag = model.update(baseline_state, pred, timestamp_ns=1_000_000_000)

    # 1. Update must be rejected
    assert diag.applied is False
    assert diag.reason == "GATE_REJECTED"
    assert diag.update_diagnostics is not None
    assert diag.update_diagnostics.applied is False
    assert diag.update_diagnostics.gating is not None
    assert diag.update_diagnostics.gating.accepted is False
    assert diag.update_diagnostics.gating.mahalanobis_sq > 25.0

    # 2. State must be strictly identical to baseline state
    np.testing.assert_array_equal(updated_state.nominal.position_enu, baseline_state.nominal.position_enu)
    np.testing.assert_array_equal(updated_state.nominal.velocity_enu, baseline_state.nominal.velocity_enu)
    np.testing.assert_array_equal(updated_state.nominal.q, baseline_state.nominal.q)
    np.testing.assert_array_equal(updated_state.nominal.accel_bias, baseline_state.nominal.accel_bias)
    np.testing.assert_array_equal(updated_state.nominal.gyro_bias, baseline_state.nominal.gyro_bias)

    # 3. Covariance must be strictly unmodified
    np.testing.assert_array_equal(updated_state.covariance, baseline_state.covariance)


def test_no_direct_state_overwrite_on_valid_update(baseline_state: ESKFState) -> None:
    """Proves that even when an ML update is ACCEPTED, it does NOT overwrite state directly.
    
    The state correction must be delta_x = K * y, not x_new = z.
    """
    model = VelocityNetMeasurementModel(VelocityNetConfig(gate_threshold=16.0, use_causal_ema=False))
    
    # Speed slightly higher than current 15.0 m/s: 16.0 m/s (innovation = 1.0 m/s)
    z_speed = 16.0
    pred = VelocityNetOutput(
        speed_mps=z_speed,
        log_variance=0.0,
        variance=2.25,
        valid=True,
    )

    updated_state, diag = model.update(baseline_state, pred)
    assert diag.applied is True

    # Nominal velocity must NOT directly equal z_speed (16.0 m/s)
    # It must be nudged proportionally by Kalman gain K (e.g. between 15.0 and 16.0)
    new_vx = updated_state.nominal.velocity_enu[0]
    assert 15.0 < new_vx < 16.0
    assert abs(new_vx - 16.0) > 0.05, "Velocity was directly overwritten rather than Kalman filtered!"
