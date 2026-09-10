"""Unit tests for Gated ZUPT and Standstill Fusion Integration (Phase 11).

Validates:
1. TEST 4: Stop-and-go scenario (moving -> stop -> stationary -> moving):
   - ZUPT activates only during confirmed standstill.
   - ZUPT clamps velocity toward zero.
   - Clean transition out of stop: stationary flag resets immediately upon motion.
2. TEST 5: Low-dynamic moving scenario:
   - Low-speed motion with normal sensor noise must NEVER falsely trigger ZUPT.
3. Handshake between NHC and ZUPT:
   - When stationary, NHC is skipped (SKIPPED_STATIONARY) allowing ZUPT's 3D constraint
     to authoritatively govern the standstill.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from navigation.eskf.measurements.zupt import (
    ClassicalZUPTDetector,
    ZUPTDetectorConfig,
    ZUPTMeasurementConfig,
    ZUPTMeasurementModel,
)
from navigation.eskf.state import ESKFNominalState, ESKFState
from navigation.nhc.measurement import NHCConfig, NHCMeasurementModel
from navigation.nhc.skid_detection import NHCStatus


def test_zupt_stop_and_go_lifecycle() -> None:
    """TEST 4: moving -> stop -> stationary -> moving lifecycle."""
    detector = ClassicalZUPTDetector(ZUPTDetectorConfig(window_size=5))
    zupt_model = ZUPTMeasurementModel(ZUPTMeasurementConfig(velocity_noise_sigma=0.02))

    nom = ESKFNominalState.from_components(
        position_enu=[0.0, 0.0, 0.0],
        velocity_enu=[5.0, 0.0, 0.0],  # Moving at 5 m/s
        q=[1.0, 0.0, 0.0, 0.0],
        timestamp_ns=0,
    )
    state = ESKFState(nominal=nom, covariance=np.eye(15) * 0.1)

    # 1. Moving phase (5 samples of forward motion, slight vibrations)
    for i in range(5):
        diag_mov = detector.push(
            omega_v=np.array([0.01, -0.01, 0.02]),
            f_v=np.array([0.5, 0.1, 9.80665]),
            speed_mps=5.0,
        )
        assert diag_mov.is_stationary is False

    # 2. Transition to stop: vehicle decelerates to zero
    # Provide 5 quiescent standstill samples (omega ~ 0, f ~ g)
    for i in range(5):
        diag_stop = detector.push(
            omega_v=np.array([0.001, 0.001, 0.001]),
            f_v=np.array([0.0, 0.0, 9.80665]),
            speed_mps=0.0,
        )

    # Standstill must be confirmed
    assert diag_stop.is_stationary is True

    # 3. Apply ZUPT update
    # Injected residual velocity of 0.3 m/s before ZUPT
    nom_drifted = ESKFNominalState.from_components(
        position_enu=state.nominal.position_enu,
        velocity_enu=[0.30, -0.15, 0.05],
        q=state.nominal.q,
        timestamp_ns=1_000_000_000,
    )
    state_drifted = ESKFState(nominal=nom_drifted, covariance=state.covariance)

    v_before = float(np.linalg.norm(state_drifted.nominal.velocity_enu))
    state_zupted, update_diag = zupt_model.update(state_drifted, timestamp_ns=1_000_000_000)

    assert update_diag.applied is True
    v_after = float(np.linalg.norm(state_zupted.nominal.velocity_enu))
    # Velocity must be significantly reduced toward zero
    assert v_after < 0.05
    assert v_after < v_before

    # 4. Resume motion: push one sample with nonzero angular velocity or acceleration
    diag_resume = detector.push(
        omega_v=np.array([0.2, 0.0, 0.0]),
        f_v=np.array([2.0, 0.0, 9.80665]),
        speed_mps=2.0,
    )
    # Must immediately exit standstill
    assert diag_resume.is_stationary is False


def test_zupt_low_dynamic_no_false_trigger() -> None:
    """TEST 5: Low-dynamic moving scenario must NOT falsely trigger ZUPT."""
    detector = ClassicalZUPTDetector(ZUPTDetectorConfig(
        window_size=8,
        speed_max_mps=0.10,
        accel_dev_max_mps2=0.25,
        omega_max_rads=0.05,
    ))

    # Vehicle creeping at 0.5 m/s (above 0.1 m/s threshold) with low noise
    for i in range(15):
        diag = detector.push(
            omega_v=np.array([0.005, -0.005, 0.005]),
            f_v=np.array([0.02, 0.01, 9.80665]),
            speed_mps=0.50,
        )
        assert diag.is_stationary is False, f"False ZUPT trigger on moving vehicle at step {i}"


def test_nhc_zupt_standstill_handshake() -> None:
    """Assert NHC is cleanly skipped during standstill to let ZUPT govern 3D velocity."""
    nhc_model = NHCMeasurementModel(NHCConfig())
    nom = ESKFNominalState.from_components(
        position_enu=[0.0, 0.0, 0.0],
        velocity_enu=[0.05, 0.02, 0.0],
        q=[1.0, 0.0, 0.0, 0.0],
        timestamp_ns=1_000_000_000,
    )
    state = ESKFState(nominal=nom, covariance=np.eye(15) * 0.05)

    # When is_stationary is True, NHC must return SKIPPED_STATIONARY without applying an update
    updated_state, nhc_diag = nhc_model.update(
        state=state,
        is_stationary=True,
        timestamp_ns=1_000_000_000,
    )

    assert nhc_diag.status == NHCStatus.SKIPPED_STATIONARY
    assert nhc_diag.applied is False
    assert nhc_diag.reason == "SKIPPED_STATIONARY"
    # Covariance and nominal state remain completely untouched
    assert np.array_equal(updated_state.covariance, state.covariance)
    assert np.array_equal(updated_state.nominal.velocity_enu, state.nominal.velocity_enu)
