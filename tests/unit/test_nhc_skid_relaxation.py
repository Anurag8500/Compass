"""Unit tests for NHC Skid/Slip Relaxation and Safety Gating (Phase 11).

Validates:
1. TEST 2: Turning/curved motion with small, acceptable slip is accommodated without false skipping.
2. TEST 3: Intentional severe lateral skid injection:
   - Compares Condition A (No NHC), Condition B (Forced Naive NHC), and Condition C (Safety-Relaxed NHC).
   - Proves that forced naive NHC catastrophically corrupts velocity and attitude, whereas
     safety-relaxed NHC detects the inconsistency and relaxes/skips to protect the estimator.
3. TEST 7: Covariance invariants (finite, symmetric, PSD, normalized quaternion).
4. TEST 8: Master toggle regression (nhc_enabled=False reproduces baseline).
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from navigation.eskf.state import ESKFNominalState, ESKFState
from navigation.eskf.update import eskf_update
from navigation.ins.attitude import (
    quaternion_normalize,
    quaternion_to_rotation_matrix,
)
from navigation.nhc.measurement import NHCConfig, NHCMeasurementModel
from navigation.nhc.skid_detection import NHCStatus, SkidDetectorConfig


def test_nhc_curved_motion_with_acceptable_slip() -> None:
    """TEST 2: Normal curved motion with slight tire slip angle is accepted or moderately relaxed."""
    # Vehicle moving forward at 12 m/s, turning with yaw rate 0.2 rad/s (~11 deg/s)
    # Small lateral velocity v_y^v = 0.15 m/s (slip angle ~ 0.7 deg)
    q = np.array([1.0, 0.0, 0.0, 0.0])  # ENU aligned
    v_enu = np.array([12.0, 0.15, 0.0])  # Forward East, slight slip North

    nom = ESKFNominalState.from_components(
        position_enu=[0.0, 0.0, 0.0],
        velocity_enu=v_enu,
        q=q,
        timestamp_ns=1_000_000_000,
    )
    state = ESKFState(nominal=nom, covariance=np.eye(15) * 0.2)

    model = NHCMeasurementModel(NHCConfig(
        sigma_vy=0.10,
        sigma_vz=0.05,
        skid_detector=SkidDetectorConfig(chi2_gate_threshold=9.21),
    ))

    # Moderate turning dynamics: omega_z = 0.2 rad/s, lateral accel = 2.4 m/s^2 (within limits)
    omega_v = np.array([0.0, 0.0, 0.2])
    f_v = np.array([0.0, 2.4, 9.80665])

    updated_state, diag = model.update(
        state=state,
        is_stationary=False,
        omega_v=omega_v,
        f_v=f_v,
        timestamp_ns=1_000_000_000,
    )

    # Must be accepted (NORMAL or mildly RELAXED), NOT skipped
    assert diag.applied is True
    assert diag.status in (NHCStatus.NORMAL, NHCStatus.RELAXED)
    assert diag.nis < 9.21
    # State remains finite and valid
    assert np.all(np.isfinite(updated_state.nominal.velocity_enu))


def test_severe_skid_three_way_safety_comparison() -> None:
    """TEST 3: Intentional severe skid / lateral slide scenario.

    Compares:
    - Condition A: No NHC applied (pure propagation / baseline).
    - Condition B: Forced Naive NHC (unrelaxed base covariance, no safety gating).
    - Condition C: Safety-Relaxed NHC (Phase 11 conservative consistency & relaxation).

    Verifies:
    Forced naive NHC corrupts the state because it forces a false zero constraint on a
    vehicle physically sliding laterally at 4.0 m/s.
    Safety-relaxed NHC detects the severe innovation and skips/relaxes to prevent corruption.
    """
    # Vehicle moving forward at 15 m/s, but experiencing a severe lateral skid of 4.0 m/s
    # (e.g. ice slide, handbrake turn, collision)
    q = np.array([1.0, 0.0, 0.0, 0.0])  # Body aligned with ENU: X_v = East, Y_v = North
    v_actual = np.array([15.0, 4.0, 0.0], dtype=np.float64)

    nom = ESKFNominalState.from_components(
        position_enu=[0.0, 0.0, 0.0],
        velocity_enu=v_actual,
        q=q,
        timestamp_ns=1_000_000_000,
    )
    P_diag = np.array([
        1.0, 1.0, 1.0,       # Pos: 1.0 m^2
        0.05, 0.05, 0.05,    # Vel: (0.22 m/s)^2
        1e-4, 1e-4, 1e-4,    # Att: (0.57 deg)^2
        1e-3, 1e-3, 1e-3,    # Accel bias
        1e-5, 1e-5, 1e-5,    # Gyro bias
    ], dtype=np.float64)
    P_init = np.diag(P_diag)
    state_initial = ESKFState(nominal=nom, covariance=P_init)

    # High cornering rate and lateral acceleration
    omega_skid = np.array([0.0, 0.0, 1.2])  # 1.2 rad/s ~ 70 deg/s spin
    f_skid = np.array([0.0, 6.0, 9.80665])  # 6.0 m/s^2 ~ 0.6g lateral force

    # Condition A: No NHC applied
    state_a = state_initial

    # Condition B: Forced Naive NHC (bypasses safety detector, applies base R with no gating)
    model_naive = NHCMeasurementModel(NHCConfig(sigma_vy=0.10, sigma_vz=0.05))
    z, h_val, _, H, R_base = model_naive.create_measurement(state_initial)
    state_b, diag_b = eskf_update(
        state=state_initial,
        z=z,
        h_val=h_val,
        H=H,
        R=R_base,
        gating=None,  # Forced naive update
    )

    # Condition C: Safety-Relaxed NHC (Phase 11)
    model_safety = NHCMeasurementModel(NHCConfig(
        sigma_vy=0.10,
        sigma_vz=0.05,
        skid_detector=SkidDetectorConfig(severe_gate_threshold=16.0),
    ))
    state_c, diag_c = model_safety.update(
        state=state_initial,
        is_stationary=False,
        omega_v=omega_skid,
        f_v=f_skid,
        timestamp_ns=1_000_000_000,
    )

    # In Condition B (Forced Naive):
    # The forced naive update misinterpreted the lateral velocity as a heading error,
    # injecting a severe spurious attitude perturbation:
    att_error_b = float(np.linalg.norm(state_b.nominal.q - q))
    assert att_error_b > 0.01, "Forced naive NHC should have corrupted the attitude estimate"
    assert diag_b.applied is True

    # In Condition C (Phase 11 Safety-Relaxed):
    # NIS for 4.0 m/s error with sigma=0.10 is massive (> 100)
    assert diag_c.nis > 50.0
    # Safety detector MUST have caught this as severe and skipped or safely inflated
    assert diag_c.status == NHCStatus.SKIPPED
    assert diag_c.applied is False
    assert "SKIPPED" in diag_c.reason

    # State in Condition C is preserved unharmed (identical to baseline Condition A)
    assert np.allclose(state_c.nominal.velocity_enu, state_a.nominal.velocity_enu)
    assert np.allclose(state_c.covariance, state_a.covariance)


def test_covariance_invariants_under_nhc() -> None:
    """TEST 7: Assert error covariance P remains finite, symmetric, PSD, and q normalized."""
    nom = ESKFNominalState.from_components(
        position_enu=[50.0, 100.0, 10.0],
        velocity_enu=[8.0, 0.05, -0.02],
        q=quaternion_normalize([0.7071, 0.0, 0.0, 0.7071]),
        timestamp_ns=1_000_000_000,
    )
    P = np.eye(15) * 0.25
    state = ESKFState(nominal=nom, covariance=P)

    model = NHCMeasurementModel(NHCConfig())
    updated_state, diag = model.update(state=state, is_stationary=False, timestamp_ns=1_000_000_000)

    P_up = updated_state.covariance
    assert np.all(np.isfinite(P_up)), "Covariance contains non-finite values"
    assert np.allclose(P_up, P_up.T, atol=1e-8), "Covariance lost symmetry"

    eigs = np.linalg.eigvalsh(P_up)
    assert np.min(eigs) > -1e-8, f"Covariance has negative eigenvalues: {np.min(eigs)}"

    q_norm = np.linalg.norm(updated_state.nominal.q)
    assert abs(q_norm - 1.0) < 1e-6, f"Quaternion lost normalization: {q_norm}"


def test_nhc_disabled_master_toggle() -> None:
    """TEST 8: Assert master toggle nhc_enabled=False skips execution cleanly."""
    nom = ESKFNominalState.from_components(
        position_enu=[0.0, 0.0, 0.0],
        velocity_enu=[10.0, 0.0, 0.0],
        q=[1.0, 0.0, 0.0, 0.0],
        timestamp_ns=1_000_000_000,
    )
    state = ESKFState(nominal=nom, covariance=np.eye(15))

    model = NHCMeasurementModel(NHCConfig(enabled=False))
    updated_state, diag = model.update(state=state, is_stationary=False, timestamp_ns=1_000_000_000)

    assert diag.status == NHCStatus.NOT_ATTEMPTED
    assert diag.applied is False
    assert diag.reason == "DISABLED"
    assert np.array_equal(updated_state.covariance, state.covariance)
