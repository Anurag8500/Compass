"""Unit tests for Non-Holonomic Constraints (NHC) in Straight and Rotated Driving (Phase 11).

Validates:
1. TEST 1: Straight driving where v_y^v ~ 0 and v_z^v ~ 0 -> innovation is near zero,
   normally accepted, negligible correction, no false relaxation.
2. TEST 1b: Rotated vehicle attitude (non-identity R_v^n) -> verifies attitude transformation
   v^v = (R_v^n)^T @ v^n accurately resolves body frame velocity.
3. TEST 6 (HARD GATE): Numerical verification of analytical measurement Jacobian H_nhc
   against central finite differences evaluated via state.inject_error() across all 15 states.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from navigation.eskf.state import ESKFNominalState, ESKFState
from navigation.ins.attitude import (
    quaternion_normalize,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)
from navigation.nhc.measurement import NHCConfig, NHCMeasurementModel
from navigation.nhc.skid_detection import NHCStatus


def test_analytical_jacobian_vs_central_finite_differences() -> None:
    """TEST 6 (HARD GATE): Verify analytical H_nhc against central finite differences.

    Uses the repository's exact right-multiplicative body-frame attitude error convention:
        q_true = q_nom ⊗ delta_q(delta_theta^v)
    via state.inject_error(delta_x).
    """
    # Create an arbitrary, realistic nominal state with nontrivial rotation and 3D velocity
    # Heading ~ 38 deg, pitch ~ -11 deg, roll ~ 7 deg
    q_nom = quaternion_normalize(np.array([0.92387953, 0.05, -0.08, 0.37]))
    v_enu = np.array([14.2, -6.8, 1.5], dtype=np.float64)  # East, North, Up

    nom = ESKFNominalState.from_components(
        position_enu=[120.0, -45.0, 10.0],
        velocity_enu=v_enu,
        q=q_nom,
        accel_bias=[0.02, -0.01, 0.04],
        gyro_bias=[0.001, -0.002, 0.001],
        timestamp_ns=1_000_000_000,
    )
    cov = np.eye(15, dtype=np.float64) * 0.1
    state = ESKFState(nominal=nom, covariance=cov)

    model = NHCMeasurementModel(NHCConfig())
    z, h_val, v_v, H_analytical, R_base = model.create_measurement(state)

    assert H_analytical.shape == (2, 15), f"Jacobian shape mismatch: {H_analytical.shape}"

    # Central finite differences step size
    eps = 1e-7
    H_numerical = np.zeros((2, 15), dtype=np.float64)

    for i in range(15):
        # Forward perturbation
        dx_pos = np.zeros(15, dtype=np.float64)
        dx_pos[i] = eps
        state_pos_nom = nom.inject_error(dx_pos)
        v_pos = state_pos_nom.R_v_n.T @ state_pos_nom.velocity_enu
        h_pos = v_pos[1:3]

        # Backward perturbation
        dx_neg = np.zeros(15, dtype=np.float64)
        dx_neg[i] = -eps
        state_neg_nom = nom.inject_error(dx_neg)
        v_neg = state_neg_nom.R_v_n.T @ state_neg_nom.velocity_enu
        h_neg = v_neg[1:3]

        # Central difference quotient
        H_numerical[:, i] = (h_pos - h_neg) / (2.0 * eps)

    # Validate parity across all 15 states
    diff = np.abs(H_analytical - H_numerical)
    max_err = float(np.max(diff))

    # Error must be well within standard numerical differentiation tolerance (O(eps^2) ~ 1e-14 / 1e-7)
    assert max_err < 1e-5, (
        f"HARD GATE FAILURE: Analytical H_nhc does not match central finite differences!\n"
        f"Max absolute error: {max_err:.3e}\n"
        f"Analytical:\n{H_analytical[:, 3:9]}\n"
        f"Numerical:\n{H_numerical[:, 3:9]}"
    )

    # Position, accel bias, and gyro bias columns must be strictly zero
    assert np.all(H_analytical[:, 0:3] == 0.0), "Position block in H_nhc must be zero"
    assert np.all(H_analytical[:, 9:12] == 0.0), "Accel bias block in H_nhc must be zero"
    assert np.all(H_analytical[:, 12:15] == 0.0), "Gyro bias block in H_nhc must be zero"


def test_nhc_straight_driving_acceptance() -> None:
    """TEST 1: Straight driving with negligible lateral and vertical velocity.

    Asserts:
    - Innovation is near zero.
    - Status is NORMAL.
    - Update is accepted.
    - Covariance is reduced along lateral/vertical directions.
    - State is not corrupted.
    """
    # Moving due North at 10 m/s with vehicle aligned with ENU North
    # Forward is X_v -> aligned with North (+Y_n)
    # Lateral is Y_v -> aligned with West (-X_n)
    # Up is Z_v -> aligned with Up (+Z_n)
    R_v_n = np.array([
        [0.0, -1.0, 0.0],  # X_n (East) = -Y_v
        [1.0,  0.0, 0.0],  # Y_n (North) = X_v
        [0.0,  0.0, 1.0],  # Z_n (Up) = Z_v
    ], dtype=np.float64)
    q = rotation_matrix_to_quaternion(R_v_n)

    # Pure forward velocity: v_x^v = 10 m/s -> v^n = R_v^n @ [10, 0, 0]^T = [0, 10, 0]^T
    v_enu = np.array([0.0, 10.0, 0.0], dtype=np.float64)

    nom = ESKFNominalState.from_components(
        position_enu=[0.0, 0.0, 0.0],
        velocity_enu=v_enu,
        q=q,
        accel_bias=np.zeros(3),
        gyro_bias=np.zeros(3),
        timestamp_ns=1_000_000_000,
    )
    P_init = np.eye(15, dtype=np.float64) * 1.0
    state = ESKFState(nominal=nom, covariance=P_init)

    model = NHCMeasurementModel(NHCConfig(sigma_vy=0.1, sigma_vz=0.05))
    updated_state, diag = model.update(
        state=state,
        is_stationary=False,
        omega_v=np.zeros(3),
        f_v=np.array([0.0, 0.0, 9.80665]),
        timestamp_ns=1_000_000_000,
    )

    assert diag.status == NHCStatus.NORMAL
    assert diag.applied is True
    assert diag.reason == "ACCEPTED_NORMAL"
    assert np.allclose(diag.innovation, [0.0, 0.0], atol=1e-6)
    assert diag.nis < 0.01

    # Velocity must remain virtually identical
    assert np.allclose(updated_state.nominal.velocity_enu, v_enu, atol=1e-5)

    # Covariance for lateral velocity (East component) must have decreased
    # In this orientation, Y_v is aligned with -East, so East velocity variance is index 3
    assert updated_state.covariance[3, 3] < P_init[3, 3]
    # Up velocity variance is index 5
    assert updated_state.covariance[5, 5] < P_init[5, 5]


def test_nhc_rotated_arbitrary_attitude() -> None:
    """TEST 1b: Verify vehicle-frame transformation with arbitrary 3D rotation."""
    # Yaw = 120 deg, pitch = 20 deg, roll = 5 deg
    q = quaternion_normalize(np.array([0.819, 0.035, 0.174, 0.544]))
    R_v_n = quaternion_to_rotation_matrix(q)

    # Set pure body-forward velocity: v^v = [15.0, 0.0, 0.0]^T
    v_body = np.array([15.0, 0.0, 0.0], dtype=np.float64)
    v_enu = R_v_n @ v_body

    nom = ESKFNominalState.from_components(
        position_enu=[10.0, 20.0, 30.0],
        velocity_enu=v_enu,
        q=q,
        accel_bias=np.zeros(3),
        gyro_bias=np.zeros(3),
        timestamp_ns=2_000_000_000,
    )
    state = ESKFState(nominal=nom, covariance=np.eye(15) * 0.5)

    model = NHCMeasurementModel(NHCConfig())
    _, h_val, v_v, _, _ = model.create_measurement(state)

    # Predicted body velocity must recover [15.0, 0.0, 0.0]
    assert np.allclose(v_v, v_body, atol=1e-6)
    assert np.allclose(h_val, [0.0, 0.0], atol=1e-6)
