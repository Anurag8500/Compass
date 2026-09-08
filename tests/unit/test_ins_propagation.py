"""Mandatory synthetic unit tests for attitude and strapdown INS propagation (Phase 4).

Validates closed-form known answers:
- Test A: Constant forward acceleration (closed-form velocity and position)
- Test B: Stationary gravity cancellation (zero velocity and position drift)
- Test C: Constant angular velocity (exact analytical rotation angle)
- Test D: 90-degree known rotation (axis, sign, and composition order)
- Test E: Coupled rotation + acceleration (time-varying ENU acceleration rotation)
- Test F: Quaternion stability (10,000 iterations maintain unit norm and orthogonality)
- Test H: Input validation and timestep bounds
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from navigation.ins.attitude import (
    delta_quaternion,
    propagate_attitude,
    quaternion_conjugate,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_norm,
    quaternion_normalize,
    quaternion_to_euler_deg,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)
from navigation.ins.propagation import (
    INSState,
    STANDARD_GRAVITY,
    StrapdownINS,
)


class TestAttitudeAndQuaternions:
    """Unit tests for quaternion algebra, conversions, and attitude propagation."""

    def test_identity_quaternion(self) -> None:
        """Identity quaternion [1, 0, 0, 0] must yield 3x3 identity rotation matrix."""
        q_id = np.array([1.0, 0.0, 0.0, 0.0])
        R = quaternion_to_rotation_matrix(q_id)
        assert np.allclose(R, np.eye(3), atol=1e-15)

        q_rec = rotation_matrix_to_quaternion(np.eye(3))
        assert np.allclose(q_rec, q_id, atol=1e-15)

        roll, pitch, yaw = quaternion_to_euler_deg(q_id)
        assert roll == pytest.approx(0.0, abs=1e-9)
        assert pitch == pytest.approx(0.0, abs=1e-9)
        assert yaw == pytest.approx(0.0, abs=1e-9)

    def test_zero_angular_velocity(self) -> None:
        """Zero angular velocity over any timestep must yield identity delta quaternion."""
        dq = delta_quaternion([0.0, 0.0, 0.0], dt=0.1)
        assert np.allclose(dq, [1.0, 0.0, 0.0, 0.0], atol=1e-15)

        q_init = np.array([1.0, 0.0, 0.0, 0.0])
        q_next = propagate_attitude(q_init, [0.0, 0.0, 0.0], dt=0.05)
        assert np.allclose(q_next, q_init, atol=1e-15)

    def test_c_constant_angular_velocity(self) -> None:
        """TEST C: Constant angular rate for known duration produces exact analytical rotation."""
        omega_z = 0.5  # rad/s (approx 28.6 deg/s)
        duration_s = 2.0
        n_steps = 200
        dt = duration_s / n_steps

        q = np.array([1.0, 0.0, 0.0, 0.0])
        for _ in range(n_steps):
            q = propagate_attitude(q, [0.0, 0.0, omega_z], dt)

        expected_theta = omega_z * duration_s  # 1.0 radian
        expected_q = np.array([math.cos(expected_theta / 2.0), 0.0, 0.0, math.sin(expected_theta / 2.0)])

        assert np.allclose(q, expected_q, atol=1e-10)
        assert quaternion_norm(q) == pytest.approx(1.0, abs=1e-12)

        # Check Euler yaw angle
        _, _, yaw_deg = quaternion_to_euler_deg(q)
        assert yaw_deg == pytest.approx(math.degrees(expected_theta), abs=1e-6)

    def test_d_known_90deg_rotation(self) -> None:
        """TEST D: 90-degree yaw rotation maps vehicle Forward (+X_v) to North (+Y_n)."""
        # Exactly +90 degrees about Z: theta = pi/2
        theta = math.pi / 2.0
        q_90z = np.array([math.cos(theta / 2.0), 0.0, 0.0, math.sin(theta / 2.0)])

        R = quaternion_to_rotation_matrix(q_90z)

        # In vehicle frame, forward is +X_v = [1, 0, 0]
        v_forward = np.array([1.0, 0.0, 0.0])
        v_enu = R @ v_forward

        # After +90 deg rotation, vehicle forward points East -> North (+Y_n = [0, 1, 0])
        assert v_enu[0] == pytest.approx(0.0, abs=1e-12)  # East
        assert v_enu[1] == pytest.approx(1.0, abs=1e-12)  # North
        assert v_enu[2] == pytest.approx(0.0, abs=1e-12)  # Up

        # In vehicle frame, lateral left is +Y_v = [0, 1, 0]
        v_left = np.array([0.0, 1.0, 0.0])
        v_left_enu = R @ v_left
        # Lateral left points North -> West (-X_n = [-1, 0, 0])
        assert v_left_enu[0] == pytest.approx(-1.0, abs=1e-12)
        assert v_left_enu[1] == pytest.approx(0.0, abs=1e-12)
        assert v_left_enu[2] == pytest.approx(0.0, abs=1e-12)

        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-14)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-14)

    def test_quaternion_inverse_and_conjugate(self) -> None:
        """Verify quaternion inverse and conjugate properties."""
        q = quaternion_normalize(np.array([0.5, 0.5, 0.5, 0.5]))
        q_inv = quaternion_inverse(q)
        q_conj = quaternion_conjugate(q)

        assert np.allclose(q_inv, q_conj, atol=1e-15)

        # q ⊗ q^(-1) = [1, 0, 0, 0]
        prod = quaternion_multiply(q, q_inv)
        assert np.allclose(prod, [1.0, 0.0, 0.0, 0.0], atol=1e-15)

    def test_f_quaternion_stability(self) -> None:
        """TEST F: 10,000 repeated small 3D rotations maintain unit norm and matrix orthogonality."""
        np.random.seed(42)
        q = np.array([1.0, 0.0, 0.0, 0.0])

        for _ in range(10_000):
            # Random angular rate up to 1 rad/s
            omega = (np.random.rand(3) - 0.5) * 2.0
            dt = 0.01
            q = propagate_attitude(q, omega, dt)

        # Check finiteness and strict unit norm
        assert np.isfinite(q).all()
        assert quaternion_norm(q) == pytest.approx(1.0, abs=1e-12)

        R = quaternion_to_rotation_matrix(q)
        assert np.isfinite(R).all()
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)


class TestStrapdownINSPropagation:
    """Unit tests for deterministic strapdown INS propagation in local ENU frame."""

    def test_b_stationary_gravity_cancellation(self) -> None:
        """TEST B: Stationary vehicle level on horizontal surface experiences zero coordinate acceleration.

        f_m^v = [0, 0, +g]
        b_a^v = [0, 0, 0]
        g^n   = [0, 0, -g]
        => a_true^n = [0, 0, 0] identically. Position and velocity remain zero.
        """
        ins = StrapdownINS(
            initial_position=[0.0, 0.0, 0.0],
            initial_velocity=[0.0, 0.0, 0.0],
            initial_q=[1.0, 0.0, 0.0, 0.0],
            gravity_magnitude=STANDARD_GRAVITY,
        )

        dt = 0.1  # 10 Hz
        f_stationary = np.array([0.0, 0.0, STANDARD_GRAVITY])
        w_stationary = np.array([0.0, 0.0, 0.0])

        # Propagate for 1,000 steps (100 seconds)
        for k in range(1, 1001):
            state = ins.step(
                f_m_v=f_stationary,
                omega_m_v=w_stationary,
                dt=dt,
                timestamp_ns=k * 100_000_000,
            )

        assert np.allclose(state.position_enu, [0.0, 0.0, 0.0], atol=1e-12)
        assert np.allclose(state.velocity_enu, [0.0, 0.0, 0.0], atol=1e-12)
        assert np.allclose(state.coordinate_accel_enu, [0.0, 0.0, 0.0], atol=1e-12)
        assert np.allclose(state.q, [1.0, 0.0, 0.0, 0.0], atol=1e-12)

    def test_a_constant_acceleration_closed_form(self) -> None:
        """TEST A: Known constant forward acceleration yields exact closed-form velocity and position.

        Vehicle level (R_v^n = I). Forward axis is +X_v (East in ENU).
        f_m^v = [a_const, 0, +g]
        a_true^n = [a_const, 0, 0]
        Exact:
            v_E(T) = a_const * T
            p_E(T) = 0.5 * a_const * T^2
        """
        a_const = 2.0  # m/s^2 forward acceleration
        duration_s = 10.0
        dt = 0.05  # 20 Hz
        n_steps = int(duration_s / dt)

        ins = StrapdownINS(
            initial_position=[0.0, 0.0, 0.0],
            initial_velocity=[0.0, 0.0, 0.0],
            initial_q=[1.0, 0.0, 0.0, 0.0],
            gravity_magnitude=STANDARD_GRAVITY,
        )

        f_accel = np.array([a_const, 0.0, STANDARD_GRAVITY])
        w_zero = np.array([0.0, 0.0, 0.0])

        for k in range(1, n_steps + 1):
            state = ins.step(
                f_m_v=f_accel,
                omega_m_v=w_zero,
                dt=dt,
                timestamp_ns=int(k * dt * 1e9),
            )

        expected_v_east = a_const * duration_s             # 20.0 m/s
        expected_p_east = 0.5 * a_const * (duration_s ** 2)  # 100.0 m

        assert state.velocity_enu[0] == pytest.approx(expected_v_east, rel=1e-5)
        assert state.position_enu[0] == pytest.approx(expected_p_east, rel=1e-5)

        # Cross-axes must remain zero
        assert state.velocity_enu[1] == pytest.approx(0.0, abs=1e-10)
        assert state.velocity_enu[2] == pytest.approx(0.0, abs=1e-10)
        assert state.position_enu[1] == pytest.approx(0.0, abs=1e-10)
        assert state.position_enu[2] == pytest.approx(0.0, abs=1e-10)

    def test_e_coupled_rotation_and_acceleration(self) -> None:
        """TEST E: Coupled rotation + acceleration verifies frame-order and body-to-ENU transformation.

        Vehicle begins facing East (q_0 = [1,0,0,0]).
        Yaw rate: omega_z = 0.1 rad/s.
        Vehicle forward acceleration: f_x^v = 1.0 m/s^2.
        As vehicle turns:
            a_E(t) = a_0 * cos(omega_z * t)
            a_N(t) = a_0 * sin(omega_z * t)
        Exact integrals:
            v_E(T) = (a_0 / omega_z) * sin(omega_z * T)
            v_N(T) = (a_0 / omega_z) * (1 - cos(omega_z * T))
            p_E(T) = (a_0 / omega_z^2) * (1 - cos(omega_z * T))
            p_N(T) = (a_0 / omega_z) * T - (a_0 / omega_z^2) * sin(omega_z * T)
        """
        a_0 = 1.0       # m/s^2
        omega_z = 0.1   # rad/s
        duration_s = 5.0
        dt = 0.001      # High-rate discrete step to test numerical integration convergence
        n_steps = int(duration_s / dt)

        ins = StrapdownINS(
            initial_position=[0.0, 0.0, 0.0],
            initial_velocity=[0.0, 0.0, 0.0],
            initial_q=[1.0, 0.0, 0.0, 0.0],
            gravity_magnitude=STANDARD_GRAVITY,
        )

        f_coupled = np.array([a_0, 0.0, STANDARD_GRAVITY])
        w_coupled = np.array([0.0, 0.0, omega_z])

        for k in range(1, n_steps + 1):
            state = ins.step(
                f_m_v=f_coupled,
                omega_m_v=w_coupled,
                dt=dt,
                timestamp_ns=int(k * dt * 1e9),
            )

        # Analytical answers
        T = duration_s
        expected_v_E = (a_0 / omega_z) * math.sin(omega_z * T)
        expected_v_N = (a_0 / omega_z) * (1.0 - math.cos(omega_z * T))
        expected_p_E = (a_0 / (omega_z ** 2)) * (1.0 - math.cos(omega_z * T))
        expected_p_N = (a_0 / omega_z) * T - (a_0 / (omega_z ** 2)) * math.sin(omega_z * T)

        # Agreement within 0.1% due to discrete 1st-order step
        assert state.velocity_enu[0] == pytest.approx(expected_v_E, rel=1e-3)
        assert state.velocity_enu[1] == pytest.approx(expected_v_N, rel=1e-3)
        assert state.position_enu[0] == pytest.approx(expected_p_E, rel=1e-3)
        assert state.position_enu[1] == pytest.approx(expected_p_N, rel=1e-3)
        assert state.position_enu[2] == pytest.approx(0.0, abs=1e-6)

    def test_h_input_validation_and_bounds(self) -> None:
        """TEST H: Strict validation of timesteps and finite numbers."""
        ins = StrapdownINS()

        # Non-positive dt
        with pytest.raises(ValueError, match="strictly positive"):
            ins.step([0.0, 0.0, 9.8], [0.0, 0.0, 0.0], dt=0.0, timestamp_ns=100)

        with pytest.raises(ValueError, match="strictly positive"):
            ins.step([0.0, 0.0, 9.8], [0.0, 0.0, 0.0], dt=-0.1, timestamp_ns=100)

        # Excessive dt (> 1.0s)
        with pytest.raises(ValueError, match="exceeds maximum allowable limit"):
            ins.step([0.0, 0.0, 9.8], [0.0, 0.0, 0.0], dt=1.5, timestamp_ns=100)

        # Non-finite values
        with pytest.raises(ValueError, match="non-finite"):
            ins.step([float("nan"), 0.0, 9.8], [0.0, 0.0, 0.0], dt=0.1, timestamp_ns=100)

        with pytest.raises(ValueError, match="non-finite"):
            ins.step([0.0, 0.0, 9.8], [0.0, float("inf"), 0.0], dt=0.1, timestamp_ns=100)

    def test_batch_trajectory_propagation(self) -> None:
        """Verify propagate_trajectory batch interface handles masks and sequence correctly."""
        n_samples = 50
        timestamps = np.arange(n_samples, dtype=np.int64) * 100_000_000  # 10 Hz
        f_m_v = np.tile([0.0, 0.0, STANDARD_GRAVITY], (n_samples, 1))
        omega_m_v = np.zeros((n_samples, 3))

        ins = StrapdownINS()
        traj = ins.propagate_trajectory(timestamps, f_m_v, omega_m_v)

        assert traj.positions_enu.shape == (n_samples, 3)
        assert traj.velocities_enu.shape == (n_samples, 3)
        assert traj.quaternions.shape == (n_samples, 4)
        assert traj.steps_integrated == n_samples
        assert traj.steps_skipped == 0
        assert np.allclose(traj.positions_enu, 0.0, atol=1e-12)
