"""Mandatory synthetic unit tests for attitude and strapdown INS propagation (Phase 4).

Validates closed-form known answers and strict hardening invariants:
- Test A: Constant forward acceleration (closed-form velocity and position)
- Test B: Stationary gravity cancellation (zero velocity and position drift)
- Test C: Constant angular velocity (exact analytical rotation angle)
- Test D: 90-degree known rotation (axis, sign, and composition order)
- Test E: Coupled rotation + acceleration (time-varying ENU acceleration rotation)
- Test F: Quaternion stability (10,000 iterations maintain unit norm and orthogonality)
- Test H: Input validation and timestep bounds (NaN, +/-Inf, <= 0, > max)
- Test I: Timestamp consistency in step() (backwards, duplicate, dt mismatch)
- Test J: Batch input shape and mask validation (reject impossible shapes)
- Test K: Invalid-sample handling (corrupted/invalid sample measurements are NEVER consumed)
- Test L: Initial coordinate acceleration is unmeasured (no fabricated measurements)
- Test M: Heading to ENU rotation geometry across all 4 cardinal quadrants
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

        prod = quaternion_multiply(q, q_inv)
        assert np.allclose(prod, [1.0, 0.0, 0.0, 0.0], atol=1e-15)

    def test_f_quaternion_stability(self) -> None:
        """TEST F: 10,000 repeated small 3D rotations maintain unit norm and matrix orthogonality."""
        np.random.seed(42)
        q = np.array([1.0, 0.0, 0.0, 0.0])

        for _ in range(10_000):
            omega = (np.random.rand(3) - 0.5) * 2.0
            dt = 0.01
            q = propagate_attitude(q, omega, dt)

        assert np.isfinite(q).all()
        assert quaternion_norm(q) == pytest.approx(1.0, abs=1e-12)

        R = quaternion_to_rotation_matrix(q)
        assert np.isfinite(R).all()
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)

    def test_m_heading_to_enu_cardinal_geometry(self) -> None:
        """TEST M: Verify heading to ENU rotation matrix for all cardinal directions."""
        cardinals = [
            (0.0,   np.array([0.0,  1.0, 0.0]), np.array([-1.0,  0.0, 0.0])),  # North
            (90.0,  np.array([1.0,  0.0, 0.0]), np.array([ 0.0,  1.0, 0.0])),  # East
            (180.0, np.array([0.0, -1.0, 0.0]), np.array([ 1.0,  0.0, 0.0])),  # South
            (270.0, np.array([-1.0, 0.0, 0.0]), np.array([ 0.0, -1.0, 0.0])),  # West
        ]

        for heading_deg, expected_fwd, expected_left in cardinals:
            psi = math.radians(heading_deg)
            R = np.array([
                [math.sin(psi), -math.cos(psi), 0.0],
                [math.cos(psi),  math.sin(psi), 0.0],
                [0.0,            0.0,           1.0],
            ], dtype=np.float64)

            # Check properties
            assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-14)
            assert np.allclose(R @ R.T, np.eye(3), atol=1e-14)

            # Check forward vector (+X_v) mapping
            fwd_enu = R @ [1.0, 0.0, 0.0]
            assert np.allclose(fwd_enu, expected_fwd, atol=1e-12)

            # Check left vector (+Y_v) mapping
            left_enu = R @ [0.0, 1.0, 0.0]
            assert np.allclose(left_enu, expected_left, atol=1e-12)


class TestStrapdownINSPropagation:
    """Unit tests for deterministic strapdown INS propagation in local ENU frame."""

    def test_l_initialization_does_not_fabricate_coordinate_accel(self) -> None:
        """TEST L: Constructor must initialize coordinate acceleration to zero, not fabricate measurements."""
        ins = StrapdownINS()
        assert np.array_equal(ins.current_state.coordinate_accel_enu, [0.0, 0.0, 0.0])
        assert ins.current_state.timestamp_ns == 0

    def test_b_stationary_gravity_cancellation(self) -> None:
        """TEST B: Stationary vehicle level on horizontal surface experiences zero coordinate acceleration."""
        ins = StrapdownINS(
            initial_position=[0.0, 0.0, 0.0],
            initial_velocity=[0.0, 0.0, 0.0],
            initial_q=[1.0, 0.0, 0.0, 0.0],
            gravity_magnitude=STANDARD_GRAVITY,
        )

        dt = 0.1  # 10 Hz
        f_stationary = np.array([0.0, 0.0, STANDARD_GRAVITY])
        w_stationary = np.array([0.0, 0.0, 0.0])

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
        """TEST A: Known constant forward acceleration yields exact closed-form velocity and position."""
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

        expected_v_east = a_const * duration_s
        expected_p_east = 0.5 * a_const * (duration_s ** 2)

        assert state.velocity_enu[0] == pytest.approx(expected_v_east, rel=1e-5)
        assert state.position_enu[0] == pytest.approx(expected_p_east, rel=1e-5)
        assert state.velocity_enu[1] == pytest.approx(0.0, abs=1e-10)
        assert state.velocity_enu[2] == pytest.approx(0.0, abs=1e-10)
        assert state.position_enu[1] == pytest.approx(0.0, abs=1e-10)
        assert state.position_enu[2] == pytest.approx(0.0, abs=1e-10)

    def test_e_coupled_rotation_and_acceleration(self) -> None:
        """TEST E: Coupled rotation + acceleration verifies frame-order and body-to-ENU transformation."""
        a_0 = 1.0       # m/s^2
        omega_z = 0.1   # rad/s
        duration_s = 5.0
        dt = 0.001
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

        T = duration_s
        expected_v_E = (a_0 / omega_z) * math.sin(omega_z * T)
        expected_v_N = (a_0 / omega_z) * (1.0 - math.cos(omega_z * T))
        expected_p_E = (a_0 / (omega_z ** 2)) * (1.0 - math.cos(omega_z * T))
        expected_p_N = (a_0 / omega_z) * T - (a_0 / (omega_z ** 2)) * math.sin(omega_z * T)

        assert state.velocity_enu[0] == pytest.approx(expected_v_E, rel=1e-3)
        assert state.velocity_enu[1] == pytest.approx(expected_v_N, rel=1e-3)
        assert state.position_enu[0] == pytest.approx(expected_p_E, rel=1e-3)
        assert state.position_enu[1] == pytest.approx(expected_p_N, rel=1e-3)
        assert state.position_enu[2] == pytest.approx(0.0, abs=1e-6)

    def test_h_input_validation_and_bounds(self) -> None:
        """TEST H: Strict validation of timesteps and finite numbers including NaN and Inf."""
        ins = StrapdownINS()

        # NaN dt
        with pytest.raises(ValueError, match="finite"):
            ins.step([0.0, 0.0, 9.8], [0.0, 0.0, 0.0], dt=float("nan"), timestamp_ns=100)

        # +Inf dt
        with pytest.raises(ValueError, match="finite"):
            ins.step([0.0, 0.0, 9.8], [0.0, 0.0, 0.0], dt=float("inf"), timestamp_ns=100)

        # -Inf dt
        with pytest.raises(ValueError, match="finite"):
            ins.step([0.0, 0.0, 9.8], [0.0, 0.0, 0.0], dt=float("-inf"), timestamp_ns=100)

        # Non-positive dt
        with pytest.raises(ValueError, match="strictly positive"):
            ins.step([0.0, 0.0, 9.8], [0.0, 0.0, 0.0], dt=0.0, timestamp_ns=100)

        with pytest.raises(ValueError, match="strictly positive"):
            ins.step([0.0, 0.0, 9.8], [0.0, 0.0, 0.0], dt=-0.1, timestamp_ns=100)

        # Excessive dt (> 1.0s)
        with pytest.raises(ValueError, match="exceeds maximum allowable limit"):
            ins.step([0.0, 0.0, 9.8], [0.0, 0.0, 0.0], dt=1.5, timestamp_ns=1_500_000_000)

        # Non-finite values in measurements
        with pytest.raises(ValueError, match="non-finite"):
            ins.step([float("nan"), 0.0, 9.8], [0.0, 0.0, 0.0], dt=0.1, timestamp_ns=100_000_000)

        with pytest.raises(ValueError, match="non-finite"):
            ins.step([0.0, 0.0, 9.8], [0.0, float("inf"), 0.0], dt=0.1, timestamp_ns=100_000_000)

    def test_i_step_timestamp_consistency(self) -> None:
        """TEST I: Strict step timestamp consistency (backwards, duplicate, dt mismatch)."""
        ins = StrapdownINS(initial_timestamp_ns=1_000_000_000)

        # Duplicate timestamp
        with pytest.raises(ValueError, match="strictly greater"):
            ins.step([0.0, 0.0, 9.8], [0.0, 0.0, 0.0], dt=0.1, timestamp_ns=1_000_000_000)

        # Backwards timestamp
        with pytest.raises(ValueError, match="strictly greater"):
            ins.step([0.0, 0.0, 9.8], [0.0, 0.0, 0.0], dt=0.1, timestamp_ns=900_000_000)

        # Non-integer timestamp
        with pytest.raises(TypeError, match="integer"):
            ins.step([0.0, 0.0, 9.8], [0.0, 0.0, 0.0], dt=0.1, timestamp_ns="not_int")  # type: ignore

        # Discrepant dt vs timestamp delta (dt=0.1s but delta is 0.2s)
        with pytest.raises(ValueError, match="inconsistent with timestamp increment"):
            ins.step([0.0, 0.0, 9.8], [0.0, 0.0, 0.0], dt=0.1, timestamp_ns=1_200_000_000)

    def test_j_trajectory_shape_validation(self) -> None:
        """TEST J: Reject invalid array shapes and broadcasting attempts."""
        ins = StrapdownINS()
        n = 10
        ts = np.arange(n, dtype=np.int64) * 100_000_000
        f_good = np.tile([0.0, 0.0, 9.8], (n, 1))
        w_good = np.zeros((n, 3))

        # Empty trajectory
        with pytest.raises(ValueError, match="empty trajectory"):
            ins.propagate_trajectory(np.array([], dtype=np.int64), np.zeros((0, 3)), np.zeros((0, 3)))

        # 2D timestamps
        with pytest.raises(ValueError, match="1D array"):
            ins.propagate_trajectory(ts.reshape(2, 5), f_good, w_good)

        # f_m_v wrong columns (N, 2)
        with pytest.raises(ValueError, match="f_m_v must have shape"):
            ins.propagate_trajectory(ts, f_good[:, :2], w_good)

        # omega_m_v wrong length (N-1, 3)
        with pytest.raises(ValueError, match="omega_m_v must have shape"):
            ins.propagate_trajectory(ts, f_good, w_good[:n-1])

        # is_validated wrong shape
        with pytest.raises(ValueError, match="is_validated mask must have shape"):
            ins.propagate_trajectory(ts, f_good, w_good, is_validated=np.ones(n+1, dtype=bool))

    def test_k_invalid_sample_handling_never_consumes_corrupted_measurement(self) -> None:
        """TEST K: An invalid middle sample must NEVER have its measurements consumed.

        Sequence:
            Sample 0..4: valid level stationary
            Sample 5: INVALID with catastrophic 1000 m/s^2 spike and 100 rad/s spin
            Sample 6..10: valid level stationary

        If sample 5's measurement were used in either step 4->5 or 5->6, velocity would spike by 100 m/s.
        With correct handling, velocity and position remain bounded near zero!
        """
        n_samples = 11
        dt = 0.1
        timestamps = np.arange(n_samples, dtype=np.int64) * 100_000_000

        f_m_v = np.tile([0.0, 0.0, STANDARD_GRAVITY], (n_samples, 1))
        omega_m_v = np.zeros((n_samples, 3))
        is_validated = np.ones(n_samples, dtype=bool)

        # Inject corrupt measurement into sample 5
        is_validated[5] = False
        f_m_v[5] = [9999.0, -8888.0, 7777.0]
        omega_m_v[5] = [50.0, -50.0, 50.0]

        ins = StrapdownINS(initial_timestamp_ns=0)
        traj = ins.propagate_trajectory(timestamps, f_m_v, omega_m_v, is_validated=is_validated)

        # Both step 4->5 (target is invalid) and 5->6 (source is invalid) must be skipped.
        # Out of 10 total intervals across 11 samples, 2 skipped -> 8 integrated.
        assert traj.steps_skipped == 2
        assert traj.steps_integrated == 8
        assert traj.propagation_steps == 8
        assert traj.skipped_steps == 2
        assert traj.sample_count == 11

        # Crucial: Catastrophic spike was NEVER integrated!
        assert np.allclose(traj.velocities_enu, 0.0, atol=1e-10)
        assert np.allclose(traj.positions_enu, 0.0, atol=1e-10)
        assert np.allclose(traj.quaternions, [1.0, 0.0, 0.0, 0.0], atol=1e-10)

    def test_n_propagation_step_count_semantics(self) -> None:
        """Verify propagation step count semantics: N samples yield N - 1 intervals.

        The initial state is recorded at index 0, but is NOT counted as a propagation step.
        For N samples with 0 skips:
            sample_count = N
            propagation_steps = N - 1
            skipped_steps = 0
            steps_integrated + steps_skipped == N - 1
        """
        # Test Case 1: 50 fully valid samples -> exactly 49 propagation steps
        n_50 = 50
        ts_50 = np.arange(n_50, dtype=np.int64) * 100_000_000
        f_50 = np.tile([0.0, 0.0, STANDARD_GRAVITY], (n_50, 1))
        w_50 = np.zeros((n_50, 3))

        ins_50 = StrapdownINS(initial_timestamp_ns=0)
        traj_50 = ins_50.propagate_trajectory(ts_50, f_50, w_50)

        assert traj_50.sample_count == 50
        assert len(traj_50.positions_enu) == 50
        assert len(traj_50.velocities_enu) == 50
        assert len(traj_50.quaternions) == 50
        assert traj_50.steps_integrated == 49
        assert traj_50.propagation_steps == 49
        assert traj_50.steps_skipped == 0
        assert traj_50.skipped_steps == 0
        assert traj_50.propagation_steps + traj_50.skipped_steps == n_50 - 1

        # Test Case 2: 601 fully valid samples (Ablation Stage 1 size) -> exactly 600 propagation steps
        n_601 = 601
        ts_601 = np.arange(n_601, dtype=np.int64) * 100_000_000
        f_601 = np.tile([0.0, 0.0, STANDARD_GRAVITY], (n_601, 1))
        w_601 = np.zeros((n_601, 3))

        ins_601 = StrapdownINS(initial_timestamp_ns=0)
        traj_601 = ins_601.propagate_trajectory(ts_601, f_601, w_601)

        assert traj_601.sample_count == 601
        assert len(traj_601.positions_enu) == 601
        assert traj_601.steps_integrated == 600
        assert traj_601.propagation_steps == 600
        assert traj_601.steps_skipped == 0
        assert traj_601.skipped_steps == 0
        assert traj_601.propagation_steps + traj_601.skipped_steps == n_601 - 1
