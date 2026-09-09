"""Synthetic unit tests for Error-State Kalman Filter (ESKF) core mechanics (Phase 5).

Validates:
- 16-nominal / 15-error state dimensions, slicing, and right-multiplicative injection.
- Analytical vs. Numerical Finite-Difference Jacobian (F_d) validation.
- Covariance propagation symmetry and positive-semidefiniteness (PSD).
- Uncertainty reduction along observed directions under generic updates.
- Mahalanobis innovation gating and state/covariance immutability on outlier rejection.
- Synthetic trajectory convergence under simulated motion with noisy GNSS.
- Multi-measurement sequential update stability and Joseph-form covariance preservation.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from navigation.eskf.gating import MahalanobisGating
from navigation.eskf.predict import (
    ProcessNoiseConfig,
    compute_discrete_F,
    compute_discrete_Q,
    predict_eskf,
)
from navigation.eskf.state import (
    ESKFNominalState,
    ESKFState,
    inject_error,
    skew,
)
from navigation.eskf.update import eskf_update
from navigation.frames.local_geo import GeoReference
from navigation.eskf.measurements.gnss import GNSSMeasurementModel, GNSSUpdateConfig
from navigation.eskf.measurements.zupt import ClassicalZUPTDetector, ZUPTMeasurementModel
from navigation.ins.attitude import (
    delta_quaternion,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_normalize,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)


class TestESKFStateAndAlgebra:
    """Validates state representations, dimensions, and manifold operations."""

    def test_state_dimensions_and_slicing(self) -> None:
        """Nominal state must have 16 dimensions; error state and covariance 15 dimensions."""
        nom = ESKFNominalState.from_components(
            position_enu=[1.0, 2.0, 3.0],
            velocity_enu=[4.0, 5.0, 6.0],
            q=[1.0, 0.0, 0.0, 0.0],
            accel_bias=[0.01, -0.02, 0.03],
            gyro_bias=[0.001, -0.002, 0.003],
            timestamp_ns=1_000_000_000,
        )
        assert nom.nominal_vector.shape == (16,)
        assert nom.error_state_dim == 15

        P0 = np.eye(15, dtype=np.float64) * 0.1
        state = ESKFState(nominal=nom, covariance=P0)
        assert state.covariance.shape == (15, 15)
        assert state.pos_cov.shape == (3, 3)
        assert state.vel_cov.shape == (3, 3)
        assert state.att_cov.shape == (3, 3)
        assert state.accel_bias_cov.shape == (3, 3)
        assert state.gyro_bias_cov.shape == (3, 3)

    def test_skew_symmetric_properties(self) -> None:
        """Skew matrix must be anti-symmetric and produce vector cross products."""
        v = np.array([1.5, -2.3, 4.1], dtype=np.float64)
        w = np.array([-0.8, 3.2, 1.1], dtype=np.float64)

        S = skew(v)
        assert np.allclose(S.T, -S, atol=1e-15)
        assert np.allclose(S @ w, np.cross(v, w), atol=1e-15)

    def test_right_multiplicative_attitude_error_injection(self) -> None:
        """Attitude error injection must obey right-multiplicative body-frame convention:
        q_true = normalize(q_nom ⊗ delta_q(delta_theta)).
        """
        # Nominal yaw 45 deg around Z
        yaw_rad = math.radians(45.0)
        q_nom = np.array([math.cos(yaw_rad / 2), 0.0, 0.0, math.sin(yaw_rad / 2)], dtype=np.float64)

        nom = ESKFNominalState.from_components(
            position_enu=[0.0, 0.0, 0.0],
            velocity_enu=[0.0, 0.0, 0.0],
            q=q_nom,
            timestamp_ns=0,
        )

        # Small rotation perturbation around vehicle X axis
        delta_theta_x = 0.02  # rad
        dx = np.zeros(15, dtype=np.float64)
        dx[6] = delta_theta_x

        injected = nom.inject_error(dx)

        # Expected quaternion via explicit right multiplication:
        dq_expected = delta_quaternion(np.array([delta_theta_x, 0.0, 0.0]), dt=1.0)
        q_expected = quaternion_normalize(quaternion_multiply(q_nom, dq_expected))

        assert np.allclose(injected.q, q_expected, atol=1e-12)

        # Vehicle-to-navigation rotation matrix comparison:
        # R(q_true) ≈ R_nom * (I + [delta_theta]_x)
        R_nom = quaternion_to_rotation_matrix(q_nom)
        R_true_linear = R_nom @ (np.eye(3) + skew(np.array([delta_theta_x, 0.0, 0.0])))
        R_injected = injected.R_v_n
        assert np.allclose(R_injected, R_true_linear, atol=1e-3)


class TestESKFJacobianValidation:
    """Validates analytical discrete error-state transition matrix F_d against numerical finite differences."""

    def test_analytical_vs_numerical_finite_difference_jacobian(self) -> None:
        """Column-by-column numerical finite-difference check of F_d."""
        dt = 0.01  # 100 Hz / 10 ms step
        eps = 1e-6

        # Non-trivial nominal state
        yaw = math.radians(35.0)
        pitch = math.radians(5.0)
        roll = math.radians(-3.0)
        cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
        cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
        cr, sr = math.cos(roll / 2), math.sin(roll / 2)
        q0 = np.array([
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ], dtype=np.float64)
        q0 = quaternion_normalize(q0)

        nom_base = ESKFNominalState.from_components(
            position_enu=[120.5, -45.2, 10.3],
            velocity_enu=[12.4, -4.1, 0.8],
            q=q0,
            accel_bias=[0.05, -0.03, 0.08],
            gyro_bias=[0.002, -0.001, 0.004],
            timestamp_ns=1_000_000_000,
        )

        f_meas = np.array([1.2, -0.6, 9.6], dtype=np.float64)
        omega_meas = np.array([0.02, -0.05, 0.08], dtype=np.float64)

        # Analytical transition matrix
        F_d = compute_discrete_F(
            nominal=nom_base,
            f_m_v=f_meas,
            omega_m_v=omega_meas,
            dt=dt,
        )
        assert F_d.shape == (15, 15)


        # Helper to step nominal state given inputs
        def propagate_nominal(nom: ESKFNominalState) -> ESKFNominalState:
            st = ESKFState(nominal=nom, covariance=np.eye(15))
            pred = predict_eskf(
                state=st,
                f_m_v=f_meas,
                omega_m_v=omega_meas,
                dt=dt,
                timestamp_ns=nom.timestamp_ns + int(dt * 1e9),
            )
            return pred.nominal

        # Baseline propagation
        nom_next_base = propagate_nominal(nom_base)

        # Numerical Jacobian via central differences
        F_num = np.zeros((15, 15), dtype=np.float64)

        for j in range(15):
            # Positive perturbation in error space
            dx_pos = np.zeros(15, dtype=np.float64)
            dx_pos[j] = eps
            nom_pos = nom_base.inject_error(dx_pos)
            nom_next_pos = propagate_nominal(nom_pos)

            # Negative perturbation in error space
            dx_neg = np.zeros(15, dtype=np.float64)
            dx_neg[j] = -eps
            nom_neg = nom_base.inject_error(dx_neg)
            nom_next_neg = propagate_nominal(nom_neg)

            # Measure difference in error space between pos and neg perturbed trajectories
            # 1. Position difference
            dp = (nom_next_pos.position_enu - nom_next_neg.position_enu) / (2.0 * eps)
            # 2. Velocity difference
            dv = (nom_next_pos.velocity_enu - nom_next_neg.velocity_enu) / (2.0 * eps)
            # 3. Attitude difference (right-multiplicative body error):
            # q_pos ≈ q_neg ⊗ delta_q(dtheta)  =>  delta_q = q_neg^-1 ⊗ q_pos
            dq_diff = quaternion_multiply(quaternion_inverse(nom_next_neg.q), nom_next_pos.q)
            # For small angles, dq ≈ [1, 0.5 * dtheta]
            dtheta = (2.0 * dq_diff[1:4] / dq_diff[0]) / (2.0 * eps)
            # 4. Accelerometer bias difference
            dba = (nom_next_pos.accel_bias - nom_next_neg.accel_bias) / (2.0 * eps)
            # 5. Gyroscope bias difference
            dbg = (nom_next_pos.gyro_bias - nom_next_neg.gyro_bias) / (2.0 * eps)

            F_num[0:3, j] = dp
            F_num[3:6, j] = dv
            F_num[6:9, j] = dtheta
            F_num[9:12, j] = dba
            F_num[12:15, j] = dbg

        # Compare analytical F_d to numerical F_num
        max_abs_diff = float(np.max(np.abs(F_d - F_num)))
        # Note: delta_t = 0.01s, second order terms ~ dt^2 ~ 1e-4, eps = 1e-6
        assert max_abs_diff < 1e-4, f"F_d finite-difference discrepancy: max diff = {max_abs_diff:.2e}"


class TestESKFPredictionAndCovariance:
    """Validates covariance propagation, symmetry, and PSD maintenance."""

    def test_covariance_symmetry_and_psd_over_long_propagation(self) -> None:
        """Propagate covariance over 500 steps; assert strict symmetry and PSD at every step."""
        nom = ESKFNominalState.from_components(
            position_enu=[0.0, 0.0, 0.0],
            velocity_enu=[15.0, 0.0, 0.0],
            q=[1.0, 0.0, 0.0, 0.0],
            timestamp_ns=0,
        )
        P0 = np.diag([
            1.0, 1.0, 4.0,  # pos
            0.1, 0.1, 0.1,  # vel
            1e-4, 1e-4, 1e-4,  # att
            1e-3, 1e-3, 1e-3,  # ba
            1e-5, 1e-5, 1e-5,  # bg
        ]).astype(np.float64)

        state = ESKFState(nominal=nom, covariance=P0)
        dt = 0.1

        f_m = np.array([0.0, 0.0, 9.80665], dtype=np.float64)
        omega_m = np.array([0.0, 0.0, 0.01], dtype=np.float64)

        for step in range(500):
            t_ns = int((step + 1) * dt * 1e9)
            state = predict_eskf(state, f_m_v=f_m, omega_m_v=omega_m, dt=dt, timestamp_ns=t_ns)

            # Check symmetry: max |P - P^T| must be machine precision
            asym = np.max(np.abs(state.covariance - state.covariance.T))
            assert asym < 1e-12, f"Asymmetry at step {step}: {asym}"

            # Check PSD: minimum eigenvalue must not be significantly negative
            eigvals = np.linalg.eigvalsh(state.covariance)
            min_eig = np.min(eigvals)
            assert min_eig >= -1e-12, f"Loss of PSD at step {step}: min eig = {min_eig}"


class TestESKFGatedUpdates:
    """Validates generic update mechanics, gating rejection, and uncertainty reduction."""

    def test_position_update_reduces_uncertainty(self) -> None:
        """Generic position update must strictly reduce position variance."""
        nom = ESKFNominalState.from_components(
            position_enu=[0.0, 0.0, 0.0],
            velocity_enu=[0.0, 0.0, 0.0],
            q=[1.0, 0.0, 0.0, 0.0],
            timestamp_ns=0,
        )
        P0 = np.eye(15, dtype=np.float64) * 10.0
        state = ESKFState(nominal=nom, covariance=P0)

        # Position observation at [1.0, 1.0, 0.0] with R = 4.0 * I
        z = np.array([1.0, 1.0, 0.0], dtype=np.float64)
        h_val = state.position_enu
        H = np.zeros((3, 15), dtype=np.float64)
        H[0:3, 0:3] = np.eye(3)
        R = np.eye(3, dtype=np.float64) * 4.0

        gating = MahalanobisGating(confidence_level=0.99)
        updated_state, diag = eskf_update(state, z=z, h_val=h_val, H=H, R=R, gating=gating)

        assert diag.applied is True
        assert diag.gating is not None
        assert diag.gating.accepted is True
        # Position variance must decrease
        assert updated_state.covariance[0, 0] < P0[0, 0]
        assert updated_state.covariance[1, 1] < P0[1, 1]
        assert updated_state.covariance[2, 2] < P0[2, 2]

        # Theoretical scalar Kalman posterior variance check: P_pos = (10 * 4) / (10 + 4) = 40/14 ≈ 2.857
        expected_var = 40.0 / 14.0
        assert updated_state.covariance[0, 0] == pytest.approx(expected_var, rel=1e-4)

    def test_mahalanobis_outlier_rejection_leaves_state_and_covariance_untouched(self) -> None:
        """Gross outlier measurement must be rejected; state and covariance remain identical."""
        nom = ESKFNominalState.from_components(
            position_enu=[0.0, 0.0, 0.0],
            velocity_enu=[0.0, 0.0, 0.0],
            q=[1.0, 0.0, 0.0, 0.0],
            timestamp_ns=1000,
        )
        P0 = np.eye(15, dtype=np.float64) * 1.0
        state = ESKFState(nominal=nom, covariance=P0)

        # Impossible 500-meter jump with 1-meter noise sigma
        z_outlier = np.array([500.0, -400.0, 100.0], dtype=np.float64)
        h_val = state.position_enu
        H = np.zeros((3, 15), dtype=np.float64)
        H[0:3, 0:3] = np.eye(3)
        R = np.eye(3, dtype=np.float64) * 1.0

        gating = MahalanobisGating(confidence_level=0.99)
        updated_state, diag = eskf_update(state, z=z_outlier, h_val=h_val, H=H, R=R, gating=gating)

        assert diag.applied is False
        assert diag.gating is not None
        assert diag.gating.accepted is False
        assert diag.gating.mahalanobis_sq > diag.gating.threshold

        # Assert strict immutability
        assert np.array_equal(updated_state.nominal.nominal_vector, nom.nominal_vector)
        assert np.array_equal(updated_state.covariance, P0)


class TestESKFSyntheticTrajectoryConvergence:
    """Validates filter convergence under simulated motion with noisy GNSS aiding."""

    def test_trajectory_convergence_under_noisy_gnss(self) -> None:
        """Simulate kinematic motion with noisy GNSS; filter must converge to true position."""
        geo_ref = GeoReference(lat_ref=12.9716, lon_ref=77.5946, alt_ref=920.0)
        gnss_model = GNSSMeasurementModel(
            geo_reference=geo_ref,
            gating=MahalanobisGating(confidence_level=0.99),
        )

        total_time = 15.0  # seconds
        dt_imu = 0.05  # 20 Hz IMU
        dt_gnss = 1.0  # 1 Hz GNSS

        v0 = 12.0  # m/s
        accel_x = 0.5  # m/s^2 forward acceleration

        # Deliberate initial position error: +8 m East, -6 m North
        initial_pos_err = np.array([8.0, -6.0, 0.0])
        nom_init = ESKFNominalState.from_components(
            position_enu=initial_pos_err,
            velocity_enu=[v0, 0.0, 0.0],
            q=[1.0, 0.0, 0.0, 0.0],
            timestamp_ns=0,
        )
        P0 = np.eye(15, dtype=np.float64) * 0.1
        P0[0:3, 0:3] = np.eye(3) * 64.0  # 8m 1-sigma initial position uncertainty
        P0[3:6, 3:6] = np.eye(3) * 1.0

        state = ESKFState(nominal=nom_init, covariance=P0)

        t = 0.0
        gnss_noise_std = 1.0
        rng = np.random.RandomState(42)

        # True specific force: forward acceleration + gravity
        f_v = np.array([accel_x, 0.0, 9.80665], dtype=np.float64)
        w_v = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        while t < total_time:
            t += dt_imu
            t_ns = int(t * 1e9)

            # Propagate ESKF
            state = predict_eskf(state, f_m_v=f_v, omega_m_v=w_v, dt=dt_imu, timestamp_ns=t_ns)

            # 1 Hz GNSS update
            if abs((t % dt_gnss)) < dt_imu / 2 or abs((t % dt_gnss) - dt_gnss) < dt_imu / 2:
                true_pos_x = v0 * t + 0.5 * accel_x * (t ** 2)
                true_pos = np.array([true_pos_x, 0.0, 0.0])

                meas_pos = true_pos + rng.normal(0, gnss_noise_std, size=3)
                meas_pos[2] = 0.0  # clean altitude
                lat, lon, alt = geo_ref.enu_to_geodetic(meas_pos[0], meas_pos[1], meas_pos[2])

                state, diag = gnss_model.update_position(
                    state=state,
                    lat=lat,
                    lon=lon,
                    alt=alt,
                    accuracy_h_m=gnss_noise_std,
                    timestamp_ns=t_ns,
                )
                assert diag.applied is True

        # Final true position
        final_true_x = v0 * total_time + 0.5 * accel_x * (total_time ** 2)
        final_true_pos = np.array([final_true_x, 0.0, 0.0])
        final_err = float(np.linalg.norm(state.position_enu[0:2] - final_true_pos[0:2]))
        # Final error must be significantly reduced from initial 10m error (to < 2.0m)
        assert final_err < 2.0, f"Filter failed to converge; final horizontal error = {final_err:.2f} m"

    def test_sequential_multi_measurement_stability(self) -> None:
        """Sequential application of Position, Velocity, and ZUPT updates maintains PSD and symmetry."""
        geo_ref = GeoReference(lat_ref=12.9716, lon_ref=77.5946, alt_ref=920.0)
        gnss_model = GNSSMeasurementModel(geo_reference=geo_ref)
        zupt_model = ZUPTMeasurementModel()

        nom = ESKFNominalState.from_components(
            position_enu=[0.0, 0.0, 0.0],
            velocity_enu=[0.02, -0.01, 0.00],
            q=[1.0, 0.0, 0.0, 0.0],
            timestamp_ns=1_000_000_000,
        )
        P0 = np.eye(15, dtype=np.float64) * 2.0
        state = ESKFState(nominal=nom, covariance=P0)

        # 1. Update position
        lat, lon, alt = geo_ref.enu_to_geodetic(0.1, -0.05, 0.0)
        state, diag_pos = gnss_model.update_position(state, lat=lat, lon=lon, alt=alt, accuracy_h_m=2.0)
        assert diag_pos.applied is True

        # 2. Update velocity
        state, diag_vel = gnss_model.update_velocity(state, v_east=0.01, v_north=0.0, v_up=0.0, accuracy_speed_mps=0.2)
        assert diag_vel.applied is True

        # 3. Update ZUPT
        state, diag_zupt = zupt_model.update(state)
        assert diag_zupt.applied is True

        # Assert symmetry and PSD
        asym = np.max(np.abs(state.covariance - state.covariance.T))
        assert asym < 1e-12
        min_eig = np.min(np.linalg.eigvalsh(state.covariance))
        assert min_eig > 0.0


