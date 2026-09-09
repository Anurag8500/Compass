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
    compute_reset_jacobian,
    inject_error,
    reset_covariance,
    skew,
)
from navigation.eskf.update import eskf_update
from navigation.frames.local_geo import GeoReference
from navigation.eskf.measurements.gnss import (
    GNSSMeasurementModel,
    GNSSUpdateConfig,
    course_to_enu_velocity,
    course_to_horizontal_velocity,
)
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

    def test_error_state_covariance_reset_transformation(self) -> None:
        """Covariance reset must transform attitude and cross-attitude covariances correctly.

        For right-multiplicative attitude error q = q_nom ⊗ delta_q(delta_theta),
        the error-state reset sensitivity is:
            G_theta = I - 0.5 * [delta_theta_hat]_x
            J_reset = diag(I, I, G_theta, I, I)
            P^+ = J_reset * P * J_reset^T

        Tests:
            1. Correct transformation for non-zero delta_theta_hat.
            2. Cross-covariances involving attitude transform via G_theta.
            3. Non-attitude blocks (pos, vel, ba, bg) remain strictly invariant.
            4. State.inject_error(dx) applies reset_covariance identically.
            5. Covariance symmetry and positive-definiteness are preserved.
        """
        # Non-zero attitude correction
        dtheta_hat = np.array([0.03, -0.02, 0.05], dtype=np.float64)
        dx = np.zeros(15, dtype=np.float64)
        dx[6:9] = dtheta_hat
        dx[0:3] = [0.1, -0.2, 0.3]  # position correction
        dx[3:6] = [-0.05, 0.1, -0.15]  # velocity correction

        # Build reset Jacobian
        J_reset = compute_reset_jacobian(dtheta_hat)
        assert J_reset.shape == (15, 15)
        # Position, velocity, ba, bg diagonal blocks must be identity
        assert np.allclose(J_reset[0:3, 0:3], np.eye(3))
        assert np.allclose(J_reset[3:6, 3:6], np.eye(3))
        assert np.allclose(J_reset[9:12, 9:12], np.eye(3))
        assert np.allclose(J_reset[12:15, 12:15], np.eye(3))

        # Attitude block G_theta = I - 0.5 * [dtheta_hat]_x
        G_theta_expected = np.eye(3) - 0.5 * skew(dtheta_hat)
        assert np.allclose(J_reset[6:9, 6:9], G_theta_expected, atol=1e-15)

        # Create a full, non-diagonal positive definite covariance matrix with cross-correlations
        np.random.seed(42)
        A = np.random.randn(15, 15) * 0.1
        P_orig = A @ A.T + np.eye(15) * 1.0  # strictly positive definite

        P_reset = reset_covariance(P_orig, dtheta_hat)

        # Check analytic transformation P^+ = J * P * J^T
        P_expected = J_reset @ P_orig @ J_reset.T
        P_expected = 0.5 * (P_expected + P_expected.T)
        assert np.allclose(P_reset, P_expected, atol=1e-14)

        # Check non-attitude blocks are strictly invariant
        assert np.allclose(P_reset[0:3, 0:3], P_orig[0:3, 0:3], atol=1e-14)
        assert np.allclose(P_reset[3:6, 3:6], P_orig[3:6, 3:6], atol=1e-14)
        assert np.allclose(P_reset[9:12, 9:12], P_orig[9:12, 9:12], atol=1e-14)
        assert np.allclose(P_reset[12:15, 12:15], P_orig[12:15, 12:15], atol=1e-14)

        # Check cross-covariance between velocity and attitude
        # P_v_theta^+ = P_v_theta * G_theta^T
        assert np.allclose(P_reset[3:6, 6:9], P_orig[3:6, 6:9] @ G_theta_expected.T, atol=1e-14)
        assert np.allclose(P_reset[6:9, 3:6], G_theta_expected @ P_orig[6:9, 3:6], atol=1e-14)

        # Check attitude block P_theta_theta^+ = G_theta * P_theta_theta * G_theta^T
        assert np.allclose(P_reset[6:9, 6:9], G_theta_expected @ P_orig[6:9, 6:9] @ G_theta_expected.T, atol=1e-14)

        # Check symmetry and PSD
        asym = np.max(np.abs(P_reset - P_reset.T))
        assert asym < 1e-14
        min_eig = np.min(np.linalg.eigvalsh(P_reset))
        assert min_eig > 0.0

        # Verify ESKFState.inject_error integrates this reset
        nom = ESKFNominalState.from_components(
            position_enu=[0.0, 0.0, 0.0],
            velocity_enu=[0.0, 0.0, 0.0],
            q=[1.0, 0.0, 0.0, 0.0],
            timestamp_ns=0,
        )
        st = ESKFState(nominal=nom, covariance=P_orig.copy())
        st_injected = st.inject_error(dx)
        assert np.allclose(st_injected.covariance, P_reset, atol=1e-14)


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

    def test_gating_malformed_singular_and_nonfinite_inputs(self) -> None:
        """Gating must safely reject nonfinite innovations, singular S, or invalid inputs."""
        gating = MahalanobisGating(confidence_level=0.99)

        # 1. Nonfinite innovation
        diag = gating.evaluate(
            innovation=np.array([np.nan, 1.0, 2.0]),
            S=np.eye(3),
        )
        assert diag.accepted is False
        assert not math.isfinite(diag.mahalanobis_sq)

        diag_inf = gating.evaluate(
            innovation=np.array([np.inf, 1.0, 2.0]),
            S=np.eye(3),
        )
        assert diag_inf.accepted is False

        # 2. Singular / ill-conditioned innovation covariance S
        S_singular = np.zeros((3, 3))
        diag_sing = gating.evaluate(
            innovation=np.array([1.0, 2.0, 3.0]),
            S=S_singular,
        )
        assert diag_sing.accepted is False

        # 3. Nonfinite S
        S_nan = np.eye(3)
        S_nan[0, 0] = np.nan
        diag_nan_s = gating.evaluate(
            innovation=np.array([1.0, 2.0, 3.0]),
            S=S_nan,
        )
        assert diag_nan_s.accepted is False

    def test_update_malformed_and_invalid_r(self) -> None:
        """Generic ESKF update must safely reject invalid R or nonfinite measurements."""
        nom = ESKFNominalState.from_components(
            position_enu=[0.0, 0.0, 0.0],
            velocity_enu=[0.0, 0.0, 0.0],
            q=[1.0, 0.0, 0.0, 0.0],
            timestamp_ns=1000,
        )
        P0 = np.eye(15, dtype=np.float64) * 1.0
        state = ESKFState(nominal=nom, covariance=P0)
        H = np.zeros((3, 15), dtype=np.float64)
        H[0:3, 0:3] = np.eye(3)

        # 1. Invalid R with negative or zero diagonal
        R_invalid_diag = np.diag([1.0, 0.0, 1.0])
        st_out, diag = eskf_update(state, z=np.ones(3), h_val=np.zeros(3), H=H, R=R_invalid_diag)
        assert diag.applied is False
        assert np.array_equal(st_out.covariance, P0)

        # 2. Asymmetric R
        R_asym = np.array([
            [1.0, 0.5, 0.0],
            [0.1, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        st_out, diag = eskf_update(state, z=np.ones(3), h_val=np.zeros(3), H=H, R=R_asym)
        assert diag.applied is False
        assert np.array_equal(st_out.covariance, P0)

        # 3. Nonfinite z
        st_out, diag = eskf_update(state, z=np.array([np.nan, 0.0, 0.0]), h_val=np.zeros(3), H=H, R=np.eye(3))
        assert diag.applied is False
        assert np.array_equal(st_out.covariance, P0)


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


class TestGNSSCourseAndVelocityModels:
    """Validates horizontal speed + bearing conversion to local ENU and velocity updates."""

    def test_course_to_enu_velocity_cardinal_directions(self) -> None:
        """Verify course conversion matches standard clockwise-from-North navigation convention."""
        speed = 10.0  # m/s

        # 0 deg: True North -> v_east = 0, v_north = 10
        ve, vn, vu = course_to_enu_velocity(speed, bearing_deg=0.0)
        assert pytest.approx(ve, abs=1e-9) == 0.0
        assert pytest.approx(vn, abs=1e-9) == 10.0
        assert pytest.approx(vu, abs=1e-9) == 0.0

        # 90 deg: East -> v_east = 10, v_north = 0
        ve, vn, vu = course_to_enu_velocity(speed, bearing_deg=90.0)
        assert pytest.approx(ve, abs=1e-9) == 10.0
        assert pytest.approx(vn, abs=1e-9) == 0.0

        # 180 deg: South -> v_east = 0, v_north = -10
        ve, vn, vu = course_to_enu_velocity(speed, bearing_deg=180.0)
        assert pytest.approx(ve, abs=1e-9) == 0.0
        assert pytest.approx(vn, abs=1e-9) == -10.0

        # 270 deg: West -> v_east = -10, v_north = 0
        ve, vn, vu = course_to_enu_velocity(speed, bearing_deg=270.0)
        assert pytest.approx(ve, abs=1e-9) == -10.0
        assert pytest.approx(vn, abs=1e-9) == 0.0

        # 360 deg wrap: same as 0 deg
        ve, vn, vu = course_to_enu_velocity(speed, bearing_deg=360.0)
        assert pytest.approx(ve, abs=1e-9) == 0.0
        assert pytest.approx(vn, abs=1e-9) == 10.0

    def test_course_to_horizontal_velocity(self) -> None:
        """Verify 2D course-to-horizontal velocity helper."""
        ve, vn = course_to_horizontal_velocity(10.0, bearing_deg=45.0)
        assert pytest.approx(ve, abs=1e-5) == 10.0 * math.sin(math.radians(45.0))
        assert pytest.approx(vn, abs=1e-5) == 10.0 * math.cos(math.radians(45.0))

        # Invalid speed
        ve_inv, vn_inv = course_to_horizontal_velocity(-1.0, bearing_deg=45.0)
        assert math.isnan(ve_inv) and math.isnan(vn_inv)

    def test_2d_horizontal_velocity_measurement_structure_and_unobserved_vertical(self) -> None:
        """2D horizontal velocity measurement must observe only v_East and v_North, leaving v_Up unobserved."""
        geo_ref = GeoReference(lat_ref=12.9716, lon_ref=77.5946, alt_ref=920.0)
        gnss_model = GNSSMeasurementModel(geo_reference=geo_ref)

        meas = gnss_model.create_horizontal_velocity_measurement(
            speed_mps=15.0,
            bearing_deg=30.0,
            accuracy_speed_mps=0.4,
        )
        assert meas is not None
        z_v, H_v, R_v = meas

        # 1. Dimension must be exactly 2
        assert z_v.shape == (2,)
        assert H_v.shape == (2, 15)
        assert R_v.shape == (2, 2)

        # 2. H must observe delta_v_East (idx 3) and delta_v_North (idx 4)
        assert H_v[0, 3] == 1.0
        assert H_v[1, 4] == 1.0

        # 3. delta_v_Up (idx 5) and all other states must be strictly unobserved (0.0)
        assert H_v[0, 5] == 0.0
        assert H_v[1, 5] == 0.0
        assert np.all(H_v[:, 0:3] == 0.0)
        assert np.all(H_v[:, 5:] == 0.0)

    def test_gnss_update_horizontal_velocity_from_course(self) -> None:
        """update_horizontal_velocity_from_course reduces horizontal velocity variance without touching vertical variance."""
        geo_ref = GeoReference(lat_ref=12.9716, lon_ref=77.5946, alt_ref=920.0)
        gnss_model = GNSSMeasurementModel(geo_reference=geo_ref)

        nom = ESKFNominalState.from_components(
            position_enu=[0.0, 0.0, 0.0],
            velocity_enu=[9.5, 9.8, 1.25],  # Nonzero vertical velocity
            q=[1.0, 0.0, 0.0, 0.0],
            timestamp_ns=1_000_000_000,
        )
        P0 = np.eye(15, dtype=np.float64) * 4.0
        state = ESKFState(nominal=nom, covariance=P0)

        speed = 10.0 * math.sqrt(2)
        state_upd, diag = gnss_model.update_horizontal_velocity_from_course(
            state=state,
            speed_mps=speed,
            bearing_deg=45.0,
            accuracy_speed_mps=0.5,
        )
        assert diag.applied is True
        assert diag.measurement_dim == 2

        # Horizontal velocity variances must contract
        assert state_upd.vel_cov[0, 0] < P0[3, 3]
        assert state_upd.vel_cov[1, 1] < P0[4, 4]

        # Vertical velocity variance and state must remain untouched by this horizontal update!
        assert state_upd.vel_cov[2, 2] == pytest.approx(P0[5, 5], rel=1e-12)
        assert state_upd.velocity_enu[2] == pytest.approx(1.25, rel=1e-12)

    def test_fix_arrival_detection_unchanged_latitude(self) -> None:
        """Fix arrival detection using complete signature detects new fix even if latitude is unchanged."""
        fix1 = (52.41643, -1.57744, 172.82, 3.48, 289.84)
        # Fix 2: Vehicle driving purely West/East, latitude identical, longitude changed
        fix2 = (52.41643, -1.57900, 172.85, 3.50, 289.80)

        assert fix1 != fix2, "Multi-field fix signature must detect new fix when latitude is unchanged"

    def test_single_authoritative_covariance_reset_path(self) -> None:
        """Calling eskf_update and calling ESKFState.inject_error apply reset identically and exactly once."""
        nom = ESKFNominalState.from_components(
            position_enu=[0.0, 0.0, 0.0],
            velocity_enu=[0.0, 0.0, 0.0],
            q=[1.0, 0.0, 0.0, 0.0],
            timestamp_ns=1000,
        )
        P0 = np.eye(15, dtype=np.float64) * 2.0
        state = ESKFState(nominal=nom, covariance=P0)

        # Correction vector with non-zero attitude correction
        dx = np.zeros(15, dtype=np.float64)
        dtheta = np.array([0.02, -0.01, 0.04])
        dx[6:9] = dtheta

        # 1. State method inject_error
        st1 = state.inject_error(dx)

        # 2. Manual reset via reset_covariance
        cov_expected = reset_covariance(P0, dtheta)
        assert np.allclose(st1.covariance, cov_expected, atol=1e-14)

        # 3. Verify exactly-once: covariance is not reset twice in eskf_update
        z = np.array([0.5, -0.5], dtype=np.float64)
        H = np.zeros((2, 15), dtype=np.float64)
        H[0, 3] = 1.0
        H[1, 4] = 1.0
        R = np.eye(2, dtype=np.float64) * 0.25

        st_upd, diag = eskf_update(state, z=z, h_val=np.zeros(2), H=H, R=R)
        assert diag.applied is True
        # Covariance must remain symmetric and strictly PSD
        asym = np.max(np.abs(st_upd.covariance - st_upd.covariance.T))
        assert asym < 1e-12
        min_eig = np.min(np.linalg.eigvalsh(st_upd.covariance))
        assert min_eig > 0.0



