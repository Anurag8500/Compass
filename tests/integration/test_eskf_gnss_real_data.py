"""Integration test verifying ESKF + GNSS + Gated ZUPT on real IO-VNBD Trip S1 data (Phase 5).

Verifies:
1. Phase 5 ESKF + GNSS fusion produces a statistically measurable improvement
   over the frozen Phase 4 open-loop baseline (3,249.32 m).
2. Covariance remains symmetric, positive-definite, and bounded.
3. Innovation gating functions correctly on real sensor data.
4. Classical Gated ZUPT suppresses stationary drift on real standstill data.
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pytest

from data.pipeline.sync import SynchronizedTrip
from data.pipeline.stationary_detect import StationaryDetector
from navigation.frames.local_geo import GeoReference
from navigation.ins.attitude import rotation_matrix_to_quaternion
from navigation.preprocessing.pipeline import PreprocessingPipeline
from navigation.eskf.gating import MahalanobisGating
from navigation.eskf.predict import ProcessNoiseConfig, predict_eskf
from navigation.eskf.state import ESKFNominalState, ESKFState
from navigation.eskf.measurements.gnss import GNSSMeasurementModel, GNSSUpdateConfig
from navigation.eskf.measurements.zupt import (
    ClassicalZUPTDetector,
    ZUPTMeasurementConfig,
    ZUPTMeasurementModel,
)


class TestESKFRealDataIntegration:
    """Integration test suite executing ESKF fusion on real driving datasets."""

    @classmethod
    def setup_class(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[2]
        cls.npz_path = cls.project_root / "data" / "cache" / "iovnbd" / "Uncategorised_S1.npz"

        if not cls.npz_path.exists():
            pytest.skip(f"Authoritative cached trip not found at {cls.npz_path}")

        cls.trip = SynchronizedTrip.load_npz(cls.npz_path)

        # Run Phase 3 Preprocessing once for the trip
        detector = StationaryDetector()
        segs, stat_mask = detector.detect(cls.trip.timestamps_ns, cls.trip.accel_raw, cls.trip.gyro_raw)

        pipeline = PreprocessingPipeline(sampling_rate_hz=10.0, filter_cutoff_hz=3.0, median_window_size=3)
        cls.preprocessed = pipeline.process_trip(cls.trip, stationary_mask=stat_mask)

    def test_eskf_gnss_measurable_improvement_over_phase4_baseline(self) -> None:
        """Phase 5 ESKF + GNSS + gated ZUPT must demonstrate statistically measurable

        improvement over the frozen Phase 4 open-loop baseline (3,249.32 m error, 1,487.53 m RMSE).
        """
        start_sample_idx = 19500
        end_sample_idx = 20100
        win_slice = slice(start_sample_idx, end_sample_idx + 1)

        ts_win = self.preprocessed.timestamps_ns[win_slice]
        f_win = self.preprocessed.f_m_v[win_slice]
        w_win = self.preprocessed.omega_m_v[win_slice]

        # 1. Ground Truth Setup (matches Phase 4 ablation exactly)
        v_alt_true_m = self.trip.v_ref_alt_m[win_slice] / 1000.0
        ref_lat = float(self.trip.v_ref_lat[start_sample_idx])
        ref_lon = float(self.trip.v_ref_lon[start_sample_idx])
        ref_alt = float(v_alt_true_m[0])
        geo_ref = GeoReference(lat_ref=ref_lat, lon_ref=ref_lon, alt_ref=ref_alt)

        gt_east, gt_north, gt_up = geo_ref.geodetic_to_enu(
            self.trip.v_ref_lat[win_slice],
            self.trip.v_ref_lon[win_slice],
            v_alt_true_m,
        )

        # 2. Oracle Initial Conditions (matches Phase 4 ablation exactly)
        init_speed_mps = float(self.trip.v_ref_speed_mps[start_sample_idx])
        init_heading_deg = float(self.trip.v_ref_heading_deg[start_sample_idx])
        psi_track = math.radians(init_heading_deg)
        v_e0 = float(init_speed_mps * math.sin(psi_track))
        v_n0 = float(init_speed_mps * math.cos(psi_track))
        v_u0 = 0.0

        R_init = np.array([
            [math.sin(psi_track), -math.cos(psi_track), 0.0],
            [math.cos(psi_track),  math.sin(psi_track), 0.0],
            [0.0,                  0.0,                 1.0],
        ], dtype=np.float64)
        q_init = rotation_matrix_to_quaternion(R_init)

        # 3. ESKF State & Models Initialization
        nom = ESKFNominalState.from_components(
            position_enu=[0.0, 0.0, 0.0],
            velocity_enu=[v_e0, v_n0, v_u0],
            q=q_init,
            gyro_bias=self.preprocessed.calibration.gyro_bias,
            timestamp_ns=int(ts_win[0]),
        )
        P0 = np.eye(15, dtype=np.float64) * 0.1
        P0[0:3, 0:3] = np.eye(3) * 10.0
        P0[3:6, 3:6] = np.eye(3) * 1.0
        P0[6:9, 6:9] = np.eye(3) * 0.01

        state = ESKFState(nominal=nom, covariance=P0)

        # GNSS config with realistic vertical uncertainty for consumer smartphone
        cfg = GNSSUpdateConfig(min_vertical_accuracy_m=35.0, default_vertical_accuracy_m=50.0)
        gnss_model = GNSSMeasurementModel(
            geo_reference=geo_ref,
            config=cfg,
            gating=MahalanobisGating(confidence_level=0.999),
        )
        proc_noise = ProcessNoiseConfig(gyro_noise_std=0.04, accel_noise_std=0.4)


        zupt_detector = ClassicalZUPTDetector()
        zupt_model = ZUPTMeasurementModel()

        est_positions = [state.position_enu.copy()]
        pos_accepted = 0
        pos_rejected = 0
        vel_accepted = 0
        vel_rejected = 0
        zupt_accepted = 0
        zupt_rejected = 0

        prev_fix_sig = None

        # 4. Filter Loop over Evaluation Window
        for k in range(len(ts_win) - 1):
            dt = float((ts_win[k+1] - ts_win[k]) * 1e-9)
            t_ns = int(ts_win[k+1])

            # Prediction
            state = predict_eskf(
                state=state,
                f_m_v=f_win[k],
                omega_m_v=w_win[k],
                dt=dt,
                timestamp_ns=t_ns,
                process_noise=proc_noise,
            )

            # GNSS Fix Arrival Check: robust multi-field comparison (lat, lon, alt, speed, bearing)
            # avoids fragile lat-only checks and avoids duplicate fusion of held samples.
            lat_k = float(self.trip.s_gnss_lat[start_sample_idx + k + 1])
            lon_k = float(self.trip.s_gnss_lon[start_sample_idx + k + 1])
            alt_k = float(self.trip.s_gnss_alt[start_sample_idx + k + 1])
            acc_h = float(self.trip.s_gnss_accuracy_m[start_sample_idx + k + 1])
            spd_k = float(self.trip.s_gnss_speed_mps[start_sample_idx + k + 1])
            brg_k = float(self.trip.s_gnss_bearing_deg[start_sample_idx + k + 1])

            fix_sig = (lat_k, lon_k, alt_k, spd_k, brg_k)
            if prev_fix_sig is None:
                prev_fix_sig = fix_sig
            elif fix_sig != prev_fix_sig:
                prev_fix_sig = fix_sig

                # 4a. GNSS 3D Position Update
                state, d_pos = gnss_model.update_position(
                    state=state,
                    lat=lat_k,
                    lon=lon_k,
                    alt=alt_k,
                    accuracy_h_m=acc_h,
                    timestamp_ns=t_ns,
                )
                if d_pos.applied:
                    pos_accepted += 1
                else:
                    pos_rejected += 1

                # 4b. GNSS 2D Horizontal Velocity Update (from course speed + bearing)
                # Note: Phone GPS provides horizontal speed and course over ground; it does
                # not provide measured vertical velocity. Fusing as a 2D measurement (v_E, v_N)
                # correctly updates horizontal states without falsely constraining vertical velocity.
                # In raw Android S-file, speed was logged in m/s under 'GPS SPEED (Kmh)' label;
                # Phase 2 ingestion divided by 3.6 per column header. Multiplying by 3.6
                # restores the true physical speed in m/s without mutating Phase 2 cache.
                if math.isfinite(spd_k) and math.isfinite(brg_k) and spd_k >= 0.0:
                    true_speed_mps = spd_k * 3.6
                    state, d_vel = gnss_model.update_horizontal_velocity_from_course(
                        state=state,
                        speed_mps=true_speed_mps,
                        bearing_deg=brg_k,
                        accuracy_speed_mps=0.5,
                        timestamp_ns=t_ns,
                    )
                    if d_vel.applied:
                        vel_accepted += 1
                    else:
                        vel_rejected += 1

            # Classical ZUPT Check
            diag_z = zupt_detector.push(omega_v=w_win[k+1], f_v=f_win[k+1])
            if diag_z.is_stationary:
                state, d_z = zupt_model.update(state, timestamp_ns=t_ns)
                if d_z.applied:
                    zupt_accepted += 1
                else:
                    zupt_rejected += 1

            # Assert covariance integrity at each step
            asym = np.max(np.abs(state.covariance - state.covariance.T))
            assert asym < 1e-11, f"Covariance asymmetry at step {k}: {asym}"
            min_eig = np.min(np.linalg.eigvalsh(state.covariance))
            assert min_eig >= -1e-11, f"Covariance lost PSD at step {k}: min_eig={min_eig}"

            est_positions.append(state.position_enu.copy())

        # 5. Error Metrics vs. Phase 4 Baseline
        est_positions_arr = np.array(est_positions)
        err_e = est_positions_arr[:, 0] - gt_east
        err_n = est_positions_arr[:, 1] - gt_north
        err_u = est_positions_arr[:, 2] - gt_up
        err_horiz = np.sqrt(err_e ** 2 + err_n ** 2)
        err_3d = np.sqrt(err_e ** 2 + err_n ** 2 + err_u ** 2)

        final_error_m = float(err_horiz[-1])
        max_error_m = float(np.max(err_horiz))
        rmse_error_m = float(np.sqrt(np.mean(err_horiz ** 2)))
        final_vert_err_m = float(abs(err_u[-1]))
        vert_rmse_m = float(np.sqrt(np.mean(err_u ** 2)))
        final_3d_err_m = float(err_3d[-1])
        rmse_3d_m = float(np.sqrt(np.mean(err_3d ** 2)))

        phase4_baseline_final_err = 3249.32
        phase4_baseline_rmse = 1487.53
        phase4_baseline_final_3d_err = 3292.29
        phase4_baseline_3d_rmse = 1502.42

        improvement_pct = (1.0 - final_error_m / phase4_baseline_final_err) * 100.0
        improvement_rmse_pct = (1.0 - rmse_error_m / phase4_baseline_rmse) * 100.0
        improvement_3d_final_pct = (1.0 - final_3d_err_m / phase4_baseline_final_3d_err) * 100.0
        improvement_3d_rmse_pct = (1.0 - rmse_3d_m / phase4_baseline_3d_rmse) * 100.0

        print("\n--- Phase 5 Real Data Validation Results ---")
        print(f"Final Horizontal Error: {final_error_m:.2f} m (Phase 4 baseline: {phase4_baseline_final_err:.2f} m)")
        print(f"Horizontal RMSE: {rmse_error_m:.2f} m (Phase 4 baseline: {phase4_baseline_rmse:.2f} m)")
        print(f"Max Horizontal Error: {max_error_m:.2f} m")
        print(f"Final Vertical Error: {final_vert_err_m:.2f} m")
        print(f"Vertical RMSE: {vert_rmse_m:.2f} m")
        print(f"Final 3D Error: {final_3d_err_m:.2f} m (Phase 4 baseline: {phase4_baseline_final_3d_err:.2f} m)")
        print(f"3D RMSE: {rmse_3d_m:.2f} m (Phase 4 baseline: {phase4_baseline_3d_rmse:.2f} m)")
        print(f"Improvement over Phase 4 (Final Horizontal): {improvement_pct:.2f}%")
        print(f"Improvement over Phase 4 (Horizontal RMSE): {improvement_rmse_pct:.2f}%")
        print(f"Improvement over Phase 4 (Final 3D): {improvement_3d_final_pct:.2f}%")
        print(f"Improvement over Phase 4 (3D RMSE): {improvement_3d_rmse_pct:.2f}%")
        print(f"GNSS Position Accepted: {pos_accepted}, Rejected: {pos_rejected}")
        print(f"GNSS Horizontal Velocity Accepted: {vel_accepted}, Rejected: {vel_rejected}")
        print(f"ZUPT Updates Accepted: {zupt_accepted}, Rejected: {zupt_rejected}")

        # Assert both position and 2D horizontal velocity updates were accepted on real data
        assert pos_accepted > 0, "No GNSS position updates were accepted"
        assert vel_accepted > 0, "No GNSS horizontal velocity updates were accepted"

        # Assert statistically measurable improvement over Phase 4
        assert final_error_m < phase4_baseline_final_err, (
            f"Phase 5 final error ({final_error_m:.2f} m) did not improve upon "
            f"Phase 4 open-loop baseline ({phase4_baseline_final_err:.2f} m)"
        )
        assert rmse_error_m < phase4_baseline_rmse, (
            f"Phase 5 RMSE ({rmse_error_m:.2f} m) did not improve upon "
            f"Phase 4 open-loop baseline ({phase4_baseline_rmse:.2f} m)"
        )
        assert final_3d_err_m < phase4_baseline_final_3d_err, (
            f"Phase 5 final 3D error ({final_3d_err_m:.2f} m) did not improve upon "
            f"Phase 4 open-loop 3D baseline ({phase4_baseline_final_3d_err:.2f} m)"
        )
        assert rmse_3d_m < phase4_baseline_3d_rmse, (
            f"Phase 5 3D RMSE ({rmse_3d_m:.2f} m) did not improve upon "
            f"Phase 4 open-loop 3D baseline ({phase4_baseline_3d_rmse:.2f} m)"
        )
        # Verify improvement is substantial (> 20%)
        assert improvement_pct > 20.0, f"Expected >20% horizontal final improvement, got {improvement_pct:.2f}%"
        assert improvement_3d_final_pct > 20.0, f"Expected >20% 3D final improvement, got {improvement_3d_final_pct:.2f}%"

    def test_eskf_zupt_standstill_suppression_on_real_data(self) -> None:
        """Classical Gated ZUPT suppresses stationary drift during vehicle rest."""
        # Initial rest segment of Trip S1: samples 50 to 150 (quiescent parking lot)
        start_idx = 50
        end_idx = 150
        win_slice = slice(start_idx, end_idx + 1)

        ts_win = self.preprocessed.timestamps_ns[win_slice]
        f_win = self.preprocessed.f_m_v[win_slice]
        w_win = self.preprocessed.omega_m_v[win_slice]

        # Initialize at zero velocity and position
        nom = ESKFNominalState.from_components(
            position_enu=[0.0, 0.0, 0.0],
            velocity_enu=[0.0, 0.0, 0.0],
            q=[1.0, 0.0, 0.0, 0.0],
            gyro_bias=self.preprocessed.calibration.gyro_bias,
            timestamp_ns=int(ts_win[0]),
        )
        P0 = np.eye(15, dtype=np.float64) * 0.01
        state = ESKFState(nominal=nom, covariance=P0)

        zupt_detector = ClassicalZUPTDetector()
        zupt_model = ZUPTMeasurementModel(ZUPTMeasurementConfig(velocity_noise_sigma=0.02))

        zupt_applied_count = 0

        for k in range(len(ts_win) - 1):
            dt = float((ts_win[k+1] - ts_win[k]) * 1e-9)
            t_ns = int(ts_win[k+1])

            state = predict_eskf(state, f_m_v=f_win[k], omega_m_v=w_win[k], dt=dt, timestamp_ns=t_ns)

            diag_z = zupt_detector.push(omega_v=w_win[k+1], f_v=f_win[k+1])
            if diag_z.is_stationary:
                state, d_z = zupt_model.update(state, timestamp_ns=t_ns)
                if d_z.applied:
                    zupt_applied_count += 1

        # In pure rest with ZUPT, velocity remains pegged near zero
        v_final_norm = np.linalg.norm(state.velocity_enu)
        assert zupt_applied_count > 50, f"Expected ZUPT to trigger frequently, triggered {zupt_applied_count} times"
        assert v_final_norm < 0.05, f"Expected velocity norm < 0.05 m/s, got {v_final_norm:.4f} m/s"
        pos_drift = np.linalg.norm(state.position_enu[0:2])
        assert pos_drift < 0.5, f"Expected horizontal drift < 0.5 m during standstill, got {pos_drift:.4f} m"
