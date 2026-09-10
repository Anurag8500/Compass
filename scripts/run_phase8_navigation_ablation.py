"""Phase 8 Controlled ESKF Navigation Outage Ablations.

Evaluates navigation performance during synthetic GNSS outages across 4 conditions:
A. Pure Phase 5 ESKF (INS propagation + gated ZUPT)
B. ESKF + VelocityNet v1.1 (1D-CNN + causal EMA alpha=0.2)
C. ESKF + VelocityNet v1.1 + BiasNet v1 (both models fused via innovation gating)
D. ESKF + BiasNet v1 (diagnostic: BiasNet without VelocityNet)

Outage durations: 10s, 30s, 60s.
Evaluates horizontal RMSE, final error, velocity RMSE, max excursion, and NIS.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import torch

from data.pipeline.sync import SynchronizedTrip
from data.pipeline.stationary_detect import StationaryDetector
from navigation.frames.local_geo import GeoReference
from navigation.ins.attitude import rotation_matrix_to_quaternion
from navigation.preprocessing.pipeline import PreprocessingPipeline
from navigation.eskf.gating import MahalanobisGating
from navigation.eskf.predict import predict_eskf
from navigation.eskf.state import ESKFNominalState, ESKFState
from navigation.eskf.update import eskf_update
from navigation.eskf.measurements.zupt import ClassicalZUPTDetector, ZUPTMeasurementConfig, ZUPTMeasurementModel
from ml.data.features import compute_canonical_features
from ml.data.normalization import FeatureNormalizer
from ml.data.resample import resample_to_canonical_10hz
from ml.models.baselines.cnn1d_velocity import CNN1DVelocityBaseline
from ml.models.biasnet import BiasNet


class CausalEMA:
    """Causal Exponential Moving Average for smoothing neural predictions online."""
    def __init__(self, alpha: float = 0.2) -> None:
        self.alpha = float(alpha)
        self.state: Optional[float] = None

    def update(self, val: float) -> float:
        if self.state is None:
            self.state = float(val)
        else:
            self.state = self.alpha * float(val) + (1.0 - self.alpha) * self.state
        return self.state

    def reset(self) -> None:
        self.state = None


def run_outage_simulation(
    trip_path: str | Path,
    outage_start_s: float = 30.0,
    outage_durations: Sequence[float] = (10.0, 30.0, 60.0),
) -> Dict[str, Any]:
    """Execute controlled GNSS outage ablations on a driving segment."""
    root = Path(".")
    trip = SynchronizedTrip.load_npz(trip_path)
    detector = StationaryDetector()
    _, stat_mask = detector.detect(trip.timestamps_ns, trip.accel_raw, trip.gyro_raw)
    pipeline = PreprocessingPipeline(sampling_rate_hz=10.0, filter_cutoff_hz=3.0, median_window_size=3)
    preprocessed = pipeline.process_trip(trip, stationary_mask=stat_mask)

    aux = {
        "v_ref_speed_mps": trip.v_ref_speed_mps,
        "v_ref_lat": trip.v_ref_lat,
        "v_ref_lon": trip.v_ref_lon,
        "v_ref_alt_m": trip.v_ref_alt_m / 1000.0,
        "v_ref_heading_deg": trip.v_ref_heading_deg,
    }

    res = resample_to_canonical_10hz(
        timestamps_ns=preprocessed.timestamps_ns,
        f_m_v=preprocessed.f_m_v,
        omega_m_v=preprocessed.omega_m_v,
        is_validated=preprocessed.is_validated,
        aux_signals=aux,
    )

    f_10hz = res.f_m_v
    w_10hz = res.omega_m_v
    ts_10hz = res.timestamps_ns
    val_10hz = res.is_validated

    lat_10hz = res.aux_signals["v_ref_lat"]
    lon_10hz = res.aux_signals["v_ref_lon"]
    alt_10hz = res.aux_signals["v_ref_alt_m"]
    spd_10hz = res.aux_signals["v_ref_speed_mps"]
    hdg_10hz = res.aux_signals["v_ref_heading_deg"]

    # Features and normalization for ML models
    features_9ch = compute_canonical_features(ts_10hz, f_10hz, w_10hz)
    norm_path = root / "data" / "ml_dataset_v1" / "normalization.json"
    normalizer = FeatureNormalizer.load_json(norm_path)
    norm_features = normalizer.transform(features_9ch).astype(np.float32)

    # Load frozen VelocityNet v1.1
    vnet = CNN1DVelocityBaseline(input_dim=9, channels=(48, 64, 64), kernel_size=3, dense_dim=32)
    vnet_ckpt = torch.load(root / "models" / "velocitynet_v1_1_best.pt", map_location="cpu", weights_only=True)
    vnet_sd = vnet_ckpt["model_state_dict"] if isinstance(vnet_ckpt, dict) and "model_state_dict" in vnet_ckpt else vnet_ckpt
    vnet.load_state_dict(vnet_sd)
    vnet.eval()

    # Load frozen BiasNet v1
    bnet = BiasNet(input_dim=9, hidden_dim=48, num_layers=2, dense_dim=24)
    bnet_ckpt = torch.load(root / "models" / "biasnet_v1_best.pt", map_location="cpu", weights_only=True)
    bnet_sd = bnet_ckpt["model_state_dict"] if isinstance(bnet_ckpt, dict) and "model_state_dict" in bnet_ckpt else bnet_ckpt
    bnet.load_state_dict(bnet_sd)
    bnet.eval()

    # Ground truth reference ENU
    geo_ref = GeoReference(lat_ref=float(lat_10hz[0]), lon_ref=float(lon_10hz[0]), alt_ref=float(alt_10hz[0]))
    gt_e, gt_n, gt_u = geo_ref.geodetic_to_enu(lat_10hz, lon_10hz, alt_10hz)

    zupt_model = ZUPTMeasurementModel(config=ZUPTMeasurementConfig())
    zupt_det = ClassicalZUPTDetector()

    outage_results = {}

    for dur in outage_durations:
        start_idx = int(round(outage_start_s * 10.0))
        end_idx = start_idx + int(round(dur * 10.0))
        if end_idx >= len(ts_10hz):
            continue

        cond_results = {}
        for cond in ["A_pure_eskf", "B_eskf_vnet", "C_eskf_vnet_bnet", "D_eskf_bnet"]:
            # Initialize state at start_idx
            psi0 = math.radians(float(hdg_10hz[start_idx]))
            spd0 = float(spd_10hz[start_idx])
            v_init = np.array([spd0 * math.sin(psi0), spd0 * math.cos(psi0), 0.0], dtype=np.float64)
            R0 = np.array([
                [math.sin(psi0), -math.cos(psi0), 0.0],
                [math.cos(psi0),  math.sin(psi0), 0.0],
                [0.0,             0.0,            1.0],
            ])
            q0 = rotation_matrix_to_quaternion(R0)
            p0 = np.array([float(gt_e[start_idx]), float(gt_n[start_idx]), float(gt_u[start_idx])], dtype=np.float64)

            nom = ESKFNominalState.from_components(
                position_enu=p0,
                velocity_enu=v_init,
                q=q0,
                gyro_bias=preprocessed.calibration.gyro_bias,
                timestamp_ns=int(ts_10hz[start_idx]),
            )
            P0 = np.diag([1.0, 1.0, 4.0, 0.1, 0.1, 0.5, 0.01, 0.01, 0.05, 0.05, 0.05, 0.05, 0.005, 0.005, 0.005]) ** 2
            state = ESKFState(nominal=nom, covariance=P0)

            ema = CausalEMA(alpha=0.2)
            pos_history = [state.nominal.position_enu]
            vel_history = [state.nominal.velocity_enu]
            nis_list: List[float] = []

            # Conservative empirical measurement covariances
            R_v_val = 2.25  # (1.5 m/s)^2
            R_b_diag = np.array([1.21, 1.21, 1.21, 0.0025, 0.0025, 0.0025], dtype=np.float64)
            R_b_mat = np.diag(R_b_diag)

            for k in range(start_idx, end_idx):
                dt = (ts_10hz[k + 1] - ts_10hz[k]) * 1e-9
                state = predict_eskf(state, f_10hz[k], w_10hz[k], dt, int(ts_10hz[k + 1]))

                # Standstill ZUPT check
                zupt_diag = zupt_det.push(omega_v=w_10hz[k], f_v=f_10hz[k])
                if zupt_diag.is_stationary:
                    state, d_z = zupt_model.update(state, timestamp_ns=int(ts_10hz[k + 1]))

                # Neural updates every 5 samples (0.5 s) when window is available
                if (k + 1 - start_idx) % 5 == 0 and (k + 1) >= 20:
                    win_slice = norm_features[k + 1 - 20 : k + 1]  # (20, 9)
                    win_tensor = torch.tensor(win_slice[np.newaxis, ...], dtype=torch.float32)

                    # 1. VelocityNet Update (Conditions B & C)
                    if cond in ["B_eskf_vnet", "C_eskf_vnet_bnet"]:
                        with torch.no_grad():
                            raw_spd, _ = vnet(win_tensor)
                            spd_pred = float(raw_spd.item())
                        spd_smoothed = ema.update(spd_pred)

                        if spd_smoothed > 0.5:  # Motion gating: suppress near standstill
                            R_v_n = state.nominal.R_v_n
                            fwd_n = R_v_n[:, 0]  # vehicle forward in ENU

                            # Measurement: forward speed projection along fwd_n
                            z_v = np.array([spd_smoothed], dtype=np.float64)
                            h_v = np.array([float(np.dot(fwd_n, state.nominal.velocity_enu))], dtype=np.float64)
                            H_v = np.zeros((1, 15), dtype=np.float64)
                            H_v[0, 3:6] = fwd_n
                            R_v = np.array([[R_v_val]], dtype=np.float64)

                            gating = MahalanobisGating(threshold_override=16.0)
                            state, d_v = eskf_update(state, z_v, h_v, H_v, R_v, gating=gating)
                            if d_v.applied and d_v.gating is not None:
                                nis_list.append(float(d_v.gating.mahalanobis_sq))

                    # 2. BiasNet Update (Conditions C & D)
                    if cond in ["C_eskf_vnet_bnet", "D_eskf_bnet"]:
                        with torch.no_grad():
                            db_pred = bnet(win_tensor).cpu().numpy()[0]  # (6,)

                        # Pseudo-measurement of bias: z_b = b_nom + db_pred
                        z_b = np.concatenate([state.nominal.accel_bias, state.nominal.gyro_bias]) + db_pred
                        h_b = np.concatenate([state.nominal.accel_bias, state.nominal.gyro_bias])
                        H_b = np.zeros((6, 15), dtype=np.float64)
                        H_b[0:3, 9:12] = np.eye(3)
                        H_b[3:6, 12:15] = np.eye(3)

                        gating_b = MahalanobisGating(threshold_override=25.0)
                        state, d_b = eskf_update(state, z_b, h_b, H_b, R_b_mat, gating=gating_b)
                        if d_b.applied and d_b.gating is not None:
                            nis_list.append(float(d_b.gating.mahalanobis_sq))

                pos_history.append(state.nominal.position_enu)
                vel_history.append(state.nominal.velocity_enu)

            # Compute Outage Metrics
            pos_arr = np.array(pos_history)
            gt_pos = np.column_stack([gt_e[start_idx : end_idx + 1], gt_n[start_idx : end_idx + 1], gt_u[start_idx : end_idx + 1]])

            h_err = np.linalg.norm(pos_arr[:, 0:2] - gt_pos[:, 0:2], axis=1)
            final_h_err = float(h_err[-1])
            rmse_h = float(math.sqrt(np.mean(h_err ** 2)))
            max_h_err = float(np.max(h_err))

            gt_spd = spd_10hz[start_idx : end_idx + 1]
            pred_spd = np.linalg.norm(np.array(vel_history)[:, 0:2], axis=1)
            rmse_vel = float(math.sqrt(np.mean((pred_spd - gt_spd) ** 2)))

            cond_results[cond] = {
                "final_horizontal_error_m": final_h_err,
                "horizontal_rmse_m": rmse_h,
                "max_horizontal_excursion_m": max_h_err,
                "velocity_rmse_mps": rmse_vel,
                "mean_nis": float(np.mean(nis_list)) if nis_list else 0.0,
                "p95_nis": float(np.percentile(nis_list, 95)) if nis_list else 0.0,
                "updates_applied": len(nis_list),
            }

        outage_results[f"outage_{int(dur)}s"] = cond_results

    return outage_results


def main() -> None:
    test_trip = "data/cache/iovnbd/Categorised_S1.npz"
    print(f"=== Running Phase 8 Navigation Outage Ablations on {test_trip} ===")
    results = run_outage_simulation(test_trip, outage_start_s=25.0, outage_durations=[10.0, 30.0, 60.0])

    print("\n" + "=" * 90)
    print(f"{'Outage':<10} | {'Metric':<25} | {'A (Pure ESKF)':<13} | {'B (+VNet)':<13} | {'C (+VNet+BNet)':<15} | {'D (+BNet only)':<13}")
    print("=" * 90)

    for out_name, conds in results.items():
        dur_s = out_name.replace("outage_", "")
        print(f"\n--- GNSS OUTAGE DURATION: {dur_s} ---")
        for metric_name, label in [
            ("horizontal_rmse_m", "Horizontal RMSE (m)"),
            ("final_horizontal_error_m", "Final Error (m)"),
            ("max_horizontal_excursion_m", "Max Excursion (m)"),
            ("velocity_rmse_mps", "Velocity RMSE (m/s)"),
            ("mean_nis", "Mean NIS"),
        ]:
            val_a = conds["A_pure_eskf"][metric_name]
            val_b = conds["B_eskf_vnet"][metric_name]
            val_c = conds["C_eskf_vnet_bnet"][metric_name]
            val_d = conds["D_eskf_bnet"][metric_name]
            print(f"{dur_s:<10} | {label:<25} | {val_a:<13.3f} | {val_b:<13.3f} | {val_c:<15.3f} | {val_d:<13.3f}")

    out_file = Path("docs/biasnet_navigation_ablation_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved ablation metrics to {out_file}")


if __name__ == "__main__":
    main()
