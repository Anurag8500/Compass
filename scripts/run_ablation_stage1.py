"""Ablation Stage 1: Classical Open-Loop Strapdown INS Drift Benchmark (Phase 4).

Executes pure unassisted dead-reckoning on a real IO-VNBD driving segment:
    - NO GNSS Kalman updates
    - NO ESKF covariance/corrections
    - NO VelocityNet
    - NO BiasNet
    - NO Non-Holonomic Constraints (NHC)
    - NO Map Matching
    - NO Zero-Velocity Updates (ZUPT)

Quantifies the empirical baseline drift of low-cost smartphone inertial sensors
under open-loop double-integration, establishing the authoritative Level 1
ablation baseline for SIH 2026 Problem Statement 26168 (ISRO).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.pipeline.stationary_detect import StationaryDetector
from data.pipeline.sync import SynchronizedTrip
from navigation.frames.local_geo import GeoReference
from navigation.ins.attitude import rotation_matrix_to_quaternion
from navigation.ins.propagation import StrapdownINS
from navigation.preprocessing.pipeline import PreprocessingPipeline


def run_ablation_stage1(
    trip_path: Path,
    start_sample_idx: int = 19500,
    duration_seconds: float = 60.0,
    output_plot_path: Optional[Path] = None,
    output_json_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute Ablation Stage 1 open-loop drift experiment on a real driving sequence."""

    # 1. Load real synchronized trip
    print(f"[Ablation Stage 1] Loading synchronized trip from: {trip_path}")
    trip = SynchronizedTrip.load_npz(trip_path)
    total_samples = trip.row_count

    # 2. Run Phase 3 Preprocessing (Calibration, Tilt Alignment, Denoising Filter)
    print("[Ablation Stage 1] Running Phase 3 Classical Preprocessing...")
    detector = StationaryDetector()
    segs, stat_mask = detector.detect(trip.timestamps_ns, trip.accel_raw, trip.gyro_raw)

    pipeline = PreprocessingPipeline(
        sampling_rate_hz=10.0,
        filter_cutoff_hz=3.0,
        median_window_size=3,
    )
    preprocessed = pipeline.process_trip(trip, stationary_mask=stat_mask)
    calib = preprocessed.calibration

    # 3. Define Evaluation Window
    t_ns = preprocessed.timestamps_ns
    dt_from_start = (t_ns - t_ns[start_sample_idx]) * 1e-9
    end_indices = np.where(dt_from_start >= duration_seconds)[0]
    end_sample_idx = int(end_indices[0]) if len(end_indices) > 0 else total_samples - 1

    actual_duration = (t_ns[end_sample_idx] - t_ns[start_sample_idx]) * 1e-9
    sample_count = end_sample_idx - start_sample_idx + 1

    print(
        f"[Ablation Stage 1] Evaluation Window: [{start_sample_idx}, {end_sample_idx}] "
        f"({sample_count} samples, {actual_duration:.2f} seconds)"
    )

    # Slice evaluation arrays
    win_slice = slice(start_sample_idx, end_sample_idx + 1)
    ts_win = preprocessed.timestamps_ns[win_slice]
    f_win = preprocessed.f_m_v[win_slice]
    w_win = preprocessed.omega_m_v[win_slice]
    valid_win = preprocessed.is_validated[win_slice]

    # 4. Initialize Rigid Session Origin & Local ENU Frame
    # Use consistent Racelogic VBOX ground-truth coordinates at the evaluation start sample
    ref_lat = float(trip.v_ref_lat[start_sample_idx])
    ref_lon = float(trip.v_ref_lon[start_sample_idx])
    ref_alt = float(trip.v_ref_alt_m[start_sample_idx])
    geo_ref = GeoReference(lat_ref=ref_lat, lon_ref=ref_lon, alt_ref=ref_alt)

    # Convert Ground Truth reference trajectory to ENU using consistent VBOX fields
    gt_east, gt_north, gt_up = geo_ref.geodetic_to_enu(
        trip.v_ref_lat[win_slice],
        trip.v_ref_lon[win_slice],
        trip.v_ref_alt_m[win_slice],
    )

    # 5. Initialization Policy (Oracle Initial Condition for Open-Loop Benchmark)
    # NOTE (Issues 10 & 11): Benchmark evaluates open-loop strapdown INS divergence
    # starting from a controlled initial condition. Initial velocity and heading are initialized
    # from VBOX ground truth at t=0 (oracle initialization) so that divergence reflects
    # sensor integration errors rather than startup alignment error. After t=0, NO aiding
    # or GNSS data is consumed.
    init_speed_mps = float(trip.v_ref_speed_mps[start_sample_idx])
    init_heading_deg = float(trip.v_ref_heading_deg[start_sample_idx])
    psi_track = math.radians(init_heading_deg)

    # In ENU (X=East, Y=North), vehicle forward is [sin(psi), cos(psi), 0]
    v_e0 = float(init_speed_mps * math.sin(psi_track))
    v_n0 = float(init_speed_mps * math.cos(psi_track))
    v_u0 = 0.0

    R_init = np.array([
        [math.sin(psi_track), -math.cos(psi_track), 0.0],
        [math.cos(psi_track),  math.sin(psi_track), 0.0],
        [0.0,                  0.0,                 1.0],
    ], dtype=np.float64)
    q_init = rotation_matrix_to_quaternion(R_init)

    print("[Ablation Stage 1] Benchmark Mode: Open-loop propagation with oracle initial velocity and heading.")
    print(f"[Ablation Stage 1] Session Reference Origin: lat={ref_lat:.7f}°, lon={ref_lon:.7f}°, alt={ref_alt:.2f} m")
    print(f"[Ablation Stage 1] Initial Velocity: [{v_e0:.3f}, {v_n0:.3f}, {v_u0:.3f}] m/s (from VBOX speed {init_speed_mps:.3f} m/s and heading {init_heading_deg:.2f}°)")
    print(f"[Ablation Stage 1] Initial Heading: {init_heading_deg:.2f}° (VBOX Ground Truth)")
    print(f"[Ablation Stage 1] Initial Quaternion: {q_init}")

    # 6. Execute Open-Loop Strapdown INS Propagation
    ins = StrapdownINS(
        initial_position=[0.0, 0.0, 0.0],
        initial_velocity=[v_e0, v_n0, v_u0],
        initial_q=q_init,
        initial_accel_bias=[0.0, 0.0, 0.0],  # Nominal prior
        initial_timestamp_ns=int(ts_win[0]),
    )

    traj = ins.propagate_trajectory(
        timestamps_ns=ts_win,
        f_m_v=f_win,
        omega_m_v=w_win,
        is_validated=valid_win,
    )

    # 7. Compute Drift vs. Time
    pos_prop = traj.positions_enu
    err_e = pos_prop[:, 0] - gt_east
    err_n = pos_prop[:, 1] - gt_north
    err_u = pos_prop[:, 2] - gt_up

    err_horiz = np.sqrt(err_e ** 2 + err_n ** 2)
    err_3d = np.sqrt(err_e ** 2 + err_n ** 2 + err_u ** 2)
    elapsed_time = (ts_win - ts_win[0]) * 1e-9

    # Key Checkpoint Drifts
    checkpoints = {}
    for target_t in [5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0]:
        if target_t <= actual_duration:
            k = int(np.argmin(np.abs(elapsed_time - target_t)))
            checkpoints[f"drift_{int(target_t)}s_m"] = round(float(err_horiz[k]), 2)

    final_horiz_err = float(err_horiz[-1])
    final_3d_err = float(err_3d[-1])
    rmse_horiz = float(np.sqrt(np.mean(err_horiz ** 2)))
    max_horiz_err = float(np.max(err_horiz))

    print("\n" + "=" * 60)
    print("ABLATION STAGE 1: REAL-DATA OPEN-LOOP RESULTS")
    print("=" * 60)
    print(f"Trip ID:                      {trip.trip_id}")
    print(f"Duration:                     {actual_duration:.2f} s")
    print(f"Samples Integrated:           {traj.steps_integrated}")
    print(f"Final Horizontal Error:       {final_horiz_err:.2f} m")
    print(f"Final 3D Error:               {final_3d_err:.2f} m")
    print(f"Horizontal RMSE:              {rmse_horiz:.2f} m")
    print(f"Max Horizontal Error:         {max_horiz_err:.2f} m")
    for cp_key, cp_val in checkpoints.items():
        print(f"  - Checkpoint {cp_key}: {cp_val} m")
    print("=" * 60 + "\n")

    # 8. Generate Drift vs Time Plot
    if output_plot_path is not None:
        output_plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Subplot 1: Drift vs Time
        axes[0].plot(elapsed_time, err_horiz, label="Horizontal Drift (2D)", color="#d9534f", linewidth=2)
        axes[0].plot(elapsed_time, np.abs(err_u), label="Vertical Drift (Up)", color="#337ab7", linewidth=1.5, linestyle="--")
        axes[0].set_ylabel("Position Error (m)", fontsize=12)
        axes[0].set_title(
            f"Ablation Stage 1: Classical Open-Loop Strapdown INS Drift\n"
            f"Trip: {trip.trip_id} | Oracle Initialized (v0, q0) | Zero aiding afterwards",
            fontsize=13,
            fontweight="bold",
        )
        axes[0].grid(True, linestyle=":", alpha=0.6)
        axes[0].legend(loc="upper left")

        # Subplot 2: Trajectory Comparison in Local ENU
        axes[1].plot(elapsed_time, pos_prop[:, 0], label="INS East (prop)", color="#5cb85c")
        axes[1].plot(elapsed_time, gt_east, label="VBOX East (ref)", color="#5cb85c", linestyle=":")
        axes[1].plot(elapsed_time, pos_prop[:, 1], label="INS North (prop)", color="#f0ad4e")
        axes[1].plot(elapsed_time, gt_north, label="VBOX North (ref)", color="#f0ad4e", linestyle=":")
        axes[1].set_xlabel("Elapsed Time (s)", fontsize=12)
        axes[1].set_ylabel("Local Coordinate (m)", fontsize=12)
        axes[1].grid(True, linestyle=":", alpha=0.6)
        axes[1].legend(loc="upper left", ncol=2)

        fig.tight_layout()
        fig.savefig(output_plot_path, dpi=200)
        plt.close(fig)
        print(f"[Ablation Stage 1] Drift plot saved to: {output_plot_path}")

    # 9. Structure Results
    results: Dict[str, Any] = {
        "trip_id": trip.trip_id,
        "benchmark_mode": "open_loop_strapdown_ins_with_oracle_initialization",
        "ground_truth_source": "Racelogic VBOX (v_ref_lat, v_ref_lon, v_ref_alt_m, v_ref_speed_mps, v_ref_heading_deg)",
        "evaluation_window": {
            "start_sample_idx": start_sample_idx,
            "end_sample_idx": end_sample_idx,
            "sample_count": sample_count,
            "duration_s": round(actual_duration, 2),
        },
        "initial_conditions": {
            "ref_lat": ref_lat,
            "ref_lon": ref_lon,
            "ref_alt_m": ref_alt,
            "initial_speed_mps": round(init_speed_mps, 3),
            "initial_velocity_enu": [round(v_e0, 3), round(v_n0, 3), round(v_u0, 3)],
            "initial_heading_deg": round(init_heading_deg, 2),
            "initial_quaternion": [round(float(x), 6) for x in q_init],
            "accelerometer_bias_prior": [0.0, 0.0, 0.0],
            "gyro_bias_estimated": [round(float(x), 6) for x in calib.gyro_bias],
        },
        "drift_metrics": {
            "final_horizontal_error_m": round(final_horiz_err, 2),
            "final_3d_error_m": round(final_3d_err, 2),
            "horizontal_rmse_m": round(rmse_horiz, 2),
            "max_horizontal_error_m": round(max_horiz_err, 2),
            "checkpoints": checkpoints,
        },
        "steps_integrated": traj.steps_integrated,
        "steps_skipped": traj.steps_skipped,
    }

    if output_json_path is not None:
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[Ablation Stage 1] Numeric metrics saved to: {output_json_path}")

    return results


def main() -> None:
    """CLI entry point for running Ablation Stage 1 benchmark."""
    parser = argparse.ArgumentParser(description="Run Ablation Stage 1 Open-Loop INS Drift Experiment")
    parser.add_argument("--trip-file", type=str, default="data/cache/iovnbd/Categorised_S1.npz", help="Path to synchronized trip .npz")
    parser.add_argument("--start-idx", type=int, default=19500, help="Start sample index of evaluation window")
    parser.add_argument("--duration-s", type=float, default=60.0, help="Target evaluation window duration in seconds")
    parser.add_argument("--output-plot", type=str, default="docs/images/ablation_stage1_drift.png", help="Path to output plot PNG")
    parser.add_argument("--output-json", type=str, default="docs/ablation_stage1_metrics.json", help="Path to output metrics JSON")
    args = parser.parse_args()

    proj_root = Path(__file__).resolve().parents[1]
    trip_path = proj_root / args.trip_file
    plot_path = proj_root / args.output_plot
    json_path = proj_root / args.output_json

    run_ablation_stage1(
        trip_path=trip_path,
        start_sample_idx=args.start_idx,
        duration_seconds=args.duration_s,
        output_plot_path=plot_path,
        output_json_path=json_path,
    )


if __name__ == "__main__":
    main()
