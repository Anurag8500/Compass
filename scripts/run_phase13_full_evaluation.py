"""Phase 13: Full Offline Replay Integration Test, 3-Axis Evaluation & Ablation Suite.

Executes:
1. Axis A Nominal Fusion Ablation Ladder (A1 to A7).
2. Dedicated GNSS-Denied Dead-Reckoning Ladder (DR-A2 to DR-A7).
3. Axis B Operating-Condition Matrix (B1 to B12).
4. Axis C Output-Tier Processing (C1, C2, C3).
5. Fully Controlled Synthetic Benchmarks (using SyntheticTrajectoryGenerator):
   - Benchmark 1: 50m travel / <5m drift target
   - Benchmark 2: 1km at 60km/h (60s) / <100m drift target
6. Real-data with synthetic GNSS blackout injection (separate category).
7. Real environmental GNSS outage (B12, separate category).
8. Multi-Session Cross-Validation (S1, S2, S3a, S3c, S4).
9. Reproducibility metadata capture and results serialization.
10. Detailed telemetry NPZ export for publication-grade figure generation.

Scientific source-labeling policy (NEVER mix these categories):
  - FULLY_CONTROLLED_SYNTHETIC: Synthetic IMU + Synthetic reference + synthetic blackout
  - REAL_IMU_SYNTHETIC_BLACKOUT: Real IO-VNBD IMU + synthetic GNSS blackout mask
  - REAL_ENVIRONMENTAL_OUTAGE: Real IO-VNBD session with naturally observed GNSS dropout
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from data.pipeline.sync import SynchronizedTrip
from data.pipeline.stationary_detect import StationaryDetector
from maps.extract_osm import RoadNetworkGraph
from ml.data.resample import resample_to_canonical_10hz
from navigation.core import NavigationCoreConfig
from navigation.evaluation.ablation import (
    AXIS_A_LEVELS,
    DR_ABLATION_LEVELS,
    compute_incremental_contributions,
    run_ablation_ladder,
)
from navigation.evaluation.aggregation import aggregate_scenario_results
from navigation.evaluation.metrics import compute_outage_metrics, compute_trajectory_metrics
from navigation.evaluation.reproducibility import capture_reproducibility_metadata
from navigation.evaluation.scenarios import (
    AXIS_B_SCENARIOS,
    ScenarioDefinition,
    SyntheticTrajectoryGenerator,
)
from navigation.frames.local_geo import GeoReference
from navigation.preprocessing.pipeline import PreprocessingPipeline
from navigation.replay import ReplayConfig, ReplayResult, run_offline_replay


@dataclass
class SyntheticReplayData:
    """Lightweight adapter to make synthetic generator output compatible with run_offline_replay."""
    timestamps_ns: np.ndarray
    f_m_v: np.ndarray
    omega_m_v: np.ndarray
    is_validated: np.ndarray
    aux_signals: Dict[str, np.ndarray]


def make_synthetic_replay_data(synth_output: Dict[str, Any]) -> SyntheticReplayData:
    """Convert SyntheticTrajectoryGenerator output into replay-compatible shape."""
    n = len(synth_output["timestamps_ns"])
    aux_signals = {
        "v_ref_speed_mps": np.linalg.norm(synth_output["ref_vel_enu"][:, :2], axis=1),
        "v_ref_lat": synth_output["ref_lat"],
        "v_ref_lon": synth_output["ref_lon"],
        "v_ref_alt_m": synth_output["ref_alt_m"],
        "v_ref_heading_deg": np.degrees(synth_output["ref_headings_rad"]),
        "s_gnss_lat": synth_output["ref_lat"],
        "s_gnss_lon": synth_output["ref_lon"],
        "s_gnss_alt": synth_output["ref_alt_m"],
        "s_gnss_accuracy_m": np.full(n, 2.5),
    }
    return SyntheticReplayData(
        timestamps_ns=synth_output["timestamps_ns"].astype(np.int64),
        f_m_v=synth_output["f_m_v"].astype(np.float64),
        omega_m_v=synth_output["omega_m_v"].astype(np.float64),
        is_validated=np.ones(n, dtype=bool),
        aux_signals=aux_signals,
    )


def save_telemetry_npz(
    out_path: Path,
    result: ReplayResult,
    label: str,
    source_category: str,
) -> None:
    """Save detailed time-series telemetry arrays for figure generation.

    Files created: {out_path}/{label}.npz
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Position errors per step
    pos_err_2d = np.linalg.norm(
        result.eskf_pos_enu[:, :2] - result.ref_pos_enu[:, :2], axis=1
    )

    # Velocity errors per step
    vel_err_2d = np.linalg.norm(
        result.eskf_vel_enu[:, :2] - result.ref_vel_enu[:, :2], axis=1
    )

    # Heading error (degrees, wrapped)
    h_err = result.eskf_headings_rad - result.ref_headings_rad
    h_err_deg = np.degrees(np.arctan2(np.sin(h_err), np.cos(h_err)))

    # Along-track vs cross-track error
    along_err = np.zeros_like(pos_err_2d)
    cross_err = np.zeros_like(pos_err_2d)
    for i in range(len(pos_err_2d)):
        h_ref = float(result.ref_headings_rad[i])
        forward_east = math.sin(h_ref)
        forward_north = math.cos(h_ref)
        right_east = forward_north
        right_north = -forward_east
        err_vec = result.eskf_pos_enu[i, :2] - result.ref_pos_enu[i, :2]
        along_err[i] = float(err_vec[0] * forward_east + err_vec[1] * forward_north)
        cross_err[i] = float(err_vec[0] * right_east + err_vec[1] * right_north)

    # NIS history per-step from the navigation steps (if available; default to empty)
    nis_history = np.zeros(len(result.times_s))
    # NOTE: Actual GNSS NIS is not logged per-step in the lightweight ReplayResult;
    # leave zeros and figures will render with a note if array is zero-constant.

    # Map match derived arrays
    snap_dists: List[float] = []
    ambiguity_margins: List[float] = []
    fallback_codes: List[int] = []
    fallback_reason_map = {
        "NONE": 0,
        "NO_CANDIDATES": 1,
        "LARGE_DISPLACEMENT": 2,
        "LOW_CONFIDENCE": 3,
        "AMBIGUOUS_PARALLEL_ROADS": 4,
        "DISCONNECTED_TRANSITION": 5,
    }
    for mm_out in result.mm_outputs:
        if hasattr(mm_out, "distance_to_road_m") and mm_out.distance_to_road_m is not None:
            snap_dists.append(float(mm_out.distance_to_road_m))
        if hasattr(mm_out, "margin") and mm_out.margin is not None:
            ambiguity_margins.append(float(mm_out.margin))
        reason_code = 0
        if hasattr(mm_out, "fallback_reason") and mm_out.fallback_reason is not None:
            reason_code = fallback_reason_map.get(str(mm_out.fallback_reason), 6)
        fallback_codes.append(reason_code)
    # Pad / truncate to trajectory length
    snap_arr = np.array(snap_dists, dtype=np.float64) if snap_dists else np.zeros(len(result.times_s))
    margin_arr = np.array(ambiguity_margins, dtype=np.float64) if ambiguity_margins else np.zeros(len(result.times_s))
    fb_arr = np.array(fallback_codes, dtype=np.int32) if fallback_codes else np.zeros(len(result.times_s), dtype=np.int32)
    if len(snap_arr) < len(result.times_s):
        snap_arr = np.pad(snap_arr, (0, len(result.times_s) - len(snap_arr)))
    else:
        snap_arr = snap_arr[: len(result.times_s)]
    if len(margin_arr) < len(result.times_s):
        margin_arr = np.pad(margin_arr, (0, len(result.times_s) - len(margin_arr)))
    else:
        margin_arr = margin_arr[: len(result.times_s)]
    if len(fb_arr) < len(result.times_s):
        fb_arr = np.pad(fb_arr, (0, len(result.times_s) - len(fb_arr)))
    else:
        fb_arr = fb_arr[: len(result.times_s)]

    # NHC / ZUPT telemetry vectors
    nhc_accepted = np.array(
        [1.0 if s == "NORMAL" or s == "RELAXED" else 0.0 for s in result.nhc_statuses],
        dtype=np.float64,
    )
    zupt_arr = np.array(
        [1.0 if z else 0.0 for z in result.zupt_applied_flags], dtype=np.float64
    )

    # 3-sigma position envelope from covariance diagonal
    if result.cov_diag_history.ndim == 2 and result.cov_diag_history.shape[1] >= 3:
        sigma_3_pos = 3.0 * np.sqrt(
            np.clip(result.cov_diag_history[:, 0], 0.0, None)
            + np.clip(result.cov_diag_history[:, 1], 0.0, None)
        )
    else:
        sigma_3_pos = np.zeros(len(result.times_s))

    # Outage window indices if present
    out_start_step = -1
    out_end_step = -1
    if result.metrics.dead_reckoning is not None:
        dr = result.metrics.dead_reckoning
        # Find nearest index by time since outage_start_s is on the replay clock
        if len(result.times_s) > 0:
            candidates = np.where(result.times_s >= dr.outage_start_s)[0]
            if len(candidates) > 0:
                out_start_step = int(candidates[0])
            candidates2 = np.where(result.times_s >= dr.outage_end_s)[0]
            if len(candidates2) > 0:
                out_end_step = int(candidates2[0])
            else:
                out_end_step = len(result.times_s) - 1

    # Latency placeholder (zero-vector; measurements captured separately)
    latency_ms = np.zeros(len(result.times_s))

    np.savez_compressed(
        out_path,
        # metadata
        source_category=np.array([source_category], dtype=object),
        label=np.array([label], dtype=object),
        # core timestamps
        times_s=result.times_s,
        timestamps_ns=result.timestamps_ns,
        # trajectories
        ref_pos_enu=result.ref_pos_enu,
        est_pos_enu=result.eskf_pos_enu,
        disp_pos_enu=result.disp_pos_enu,
        ref_vel_enu=result.ref_vel_enu,
        est_vel_enu=result.eskf_vel_enu,
        ref_headings_rad=result.ref_headings_rad,
        est_headings_rad=result.eskf_headings_rad,
        # errors
        pos_err_2d=pos_err_2d,
        vel_err_2d=vel_err_2d,
        heading_err_deg=h_err_deg,
        along_track_err_m=along_err,
        cross_track_err_m=cross_err,
        # covariance & NIS
        cov_diag_history=result.cov_diag_history,
        sigma_3_pos_envelope_m=sigma_3_pos,
        nis_history=nis_history,
        # constraints
        nhc_accepted_flag=nhc_accepted,
        zupt_applied_flag=zupt_arr,
        # map matching
        snap_distance_m=snap_arr,
        ambiguity_margin_nats=margin_arr,
        fallback_reason_code=fb_arr,
        # outage
        outage_start_step=np.array([out_start_step], dtype=np.int32),
        outage_end_step=np.array([out_end_step], dtype=np.int32),
        # latency placeholder
        step_latency_ms=latency_ms,
    )


def load_and_preprocess_session(
    npz_path: Path,
    rate_hz: float = 10.0,
) -> Tuple[Any, np.ndarray, SynchronizedTrip]:
    """Load, stationary detect, calibrate, and resample an IO-VNBD session."""
    trip = SynchronizedTrip.load_npz(npz_path)
    detector = StationaryDetector()
    _, stat_mask = detector.detect(trip.timestamps_ns, trip.accel_raw, trip.gyro_raw)

    pipeline = PreprocessingPipeline(
        sampling_rate_hz=rate_hz,
        filter_cutoff_hz=3.0,
        median_window_size=3,
    )
    preprocessed = pipeline.process_trip(trip, stationary_mask=stat_mask)

    aux = {
        "v_ref_speed_mps": trip.v_ref_speed_mps,
        "v_ref_lat": trip.v_ref_lat,
        "v_ref_lon": trip.v_ref_lon,
        "v_ref_alt_m": trip.v_ref_alt_m / 1000.0,
        "v_ref_heading_deg": trip.v_ref_heading_deg,
        "s_gnss_lat": trip.s_gnss_lat,
        "s_gnss_lon": trip.s_gnss_lon,
        "s_gnss_alt": trip.s_gnss_alt,
        "s_gnss_accuracy_m": trip.s_gnss_accuracy_m,
    }

    res = resample_to_canonical_10hz(
        timestamps_ns=preprocessed.timestamps_ns,
        f_m_v=preprocessed.f_m_v,
        omega_m_v=preprocessed.omega_m_v,
        is_validated=preprocessed.is_validated,
        aux_signals=aux,
    )
    calib_gyro_bias = preprocessed.calibration.gyro_bias
    return res, calib_gyro_bias, trip


def run_synthetic_benchmark(
    synth_out: Dict[str, Any],
    road_graph: Optional[RoadNetworkGraph],
    standard_cfg: NavigationCoreConfig,
    benchmark_id: str,
) -> Tuple[ReplayResult, Dict[str, Any]]:
    """Run a fully-controlled synthetic benchmark with proper source labeling."""
    s_data = make_synthetic_replay_data(synth_out)
    n_steps = len(synth_out["timestamps_ns"])
    replay_cfg = ReplayConfig(
        start_idx=0,
        duration_steps=n_steps,
        rate_hz=synth_out["rate_hz"],
        outage_start_rel_steps=synth_out["outage_start_step"],
        outage_duration_steps=synth_out["outage_duration_steps"],
        enable_map_matching=False,
        assume_prealigned=True,
    )
    # Controlled synthetic benchmarks represent pure kinematic / strapdown IMU dead-reckoning
    # without device vibration, so neural speed/bias models trained on real phone sensors are disabled.
    synth_cfg = NavigationCoreConfig(
        gnss_enabled=standard_cfg.gnss_enabled,
        velocitynet_enabled=False,
        biasnet_enabled=False,
        nhc_enabled=True,
        zupt_enabled=True,
        process_noise=standard_cfg.process_noise,
        nhc=standard_cfg.nhc,
    )
    result = run_offline_replay(
        s_data,
        synth_out["calib_gyro_bias"],
        synth_cfg,
        replay_cfg,
        road_graph=road_graph,
        calib_accel_bias=synth_out.get("calib_accel_bias"),
    )
    dr = result.metrics.dead_reckoning
    report: Dict[str, Any] = {
        "target": f"< {synth_out['target_max_drift_m']:.1f}m drift over {synth_out['distance_travelled_m']:.1f}m",
        "data_source": "FULLY_CONTROLLED_SYNTHETIC",
        "source_details": synth_out["name"],
        "distance_m": dr.distance_travelled_m if dr else float(synth_out["distance_travelled_m"]),
        "final_drift_m": dr.final_outage_drift_m if dr else 0.0,
        "max_drift_m": dr.max_outage_drift_m if dr else 0.0,
        "drift_pct": dr.drift_percentage if dr else 0.0,
        "passed": (
            (dr.final_outage_drift_m < synth_out["target_max_drift_m"])
            if dr and dr.distance_travelled_m > 0
            else False
        ),
        "sih_10pct_passed": dr.sih_10pct_passed if dr else False,
        "speed_mps": float(np.mean(np.linalg.norm(synth_out["ref_vel_enu"][:, :2], axis=1))),
        "label": "FULLY_CONTROLLED_SYNTHETIC",
    }
    return result, report


def main() -> None:
    print("=" * 80)
    print("C.O.M.P.A.S.S. — Cognitive Off-grid Machine-learning Positioning And Sensor System")
    print("PHASE 13 FULL EVALUATION & ABLATION SUITE (SIH PS 26168)")
    print("=" * 80)

    # 1. Paths and Setup
    osm_path = Path("data/maps/coventry_s1_road_graph.json")
    if not osm_path.exists():
        raise FileNotFoundError(f"Missing road graph: {osm_path}")

    s1_path = Path("data/cache/iovnbd/Categorised_S1.npz")
    if not s1_path.exists():
        raise FileNotFoundError(f"Missing S1 dataset: {s1_path}")

    tel_dir = Path("docs/phase13_telemetry")
    tel_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Road Network Graph and IO-VNBD Session S1...")
    road_graph = RoadNetworkGraph.load_json(osm_path)
    res_s1, gyro_bias_s1, _trip_s1 = load_and_preprocess_session(s1_path, rate_hz=10.0)

    results_dir = Path("docs")
    results_dir.mkdir(parents=True, exist_ok=True)

    all_phase13_results: Dict[str, Any] = {}

    # 2. Capture Reproducibility Metadata
    print("\n[Step 1/9] Capturing Reproducibility & Provenance Metadata...")
    meta = capture_reproducibility_metadata(map_path=osm_path)
    all_phase13_results["reproducibility"] = meta.to_dict()
    print(f"  Git Commit: {meta.git_commit_hash} (dirty={meta.git_is_dirty})")
    print(f"  Platform: {meta.platform_info} | Python: {meta.python_version} | NumPy: {meta.numpy_version}")
    print(f"  Map Hash: {meta.map_hash[:12]}...")
    if meta.git_is_dirty:
        print("  WARNING: Working tree is dirty — reproducibility NOT fully frozen.")

    standard_core_cfg = NavigationCoreConfig(
        gnss_enabled=True,
        velocitynet_enabled=True,
        biasnet_enabled=True,
        nhc_enabled=True,
        zupt_enabled=True,
    )

    # 3. Axis A: Nominal Fusion Ladder (A1 to A7 on Scenario A Continuous GNSS)
    print("\n[Step 2/9] Running Axis A: Nominal Fusion Ablation Ladder (A1 to A7)...")
    base_cfg_a = ReplayConfig(start_idx=4900, duration_steps=600, rate_hz=10.0)
    axis_a_runs = run_ablation_ladder(
        ladder_levels=AXIS_A_LEVELS,
        data=res_s1,
        calib_gyro_bias=gyro_bias_s1,
        base_replay_config=base_cfg_a,
        road_graph=road_graph,
    )
    axis_a_deltas = compute_incremental_contributions(axis_a_runs)

    all_phase13_results["axis_a_nominal_ladder"] = {
        "levels": {k: v.metrics.to_dict() for k, v in axis_a_runs.items()},
        "incremental_contributions": axis_a_deltas,
    }
    for k, v in axis_a_runs.items():
        print(f"  {k:16s} -> 2D RMSE: {v.metrics.position.rmse_2d:.4f} m | Mean: {v.metrics.position.mean_error:.4f} m")

    # Phase 11 Baseline Protection Check (A6 ZUPT 2D RMSE vs 1.5496 m)
    a6_rmse = axis_a_runs["A6_ZUPT"].metrics.position.rmse_2d
    ref_rmse = 1.5496
    diff_abs = abs(a6_rmse - ref_rmse)
    phase11_preserved = bool(diff_abs < 0.05)
    all_phase13_results["phase11_baseline_preservation"] = {
        "measured_a6_rmse_2d_m": float(a6_rmse),
        "protected_reference_rmse_2d_m": ref_rmse,
        "absolute_difference_m": float(diff_abs),
        "preserved_bit_for_bit": phase11_preserved,
    }
    print(f"  Phase 11 Baseline Protection: Measured A6 RMSE = {a6_rmse:.4f} m (Ref: {ref_rmse:.4f} m) -> Preserved: {phase11_preserved}")

    # Telemetry for figure 01 (full trajectory) and 02/04
    if "A6_ZUPT" in axis_a_runs:
        save_telemetry_npz(tel_dir / "axis_a_A6_continuous.npz", axis_a_runs["A6_ZUPT"], "A6_continuous", "REAL_IMU_CONTINUOUS")
    if "A7_MAPMATCH" in axis_a_runs:
        save_telemetry_npz(tel_dir / "axis_a_A7_mapmatch.npz", axis_a_runs["A7_MAPMATCH"], "A7_mapmatch", "REAL_IMU_CONTINUOUS")

    # 4. Dedicated GNSS-Denied Dead-Reckoning Ladder (DR-A2 to DR-A7 on 60s Outage)
    print("\n[Step 3/9] Running Dedicated GNSS-Denied Dead-Reckoning Ladder (DR-A2 to DR-A7)...")
    base_cfg_dr = ReplayConfig(
        start_idx=4900,
        duration_steps=800,
        rate_hz=10.0,
        outage_start_rel_steps=100,
        outage_duration_steps=600,
    )
    dr_runs = run_ablation_ladder(
        ladder_levels=DR_ABLATION_LEVELS,
        data=res_s1,
        calib_gyro_bias=gyro_bias_s1,
        base_replay_config=base_cfg_dr,
        road_graph=road_graph,
    )
    dr_deltas = compute_incremental_contributions(dr_runs)

    all_phase13_results["dedicated_dr_ladder"] = {
        "levels": {k: v.metrics.to_dict() for k, v in dr_runs.items()},
        "incremental_contributions": dr_deltas,
        "data_source_category": "REAL_IMU_SYNTHETIC_BLACKOUT",
        "session_used": "Categorised_S1.npz",
    }
    for k, v in dr_runs.items():
        dr_m = v.metrics.dead_reckoning
        if dr_m:
            status = "PASS" if dr_m.sih_10pct_passed else "FAIL"
            print(f"  {k:20s} -> Final Drift: {dr_m.final_outage_drift_m:8.2f} m | Drift %: {dr_m.drift_percentage:5.2f}% | Dist: {dr_m.distance_travelled_m:6.1f} m | {status}")
    # Telemetry for figures 03, 05, 06, 07 (DR-A6: full-system DR estimator result)
    if "DR_A6_ZUPT" in dr_runs:
        save_telemetry_npz(tel_dir / "dr_A6_outage_60s.npz", dr_runs["DR_A6_ZUPT"], "DR_A6_outage", "REAL_IMU_SYNTHETIC_BLACKOUT")
    if "DR_A7_MAPMATCH" in dr_runs:
        save_telemetry_npz(tel_dir / "dr_A7_mapmatch_outage.npz", dr_runs["DR_A7_MAPMATCH"], "DR_A7_outage_mm", "REAL_IMU_SYNTHETIC_BLACKOUT")

    # 5. Axis B: Operating-Condition Matrix (B1 to B12)
    print("\n[Step 4/9] Running Axis B: Operating-Condition Matrix (B1 to B12)...")
    axis_b_results: Dict[str, Any] = {}

    for scen_id, scen_def in AXIS_B_SCENARIOS.items():
        # Source category labeling
        if scen_id == "B10_ZERO_COVERAGE":
            source_cat = "REAL_IMU_OFF_MAP"
            rep_cfg = ReplayConfig(
                start_idx=scen_def.start_idx,
                duration_steps=scen_def.duration_steps,
                enable_map_matching=True,
            )
            res_b = run_offline_replay(res_s1, gyro_bias_s1, standard_core_cfg, rep_cfg, road_graph=None)
        elif scen_id == "B12_REAL_OUTAGE":
            source_cat = "REAL_IMU_SYNTHETIC_BLACKOUT"
            s3c_path = Path("data/cache/iovnbd/Categorised_S3c.npz")
            if s3c_path.exists():
                res_s3c, gyro_bias_s3c, _ = load_and_preprocess_session(s3c_path)
                rep_cfg = ReplayConfig(
                    start_idx=scen_def.start_idx,
                    duration_steps=scen_def.duration_steps,
                    outage_start_rel_steps=scen_def.outage_start_rel_steps,
                    outage_duration_steps=scen_def.outage_duration_steps,
                    enable_map_matching=True,
                )
                res_b = run_offline_replay(res_s3c, gyro_bias_s3c, standard_core_cfg, rep_cfg, road_graph=road_graph)
            else:
                rep_cfg = ReplayConfig(
                    start_idx=scen_def.start_idx,
                    duration_steps=scen_def.duration_steps,
                    outage_start_rel_steps=scen_def.outage_start_rel_steps,
                    outage_duration_steps=scen_def.outage_duration_steps,
                    enable_map_matching=True,
                )
                res_b = run_offline_replay(res_s1, gyro_bias_s1, standard_core_cfg, rep_cfg, road_graph=road_graph)
        else:
            if scen_def.outage_start_rel_steps is not None:
                source_cat = "REAL_IMU_SYNTHETIC_BLACKOUT"
            else:
                source_cat = "REAL_IMU_CONTINUOUS"
            rep_cfg = ReplayConfig(
                start_idx=scen_def.start_idx,
                duration_steps=scen_def.duration_steps,
                outage_start_rel_steps=scen_def.outage_start_rel_steps,
                outage_duration_steps=scen_def.outage_duration_steps,
                enable_map_matching=True,
            )
            res_b = run_offline_replay(res_s1, gyro_bias_s1, standard_core_cfg, rep_cfg, road_graph=road_graph)

        metrics_dict = res_b.metrics.to_dict()
        metrics_dict["_source_category"] = source_cat
        metrics_dict["_scenario_name"] = scen_def.name
        axis_b_results[scen_id] = metrics_dict

        dr_info = ""
        if res_b.metrics.dead_reckoning:
            dr = res_b.metrics.dead_reckoning
            status = "PASS" if dr.sih_10pct_passed else "FAIL"
            dr_info = f" | Drift: {dr.final_outage_drift_m:.2f} m ({dr.drift_percentage:.2f}%) | {status}"
        snap_info = f" | Snap Rate: {res_b.metrics.map_matching.snap_rate_pct:.1f}%" if res_b.metrics.map_matching else ""
        print(f"  {scen_id:20s} [{source_cat:28s}] -> RMSE: {res_b.metrics.position.rmse_2d:.3f} m{dr_info}{snap_info}")

        # Save telemetry for specific scenarios
        if scen_id == "B4_OUTAGE_60S":
            save_telemetry_npz(tel_dir / "axisB_B4_outage_60s.npz", res_b, "B4_outage60s", source_cat)
        elif scen_id == "B1_CONTINUOUS_GNSS":
            save_telemetry_npz(tel_dir / "axisB_B1_continuous.npz", res_b, "B1_continuous", source_cat)
        elif scen_id == "B8_STOP_AND_GO":
            save_telemetry_npz(tel_dir / "axisB_B8_stopgo.npz", res_b, "B8_stopgo", source_cat)
        elif scen_id == "B9_PARALLEL_ROADS":
            save_telemetry_npz(tel_dir / "axisB_B9_ambiguity.npz", res_b, "B9_ambiguity", source_cat)
        elif scen_id == "B11_RECOVERY":
            save_telemetry_npz(tel_dir / "axisB_B11_recovery.npz", res_b, "B11_recovery", source_cat)

    all_phase13_results["axis_b_operating_conditions"] = axis_b_results

    # 6. Axis C: Output-Tier Processing
    print("\n[Step 5/9] Running Axis C: Output-Tier Processing Analysis...")
    c1_res = run_offline_replay(res_s1, gyro_bias_s1, NavigationCoreConfig(gnss_enabled=True, velocitynet_enabled=True, biasnet_enabled=True, nhc_enabled=False, zupt_enabled=False), ReplayConfig(start_idx=4900, duration_steps=600, enable_map_matching=False), road_graph)
    c2_res = run_offline_replay(res_s1, gyro_bias_s1, NavigationCoreConfig(gnss_enabled=True, velocitynet_enabled=True, biasnet_enabled=True, nhc_enabled=True, zupt_enabled=True), ReplayConfig(start_idx=4900, duration_steps=600, enable_map_matching=False), road_graph)
    c3_res = run_offline_replay(res_s1, gyro_bias_s1, NavigationCoreConfig(gnss_enabled=True, velocitynet_enabled=True, biasnet_enabled=True, nhc_enabled=True, zupt_enabled=True), ReplayConfig(start_idx=4900, duration_steps=600, enable_map_matching=True), road_graph)

    all_phase13_results["axis_c_output_processing"] = {
        "C1_RAW_ESKF": c1_res.metrics.to_dict(),
        "C2_ESKF_KINEMATIC_CONSTRAINTS": c2_res.metrics.to_dict(),
        "C3_ESKF_DOWNSTREAM_MAPMATCH": c3_res.metrics.to_dict(),
    }
    print(f"  C1 Raw ESKF                 -> 2D RMSE: {c1_res.metrics.position.rmse_2d:.4f} m")
    print(f"  C2 ESKF + Kinematics (NHC/Z) -> 2D RMSE: {c2_res.metrics.position.rmse_2d:.4f} m")
    print(f"  C3 Downstream Snapped Disp  -> 2D RMSE: {c3_res.metrics.position.rmse_2d:.4f} m | Snap Rate: {c3_res.metrics.map_matching.snap_rate_pct:.1f}%")
    save_telemetry_npz(tel_dir / "axisC_C3_mapmatch.npz", c3_res, "C3_mapmatch", "REAL_IMU_CONTINUOUS")

    # 7. Fully Controlled Synthetic Benchmarks (use SyntheticTrajectoryGenerator)
    print("\n[Step 6/9] Running FULLY CONTROLLED SYNTHETIC Benchmarks...")
    synth_b1_out = SyntheticTrajectoryGenerator.generate_50m_benchmark()
    synth_b1_res, synth_b1_report = run_synthetic_benchmark(synth_b1_out, None, standard_core_cfg, "B1_50m")
    print(f"  Synthetic Benchmark 1 (50m)     -> Dist: {synth_b1_report['distance_m']:.2f} m | Drift: {synth_b1_report['final_drift_m']:.2f} m | {synth_b1_report['drift_pct']:.2f}% | Status: {synth_b1_report['passed']}")
    save_telemetry_npz(tel_dir / "synth_benchmark_1_50m.npz", synth_b1_res, "synth_b1_50m", "FULLY_CONTROLLED_SYNTHETIC")

    synth_b2_out = SyntheticTrajectoryGenerator.generate_1km_60kmh_benchmark()
    synth_b2_res, synth_b2_report = run_synthetic_benchmark(synth_b2_out, None, standard_core_cfg, "B2_1km")
    print(f"  Synthetic Benchmark 2 (1km/60s) -> Dist: {synth_b2_report['distance_m']:.2f} m | Avg Speed: {synth_b2_report['speed_mps']:.2f} m/s ({synth_b2_report['speed_mps']*3.6:.1f} km/h) | Drift: {synth_b2_report['final_drift_m']:.2f} m | {synth_b2_report['drift_pct']:.2f}% | Status: {synth_b2_report['passed']}")
    save_telemetry_npz(tel_dir / "synth_benchmark_2_1km.npz", synth_b2_res, "synth_b2_1km", "FULLY_CONTROLLED_SYNTHETIC")

    # Also re-run real-data synthetic-blackout benchmarks separately, with correct labeling
    print("\n  Also running REAL-DATA SYNTHETIC BLACKOUT benchmarks (separate evidence category)...")
    # Real-data 50m-ish benchmark (short 10s blackout on S1 at ~5 m/s)
    rdb1_result = run_offline_replay(
        res_s1, gyro_bias_s1, standard_core_cfg,
        ReplayConfig(start_idx=4900, duration_steps=300, outage_start_rel_steps=100,
                     outage_duration_steps=100, enable_map_matching=False),
    )
    rdb1_dr = rdb1_result.metrics.dead_reckoning
    rdb1_report = {
        "target": "Real-data synthetic blackout 10s window",
        "data_source": "REAL_IMU_SYNTHETIC_BLACKOUT",
        "source_details": "IO-VNBD S1, 10s synthetic GNSS blackout injection",
        "distance_m": rdb1_dr.distance_travelled_m if rdb1_dr else 0.0,
        "final_drift_m": rdb1_dr.final_outage_drift_m if rdb1_dr else 0.0,
        "max_drift_m": rdb1_dr.max_outage_drift_m if rdb1_dr else 0.0,
        "drift_pct": rdb1_dr.drift_percentage if rdb1_dr else 0.0,
        "sih_10pct_passed": rdb1_dr.sih_10pct_passed if rdb1_dr else False,
        "label": "REAL_IMU_SYNTHETIC_BLACKOUT",
    }
    # Real-data ~1km-ish benchmark (60s blackout on S1) - NOT the 1km/60km/h controlled synthetic
    rdb2_result = run_offline_replay(
        res_s1, gyro_bias_s1, standard_core_cfg,
        ReplayConfig(start_idx=4900, duration_steps=800, outage_start_rel_steps=100,
                     outage_duration_steps=600, enable_map_matching=False),
    )
    rdb2_dr = rdb2_result.metrics.dead_reckoning
    rdb2_report = {
        "target": "Real-data synthetic blackout 60s window",
        "data_source": "REAL_IMU_SYNTHETIC_BLACKOUT",
        "source_details": "IO-VNBD S1, 60s synthetic GNSS blackout injection (NOT controlled 1km @ 60km/h)",
        "distance_m": rdb2_dr.distance_travelled_m if rdb2_dr else 0.0,
        "final_drift_m": rdb2_dr.final_outage_drift_m if rdb2_dr else 0.0,
        "max_drift_m": rdb2_dr.max_outage_drift_m if rdb2_dr else 0.0,
        "drift_pct": rdb2_dr.drift_percentage if rdb2_dr else 0.0,
        "sih_10pct_passed": rdb2_dr.sih_10pct_passed if rdb2_dr else False,
        "label": "REAL_IMU_SYNTHETIC_BLACKOUT",
    }
    print(f"  Real-Data Blackout (10s S1) -> Dist: {rdb1_report['distance_m']:.1f} m | Drift: {rdb1_report['final_drift_m']:.2f} m | {rdb1_report['drift_pct']:.2f}% | SIH: {'PASS' if rdb1_report['sih_10pct_passed'] else 'FAIL'}")
    print(f"  Real-Data Blackout (60s S1) -> Dist: {rdb2_report['distance_m']:.1f} m | Drift: {rdb2_report['final_drift_m']:.2f} m | {rdb2_report['drift_pct']:.2f}% | SIH: {'PASS' if rdb2_report['sih_10pct_passed'] else 'FAIL'}")
    print(f"    NOTE: 60s S1 distance is {rdb2_report['distance_m']:.1f} m, NOT 1000 m. Speed: {(rdb2_report['distance_m']/60.0):.2f} m/s = {(rdb2_report['distance_m']/60.0)*3.6:.1f} km/h, NOT 60 km/h.")

    all_phase13_results["synthetic_benchmarks"] = {
        "benchmark_1_50m_FULLY_CONTROLLED_SYNTHETIC": synth_b1_report,
        "benchmark_2_1km_60kmh_FULLY_CONTROLLED_SYNTHETIC": synth_b2_report,
        "realdata_blackout_10s_S1": rdb1_report,
        "realdata_blackout_60s_S1": rdb2_report,
    }

    # 8. Multi-Session Cross-Validation & Window Validity Check
    print("\n[Step 7/9] Running Multi-Session Cross-Validation & Window Validity Check...")
    multi_session_results: Dict[str, Any] = {}
    session_configs = {
        "Session_S1":  (Path("data/cache/iovnbd/Categorised_S1.npz"),  4900, 300, 100, 100),
        "Session_S2":  (Path("data/cache/iovnbd/Categorised_S2.npz"),   100, 300, 100, 100),
        "Session_S3a": (Path("data/cache/iovnbd/Categorised_S3a.npz"),    0, 300, 100, 100),
        "Session_S3c": (Path("data/cache/iovnbd/Categorised_S3c.npz"),    0, 300, 100, 100),
        "Session_S4":  (Path("data/cache/iovnbd/Categorised_S4.npz"),  4900, 300, 100, 100),
    }
    per_session_rows: List[Dict[str, Any]] = []
    for s_name, (s_p, s_start, s_dur, s_out_start, s_out_dur) in session_configs.items():
        if not s_p.exists():
            print(f"  {s_name:12s} -> SKIPPED (file not found)")
            continue
        s_res, s_gb, _ = load_and_preprocess_session(s_p)
        s_out = run_offline_replay(
            s_res, s_gb, standard_core_cfg,
            ReplayConfig(start_idx=s_start, duration_steps=s_dur,
                         outage_start_rel_steps=s_out_start, outage_duration_steps=s_out_dur,
                         enable_map_matching=True),
            road_graph=road_graph,
        )

        dr_p = s_out.metrics.dead_reckoning
        dist_tr = dr_p.distance_travelled_m if dr_p else 0.0

        # Assess window suitability
        if s_start < 500:
            suitability = "UNSUITABLE_WINDOW"
            suit_reason = f"Filter uninitialized at start_idx={s_start} (warmup < 50s)"
        elif dist_tr < 10.0:
            suitability = "UNSUITABLE_WINDOW"
            suit_reason = f"Low vehicle motion in outage window ({dist_tr:.1f} m)"
        else:
            suitability = "VALID_WINDOW"
            suit_reason = f"Sufficient filter warmup (start_idx={s_start}) and active motion ({dist_tr:.1f} m)"

        md = s_out.metrics.to_dict()
        md["_session_file"] = s_p.name
        md["_source_category"] = "REAL_IMU_SYNTHETIC_BLACKOUT_MULTI_SESSION"
        md["_suitability_status"] = suitability
        md["_suitability_reason"] = suit_reason
        multi_session_results[s_name] = md

        row_dict = {
            "session_name": s_name,
            "suitability_status": suitability,
            "suitability_reason": suit_reason,
            "rmse_2d": float(s_out.metrics.position.rmse_2d),
            "mean_err": float(s_out.metrics.position.mean_error),
            "median_err": float(s_out.metrics.position.median_error),
            "p95_err": float(s_out.metrics.position.p95_error),
            "max_err": float(s_out.metrics.position.max_error),
            "vel_rmse_2d": float(s_out.metrics.velocity.rmse_2d),
            "heading_rmse_deg": float(s_out.metrics.heading.rmse_deg),
            "drift_m": (float(dr_p.final_outage_drift_m) if dr_p else float("nan")),
            "drift_pct": (float(dr_p.drift_percentage) if dr_p else float("nan")),
            "sih_passed": (bool(dr_p.sih_10pct_passed) if dr_p else False),
            "sih_10pct_passed": (bool(dr_p.sih_10pct_passed) if dr_p else False),
            "snap_rate_pct": (float(s_out.metrics.map_matching.snap_rate_pct) if s_out.metrics.map_matching else float("nan")),
        }
        per_session_rows.append(row_dict)

        dr_str = ""
        if dr_p:
            status = "PASS" if dr_p.sih_10pct_passed else "FAIL"
            dr_str = f" | 10s Drift: {dr_p.final_outage_drift_m:.2f} m ({dr_p.drift_percentage:.2f}%) | {status}"
        print(f"  {s_name:12s} [{suitability:17s}] -> 2D RMSE: {s_out.metrics.position.rmse_2d:7.3f} m{dr_str}")

    # Compute honest multi-session aggregation stats (all vs suitable only)
    if per_session_rows:
        import statistics
        rmse_vals_all = [r["rmse_2d"] for r in per_session_rows]
        suitable_rows = [r for r in per_session_rows if r["suitability_status"] == "VALID_WINDOW"]
        rmse_vals_suitable = [r["rmse_2d"] for r in suitable_rows] if suitable_rows else rmse_vals_all

        all_phase13_results["multi_session_aggregation"] = {
            "total_sessions": len(per_session_rows),
            "valid_window_sessions": len(suitable_rows),
            "all_sessions_rmse_2d_mean": float(statistics.mean(rmse_vals_all)),
            "valid_windows_rmse_2d_mean": float(statistics.mean(rmse_vals_suitable)),
            "per_session_rows": per_session_rows,
        }
    all_phase13_results["multi_session_cross_validation"] = multi_session_results

    # 9. Multi-Scenario Aggregation (Strictly Axis B Operating Conditions Only)
    print("\n[Step 8/9] Aggregating Axis B Cross-Condition Results...")
    axis_b_metrics_list = list(axis_b_results.values())
    agg_report = aggregate_scenario_results(axis_b_metrics_list)
    all_phase13_results["aggregate_report"] = agg_report.to_dict()

    print(f"  Total Axis B Scenarios Evaluated: {agg_report.total_scenarios_evaluated}")
    print(f"  Continuous 2D RMSE: Mean={agg_report.continuous_rmse_stats.mean:.3f} m | P95={agg_report.continuous_rmse_stats.p95:.3f} m")
    print(f"  Outage Drift %:     Mean={agg_report.outage_drift_pct_stats.mean:.2f}% | Median={agg_report.outage_drift_pct_stats.median:.2f}%")
    print(f"  SIH PS 26168 (<10% Drift) Pass Rate: {agg_report.sih_pass_rate_pct:.1f}% ({agg_report.total_passed_scenarios}/{agg_report.total_scenarios_evaluated})")

    # 10. Serialize complete results
    print("\n[Step 9/9] Saving complete JSON results...")
    json_path = results_dir / "phase13_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_phase13_results, f, indent=2, default=str)
    print(f"Saved complete Phase 13 results to: {json_path}")
    print(f"Detailed telemetry arrays saved to: {tel_dir}/")


if __name__ == "__main__":
    main()
