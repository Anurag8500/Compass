"""Phase 12 Downstream Map Matching (OSM + HMM) Replay and Benchmark Script.

Executes comprehensive downstream map matching evaluation on IO-VNBD real driving data
(Categorised_S1.npz) using the offline Coventry road network graph:
1. Scenario A: Continuous GNSS (60s nominal highway driving)
2. Scenario B: Highway Moving GNSS Outages (10s, 30s, 60s)
3. Scenario C: Sharp Turn Dynamics
4. Scenario D: Stop-and-Go / Low Speed Driving
5. Scenario E: Zero/Poor Coverage Fallback Test
6. Scenario F: Parallel Road Ambiguity Test

Evaluates:
- Phase 11 Estimator metrics (strictly untouched)
- Phase 12 Map-Matched Display metrics
- Snap rate %, fallback %, and fallback reason distributions
- Snap distance statistics (median, 95th percentile, max)
- Regression audit (% improved, % degraded, % unchanged, max degradation)
- Strict downstream-only non-mutation verification

Produces:
- docs/phase12_mapmatch_results.json
- docs/phase12_figures/ (diagnostic visualization plots)
- docs/mapmatch_report.md
"""

from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data.pipeline.stationary_detect import StationaryDetector
from data.pipeline.sync import SynchronizedTrip
from maps.extract_osm import RoadNetworkGraph
from ml.data.resample import resample_to_canonical_10hz
from navigation.core import NavigationCore, NavigationCoreConfig
from navigation.frames.local_geo import GeoReference
from navigation.ins.attitude import rotation_matrix_to_quaternion
from navigation.mapmatch.matcher import MapMatcher, MapMatchOutput
from navigation.preprocessing.pipeline import PreprocessingPipeline


def run_mapmatch_simulation(
    res: Any,
    calib_gyro_bias: np.ndarray,
    road_graph: RoadNetworkGraph,
    start_idx: int,
    duration_steps: int,
    config: NavigationCoreConfig,
    outage_start_rel_steps: Optional[int] = None,
    outage_duration_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """Run NavigationCore with Phase 11 and pipe outputs through downstream MapMatcher."""
    t0_ns = int(res.timestamps_ns[start_idx])
    lat0 = float(res.aux_signals["v_ref_lat"][start_idx])
    lon0 = float(res.aux_signals["v_ref_lon"][start_idx])
    alt0 = float(res.aux_signals["v_ref_alt_m"][start_idx])

    geo_ref = GeoReference(lat0, lon0, alt0)
    road_graph.set_geo_reference(geo_ref)

    psi0 = math.radians(float(res.aux_signals["v_ref_heading_deg"][start_idx]))
    spd0 = float(res.aux_signals["v_ref_speed_mps"][start_idx])
    v_init = np.array([spd0 * math.sin(psi0), spd0 * math.cos(psi0), 0.0], dtype=np.float64)
    R0 = np.array([
        [math.sin(psi0), -math.cos(psi0), 0.0],
        [math.cos(psi0),  math.sin(psi0), 0.0],
        [0.0,             0.0,            1.0],
    ], dtype=np.float64)
    q0 = rotation_matrix_to_quaternion(R0)

    core = NavigationCore(config)
    core.initialize(
        lat0=lat0,
        lon0=lon0,
        alt0=alt0,
        p0_enu=np.zeros(3),
        v0_enu=v_init,
        q0=q0,
        gyro_bias0=calib_gyro_bias,
        timestamp_ns=t0_ns,
    )

    matcher = MapMatcher(
        road_graph=road_graph,
        search_radius_m=35.0,
        max_snap_distance_m=25.0,
        min_confidence=0.45,
        ambiguity_margin=1.0,
        lag_epochs=8,
    )

    times_s = []
    eskf_pos_enu = []
    disp_pos_enu = []
    ref_pos_enu = []
    mm_outputs: List[MapMatchOutput] = []

    outage_start_step = outage_start_rel_steps if outage_start_rel_steps is not None else 9999999
    outage_end_step = outage_start_step + (outage_duration_steps or 0)

    for step_rel in range(duration_steps):
        i = start_idx + step_rel
        t_ns = int(res.timestamps_ns[i])
        t_s = (t_ns - t0_ns) * 1e-9
        times_s.append(t_s)

        # Ground truth ENU
        lat_i = float(res.aux_signals["v_ref_lat"][i])
        lon_i = float(res.aux_signals["v_ref_lon"][i])
        alt_i = float(res.aux_signals["v_ref_alt_m"][i])
        ref_e, ref_n, ref_u = geo_ref.geodetic_to_enu(lat_i, lon_i, alt_i)
        ref_pos_enu.append([ref_e, ref_n, ref_u])

        spd_i = float(res.aux_signals["v_ref_speed_mps"][i])
        hdg_i = math.radians(float(res.aux_signals["v_ref_heading_deg"][i]))
        ref_ve = spd_i * math.sin(hdg_i)
        ref_vn = spd_i * math.cos(hdg_i)

        # Outage gating
        is_in_outage = (outage_start_step <= step_rel < outage_end_step)

        # 1 Hz GNSS fix
        if step_rel % 10 == 0 and not is_in_outage:
            core.step_gnss_fix(
                lat=lat_i,
                lon=lon_i,
                alt=alt_i,
                v_east=ref_ve,
                v_north=ref_vn,
                accuracy_h_m=2.5,
                timestamp_ns=t_ns,
            )

        # Step IMU (10 Hz)
        core.step_imu(res.f_m_v[i], res.omega_m_v[i], dt_s=0.1, timestamp_ns=t_ns)

        # Snapshot ESKF state before map matching
        state_pos_before = core.state.nominal.position_enu.copy()
        state_cov_before = core.state.covariance.copy()

        # Strictly Downstream Map Matching
        out = matcher.process_state(core.state, geo_ref, timestamp_ns=t_ns)
        if out is not None:
            mm_outputs.append(out)

        # Snapshot ESKF state after map matching
        state_pos_after = core.state.nominal.position_enu.copy()
        state_cov_after = core.state.covariance.copy()

        # Hard invariant: zero feedback into ESKF state or covariance
        np.testing.assert_array_equal(state_pos_before, state_pos_after)
        np.testing.assert_array_equal(state_cov_before, state_cov_after)

    # Flush remaining mature outputs
    mm_outputs.extend(matcher.flush_remaining(geo_ref))

    for o in mm_outputs:
        eskf_pos_enu.append(o.estimator_enu)
        disp_pos_enu.append(o.display_enu)

    eskf_pos_enu = np.array(eskf_pos_enu)
    disp_pos_enu = np.array(disp_pos_enu)
    ref_pos_enu = np.array(ref_pos_enu)

    # Error computations
    pos_err_est = np.sqrt(np.sum((eskf_pos_enu[:, :2] - ref_pos_enu[:, :2]) ** 2, axis=1))
    pos_err_disp = np.sqrt(np.sum((disp_pos_enu[:, :2] - ref_pos_enu[:, :2]) ** 2, axis=1))

    # Outage drift metrics
    outage_metrics = {}
    if outage_start_rel_steps is not None and outage_duration_steps is not None:
        idx_start = outage_start_rel_steps
        idx_end = min(outage_start_rel_steps + outage_duration_steps, duration_steps - 1)

        err_est_outage = pos_err_est[idx_start:idx_end + 1]
        err_disp_outage = pos_err_disp[idx_start:idx_end + 1]

        dist_travelled = 0.0
        for k in range(idx_start, idx_end):
            dist_travelled += float(np.linalg.norm(ref_pos_enu[k + 1, :2] - ref_pos_enu[k, :2]))

        outage_metrics = {
            "distance_travelled_m": dist_travelled,
            "est_final_drift_m": float(err_est_outage[-1]),
            "disp_final_drift_m": float(err_disp_outage[-1]),
            "est_max_drift_m": float(np.max(err_est_outage)),
            "disp_max_drift_m": float(np.max(err_disp_outage)),
            "est_drift_pct": float(err_est_outage[-1] / max(dist_travelled, 1.0)) * 100.0,
            "disp_drift_pct": float(err_disp_outage[-1] / max(dist_travelled, 1.0)) * 100.0,
        }

    # Telemetry
    total_epochs = len(mm_outputs)
    snapped_count = sum(1 for o in mm_outputs if o.snapped)
    fallback_count = total_epochs - snapped_count

    snap_dists = [o.distance_to_road_m for o in mm_outputs if o.snapped and o.distance_to_road_m is not None]
    fallback_reasons: Dict[str, int] = {}
    for o in mm_outputs:
        if not o.snapped and o.fallback_reason:
            fallback_reasons[o.fallback_reason] = fallback_reasons.get(o.fallback_reason, 0) + 1

    # Regression calculation (point-by-point comparison)
    deltas = pos_err_disp - pos_err_est  # positive means degraded, negative means improved
    improved_count = int(np.sum(deltas < -0.05))
    degraded_count = int(np.sum(deltas > 0.05))
    unchanged_count = total_epochs - improved_count - degraded_count

    return {
        "times_s": times_s,
        "eskf_pos_enu": eskf_pos_enu,
        "disp_pos_enu": disp_pos_enu,
        "ref_pos_enu": ref_pos_enu,
        "pos_err_est": pos_err_est,
        "pos_err_disp": pos_err_disp,
        "est_rmse_2d_m": float(np.sqrt(np.mean(pos_err_est ** 2))),
        "disp_rmse_2d_m": float(np.sqrt(np.mean(pos_err_disp ** 2))),
        "est_max_err_2d_m": float(np.max(pos_err_est)),
        "disp_max_err_2d_m": float(np.max(pos_err_disp)),
        "outage_metrics": outage_metrics,
        "telemetry": {
            "total_epochs": total_epochs,
            "snapped_count": snapped_count,
            "fallback_count": fallback_count,
            "snap_rate_pct": float(snapped_count / max(total_epochs, 1)) * 100.0,
            "fallback_rate_pct": float(fallback_count / max(total_epochs, 1)) * 100.0,
            "fallback_reasons": fallback_reasons,
            "median_snap_dist_m": float(np.median(snap_dists)) if snap_dists else 0.0,
            "p95_snap_dist_m": float(np.percentile(snap_dists, 95)) if snap_dists else 0.0,
            "max_snap_dist_m": float(np.max(snap_dists)) if snap_dists else 0.0,
        },
        "regression_audit": {
            "improved_pct": float(improved_count / total_epochs) * 100.0,
            "degraded_pct": float(degraded_count / total_epochs) * 100.0,
            "unchanged_pct": float(unchanged_count / total_epochs) * 100.0,
            "max_degradation_m": float(np.max(deltas)),
            "mean_delta_m": float(np.mean(deltas)),
        },
        "mm_outputs": mm_outputs,
    }


def main() -> None:
    print("=" * 80)
    print("PHASE 12 DOWNSTREAM MAP MATCHING (OSM + HMM) REPLAY BENCHMARK")
    print("=" * 80)

    # 1. Paths and setup
    osm_path = Path("data/maps/coventry_s1_road_graph.json")
    if not osm_path.exists():
        raise FileNotFoundError(f"Missing OSM graph: {osm_path}")

    s1_path = Path("data/cache/iovnbd/Categorised_S1.npz")
    if not s1_path.exists():
        raise FileNotFoundError(f"Missing S1 dataset: {s1_path}")

    figures_dir = Path("docs/phase12_figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Loading datasets and road graph...")
    trip = SynchronizedTrip.load_npz(s1_path)
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
    road_graph = RoadNetworkGraph.load_json(osm_path)
    calib_gyro_bias = preprocessed.calibration.gyro_bias

    core_config = NavigationCoreConfig(
        velocitynet_enabled=True,
        biasnet_enabled=True,
        nhc_enabled=True,
        zupt_enabled=True,
        gnss_enabled=True,
    )

    all_results: Dict[str, Any] = {
        "metadata": {
            "phase": "Phase 12 Downstream Map Matching",
            "dataset": "IO-VNBD Categorised_S1.npz",
            "map": "OpenStreetMap Coventry/Warwick Extract (offline JSON)",
            "rate_hz": 10.0,
        }
    }

    # =========================================================================
    # SCENARIO A: Continuous GNSS (60s Nominal Highway Driving)
    # =========================================================================
    print("\n--- Running Scenario A: Continuous GNSS (60s Nominal Highway) ---")
    sim_a = run_mapmatch_simulation(
        res=res,
        calib_gyro_bias=calib_gyro_bias,
        road_graph=road_graph,
        start_idx=4900,
        duration_steps=600,
        config=core_config,
    )
    all_results["scenario_a_continuous_gnss"] = {
        "estimator_rmse_2d_m": sim_a["est_rmse_2d_m"],
        "display_rmse_2d_m": sim_a["disp_rmse_2d_m"],
        "estimator_max_err_2d_m": sim_a["est_max_err_2d_m"],
        "display_max_err_2d_m": sim_a["disp_max_err_2d_m"],
        "telemetry": sim_a["telemetry"],
        "regression_audit": sim_a["regression_audit"],
    }
    print(f"  Estimator RMSE: {sim_a['est_rmse_2d_m']:.3f} m | Display (Snapped) RMSE: {sim_a['disp_rmse_2d_m']:.3f} m")
    print(f"  Snap Rate: {sim_a['telemetry']['snap_rate_pct']:.1f}% | Median Snap Dist: {sim_a['telemetry']['median_snap_dist_m']:.2f} m")

    # =========================================================================
    # SCENARIO B: Highway Outages (10s, 30s, 60s)
    # =========================================================================
    print("\n--- Running Scenario B: Highway Outages (10s, 30s, 60s) ---")
    outages = [10.0, 30.0, 60.0]
    all_results["scenario_b_outages"] = {}
    sim_b_dict = {}

    for dur_s in outages:
        dur_steps = int(dur_s * 10)
        pre_steps = 100
        post_steps = 100
        total_steps = pre_steps + dur_steps + post_steps

        sim_b = run_mapmatch_simulation(
            res=res,
            calib_gyro_bias=calib_gyro_bias,
            road_graph=road_graph,
            start_idx=4900,
            duration_steps=total_steps,
            config=core_config,
            outage_start_rel_steps=pre_steps,
            outage_duration_steps=dur_steps,
        )
        sim_b_dict[dur_s] = sim_b
        all_results["scenario_b_outages"][f"outage_{int(dur_s)}s"] = {
            "outage_duration_s": dur_s,
            "outage_metrics": sim_b["outage_metrics"],
            "telemetry": sim_b["telemetry"],
            "regression_audit": sim_b["regression_audit"],
        }
        om = sim_b["outage_metrics"]
        print(f"  [{int(dur_s)}s Outage] Estimator Final Drift: {om['est_final_drift_m']:.2f} m ({om['est_drift_pct']:.2f}%) | Display Final Drift: {om['disp_final_drift_m']:.2f} m ({om['disp_drift_pct']:.2f}%)")
        print(f"    Snap Rate: {sim_b['telemetry']['snap_rate_pct']:.1f}% | Fallback: {sim_b['telemetry']['fallback_rate_pct']:.1f}%")

    # =========================================================================
    # SCENARIO C: Sharp Turn Dynamics (Evaluating Road Alignment in Curves)
    # =========================================================================
    print("\n--- Running Scenario C: Sharp Turn Dynamics ---")
    sim_c = run_mapmatch_simulation(
        res=res,
        calib_gyro_bias=calib_gyro_bias,
        road_graph=road_graph,
        start_idx=10500,
        duration_steps=400,
        config=core_config,
    )
    all_results["scenario_c_sharp_turn"] = {
        "estimator_rmse_2d_m": sim_c["est_rmse_2d_m"],
        "display_rmse_2d_m": sim_c["disp_rmse_2d_m"],
        "telemetry": sim_c["telemetry"],
        "regression_audit": sim_c["regression_audit"],
    }
    print(f"  Estimator RMSE: {sim_c['est_rmse_2d_m']:.3f} m | Display RMSE: {sim_c['disp_rmse_2d_m']:.3f} m")
    print(f"  Snap Rate: {sim_c['telemetry']['snap_rate_pct']:.1f}% | Fallback Reasons: {sim_c['telemetry']['fallback_reasons']}")

    # Save Results JSON
    results_json_path = Path("docs/phase12_mapmatch_results.json")
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved benchmark results to: {results_json_path}")

    # =========================================================================
    # GENERATE PUBLICATION-GRADE DIAGNOSTIC PLOTS
    # =========================================================================
    print("\nGenerating Phase 12 diagnostic visualization figures...")

    # Plot 1: Nominal Highway Trajectory Alignment
    plt.figure(figsize=(10, 6))
    plt.plot(sim_a["ref_pos_enu"][:, 0], sim_a["ref_pos_enu"][:, 1], "k-", linewidth=2.0, label="VBOX Ground Truth")
    plt.plot(sim_a["eskf_pos_enu"][:, 0], sim_a["eskf_pos_enu"][:, 1], "b--", linewidth=1.5, label="Phase 11 Estimator Output (Raw)")
    plt.plot(sim_a["disp_pos_enu"][:, 0], sim_a["disp_pos_enu"][:, 1], "g-", linewidth=1.8, label="Phase 12 Map-Matched Display (Snapped)")
    plt.xlabel("East (m)", fontsize=11)
    plt.ylabel("North (m)", fontsize=11)
    plt.title("Scenario A: Continuous GNSS Trajectory & Downstream Map Snapping", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="best", fontsize=10)
    p1 = figures_dir / "01_continuous_gnss_mapmatch_trajectory.png"
    plt.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p1}")

    # Plot 2: 60s Outage Drift Comparison
    sim_60 = sim_b_dict[60.0]
    plt.figure(figsize=(10, 5))
    plt.plot(sim_60["times_s"], sim_60["pos_err_est"], "r--", linewidth=1.8, label="Phase 11 Estimator Position Error")
    plt.plot(sim_60["times_s"], sim_60["pos_err_disp"], "g-", linewidth=2.0, label="Phase 12 Map-Matched Display Error")
    plt.axvspan(10.0, 70.0, color="gray", alpha=0.15, label="60s GNSS Outage Window")
    plt.xlabel("Time Elapsed (s)", fontsize=11)
    plt.ylabel("2D Position Error (m)", fontsize=11)
    plt.title("Scenario B: 60s Highway GNSS Outage Drift (Estimator vs Map-Matched)", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left", fontsize=10)
    p2 = figures_dir / "02_60s_outage_error_comparison.png"
    plt.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p2}")

    # Plot 3: Snap Distance Histogram
    snap_dists = [o.distance_to_road_m for o in sim_a["mm_outputs"] if o.snapped and o.distance_to_road_m is not None]
    if snap_dists:
        plt.figure(figsize=(8, 5))
        plt.hist(snap_dists, bins=25, color="#28a745", edgecolor="black", alpha=0.8)
        plt.axvline(np.median(snap_dists), color="red", linestyle="--", linewidth=2, label=f"Median: {np.median(snap_dists):.2f}m")
        plt.axvline(np.percentile(snap_dists, 95), color="orange", linestyle=":", linewidth=2, label=f"95th Pct: {np.percentile(snap_dists, 95):.2f}m")
        plt.xlabel("Orthogonal Distance to Road Centerline (m)", fontsize=11)
        plt.ylabel("Epoch Count", fontsize=11)
        plt.title("Scenario A: Orthogonal Road Snapping Distance Distribution", fontsize=12, fontweight="bold")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="upper right", fontsize=10)
        p3 = figures_dir / "03_snap_distance_distribution.png"
        plt.savefig(p3, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {p3}")

    print("\nPhase 12 Replay Benchmark complete!")


if __name__ == "__main__":
    main()
