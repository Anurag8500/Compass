"""Deterministic offline replay engine for C.O.M.P.A.S.S. (Phase 13).

Orchestrates:
1. NavigationCore execution (Strapdown INS, ESKF propagation, GNSS updates, ML pseudo-measurements, NHC, ZUPT).
2. Synthetic / scheduled GNSS blackout injection.
3. Strictly downstream OSM HMM Map Matching (with verified zero feedback into estimator state).
4. Full telemetry capture (state trajectory, covariance, FSM modes, constraint activation, ML cadence).
5. Comprehensive metrics evaluation via navigation.evaluation.metrics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from maps.extract_osm import RoadNetworkGraph
from navigation.core import NavigationCore, NavigationCoreConfig
from navigation.evaluation.metrics import (
    DeadReckoningMetrics,
    TrajectoryMetrics,
    compute_trajectory_metrics,
)
from navigation.frames.local_geo import GeoReference
from navigation.ins.attitude import rotation_matrix_to_quaternion
from navigation.mapmatch.matcher import MapMatcher, MapMatchOutput


@dataclass(frozen=True)
class ReplayConfig:
    """Configuration profile for an offline replay execution."""
    start_idx: int = 0
    duration_steps: int = 600
    rate_hz: float = 10.0
    outage_start_rel_steps: Optional[int] = None
    outage_duration_steps: Optional[int] = None
    mounting_yaw_err_deg: float = 0.0
    enable_map_matching: bool = True
    map_matching_lag_epochs: int = 8
    map_matching_search_radius_m: float = 35.0


@dataclass
class ReplayResult:
    """Rich telemetry and metrics result from an offline replay run."""
    times_s: np.ndarray
    timestamps_ns: np.ndarray
    eskf_pos_enu: np.ndarray
    ref_pos_enu: np.ndarray
    disp_pos_enu: np.ndarray
    eskf_vel_enu: np.ndarray
    ref_vel_enu: np.ndarray
    ref_headings_rad: np.ndarray
    eskf_headings_rad: np.ndarray
    cov_diag_history: np.ndarray  # (N, 15)
    fsm_modes: List[str]
    nhc_statuses: List[str]
    zupt_applied_flags: List[bool]
    mm_outputs: List[MapMatchOutput]
    metrics: TrajectoryMetrics
    telemetry_summary: Dict[str, Any]


def run_offline_replay(
    data: Any,
    calib_gyro_bias: np.ndarray,
    core_config: NavigationCoreConfig,
    replay_config: ReplayConfig,
    road_graph: Optional[RoadNetworkGraph] = None,
) -> ReplayResult:
    """Execute a fully deterministic end-to-end replay on resampled sensor data.

    Args:
        data: Object with timestamps_ns, f_m_v, omega_m_v, and aux_signals (lat, lon, alt, speed, heading).
        calib_gyro_bias: 3-vector calibrated initial gyroscope bias.
        core_config: NavigationCoreConfig specifying which modules (GNSS, ML, NHC, ZUPT) are active.
        replay_config: ReplayConfig specifying slice range, outage window, and rates.
        road_graph: Optional RoadNetworkGraph for downstream map matching.

    Returns:
        ReplayResult containing full state trajectories, diagnostics, and metrics.
    """
    start_idx = replay_config.start_idx
    duration_steps = replay_config.duration_steps
    rate_hz = replay_config.rate_hz
    dt_s = 1.0 / rate_hz

    t0_ns = int(data.timestamps_ns[start_idx])
    lat0 = float(data.aux_signals["v_ref_lat"][start_idx])
    lon0 = float(data.aux_signals["v_ref_lon"][start_idx])
    alt0 = float(data.aux_signals["v_ref_alt_m"][start_idx])

    geo_ref = GeoReference(lat0, lon0, alt0)
    psi0 = math.radians(float(data.aux_signals["v_ref_heading_deg"][start_idx]))
    spd0 = float(data.aux_signals["v_ref_speed_mps"][start_idx])
    v_init = np.array([spd0 * math.sin(psi0), spd0 * math.cos(psi0), 0.0], dtype=np.float64)

    R0 = np.array([
        [math.sin(psi0), -math.cos(psi0), 0.0],
        [math.cos(psi0),  math.sin(psi0), 0.0],
        [0.0,             0.0,            1.0],
    ], dtype=np.float64)
    q0 = rotation_matrix_to_quaternion(R0)

    core = NavigationCore(core_config)
    core.initialize(
        lat0=lat0,
        lon0=lon0,
        alt0=alt0,
        p0_enu=np.zeros(3, dtype=np.float64),
        v0_enu=v_init,
        q0=q0,
        gyro_bias0=calib_gyro_bias,
        timestamp_ns=t0_ns,
    )

    matcher: Optional[MapMatcher] = None
    if replay_config.enable_map_matching and road_graph is not None:
        road_graph.set_geo_reference(geo_ref)
        matcher = MapMatcher(
            road_graph=road_graph,
            search_radius_m=replay_config.map_matching_search_radius_m,
            lag_epochs=replay_config.map_matching_lag_epochs,
        )

    # Mounting yaw misalignment matrix if requested
    dpsi = math.radians(replay_config.mounting_yaw_err_deg)
    R_mount = np.array([
        [math.cos(dpsi), -math.sin(dpsi), 0.0],
        [math.sin(dpsi),  math.cos(dpsi), 0.0],
        [0.0,             0.0,            1.0],
    ], dtype=np.float64) if replay_config.mounting_yaw_err_deg != 0.0 else np.eye(3)

    outage_start_step = replay_config.outage_start_rel_steps if replay_config.outage_start_rel_steps is not None else 99999999
    outage_end_step = outage_start_step + (replay_config.outage_duration_steps or 0)

    times_s: List[float] = []
    timestamps_ns: List[int] = []
    eskf_pos_enu: List[np.ndarray] = []
    disp_pos_enu: List[np.ndarray] = []
    ref_pos_enu: List[np.ndarray] = []
    eskf_vel_enu: List[np.ndarray] = []
    ref_vel_enu: List[np.ndarray] = []
    ref_headings_rad: List[float] = []
    eskf_headings_rad: List[float] = []
    cov_diag_history: List[np.ndarray] = []
    fsm_modes: List[str] = []
    nhc_statuses: List[str] = []
    zupt_flags: List[bool] = []
    mm_outputs: List[MapMatchOutput] = []

    # Downstream isolation assertion buffer
    for step_rel in range(duration_steps):
        i = start_idx + step_rel
        t_ns = int(data.timestamps_ns[i])
        t_s = (t_ns - t0_ns) * 1e-9

        times_s.append(t_s)
        timestamps_ns.append(t_ns)

        # Ground truth
        lat_i = float(data.aux_signals["v_ref_lat"][i])
        lon_i = float(data.aux_signals["v_ref_lon"][i])
        alt_i = float(data.aux_signals["v_ref_alt_m"][i])
        ref_e, ref_n, ref_u = geo_ref.geodetic_to_enu(lat_i, lon_i, alt_i)
        ref_pos_enu.append(np.array([ref_e, ref_n, ref_u]))

        spd_i = float(data.aux_signals["v_ref_speed_mps"][i])
        psi_i = math.radians(float(data.aux_signals["v_ref_heading_deg"][i]))
        ref_ve = spd_i * math.sin(psi_i)
        ref_vn = spd_i * math.cos(psi_i)
        ref_vel_enu.append(np.array([ref_ve, ref_vn, 0.0]))
        ref_headings_rad.append(psi_i)

        # Outage check
        is_in_outage = (outage_start_step <= step_rel < outage_end_step)

        # GNSS fix at 1 Hz cadence (every int(rate_hz) steps)
        gnss_cadence_steps = max(1, int(rate_hz))
        if step_rel % gnss_cadence_steps == 0:
            if not is_in_outage:
                core.step_gnss_fix(
                    lat=lat_i,
                    lon=lon_i,
                    alt=alt_i,
                    v_east=ref_ve,
                    v_north=ref_vn,
                    accuracy_h_m=2.5,
                    timestamp_ns=t_ns,
                )

        # Sensor step
        f_in = R_mount @ data.f_m_v[i]
        w_in = R_mount @ data.omega_m_v[i]
        imu_out = core.step_imu(f_in, w_in, dt_s=dt_s, timestamp_ns=t_ns)

        # Record estimator state
        pos_curr = core.state.nominal.position_enu.copy()
        vel_curr = core.state.nominal.velocity_enu.copy()
        cov_diag = np.diag(core.state.covariance).copy()

        eskf_pos_enu.append(pos_curr)
        eskf_vel_enu.append(vel_curr)
        cov_diag_history.append(cov_diag)
        fsm_modes.append(core.mode.value)
        zupt_flags.append(imu_out.zupt_applied)

        # Yaw heading from ESKF quaternion
        q_curr = core.state.nominal.q
        R_curr = core.state.nominal.R_v_n
        eskf_yaw = float(math.atan2(R_curr[0, 0], R_curr[1, 0])) % (2.0 * math.pi)
        eskf_headings_rad.append(eskf_yaw)

        if imu_out.nhc_diagnostics is not None:
            nhc_statuses.append(imu_out.nhc_diagnostics.status.value)
        else:
            nhc_statuses.append("NOT_ATTEMPTED")

        # Strictly Downstream Map Matching
        if matcher is not None:
            # Snapshot before
            p_before = core.state.nominal.position_enu.copy()
            cov_before = core.state.covariance.copy()

            mm_out = matcher.process_state(core.state, geo_ref, timestamp_ns=t_ns)
            if mm_out is not None:
                mm_outputs.append(mm_out)

            # Assert complete non-mutation
            np.testing.assert_array_equal(p_before, core.state.nominal.position_enu)
            np.testing.assert_array_equal(cov_before, core.state.covariance)

    # Flush remaining mature map match buffer
    if matcher is not None:
        mm_outputs.extend(matcher.flush_remaining(geo_ref))
        for o in mm_outputs:
            disp_pos_enu.append(np.array(o.display_enu))
    else:
        disp_pos_enu = [p.copy() for p in eskf_pos_enu]

    times_s_arr = np.array(times_s)
    timestamps_ns_arr = np.array(timestamps_ns)
    eskf_pos_arr = np.array(eskf_pos_enu)
    disp_pos_arr = np.array(disp_pos_enu)
    ref_pos_arr = np.array(ref_pos_enu)
    eskf_vel_arr = np.array(eskf_vel_enu)
    ref_vel_arr = np.array(ref_vel_enu)
    cov_diag_arr = np.array(cov_diag_history)

    outage_window = None
    if replay_config.outage_start_rel_steps is not None and replay_config.outage_duration_steps is not None:
        o_start = replay_config.outage_start_rel_steps
        o_end = min(duration_steps - 1, o_start + replay_config.outage_duration_steps)
        outage_window = (o_start, o_end)

    # Compute complete metrics
    metrics = compute_trajectory_metrics(
        est_pos_enu=eskf_pos_arr,
        ref_pos_enu=ref_pos_arr,
        est_vel_enu=eskf_vel_arr,
        ref_vel_enu=ref_vel_arr,
        times_s=times_s_arr,
        est_headings_rad=eskf_headings_rad,
        ref_headings_rad=ref_headings_rad,
        outage_steps=outage_window,
        mm_outputs=mm_outputs if replay_config.enable_map_matching else None,
    )

    telemetry = {
        "total_steps": duration_steps,
        "rate_hz": rate_hz,
        "duration_s": float(times_s[-1]),
        "fsm_mode_counts": {m: fsm_modes.count(m) for m in set(fsm_modes)},
        "zupt_count": sum(1 for z in zupt_flags if z),
    }

    return ReplayResult(
        times_s=times_s_arr,
        timestamps_ns=timestamps_ns_arr,
        eskf_pos_enu=eskf_pos_arr,
        ref_pos_enu=ref_pos_arr,
        disp_pos_enu=disp_pos_arr,
        eskf_vel_enu=eskf_vel_arr,
        ref_vel_enu=ref_vel_arr,
        ref_headings_rad=np.array(ref_headings_rad),
        eskf_headings_rad=np.array(eskf_headings_rad),
        cov_diag_history=cov_diag_arr,
        fsm_modes=fsm_modes,
        nhc_statuses=nhc_statuses,
        zupt_applied_flags=zupt_flags,
        mm_outputs=mm_outputs,
        metrics=metrics,
        telemetry_summary=telemetry,
    )
