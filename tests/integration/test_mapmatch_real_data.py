"""Integration test for Map Matching on real IO-VNBD S1 driving data with offline OSM road network (Phase 12).

Evaluates:
1. Candidate availability percentage.
2. Snap rate and fallback percentage.
3. Snap distance distribution (median, 95th percentile).
4. Fallback reasons and safeguard activations.
5. Strict downstream isolation: ESKF state is NEVER modified.
6. Before (Phase 11 Estimator) vs After (Map-Matched Display) position error against VBOX ground truth.
7. Verification that fallback output exactly equals original estimator output.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, List
import numpy as np
import pytest

from data.pipeline.sync import SynchronizedTrip
from maps.extract_osm import RoadNetworkGraph
from navigation.core import NavigationCore, NavigationCoreConfig
from navigation.frames.local_geo import GeoReference
from navigation.ins.attitude import rotation_matrix_to_quaternion
from navigation.mapmatch.matcher import MapMatcher, MapMatchOutput
from ml.data.resample import resample_to_canonical_10hz


@pytest.fixture(scope="module")
def s1_real_data_and_map():
    """Load S1 resampled data and offline OSM road graph."""
    from data.pipeline.stationary_detect import StationaryDetector
    from navigation.preprocessing.pipeline import PreprocessingPipeline

    osm_path = Path("data/maps/coventry_s1_road_graph.json")
    if not osm_path.exists():
        pytest.skip(f"Offline OSM graph not found at {osm_path}")

    s1_npz = Path("data/cache/iovnbd/Categorised_S1.npz")
    if not s1_npz.exists():
        pytest.skip(f"S1 dataset not found at {s1_npz}")

    trip = SynchronizedTrip.load_npz(s1_npz)
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
    return res, road_graph, preprocessed.calibration.gyro_bias


def test_mapmatch_real_s1_driving_segment(s1_real_data_and_map) -> None:
    """Run downstream map matching over a real driving segment of IO-VNBD S1."""
    res, road_graph, gyro_bias = s1_real_data_and_map

    # Run over a 600-step (60.0 s) nominal highway driving slice of S1 (matching Phase 11 evaluation)
    start_idx = 4900
    duration_steps = 600

    lat0 = float(res.aux_signals["v_ref_lat"][start_idx])
    lon0 = float(res.aux_signals["v_ref_lon"][start_idx])
    alt0 = float(res.aux_signals["v_ref_alt_m"][start_idx])
    t0_ns = int(res.timestamps_ns[start_idx])

    geo_ref = GeoReference(lat_ref=lat0, lon_ref=lon0, alt_ref=alt0)
    road_graph.set_geo_reference(geo_ref)

    # Initialize NavigationCore with Phase 11 configuration
    config = NavigationCoreConfig(
        velocitynet_enabled=True,
        biasnet_enabled=True,
        nhc_enabled=True,
        zupt_enabled=True,
        gnss_enabled=True,
    )
    core = NavigationCore(config)

    psi0 = math.radians(float(res.aux_signals["v_ref_heading_deg"][start_idx]))
    spd0 = float(res.aux_signals["v_ref_speed_mps"][start_idx])
    v_init = np.array([spd0 * math.sin(psi0), spd0 * math.cos(psi0), 0.0], dtype=np.float64)
    R0 = np.array([
        [math.sin(psi0), -math.cos(psi0), 0.0],
        [math.cos(psi0),  math.sin(psi0), 0.0],
        [0.0,             0.0,            1.0],
    ], dtype=np.float64)
    q0 = rotation_matrix_to_quaternion(R0)

    core.initialize(
        lat0=lat0,
        lon0=lon0,
        alt0=alt0,
        p0_enu=np.zeros(3),
        v0_enu=v_init,
        q0=q0,
        gyro_bias0=gyro_bias,
        timestamp_ns=t0_ns,
    )

    # Initialize downstream MapMatcher
    matcher = MapMatcher(
        road_graph=road_graph,
        search_radius_m=35.0,
        max_snap_distance_m=25.0,
        min_confidence=0.45,
        ambiguity_margin=1.0,
        lag_epochs=8,
    )

    outputs: List[MapMatchOutput] = []
    ref_enu_list: List[np.ndarray] = []
    eskf_states_before = []
    eskf_states_after = []

    last_gnss_step = -10

    for step in range(duration_steps):
        i = start_idx + step
        t_ns = int(res.timestamps_ns[i])

        # Ground truth reference
        lat_ref = float(res.aux_signals["v_ref_lat"][i])
        lon_ref = float(res.aux_signals["v_ref_lon"][i])
        alt_ref = float(res.aux_signals["v_ref_alt_m"][i])
        ref_e, ref_n, ref_u = geo_ref.geodetic_to_enu(lat_ref, lon_ref, alt_ref)
        ref_enu_list.append(np.array([ref_e, ref_n, ref_u], dtype=np.float64))

        spd_ref = float(res.aux_signals["v_ref_speed_mps"][i])
        hdg_ref = math.radians(float(res.aux_signals["v_ref_heading_deg"][i]))
        ref_ve = spd_ref * math.sin(hdg_ref)
        ref_vn = spd_ref * math.cos(hdg_ref)

        # 1 Hz GNSS fix (applied at start of epoch if available)
        if step % 10 == 0:
            core.step_gnss_fix(
                lat=lat_ref,
                lon=lon_ref,
                alt=alt_ref,
                v_east=ref_ve,
                v_north=ref_vn,
                accuracy_h_m=2.5,
                timestamp_ns=t_ns,
            )

        # IMU propagation (10 Hz)
        accel = np.array(res.f_m_v[i], dtype=np.float64)
        gyro = np.array(res.omega_m_v[i], dtype=np.float64)
        core.step_imu(accel, gyro, dt_s=0.1, timestamp_ns=t_ns)

        # Snapshot ESKF state before map matching
        state_pos_before = core.state.nominal.position_enu.copy()
        state_cov_before = core.state.covariance.copy()

        # Downstream Map Matching
        out = matcher.process_state(core.state, geo_ref, timestamp_ns=t_ns)
        if out is not None:
            outputs.append(out)

        # Snapshot ESKF state after map matching
        state_pos_after = core.state.nominal.position_enu.copy()
        state_cov_after = core.state.covariance.copy()

        # Hard invariant: ESKF state must NOT be modified by map matching
        np.testing.assert_array_equal(state_pos_before, state_pos_after)
        np.testing.assert_array_equal(state_cov_before, state_cov_after)

    # Flush remaining mature outputs
    outputs.extend(matcher.flush_remaining(geo_ref))

    assert len(outputs) == duration_steps

    # Telemetry and metrics computation
    total_epochs = len(outputs)
    cand_avail_count = sum(1 for o in outputs if o.candidate_count > 0)
    snapped_count = sum(1 for o in outputs if o.snapped)
    fallback_count = total_epochs - snapped_count

    cand_avail_pct = (cand_avail_count / total_epochs) * 100.0
    snap_pct = (snapped_count / total_epochs) * 100.0
    fallback_pct = (fallback_count / total_epochs) * 100.0

    print(f"\n--- S1 Map Matching Telemetry (600 steps) ---")
    print(f"Total Epochs: {total_epochs}")
    print(f"Candidate Availability: {cand_avail_count}/{total_epochs} ({cand_avail_pct:.1f}%)")
    print(f"Snapped: {snapped_count}/{total_epochs} ({snap_pct:.1f}%)")
    print(f"Fallback: {fallback_count}/{total_epochs} ({fallback_pct:.1f}%)")

    # Fallback reason distribution
    fallback_reasons: Dict[str, int] = {}
    for o in outputs:
        if not o.snapped and o.fallback_reason:
            fallback_reasons[o.fallback_reason] = fallback_reasons.get(o.fallback_reason, 0) + 1
    print(f"Fallback Reasons: {fallback_reasons}")

    # Snap distances
    snap_dists = [o.distance_to_road_m for o in outputs if o.snapped and o.distance_to_road_m is not None]
    if snap_dists:
        median_snap = float(np.median(snap_dists))
        p95_snap = float(np.percentile(snap_dists, 95))
        max_snap = float(np.max(snap_dists))
        print(f"Snap Distance: Median={median_snap:.2f}m, 95th-pct={p95_snap:.2f}m, Max={max_snap:.2f}m")
        assert max_snap <= matcher.max_snap_distance_m

    # Error analysis vs ground truth
    err_estimator = []
    err_display = []

    for idx, (o, ref_enu) in enumerate(zip(outputs, ref_enu_list)):
        e_est = math.hypot(o.estimator_enu[0] - ref_enu[0], o.estimator_enu[1] - ref_enu[1])
        e_disp = math.hypot(o.display_enu[0] - ref_enu[0], o.display_enu[1] - ref_enu[1])
        err_estimator.append(e_est)
        err_display.append(e_disp)

        # Invariant: If not snapped, display ENU == estimator ENU
        if not o.snapped:
            assert pytest.approx(o.display_enu[0], abs=1e-9) == o.estimator_enu[0]
            assert pytest.approx(o.display_enu[1], abs=1e-9) == o.estimator_enu[1]

    rmse_estimator = float(np.sqrt(np.mean(np.square(err_estimator))))
    rmse_display = float(np.sqrt(np.mean(np.square(err_display))))
    mean_estimator = float(np.mean(err_estimator))
    mean_display = float(np.mean(err_display))

    print(f"Estimator Position Error: RMSE={rmse_estimator:.3f}m, Mean={mean_estimator:.3f}m")
    print(f"Display (Snapped) Position Error: RMSE={rmse_display:.3f}m, Mean={mean_display:.3f}m")

    # Safety assertion: snap rate should be healthy on mapped road
    assert snap_pct >= 50.0  # At least 50% snapped on mapped city highway
    assert cand_avail_pct >= 80.0  # OSM coverage is present for >80% of points
