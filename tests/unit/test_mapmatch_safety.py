"""Unit tests for Map Matching safety, safeguards, and strict downstream isolation (Phase 12).

Tests:
1. Parallel road ambiguity safety: no erratic oscillation, low confidence, safe fallback.
2. Large displacement snap rejection: points beyond max_snap_distance_m safely fallback.
3. Disconnected graph jump rejection: transition model rejects geographically close unconnected edges.
4. Hard downstream architectural test: ESKF state, covariance, and biases are bit-for-bit identical before and after.
"""

from __future__ import annotations

import math
from typing import Tuple
import networkx as nx
import numpy as np
import pytest

from maps.extract_osm import RoadNetworkGraph
from navigation.frames.local_geo import GeoReference
from navigation.mapmatch.matcher import MapMatcher
from navigation.schemas.state import NavigationState, OrientationState


def build_parallel_road_graph() -> Tuple[RoadNetworkGraph, GeoReference]:
    """Two parallel roads separated by 12 meters."""
    geo_ref = GeoReference(lat_ref=52.4000, lon_ref=-1.5000, alt_ref=0.0)

    nodes_enu = {
        1: (0.0, 0.0),
        2: (200.0, 0.0),
        3: (0.0, 12.0),
        4: (200.0, 12.0),
    }

    g = nx.DiGraph()
    for nid, (e, n) in nodes_enu.items():
        lat, lon, _ = geo_ref.enu_to_geodetic(e, n, 0.0)
        g.add_node(nid, lat=float(lat), lon=float(lon))

    # Road 1: Y=0.0, Road 2: Y=12.0
    for u, v, eid, y_coord in [(1, 2, "road_south", 0.0), (3, 4, "road_north", 12.0)]:
        geom_latlon = []
        for x in [0.0, 200.0]:
            lat, lon, _ = geo_ref.enu_to_geodetic(x, y_coord, 0.0)
            geom_latlon.append((float(lat), float(lon)))

        g.add_edge(
            u,
            v,
            edge_id=eid,
            name=eid,
            highway="residential",
            oneway=True,
            length_m=200.0,
            azimuth_rad=0.5 * math.pi,
            geometry_latlon=geom_latlon,
        )

    return RoadNetworkGraph(g, geo_ref=geo_ref), geo_ref


def test_parallel_road_ambiguity_handling() -> None:
    """Trajectory right in the middle between parallel roads (Y=6.0m) must not oscillate wildly."""
    rng, geo_ref = build_parallel_road_graph()
    matcher = MapMatcher(
        road_graph=rng,
        search_radius_m=20.0,
        ambiguity_margin=1.0,
        min_confidence=0.45,
        lag_epochs=4,
    )

    # Equidistant points: Y=6.0m (exactly midway between Y=0 and Y=12)
    traj = [(float(x), 6.0, 0.0) for x in range(10, 100, 10)]
    outputs = []
    t_ns = 1_000_000_000
    cov = np.eye(15) * 4.0

    for pt in traj:
        state = NavigationState(
            position_local=np.array(pt, dtype=np.float64),
            velocity_local=np.array([10.0, 0.0, 0.0], dtype=np.float64),
            orientation=OrientationState(q=np.array([1.0, 0.0, 0.0, 0.0]), gyro_bias=np.zeros(3)),
            accel_bias=np.zeros(3),
            covariance=cov,
            reference_point=(geo_ref.lat_ref, geo_ref.lon_ref),
            mode="GNSS_AIDED",
            timestamp_ns=t_ns,
        )
        out = matcher.process_state(state, geo_ref)
        if out is not None:
            outputs.append(out)
        t_ns += 100_000_000
    outputs.extend(matcher.flush_remaining(geo_ref))

    # Because distance to both roads is identical (6m vs 6m), margin is ~0.0 < ambiguity_margin (1.0).
    # All or nearly all outputs must trigger AMBIGUOUS_PARALLEL_ROADS or LOW_CONFIDENCE fallback!
    fallbacks = [o for o in outputs if not o.snapped]
    assert len(fallbacks) >= len(outputs) * 0.8
    for o in fallbacks:
        assert o.fallback_reason in {"AMBIGUOUS_PARALLEL_ROADS", "LOW_CONFIDENCE"}
        # Fallback coordinate must be EXACTLY the original estimator coordinate
        assert pytest.approx(o.display_enu, abs=1e-9) == o.estimator_enu


def test_large_snap_displacement_rejection() -> None:
    """Trajectory points far from the nearest road (> max_snap_distance_m) are safely rejected."""
    rng, geo_ref = build_parallel_road_graph()
    matcher = MapMatcher(
        road_graph=rng,
        search_radius_m=60.0,
        max_snap_distance_m=25.0,
        lag_epochs=2,
    )

    # Point at Y=50.0m (nearest road is Road North at Y=12.0m, displacement = 38.0m > 25.0m)
    traj = [(50.0, 50.0, 0.0), (60.0, 50.0, 0.0), (70.0, 50.0, 0.0)]
    outputs = []
    t_ns = 1_000_000_000

    for pt in traj:
        state = NavigationState(
            position_local=np.array(pt, dtype=np.float64),
            velocity_local=np.array([10.0, 0.0, 0.0], dtype=np.float64),
            orientation=OrientationState(q=np.array([1.0, 0.0, 0.0, 0.0]), gyro_bias=np.zeros(3)),
            accel_bias=np.zeros(3),
            covariance=np.eye(15),
            reference_point=(geo_ref.lat_ref, geo_ref.lon_ref),
            mode="GNSS_AIDED",
            timestamp_ns=t_ns,
        )
        out = matcher.process_state(state, geo_ref)
        if out is not None:
            outputs.append(out)
        t_ns += 100_000_000
    outputs.extend(matcher.flush_remaining(geo_ref))

    for o in outputs:
        assert not o.snapped
        assert o.fallback_reason == "LARGE_DISPLACEMENT"
        assert pytest.approx(o.display_enu, abs=1e-9) == o.estimator_enu
        assert pytest.approx(o.display_lat_lon, abs=1e-9) == o.estimator_lat_lon


def test_disconnected_graph_jump_rejection() -> None:
    """Transition between geographically close but topologically unconnected edges is rejected."""
    rng, geo_ref = build_parallel_road_graph()
    matcher = MapMatcher(road_graph=rng, search_radius_m=20.0, lag_epochs=2)

    # Step 1: on Road South (Y=0.0m). Step 2: on Road North (Y=12.0m) 100ms later.
    # Because roads are parallel and have no connecting edges, the transition is physically disconnected.
    pts = [(50.0, 0.0, 0.0), (52.0, 12.0, 0.0)]
    outputs = []
    t_ns = 1_000_000_000
    cov = np.eye(15) * 1.0

    for pt in pts:
        state = NavigationState(
            position_local=np.array(pt, dtype=np.float64),
            velocity_local=np.array([10.0, 0.0, 0.0], dtype=np.float64),
            orientation=OrientationState(q=np.array([1.0, 0.0, 0.0, 0.0]), gyro_bias=np.zeros(3)),
            accel_bias=np.zeros(3),
            covariance=cov,
            reference_point=(geo_ref.lat_ref, geo_ref.lon_ref),
            mode="GNSS_AIDED",
            timestamp_ns=t_ns,
        )
        out = matcher.process_state(state, geo_ref)
        if out is not None:
            outputs.append(out)
        t_ns += 100_000_000

    outputs.extend(matcher.flush_remaining(geo_ref))
    assert len(outputs) == 2
    # At least one step must reject the disconnected leap
    has_rejected_or_fallback = any(
        not o.snapped or o.fallback_reason in {"DISCONNECTED_TRANSITION", "LOW_CONFIDENCE", "AMBIGUOUS_PARALLEL_ROADS"}
        for o in outputs
    )
    assert has_rejected_or_fallback


def test_downstream_strictly_non_mutating_regression() -> None:
    """Hard architectural test: ESKF nominal state, covariance, and biases are 100% untouched."""
    rng, geo_ref = build_parallel_road_graph()
    matcher = MapMatcher(road_graph=rng)

    orig_pos = np.array([50.0, 2.0, -1.5], dtype=np.float64)
    orig_vel = np.array([12.3, -0.4, 0.1], dtype=np.float64)
    orig_q = np.array([0.7071, 0.0, 0.0, 0.7071], dtype=np.float64)
    orig_ba = np.array([0.015, -0.022, 0.035], dtype=np.float64)
    orig_bg = np.array([0.001, -0.002, 0.0005], dtype=np.float64)
    orig_cov = np.eye(15, dtype=np.float64) * 0.85
    orig_cov[0, 1] = 0.12
    orig_cov[1, 0] = 0.12

    state = NavigationState(
        position_local=orig_pos.copy(),
        velocity_local=orig_vel.copy(),
        orientation=OrientationState(q=orig_q.copy(), gyro_bias=orig_bg.copy()),
        accel_bias=orig_ba.copy(),
        covariance=orig_cov.copy(),
        reference_point=(geo_ref.lat_ref, geo_ref.lon_ref),
        mode="GNSS_AIDED",
        timestamp_ns=1_500_000_000,
    )

    # Run map matching
    out = matcher.process_state(state, geo_ref)
    matcher.flush_remaining(geo_ref)

    # Assert bit-for-bit numerical identity of state before and after
    np.testing.assert_array_equal(state.position_local, orig_pos)
    np.testing.assert_array_equal(state.velocity_local, orig_vel)
    np.testing.assert_array_equal(state.orientation.q, orig_q)
    np.testing.assert_array_equal(state.orientation.gyro_bias, orig_bg)
    np.testing.assert_array_equal(state.accel_bias, orig_ba)
    np.testing.assert_array_equal(state.covariance, orig_cov)
    assert state.mode == "GNSS_AIDED"
    assert state.timestamp_ns == 1_500_000_000


def test_downstream_full_core_immutability_audit() -> None:
    """End-to-end immutability audit: running downstream MapMatcher produces ZERO feedback into NavigationCore."""
    from navigation.core import NavigationCore, NavigationCoreConfig

    rng, geo_ref = build_parallel_road_graph()
    matcher = MapMatcher(road_graph=rng, search_radius_m=35.0, lag_epochs=4)

    cfg = NavigationCoreConfig(
        velocitynet_enabled=False,
        biasnet_enabled=False,
        nhc_enabled=True,
        zupt_enabled=True,
        gnss_enabled=True,
    )

    core_with_mm = NavigationCore(cfg)
    core_without_mm = NavigationCore(cfg)

    t0_ns = 1_000_000_000_000
    p0 = np.array([0.0, 0.0, 0.0])
    v0 = np.array([15.0, 0.0, 0.0])
    q0 = np.array([1.0, 0.0, 0.0, 0.0])

    core_with_mm.initialize(
        lat0=geo_ref.lat_ref,
        lon0=geo_ref.lon_ref,
        alt0=0.0,
        p0_enu=p0,
        v0_enu=v0,
        q0=q0,
        gyro_bias0=np.zeros(3),
        accel_bias0=np.zeros(3),
        timestamp_ns=t0_ns,
    )
    core_without_mm.initialize(
        lat0=geo_ref.lat_ref,
        lon0=geo_ref.lon_ref,
        alt0=0.0,
        p0_enu=p0,
        v0_enu=v0,
        q0=q0,
        gyro_bias0=np.zeros(3),
        accel_bias0=np.zeros(3),
        timestamp_ns=t0_ns,
    )

    for step in range(50):
        t_ns = t0_ns + step * 100_000_000
        f_m = np.array([0.0, 0.0, -9.81])
        w_m = np.array([0.0, 0.0, 0.02])

        # Step both cores with exact same sensor data
        out_mm = core_with_mm.step_imu(f_m, w_m, dt_s=0.1, timestamp_ns=t_ns)
        out_no_mm = core_without_mm.step_imu(f_m, w_m, dt_s=0.1, timestamp_ns=t_ns)

        # Feed core_with_mm state into downstream map matcher
        nav_state = core_with_mm.get_navigation_state()
        mm_out = matcher.process_state(nav_state, geo_ref)

        # Hard downstream isolation assertion:
        # Every single state variable and covariance element must remain bit-for-bit identical!
        np.testing.assert_array_equal(core_with_mm.state.nominal.position_enu, core_without_mm.state.nominal.position_enu)
        np.testing.assert_array_equal(core_with_mm.state.nominal.velocity_enu, core_without_mm.state.nominal.velocity_enu)
        np.testing.assert_array_equal(core_with_mm.state.nominal.q, core_without_mm.state.nominal.q)
        np.testing.assert_array_equal(core_with_mm.state.nominal.accel_bias, core_without_mm.state.nominal.accel_bias)
        np.testing.assert_array_equal(core_with_mm.state.nominal.gyro_bias, core_without_mm.state.nominal.gyro_bias)
        np.testing.assert_array_equal(core_with_mm.state.covariance, core_without_mm.state.covariance)
        assert core_with_mm.mode == core_without_mm.mode
        assert core_with_mm.state.nominal.timestamp_ns == core_without_mm.state.nominal.timestamp_ns

