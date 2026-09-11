"""Unit tests for Map Matching on a small, deterministic synthetic road graph (Phase 12).

Tests:
1. Correct candidate projection and distance.
2. Log-Gaussian emission scores.
3. Transition model respecting connectivity.
4. Viterbi intended road selection under noise.
5. Strict causality of fixed-lag sliding window.
6. Absolute determinism across repeated runs.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple
import networkx as nx
import numpy as np
import pytest

from maps.extract_osm import RoadNetworkGraph
from navigation.frames.local_geo import GeoReference
from navigation.mapmatch.candidates import CandidateSearch
from navigation.mapmatch.emission import EmissionModel
from navigation.mapmatch.matcher import MapMatcher
from navigation.mapmatch.transition import TransitionModel, LOG_ZERO
from navigation.mapmatch.viterbi import FixedLagViterbi
from navigation.schemas.state import NavigationState, OrientationState


def build_synthetic_known_graph() -> Tuple[RoadNetworkGraph, GeoReference]:
    """Build a deterministic 3-road synthetic graph.

    Graph topology:
    - Node 1: (0, 0)
    - Node 2: (200, 0)
    - Node 3: (0, 20)  [Parallel road 20m North]
    - Node 4: (200, 20)
    - Node 5: (200, 150) [Connecting Turn from Node 2 going North]

    Edges:
    - Road A (Main): 1 -> 2 along Y=0 (length 200m)
    - Road B (Parallel): 3 -> 4 along Y=20 (length 200m)
    - Road C (Turn): 2 -> 5 along X=200 (length 150m)
    """
    geo_ref = GeoReference(lat_ref=52.4000, lon_ref=-1.5000, alt_ref=0.0)

    # Convert ENU coordinates to lat/lon for node definitions
    nodes_enu = {
        1: (0.0, 0.0),
        2: (200.0, 0.0),
        3: (0.0, 20.0),
        4: (200.0, 20.0),
        5: (200.0, 150.0),
    }

    g = nx.DiGraph()
    for nid, (e, n) in nodes_enu.items():
        lat, lon, _ = geo_ref.enu_to_geodetic(e, n, 0.0)
        g.add_node(nid, lat=float(lat), lon=float(lon))

    # Add edges
    edges_def = [
        (1, 2, "road_A_main", [(0.0, 0.0), (200.0, 0.0)], 200.0, 0.5 * math.pi),
        (3, 4, "road_B_parallel", [(0.0, 20.0), (200.0, 20.0)], 200.0, 0.5 * math.pi),
        (2, 5, "road_C_turn", [(200.0, 0.0), (200.0, 150.0)], 150.0, 0.0),
    ]

    for u, v, eid, geom_enu, length_m, az in edges_def:
        geom_latlon = []
        for e, n in geom_enu:
            lat, lon, _ = geo_ref.enu_to_geodetic(e, n, 0.0)
            geom_latlon.append((float(lat), float(lon)))

        g.add_edge(
            u,
            v,
            edge_id=eid,
            name=eid,
            highway="primary",
            oneway=True,
            length_m=length_m,
            azimuth_rad=az,
            geometry_latlon=geom_latlon,
        )

    rng = RoadNetworkGraph(g, geo_ref=geo_ref)
    return rng, geo_ref


def test_candidate_search_known_graph() -> None:
    """Verify projection onto known synthetic segments."""
    rng, geo_ref = build_synthetic_known_graph()
    cs = CandidateSearch(rng, search_radius_m=30.0)

    # Point at (50.0, 4.0): 4m North of Road A (Y=0), 16m South of Road B (Y=20)
    cands = cs.search_candidates((50.0, 4.0))
    assert len(cands) == 2

    # Best candidate must be Road A with distance ~4m
    cand0 = cands[0]
    assert cand0.edge_id == "road_A_main"
    assert pytest.approx(cand0.distance_to_road_m, abs=1e-3) == 4.0
    assert pytest.approx(cand0.distance_along_edge_m, abs=1e-3) == 50.0
    assert pytest.approx(cand0.projected_point_enu[0], abs=1e-3) == 50.0
    assert pytest.approx(cand0.projected_point_enu[1], abs=1e-3) == 0.0

    # Second candidate must be Road B with distance ~16m
    cand1 = cands[1]
    assert cand1.edge_id == "road_B_parallel"
    assert pytest.approx(cand1.distance_to_road_m, abs=1e-3) == 16.0


def test_emission_model_covariance_awareness() -> None:
    """Verify covariance-aware log-Gaussian emission model."""
    rng, geo_ref = build_synthetic_known_graph()
    cs = CandidateSearch(rng, search_radius_m=30.0)
    cands = cs.search_candidates((50.0, 3.0))
    cand_a = cands[0]  # Road A along X axis (normal along Y axis)

    em = EmissionModel(sigma_road_m=4.0)

    # Without state covariance: sigma_d = 4.0 m
    log_e_zero_cov = em.compute_log_emission(cand_a, cov_enu_2x2=None)
    expected_var = 16.0
    expected_log_e = -0.5 * math.log(2.0 * math.pi * expected_var) - (3.0 ** 2) / (2.0 * expected_var)
    assert pytest.approx(log_e_zero_cov, rel=1e-5) == expected_log_e

    # With state covariance having large uncertainty in North (normal) direction: P_yy = 9.0
    # sigma_d^2 = 9.0 + 16.0 = 25.0 m^2
    cov = np.diag([4.0, 9.0])
    log_e_cov = em.compute_log_emission(cand_a, cov_enu_2x2=cov)
    expected_var_cov = 25.0
    expected_log_cov = -0.5 * math.log(2.0 * math.pi * expected_var_cov) - (3.0 ** 2) / (2.0 * expected_var_cov)
    assert pytest.approx(log_e_cov, rel=1e-5) == expected_log_cov


def test_transition_model_connectivity_and_routing() -> None:
    """Verify transition model respects graph directionality and connectivity."""
    rng, geo_ref = build_synthetic_known_graph()
    cs = CandidateSearch(rng, search_radius_m=30.0)
    tm = TransitionModel(rng, beta_m=5.0)

    cand_a1 = cs.search_candidates((50.0, 0.0))[0]
    cand_a2 = cs.search_candidates((70.0, 0.0))[0]
    cand_b = cs.search_candidates((70.0, 20.0))[0]
    cand_c = cs.search_candidates((200.0, 30.0))[0]

    # Valid forward transition along Road A: 20m travel
    log_t_valid = tm.compute_log_transition(
        c_prev=cand_a1,
        c_curr=cand_a2,
        prev_traj_enu=(50.0, 0.0),
        curr_traj_enu=(70.0, 0.0),
    )
    # d_graph = 20m, d_traj = 20m -> delta_d = 0 -> log_t = -ln(beta)
    assert pytest.approx(log_t_valid, rel=1e-4) == -math.log(5.0)

    # Valid turn transition from Road A to connected Road C (at intersection node 2)
    # Remaining on Road A: 200 - 180 = 20m. Along Road C: 30m. Total graph = 50m.
    cand_a_end = cs.search_candidates((180.0, 0.0))[0]
    log_t_turn = tm.compute_log_transition(
        c_prev=cand_a_end,
        c_curr=cand_c,
        prev_traj_enu=(180.0, 0.0),
        curr_traj_enu=(200.0, 30.0),
    )
    assert log_t_turn > LOG_ZERO / 2

    # Disconnected jump: from parallel Road B to Road C without connection
    log_t_disc = tm.compute_log_transition(
        c_prev=cand_b,
        c_curr=cand_c,
        prev_traj_enu=(70.0, 20.0),
        curr_traj_enu=(200.0, 30.0),
    )
    assert log_t_disc == LOG_ZERO


def test_viterbi_selects_intended_path_under_noise() -> None:
    """Verify fixed-lag Viterbi picks Road A -> Road C turn despite parallel Road B noise."""
    rng, geo_ref = build_synthetic_known_graph()
    matcher = MapMatcher(
        road_graph=rng,
        search_radius_m=35.0,
        lag_epochs=5,
        min_confidence=0.45,
    )

    # Generate synthetic trajectory along Road A (0 -> 200) then Road C (0 -> 100)
    # Add lateral noise pulling slightly towards Road B (+3m North)
    traj_points = []
    # Segment 1: along Road A from X=10 to X=190
    for x in range(10, 200, 10):
        traj_points.append((float(x), 3.0, 0.0))  # Y=3.0 (closer to Road A than Road B)

    # Turn onto Road C
    for y in range(10, 100, 10):
        traj_points.append((200.0, float(y), 0.0))

    outputs = []
    t_ns = 1_000_000_000
    cov = np.eye(15) * 1.0

    for pt in traj_points:
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

    # Flush remaining
    outputs.extend(matcher.flush_remaining(geo_ref))

    assert len(outputs) == len(traj_points)

    # Check that initial segment snapped to Road A
    snapped_a = [o for o in outputs[:18] if o.snapped]
    assert len(snapped_a) > 10
    for o in snapped_a:
        assert o.matched_edge_id == "road_A_main"
        # Display ENU Y coordinate should be snapped onto Road A (Y=0.0)
        assert pytest.approx(o.display_enu[1], abs=1e-3) == 0.0

    # Check that turn segment snapped to Road C
    snapped_c = [o for o in outputs[20:] if o.snapped]
    assert len(snapped_c) > 5
    for o in snapped_c:
        assert o.matched_edge_id == "road_C_turn"
        # Display ENU X coordinate should be snapped onto Road C (X=200.0)
        assert pytest.approx(o.display_enu[0], abs=1e-3) == 200.0


def test_strict_causality_fixed_lag() -> None:
    """Verify that output at epoch t does NOT change based on data after lag window."""
    rng, geo_ref = build_synthetic_known_graph()

    # Trajectory 1: Straight drive along Road A
    traj1 = [(float(x), 2.0, 0.0) for x in range(10, 150, 10)]

    # Trajectory 2: Identical up to step 10, but then turns drastically
    traj2 = list(traj1[:10])
    for y in range(10, 60, 10):
        traj2.append((100.0, float(y), 0.0))

    def run_trajectory(pts):
        matcher = MapMatcher(road_graph=rng, lag_epochs=4)
        outs = []
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
                outs.append(out)
            t_ns += 100_000_000
        return outs

    outs1 = run_trajectory(traj1)
    outs2 = run_trajectory(traj2)

    # First 5 committed outputs (up to index 5) must be strictly identical
    assert len(outs1) >= 5
    assert len(outs2) >= 5
    for i in range(5):
        assert outs1[i].timestamp_ns == outs2[i].timestamp_ns
        assert outs1[i].snapped == outs2[i].snapped
        assert outs1[i].matched_edge_id == outs2[i].matched_edge_id
        assert pytest.approx(outs1[i].display_enu, abs=1e-9) == outs2[i].display_enu


def test_determinism_repeated_runs() -> None:
    """Verify that repeated runs of the exact same trajectory produce bit-identical results."""
    rng, geo_ref = build_synthetic_known_graph()
    traj = [(float(x), 2.0, 0.0) for x in range(10, 100, 10)]

    def run():
        matcher = MapMatcher(road_graph=rng, lag_epochs=4)
        outs = []
        t_ns = 1_000_000_000
        cov = np.eye(15) * 1.0
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
                outs.append(out)
            t_ns += 100_000_000
        outs.extend(matcher.flush_remaining(geo_ref))
        return outs

    run1 = run()
    run2 = run()

    assert len(run1) == len(run2)
    for o1, o2 in zip(run1, run2):
        assert o1.timestamp_ns == o2.timestamp_ns
        assert o1.snapped == o2.snapped
        assert o1.matched_edge_id == o2.matched_edge_id
        assert o1.fallback_reason == o2.fallback_reason
        assert pytest.approx(o1.confidence, abs=1e-12) == o2.confidence
        assert pytest.approx(o1.display_lat_lon, abs=1e-12) == o2.display_lat_lon
        assert pytest.approx(o1.display_enu, abs=1e-12) == o2.display_enu
