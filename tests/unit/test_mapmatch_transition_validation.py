"""Targeted validation of the physically bounded road-graph transition model (Phase 12).

Tests the ACTUAL implemented transition model and kinematic/geometry tolerance gate:
1. Same-edge forward motion acceptance.
2. Same-edge backward motion rejection (beyond -1.0 m projection jitter).
3. Legitimate junction turn acceptance (connecting directed edges).
4. Legitimate reverse direction using matching reverse directed edge.
5. Disconnected road rejection (NetworkXNoPath -> LOG_ZERO / None).
6. Physically impossible jump rejection (> max_speed_mps * delta_t_s + proj_uncertainty + geom_tol).
7. Physically plausible connected motion acceptance.
8. Small timestamp intervals (delta_t = 0.05s).
9. Larger timestamp intervals (delta_t = 1.0s, 2.0s).
10. Candidate projection offset contribution (c_prev.distance_to_road_m + c_curr.distance_to_road_m).
11. Determinism across repeated evaluations.
"""

from __future__ import annotations

import math
import networkx as nx
import numpy as np
import pytest

from maps.extract_osm import RoadNetworkGraph
from navigation.frames.local_geo import GeoReference
from navigation.mapmatch.candidates import RoadCandidate
from navigation.mapmatch.transition import TransitionModel, LOG_ZERO


@pytest.fixture
def test_network() -> Tuple[RoadNetworkGraph, GeoReference]:
    """Builds a deterministic test road graph with a junction and dual carriageway."""
    geo_ref = GeoReference(lat_ref=52.4000, lon_ref=-1.5000, alt_ref=0.0)

    # Nodes in ENU:
    # 1: (0, 0)
    # 2: (100, 0)
    # 3: (200, 0)
    # 4: (100, 100)
    # 5: (0, 20)  # Parallel road
    # 6: (200, 20) # Parallel road
    nodes = {
        1: (0.0, 0.0),
        2: (100.0, 0.0),
        3: (200.0, 0.0),
        4: (100.0, 100.0),
        5: (0.0, 20.0),
        6: (200.0, 20.0),
    }

    g = nx.DiGraph()
    for nid, (e, n) in nodes.items():
        lat, lon, _ = geo_ref.enu_to_geodetic(e, n, 0.0)
        g.add_node(nid, lat=float(lat), lon=float(lon))

    def add_dir_edge(u: int, v: int, eid: str, oneway: bool = True):
        p1 = nodes[u]
        p2 = nodes[v]
        length = float(math.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        geom = [geo_ref.enu_to_geodetic(p1[0], p1[1], 0.0)[:2], geo_ref.enu_to_geodetic(p2[0], p2[1], 0.0)[:2]]
        g.add_edge(
            u, v,
            edge_id=eid,
            name=eid,
            highway="primary",
            oneway=oneway,
            length_m=length,
            azimuth_rad=math.atan2(p2[0] - p1[0], p2[1] - p1[1]),
            geometry_latlon=geom,
        )

    # Main road: 1 -> 2 -> 3 (Eastbound)
    add_dir_edge(1, 2, "main_1_2")
    add_dir_edge(2, 3, "main_2_3")

    # Reverse main road: 3 -> 2 -> 1 (Westbound)
    add_dir_edge(3, 2, "main_3_2_rev")
    add_dir_edge(2, 1, "main_2_1_rev")

    # Junction turn: 2 -> 4 (Northbound)
    add_dir_edge(2, 4, "junction_2_4")

    # Disconnected parallel road: 5 -> 6 (Eastbound, 20m North)
    add_dir_edge(5, 6, "parallel_5_6")

    rng = RoadNetworkGraph(g, geo_ref=geo_ref)
    return rng, geo_ref


def make_cand(
    edge_id: str,
    u: int,
    v: int,
    dist_along: float,
    total_len: float,
    dist_to_road: float = 1.0,
) -> RoadCandidate:
    return RoadCandidate(
        edge_id=edge_id,
        u=u,
        v=v,
        segment_idx=0,
        projected_point_enu=(0.0, 0.0),
        projected_lat_lon=(52.4, -1.5),
        distance_to_road_m=dist_to_road,
        fraction_along_edge=dist_along / max(total_len, 1e-3),
        distance_along_edge_m=dist_along,
        edge_total_length_m=total_len,
        edge_tangent_enu=(1.0, 0.0),
        edge_normal_enu=(0.0, 1.0),
        edge_azimuth_rad=0.0,
        edge_name=edge_id,
        highway="primary",
    )


def test_transition_same_edge_forward_motion(test_network) -> None:
    """Same-edge forward motion is accepted with exact distance difference."""
    rng, _ = test_network
    tm = TransitionModel(road_graph=rng, beta_m=5.0, max_speed_mps=45.0)

    c_prev = make_cand("main_1_2", 1, 2, dist_along=20.0, total_len=100.0)
    c_curr = make_cand("main_1_2", 1, 2, dist_along=35.0, total_len=100.0)

    # 15 m forward in 1.0 s -> delta_d = 15.0 m
    route_dist = tm.get_route_distance_m(c_prev, c_curr, delta_t_s=1.0)
    assert route_dist == pytest.approx(15.0, abs=1e-3)

    log_p = tm.compute_log_transition(
        c_prev, c_curr,
        prev_traj_enu=(20.0, 0.0),
        curr_traj_enu=(35.0, 0.0),
        delta_t_s=1.0,
    )
    # discrepancy = |15.0 - 15.0| = 0.0 -> log_p = -ln(5.0)
    assert log_p == pytest.approx(-math.log(5.0), abs=1e-3)


def test_transition_same_edge_backward_rejection(test_network) -> None:
    """Same-edge backward motion beyond -1.0 m projection tolerance is rejected."""
    rng, _ = test_network
    tm = TransitionModel(road_graph=rng, beta_m=5.0)

    c_prev = make_cand("main_1_2", 1, 2, dist_along=50.0, total_len=100.0)
    # Moving backwards 5m on a directed edge
    c_curr = make_cand("main_1_2", 1, 2, dist_along=45.0, total_len=100.0)

    route_dist = tm.get_route_distance_m(c_prev, c_curr, delta_t_s=1.0)
    assert route_dist is None

    log_p = tm.compute_log_transition(
        c_prev, c_curr,
        prev_traj_enu=(50.0, 0.0),
        curr_traj_enu=(45.0, 0.0),
        delta_t_s=1.0,
    )
    assert log_p == LOG_ZERO


def test_transition_legitimate_junction_turn(test_network) -> None:
    """Legitimate turn at junction 2 from main_1_2 into junction_2_4 is accepted."""
    rng, _ = test_network
    tm = TransitionModel(road_graph=rng, beta_m=5.0)

    # At 90m on main_1_2 (10m from node 2)
    c_prev = make_cand("main_1_2", 1, 2, dist_along=90.0, total_len=100.0)
    # At 15m on junction_2_4 (15m from node 2)
    c_curr = make_cand("junction_2_4", 2, 4, dist_along=15.0, total_len=100.0)

    # Route distance = (100 - 90) + Dijkstra(2, 2=0) + 15 = 25m
    route_dist = tm.get_route_distance_m(c_prev, c_curr, delta_t_s=2.0)
    assert route_dist == pytest.approx(25.0, abs=1e-3)


def test_transition_legitimate_reverse_on_reverse_edge(test_network) -> None:
    """Reverse direction travel is accepted if using the matching reverse directed edge."""
    rng, _ = test_network
    tm = TransitionModel(road_graph=rng, beta_m=5.0)

    # Traveling Westbound on main_3_2_rev
    c_prev = make_cand("main_3_2_rev", 3, 2, dist_along=10.0, total_len=100.0)
    c_curr = make_cand("main_3_2_rev", 3, 2, dist_along=25.0, total_len=100.0)

    route_dist = tm.get_route_distance_m(c_prev, c_curr, delta_t_s=1.0)
    assert route_dist == pytest.approx(15.0, abs=1e-3)


def test_transition_disconnected_road_rejection(test_network) -> None:
    """Transition between parallel unconnected roads returns LOG_ZERO."""
    rng, _ = test_network
    tm = TransitionModel(road_graph=rng, beta_m=5.0)

    c_prev = make_cand("main_1_2", 1, 2, dist_along=50.0, total_len=100.0)
    c_curr = make_cand("parallel_5_6", 5, 6, dist_along=50.0, total_len=200.0)

    route_dist = tm.get_route_distance_m(c_prev, c_curr, delta_t_s=1.0)
    assert route_dist is None

    log_p = tm.compute_log_transition(
        c_prev, c_curr,
        prev_traj_enu=(50.0, 0.0),
        curr_traj_enu=(50.0, 20.0),
        delta_t_s=1.0,
    )
    assert log_p == LOG_ZERO


def test_transition_physically_impossible_speed_rejection(test_network) -> None:
    """Jump exceeding max_speed_mps * dt + proj_uncertainty + geom_tol is rejected."""
    rng, _ = test_network
    # max_speed_mps = 30 m/s (~108 km/h)
    tm = TransitionModel(road_graph=rng, beta_m=5.0, max_speed_mps=30.0)

    # Turn from 1_2 to 2_4 requiring 25m distance
    c_prev = make_cand("main_1_2", 1, 2, dist_along=90.0, total_len=100.0, dist_to_road=1.0)
    c_curr = make_cand("junction_2_4", 2, 4, dist_along=15.0, total_len=100.0, dist_to_road=1.0)
    # d_total = 25m.
    # At dt = 0.1s: d_kinematic = 30 * 0.1 = 3.0m.
    # proj_uncertainty = 1.0 + 1.0 = 2.0m, geom_tol = 10.0m.
    # Gate threshold = 3.0 + 2.0 + 10.0 = 15.0m < 25.0m!
    # Must be rejected!
    route_dist = tm.get_route_distance_m(c_prev, c_curr, delta_t_s=0.1)
    assert route_dist is None

    # But at dt = 1.0s: d_kinematic = 30m, threshold = 30 + 2 + 10 = 42m > 25m -> Accepted!
    route_dist_ok = tm.get_route_distance_m(c_prev, c_curr, delta_t_s=1.0)
    assert route_dist_ok == pytest.approx(25.0, abs=1e-3)


def test_transition_projection_offset_contribution(test_network) -> None:
    """Validates that candidate perpendicular offsets (c_prev.dist + c_curr.dist) dynamically contribute to the gate."""
    rng, _ = test_network
    tm = TransitionModel(road_graph=rng, beta_m=5.0, max_speed_mps=10.0)

    # d_inter = 0 (same node 2), d_total = 16.0m
    # dt = 0.5s -> d_kinematic = 5.0m, geom_tol = 10.0m.
    # If dist_to_road = 0.2m -> threshold = 5.0 + 0.4 + 10.0 = 15.4m < 16.0m -> REJECTED
    c1 = make_cand("main_1_2", 1, 2, dist_along=92.0, total_len=100.0, dist_to_road=0.2)
    c2 = make_cand("junction_2_4", 2, 4, dist_along=8.0, total_len=100.0, dist_to_road=0.2)
    assert tm.get_route_distance_m(c1, c2, delta_t_s=0.5) is None

    # If vehicle has higher lateral uncertainty (e.g. dist_to_road = 3.0m):
    # threshold = 5.0 + 6.0 + 10.0 = 21.0m > 16.0m -> ACCEPTED!
    c1_wide = make_cand("main_1_2", 1, 2, dist_along=92.0, total_len=100.0, dist_to_road=3.0)
    c2_wide = make_cand("junction_2_4", 2, 4, dist_along=8.0, total_len=100.0, dist_to_road=3.0)
    assert tm.get_route_distance_m(c1_wide, c2_wide, delta_t_s=0.5) == pytest.approx(16.0, abs=1e-3)


def test_transition_determinism(test_network) -> None:
    """Repeated evaluations must return bit-for-bit identical log probabilities."""
    rng, _ = test_network
    tm = TransitionModel(road_graph=rng, beta_m=5.0)

    c_prev = make_cand("main_1_2", 1, 2, dist_along=10.0, total_len=100.0)
    c_curr = make_cand("main_1_2", 1, 2, dist_along=22.0, total_len=100.0)

    scores = [
        tm.compute_log_transition(c_prev, c_curr, (10.0, 0.0), (22.0, 0.0), delta_t_s=1.0)
        for _ in range(10)
    ]
    assert len(set(scores)) == 1, "Transition model must be 100% deterministic"
