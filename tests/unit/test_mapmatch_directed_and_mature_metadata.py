"""Unit tests for Phase 12 directed edge semantics, mature-epoch metadata isolation, and Viterbi timestamps.

Tests:
1. Viterbi timestamp implicit fallback when timestamp_ns is omitted.
2. Dedicated isolation test: current epoch candidates cannot alter mature epoch decisions.
3. Directed-edge transition semantics (7 cases):
   - 1: Forward movement on forward edge
   - 2: Backward movement on forward edge (rejected)
   - 3: Forward movement on reverse edge
   - 4: Forward edge -> reverse edge transition
   - 5: Reverse edge -> forward edge transition
   - 6: One-way road behavior
   - 7: Two-way road behavior
4. Flush remaining metadata integrity.
"""

from __future__ import annotations

import math
from typing import Tuple
import networkx as nx
import numpy as np
import pytest

from maps.extract_osm import RoadNetworkGraph
from navigation.frames.local_geo import GeoReference
from navigation.mapmatch.candidates import CandidateSearch, RoadCandidate
from navigation.mapmatch.matcher import MapMatcher
from navigation.mapmatch.transition import TransitionModel, LOG_ZERO
from navigation.schemas.state import NavigationState, OrientationState


def build_bidirectional_and_oneway_test_graph() -> Tuple[RoadNetworkGraph, GeoReference]:
    """Construct a synthetic graph with both a bidirectional street and a one-way street.

    Topology:
    - Node 1: (0, 0)
    - Node 2: (100, 0)
    - Node 3: (0, 30)
    - Node 4: (100, 30)

    Edges:
    - Two-way street between 1 and 2:
      * forward: 1 -> 2 (edge_id: "two_way_fwd", length: 100m, azimuth: pi/2)
      * reverse: 2 -> 1 (edge_id: "two_way_rev", length: 100m, azimuth: 3*pi/2)
    - One-way street between 3 and 4:
      * forward: 3 -> 4 (edge_id: "one_way_fwd", length: 100m, azimuth: pi/2)
    """
    geo_ref = GeoReference(lat_ref=52.4000, lon_ref=-1.5000, alt_ref=0.0)

    nodes_enu = {
        1: (0.0, 0.0),
        2: (100.0, 0.0),
        3: (0.0, 30.0),
        4: (100.0, 30.0),
    }

    g = nx.DiGraph()
    for nid, (e, n) in nodes_enu.items():
        lat, lon, _ = geo_ref.enu_to_geodetic(e, n, 0.0)
        g.add_node(nid, lat=float(lat), lon=float(lon))

    # Two-way forward: 1 -> 2
    lat1, lon1, _ = geo_ref.enu_to_geodetic(0.0, 0.0, 0.0)
    lat2, lon2, _ = geo_ref.enu_to_geodetic(100.0, 0.0, 0.0)
    g.add_edge(
        1, 2,
        edge_id="two_way_fwd",
        name="two_way_road",
        highway="primary",
        oneway=False,
        length_m=100.0,
        azimuth_rad=0.5 * math.pi,
        geometry_latlon=[(float(lat1), float(lon1)), (float(lat2), float(lon2))],
    )

    # Two-way reverse: 2 -> 1
    g.add_edge(
        2, 1,
        edge_id="two_way_rev",
        name="two_way_road",
        highway="primary",
        oneway=False,
        length_m=100.0,
        azimuth_rad=1.5 * math.pi,
        geometry_latlon=[(float(lat2), float(lon2)), (float(lat1), float(lon1))],
    )

    # One-way street: 3 -> 4
    lat3, lon3, _ = geo_ref.enu_to_geodetic(0.0, 30.0, 0.0)
    lat4, lon4, _ = geo_ref.enu_to_geodetic(100.0, 30.0, 0.0)
    g.add_edge(
        3, 4,
        edge_id="one_way_fwd",
        name="one_way_road",
        highway="primary",
        oneway=True,
        length_m=100.0,
        azimuth_rad=0.5 * math.pi,
        geometry_latlon=[(float(lat3), float(lon3)), (float(lat4), float(lon4))],
    )

    rng = RoadNetworkGraph(g, geo_ref=geo_ref)
    return rng, geo_ref


def test_viterbi_timestamp_implicit_fallback() -> None:
    """Verify process_state handles omitted timestamp_ns correctly and causally."""
    rng, geo_ref = build_bidirectional_and_oneway_test_graph()
    matcher = MapMatcher(road_graph=rng, lag_epochs=3)

    # Ingest 6 states along two_way_fwd with distinct timestamps
    base_t_ns = 1_700_000_000_000_000_000
    dt_ns = 100_000_000  # 100 ms
    cov = np.eye(15) * 1.0

    collected_outputs = []
    for k in range(6):
        curr_t_ns = base_t_ns + k * dt_ns
        pos = np.array([10.0 + k * 10.0, 0.5, 0.0], dtype=np.float64)
        vel = np.array([10.0, 0.0, 0.0], dtype=np.float64)
        state = NavigationState(
            position_local=pos,
            velocity_local=vel,
            orientation=OrientationState(q=np.array([1.0, 0.0, 0.0, 0.0]), gyro_bias=np.zeros(3)),
            accel_bias=np.zeros(3),
            covariance=cov,
            reference_point=(geo_ref.lat_ref, geo_ref.lon_ref),
            mode="GNSS_AIDED",
            timestamp_ns=curr_t_ns,
        )
        # Call WITHOUT explicitly passing timestamp_ns
        out = matcher.process_state(state, geo_ref)
        if out is not None:
            collected_outputs.append(out)

    flushed = matcher.flush_remaining(geo_ref)
    all_outs = collected_outputs + flushed

    assert len(all_outs) == 6
    for k, o in enumerate(all_outs):
        expected_t_ns = base_t_ns + k * dt_ns
        assert o.timestamp_ns == expected_t_ns, f"Epoch {k}: expected {expected_t_ns}, got {o.timestamp_ns}"
        # Causality verification: committed epoch coordinates must strictly match the k-th state
        expected_east = 10.0 + k * 10.0
        assert pytest.approx(o.estimator_enu[0], abs=1e-5) == expected_east


def test_committed_epoch_metadata_isolation() -> None:
    """Dedicated regression test: Current epoch candidate count must NOT alter mature epoch decision.

    Scenario:
    - At epoch 0 (mature), vehicle is between 2 parallel roads with ambiguity (margin < 1.0).
    - At subsequent epochs 1..lag_epochs+1, vehicle enters a region with only 1 candidate.
    - When epoch 0 matures and commits, it MUST be evaluated using epoch 0's candidate_count (2)
      and trigger AMBIGUOUS_PARALLEL_ROADS, NOT bypass the check because current epoch has count 1.
    """
    geo_ref = GeoReference(lat_ref=52.4000, lon_ref=-1.5000, alt_ref=0.0)

    # Create two parallel roads separated by 10m
    g = nx.DiGraph()
    for nid, (e, n) in {1: (0.0, 0.0), 2: (200.0, 0.0), 3: (0.0, 10.0), 4: (200.0, 10.0)}.items():
        lat, lon, _ = geo_ref.enu_to_geodetic(e, n, 0.0)
        g.add_node(nid, lat=float(lat), lon=float(lon))

    # Road S (Y=0)
    lat1, lon1, _ = geo_ref.enu_to_geodetic(0.0, 0.0, 0.0)
    lat2, lon2, _ = geo_ref.enu_to_geodetic(200.0, 0.0, 0.0)
    g.add_edge(1, 2, edge_id="road_south", name="s", highway="primary", oneway=True, length_m=200.0, azimuth_rad=0.5 * math.pi,
               geometry_latlon=[(float(lat1), float(lon1)), (float(lat2), float(lon2))])

    # Road N (Y=10) from X=0 to X=50 only (ends at X=50 so later points only have 1 candidate!)
    lat3, lon3, _ = geo_ref.enu_to_geodetic(0.0, 10.0, 0.0)
    lat4, lon4, _ = geo_ref.enu_to_geodetic(50.0, 10.0, 0.0)
    g.add_edge(3, 4, edge_id="road_north", name="n", highway="primary", oneway=True, length_m=50.0, azimuth_rad=0.5 * math.pi,
               geometry_latlon=[(float(lat3), float(lon3)), (float(lat4), float(lon4))])

    rng = RoadNetworkGraph(g, geo_ref=geo_ref)
    matcher = MapMatcher(road_graph=rng, search_radius_m=20.0, ambiguity_margin=1.0, lag_epochs=4)

    # Epoch 0: at (25.0, 5.0) -> equidistant between South (dist 5m) and North (dist 5m). 2 candidates, margin ~ 0.
    # Epochs 1..5: at (60, 0), (70, 0), (80, 0), (90, 0), (100, 0) -> North road has ended! Only 1 candidate (South).
    cov = np.eye(15) * 1.0
    states = [
        np.array([25.0, 5.0, 0.0]),  # Epoch 0: 2 candidates, ambiguous
        np.array([60.0, 0.0, 0.0]),  # Epoch 1: 1 candidate
        np.array([70.0, 0.0, 0.0]),  # Epoch 2: 1 candidate
        np.array([80.0, 0.0, 0.0]),  # Epoch 3: 1 candidate
        np.array([90.0, 0.0, 0.0]),  # Epoch 4: 1 candidate
        np.array([100.0, 0.0, 0.0]), # Epoch 5: 1 candidate (commit for epoch 0 triggers here)
    ]

    commits_out = []
    t_ns = 1_000_000_000
    for pos in states:
        state = NavigationState(
            position_local=pos,
            velocity_local=np.array([10.0, 0.0, 0.0]),
            orientation=OrientationState(q=np.array([1.0, 0.0, 0.0, 0.0]), gyro_bias=np.zeros(3)),
            accel_bias=np.zeros(3),
            covariance=cov,
            reference_point=(geo_ref.lat_ref, geo_ref.lon_ref),
            mode="GNSS_AIDED",
            timestamp_ns=t_ns,
        )
        out = matcher.process_state(state, geo_ref)
        if out is not None:
            commits_out.append(out)
        t_ns += 100_000_000

    assert len(commits_out) >= 1
    # First commit corresponds to Epoch 0 (at X=25.0)
    out0 = commits_out[0]
    assert pytest.approx(out0.estimator_enu[0], abs=1e-5) == 25.0
    # Must reflect Epoch 0's candidate count (2), NOT current epoch's count (1)!
    assert out0.candidate_count == 2
    # Because margin < 1.0 and candidate_count == 2, it MUST safely fallback to AMBIGUOUS_PARALLEL_ROADS
    assert not out0.snapped
    assert out0.fallback_reason == "AMBIGUOUS_PARALLEL_ROADS"


def test_directed_edge_transition_semantics() -> None:
    """Explicit unit tests for all 7 directed-edge transition semantics cases."""
    rng, geo_ref = build_bidirectional_and_oneway_test_graph()
    tm = TransitionModel(rng, beta_m=5.0)
    cs = CandidateSearch(rng, search_radius_m=20.0)

    # Case 1: Forward movement on forward edge (two_way_fwd: 1 -> 2)
    c1_prev = [c for c in cs.search_candidates((20.0, 0.0)) if c.edge_id == "two_way_fwd"][0]
    c1_curr = [c for c in cs.search_candidates((40.0, 0.0)) if c.edge_id == "two_way_fwd"][0]
    d1 = tm.get_route_distance_m(c1_prev, c1_curr)
    assert d1 is not None
    assert pytest.approx(d1, abs=1e-3) == 20.0
    log_p1 = tm.compute_log_transition(c1_prev, c1_curr, (20.0, 0.0), (40.0, 0.0))
    assert log_p1 > LOG_ZERO / 2

    # Case 2: Backward movement on forward edge (rejected)
    c2_prev = [c for c in cs.search_candidates((40.0, 0.0)) if c.edge_id == "two_way_fwd"][0]
    c2_curr = [c for c in cs.search_candidates((20.0, 0.0)) if c.edge_id == "two_way_fwd"][0]
    d2 = tm.get_route_distance_m(c2_prev, c2_curr)
    assert d2 is None
    log_p2 = tm.compute_log_transition(c2_prev, c2_curr, (40.0, 0.0), (20.0, 0.0))
    assert log_p2 == LOG_ZERO

    # Case 3: Forward movement on reverse edge (two_way_rev: 2 -> 1)
    # On reverse edge, node 2 is start (X=100) and node 1 is end (X=0).
    # Moving from X=80 to X=60 is progressing forward along two_way_rev!
    c3_prev = [c for c in cs.search_candidates((80.0, 0.0)) if c.edge_id == "two_way_rev"][0]
    c3_curr = [c for c in cs.search_candidates((60.0, 0.0)) if c.edge_id == "two_way_rev"][0]
    d3 = tm.get_route_distance_m(c3_prev, c3_curr)
    assert d3 is not None
    assert pytest.approx(d3, abs=1e-3) == 20.0
    log_p3 = tm.compute_log_transition(c3_prev, c3_curr, (80.0, 0.0), (60.0, 0.0))
    assert log_p3 > LOG_ZERO / 2

    # Case 4: Forward edge -> reverse edge transition (connected at junction node 2)
    # Vehicle approaches end of forward edge at X=90, turns around and enters reverse edge at X=90.
    c4_prev = [c for c in cs.search_candidates((90.0, 0.0)) if c.edge_id == "two_way_fwd"][0]
    c4_curr = [c for c in cs.search_candidates((90.0, 0.0)) if c.edge_id == "two_way_rev"][0]
    d4 = tm.get_route_distance_m(c4_prev, c4_curr)
    # Remaining on fwd: 100 - 90 = 10m. From node 2 along rev: 10m. Total = 20m.
    assert d4 is not None
    assert pytest.approx(d4, abs=1e-3) == 20.0

    # Case 5: Reverse edge -> forward edge transition (connected at junction node 1)
    # Vehicle approaches end of reverse edge at X=10, turns around and enters forward edge at X=10.
    c5_prev = [c for c in cs.search_candidates((10.0, 0.0)) if c.edge_id == "two_way_rev"][0]
    c5_curr = [c for c in cs.search_candidates((10.0, 0.0)) if c.edge_id == "two_way_fwd"][0]
    d5 = tm.get_route_distance_m(c5_prev, c5_curr)
    # Remaining on rev: 10m. From node 1 along fwd: 10m. Total = 20m.
    assert d5 is not None
    assert pytest.approx(d5, abs=1e-3) == 20.0

    # Case 6: One-way road behavior (one_way_fwd: 3 -> 4)
    # Forward movement is valid
    c6_fwd_prev = [c for c in cs.search_candidates((20.0, 30.0)) if c.edge_id == "one_way_fwd"][0]
    c6_fwd_curr = [c for c in cs.search_candidates((40.0, 30.0)) if c.edge_id == "one_way_fwd"][0]
    d6_fwd = tm.get_route_distance_m(c6_fwd_prev, c6_fwd_curr)
    assert d6_fwd is not None
    assert pytest.approx(d6_fwd, abs=1e-3) == 20.0
    # Backward movement along one-way road is rejected
    d6_back = tm.get_route_distance_m(c6_fwd_curr, c6_fwd_prev)
    assert d6_back is None
    # No reverse edge exists for one_way road in the graph
    c6_cands = cs.search_candidates((20.0, 30.0))
    assert all(c.edge_id != "one_way_rev" for c in c6_cands)

    # Case 7: Two-way road behavior
    # Bidirectional traffic is supported simultaneously by two_way_fwd and two_way_rev
    c7_cands = cs.search_candidates((50.0, 0.0))
    edge_ids = {c.edge_id for c in c7_cands}
    assert "two_way_fwd" in edge_ids
    assert "two_way_rev" in edge_ids
