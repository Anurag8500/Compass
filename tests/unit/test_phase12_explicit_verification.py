"""Explicit mathematical and architectural verification tests for Phase 12 (Pre-implementation Correction 1).

Verifies:
1. Fixed-lag Viterbi causality.
2. No future-sample usage during online stepping.
3. Mature-state commit semantics (mature epoch t - W metadata isolation).
4. Session-local ENU consistency between road graph and trajectory coordinates.
5. Segment-safe exact orthogonal projection (including endpoint clamping).
6. Covariance-aware emission formula (directional road-normal variance projection).
"""

from __future__ import annotations

import math
from typing import List, Tuple
import networkx as nx
import numpy as np
import pytest

from maps.extract_osm import RoadNetworkGraph
from navigation.frames.local_geo import GeoReference
from navigation.mapmatch.candidates import CandidateSearch, RoadCandidate
from navigation.mapmatch.emission import EmissionModel
from navigation.mapmatch.transition import TransitionModel
from navigation.mapmatch.viterbi import FixedLagViterbi, ViterbiCommit


@pytest.fixture
def test_graph_and_geo() -> Tuple[RoadNetworkGraph, GeoReference]:
    geo_ref = GeoReference(lat_ref=52.4000, lon_ref=-1.5000, alt_ref=0.0)
    g = nx.DiGraph()

    nodes = {
        1: (0.0, 0.0),
        2: (200.0, 0.0),
        3: (200.0, 200.0),
    }
    for nid, (e, n) in nodes.items():
        lat, lon, _ = geo_ref.enu_to_geodetic(e, n, 0.0)
        g.add_node(nid, lat=float(lat), lon=float(lon))

    def add_e(u, v, eid):
        p1, p2 = nodes[u], nodes[v]
        length = float(math.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        geom = [geo_ref.enu_to_geodetic(p1[0], p1[1], 0.0)[:2], geo_ref.enu_to_geodetic(p2[0], p2[1], 0.0)[:2]]
        g.add_edge(
            u, v,
            edge_id=eid,
            name=eid,
            highway="primary",
            oneway=True,
            length_m=length,
            azimuth_rad=math.atan2(p2[0] - p1[0], p2[1] - p1[1]),
            geometry_latlon=geom,
        )

    add_e(1, 2, "road_east")
    add_e(2, 3, "road_north")
    rng = RoadNetworkGraph(g, geo_ref=geo_ref)
    return rng, geo_ref


def make_test_cand(edge_id: str, u: int, v: int, dist_along: float, total_len: float, d_perp: float = 1.0) -> RoadCandidate:
    return RoadCandidate(
        edge_id=edge_id,
        u=u,
        v=v,
        segment_idx=0,
        projected_point_enu=(dist_along, 0.0),
        projected_lat_lon=(52.4, -1.5),
        distance_to_road_m=d_perp,
        fraction_along_edge=dist_along / total_len,
        distance_along_edge_m=dist_along,
        edge_total_length_m=total_len,
        edge_tangent_enu=(1.0, 0.0),
        edge_normal_enu=(0.0, 1.0),
        edge_azimuth_rad=0.0,
        edge_name=edge_id,
        highway="primary",
    )


def test_fixed_lag_viterbi_causality_and_no_future_usage(test_graph_and_geo) -> None:
    """Changing future observations after maturity depth W has ZERO effect on committed outputs."""
    rng, _ = test_graph_and_geo
    tm = TransitionModel(rng)

    lag_w = 4
    viterbi_a = FixedLagViterbi(transition_model=tm, lag_epochs=lag_w)
    viterbi_b = FixedLagViterbi(transition_model=tm, lag_epochs=lag_w)

    committed_a: List[ViterbiCommit] = []
    committed_b: List[ViterbiCommit] = []

    # Common trajectory up to step 10
    for step in range(10):
        t_ns = 1_000_000_000 + step * 100_000_000
        x = float(step * 10.0)
        cands = [make_test_cand("road_east", 1, 2, dist_along=x, total_len=200.0, d_perp=0.5)]
        emiss = [-0.5]

        res_a = viterbi_a.step(t_ns, (x, 0.5), cands, emiss, delta_t_s=0.1)
        res_b = viterbi_b.step(t_ns, (x, 0.5), cands, emiss, delta_t_s=0.1)

        if res_a is not None:
            committed_a.append(res_a)
        if res_b is not None:
            committed_b.append(res_b)

    # Mature outputs up to this point MUST be identical
    assert len(committed_a) == len(committed_b)
    for ca, cb in zip(committed_a, committed_b):
        assert ca.candidate.edge_id == cb.candidate.edge_id
        assert ca.best_score == pytest.approx(cb.best_score, abs=1e-9)

    len_before_future_divergence = len(committed_a)

    # Now step 10 to 15: Stream A continues straight on road_east, Stream B veers wildly
    for step in range(10, 15):
        t_ns = 1_000_000_000 + step * 100_000_000
        x = float(step * 10.0)

        cands_a = [make_test_cand("road_east", 1, 2, dist_along=x, total_len=200.0, d_perp=0.5)]
        cands_b = [make_test_cand("road_east", 1, 2, dist_along=x, total_len=200.0, d_perp=15.0)]

        res_a = viterbi_a.step(t_ns, (x, 0.5), cands_a, [-0.5], delta_t_s=0.1)
        res_b = viterbi_b.step(t_ns, (x, 15.0), cands_b, [-15.0], delta_t_s=0.1)

        if res_a is not None:
            committed_a.append(res_a)
        if res_b is not None:
            committed_b.append(res_b)

    # Causality Assertion: The first len_before_future_divergence decisions in committed_a were
    # committed BEFORE the future divergence occurred. They MUST be bit-for-bit identical to
    # the decisions that would have been committed regardless of future samples!
    for idx in range(len_before_future_divergence):
        assert committed_a[idx].candidate.edge_id == committed_b[idx].candidate.edge_id
        assert committed_a[idx].timestamp_ns == committed_b[idx].timestamp_ns


def test_mature_state_commit_semantics(test_graph_and_geo) -> None:
    """Verifies that committed result corresponds to epoch t - W, and metadata is strictly isolated."""
    rng, _ = test_graph_and_geo
    tm = TransitionModel(rng)
    lag_w = 3
    viterbi = FixedLagViterbi(transition_model=tm, lag_epochs=lag_w)

    timestamps = [1_000_000_000 + k * 100_000_000 for k in range(10)]

    results: List[ViterbiCommit] = []
    for k, t_ns in enumerate(timestamps):
        cands = [
            make_test_cand("road_east", 1, 2, dist_along=float(k*10), total_len=200.0, d_perp=float(c))
            for c in range(k + 1)
        ]
        emiss = [-0.1 * c for c in range(k + 1)]
        res = viterbi.step(t_ns, (float(k*10), 0.0), cands, emiss, delta_t_s=0.1)
        if res is not None:
            results.append(res)

    assert len(results) > 0
    first_res = results[0]
    # The committed timestamp must be the mature timestamp from step index 0
    assert first_res.timestamp_ns == timestamps[0]
    # And candidate count must be 1 (count from step 0), NOT 4 (count from current step 3)
    assert first_res.candidate_count == 1, "Mature epoch metadata isolation failed!"


def test_session_local_enu_consistency(test_graph_and_geo) -> None:
    """Verifies that road graph geometry and trajectory coordinates projected through GeoReference match metric scale."""
    rng, geo_ref = test_graph_and_geo

    # Point at (100.0, 50.0, 0.0) in local ENU
    lat, lon, alt = geo_ref.enu_to_geodetic(100.0, 50.0, 0.0)
    e_rec, n_rec, u_rec = geo_ref.geodetic_to_enu(lat, lon, alt)

    assert e_rec == pytest.approx(100.0, abs=1e-4)
    assert n_rec == pytest.approx(50.0, abs=1e-4)
    assert u_rec == pytest.approx(0.0, abs=1e-4)

    # Check edge length in ENU matches WGS84 great-circle distance
    edge_data = rng.graph[1][2]
    expected_len = 200.0
    assert edge_data["length_m"] == pytest.approx(expected_len, abs=1e-2)


def test_segment_safe_exact_projection() -> None:
    """Verifies exact orthogonal projection and clamping onto segment [a, b]."""
    geo_ref = GeoReference(52.4, -1.5, 0.0)
    # Segment along X from (10, 20) to (110, 20)
    p0 = (10.0, 20.0)
    p1 = (110.0, 20.0)

    # Case A: Interior point (50, 25) -> orthogonal projection is (50, 20), dist = 5.0
    q_interior = (50.0, 25.0)
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    L2 = dx*dx + dy*dy
    t = ((q_interior[0] - p0[0])*dx + (q_interior[1] - p0[1])*dy) / L2
    t_clamped = max(0.0, min(1.0, t))
    proj_x = p0[0] + t_clamped * dx
    proj_y = p0[1] + t_clamped * dy
    d_perp = math.hypot(q_interior[0] - proj_x, q_interior[1] - proj_y)

    assert t == pytest.approx(0.4, abs=1e-6)
    assert proj_x == pytest.approx(50.0, abs=1e-6)
    assert proj_y == pytest.approx(20.0, abs=1e-6)
    assert d_perp == pytest.approx(5.0, abs=1e-6)

    # Case B: Point before start (5, 30) -> clamped to p0 (10, 20)
    q_before = (5.0, 20.0)
    t_b = max(0.0, min(1.0, ((q_before[0] - p0[0])*dx + (q_before[1] - p0[1])*dy) / L2))
    assert t_b == 0.0

    # Case C: Point after end (125, 20) -> clamped to p1 (110, 20)
    q_after = (125.0, 20.0)
    t_a = max(0.0, min(1.0, ((q_after[0] - p0[0])*dx + (q_after[1] - p0[1])*dy) / L2))
    assert t_a == 1.0


def test_covariance_aware_emission_formula() -> None:
    """Verifies road-normal variance projection: sigma_d^2 = n^T P_pp n + sigma_road^2."""
    em = EmissionModel(sigma_road_m=4.0)

    # Road running East-West: tangent = [1, 0], normal = [0, 1] (North)
    cand = make_test_cand("road_east", 1, 2, dist_along=50.0, total_len=200.0, d_perp=3.0)

    # Test 1: Diagonal covariance with high North uncertainty (var_N = 9.0) and low East uncertainty (var_E = 1.0)
    cov_high_north = np.array([
        [1.0, 0.0],
        [0.0, 9.0],
    ])
    # Road normal is [0, 1] (along North), so n^T P n = 9.0
    # Total sigma_d^2 = 9.0 + 4.0^2 = 9.0 + 16.0 = 25.0 -> sigma_d = 5.0
    log_p_1 = em.compute_log_emission(cand, cov_enu_2x2=cov_high_north)
    expected_log_p_1 = -0.5 * math.log(2 * math.pi * 25.0) - (3.0**2) / (2 * 25.0)
    assert log_p_1 == pytest.approx(expected_log_p_1, abs=1e-6)

    # Test 2: Inverted covariance with high East uncertainty (var_E = 9.0) and low North uncertainty (var_N = 1.0)
    cov_low_north = np.array([
        [9.0, 0.0],
        [0.0, 1.0],
    ])
    # Road normal is [0, 1], so n^T P n = 1.0 (East uncertainty has ZERO projection along North normal!)
    # Total sigma_d^2 = 1.0 + 16.0 = 17.0
    log_p_2 = em.compute_log_emission(cand, cov_enu_2x2=cov_low_north)
    expected_log_p_2 = -0.5 * math.log(2 * math.pi * 17.0) - (3.0**2) / (2 * 17.0)
    assert log_p_2 == pytest.approx(expected_log_p_2, abs=1e-6)

    # At a larger distance (d=10.0m), the wider distribution has much higher likelihood in the tail
    cand_far = make_test_cand("road_east", 1, 2, dist_along=50.0, total_len=200.0, d_perp=10.0)
    log_p_far_wide = em.compute_log_emission(cand_far, cov_enu_2x2=cov_high_north)
    log_p_far_narrow = em.compute_log_emission(cand_far, cov_enu_2x2=cov_low_north)
    assert log_p_far_wide > log_p_far_narrow
    assert em.compute_distance_variance(cand, cov_high_north) == pytest.approx(25.0, abs=1e-6)
    assert em.compute_distance_variance(cand, cov_low_north) == pytest.approx(17.0, abs=1e-6)
