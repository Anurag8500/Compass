"""Integration test for Map Matching graceful no-coverage fallback (Phase 12).

Verifies that when a vehicle drives outside road network coverage (e.g. open field,
offshore, unmapped path):
1. The pipeline never crashes or raises exceptions.
2. No fabricated candidates are created.
3. No forced snap occurs (snapped == False).
4. Output position exactly equals the original estimator position.
5. The fallback reason "NO_CANDIDATES" is explicitly recorded.
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


def build_isolated_graph() -> Tuple[RoadNetworkGraph, GeoReference]:
    """Build a graph with a single small road centered at the origin."""
    geo_ref = GeoReference(lat_ref=52.4000, lon_ref=-1.5000, alt_ref=0.0)
    g = nx.DiGraph()

    # Small road from (0, 0) to (50, 0)
    lat0, lon0, _ = geo_ref.enu_to_geodetic(0.0, 0.0, 0.0)
    lat1, lon1, _ = geo_ref.enu_to_geodetic(50.0, 0.0, 0.0)
    g.add_node(1, lat=float(lat0), lon=float(lon0))
    g.add_node(2, lat=float(lat1), lon=float(lon1))

    g.add_edge(
        1,
        2,
        edge_id="isolated_segment",
        name="Isolated Way",
        highway="residential",
        oneway=True,
        length_m=50.0,
        azimuth_rad=0.5 * math.pi,
        geometry_latlon=[(float(lat0), float(lon0)), (float(lat1), float(lon1))],
    )

    return RoadNetworkGraph(g, geo_ref=geo_ref), geo_ref


def test_no_coverage_graceful_fallback() -> None:
    """Drive trajectory located 500m away from the nearest road (100% no coverage)."""
    rng, geo_ref = build_isolated_graph()
    matcher = MapMatcher(
        road_graph=rng,
        search_radius_m=35.0,
        lag_epochs=5,
    )

    # 10 points driving at (e=500.0, n=500.0) to (e=600.0, n=500.0)
    traj_enu = [(500.0 + float(i) * 10.0, 500.0, 0.0) for i in range(10)]
    outputs = []
    t_ns = 2_000_000_000

    for pt in traj_enu:
        state = NavigationState(
            position_local=np.array(pt, dtype=np.float64),
            velocity_local=np.array([10.0, 0.0, 0.0], dtype=np.float64),
            orientation=OrientationState(q=np.array([1.0, 0.0, 0.0, 0.0]), gyro_bias=np.zeros(3)),
            accel_bias=np.zeros(3),
            covariance=np.eye(15) * 2.0,
            reference_point=(geo_ref.lat_ref, geo_ref.lon_ref),
            mode="GNSS_AIDED",
            timestamp_ns=t_ns,
        )
        out = matcher.process_state(state, geo_ref)
        if out is not None:
            outputs.append(out)
        t_ns += 100_000_000

    outputs.extend(matcher.flush_remaining(geo_ref))

    assert len(outputs) == len(traj_enu)

    for i, out in enumerate(outputs):
        # 1. Must NOT be snapped
        assert not out.snapped
        assert not out.result_schema.snapped

        # 2. Candidate count must be 0
        assert out.candidate_count == 0

        # 3. Fallback reason must be recorded
        assert out.fallback_reason == "NO_CANDIDATES"

        # 4. Display position must EXACTLY match estimator position
        expected_enu = traj_enu[i]
        assert pytest.approx(out.estimator_enu[0], abs=1e-9) == expected_enu[0]
        assert pytest.approx(out.estimator_enu[1], abs=1e-9) == expected_enu[1]
        assert pytest.approx(out.display_enu[0], abs=1e-9) == expected_enu[0]
        assert pytest.approx(out.display_enu[1], abs=1e-9) == expected_enu[1]

        # 5. Geodetic coordinates must match
        assert pytest.approx(out.display_lat_lon[0], abs=1e-9) == out.estimator_lat_lon[0]
        assert pytest.approx(out.display_lat_lon[1], abs=1e-9) == out.estimator_lat_lon[1]

        # 6. Confidence must be 0.0
        assert out.confidence == 0.0
        assert out.result_schema.confidence == 0.0
