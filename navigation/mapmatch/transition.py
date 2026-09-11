"""Road-graph transition probability model based on network topology routing (Phase 12).

Evaluates transition likelihood between consecutive road candidates:
1. Shortest-path routing along directed road network graph.
2. Exponential penalty on discrepancy between graph route distance and trajectory displacement:
       Delta_d = |d_graph - Delta_d_traj|
       ln p(c_t | c_{t-1}) = -ln(beta) - Delta_d / beta
3. Disconnected graph transitions or impossible backward moves on one-way streets return -inf.
4. Uses bounded Dijkstra search with memoization to maintain low streaming latency.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple
import networkx as nx
import numpy as np

from maps.extract_osm import RoadNetworkGraph
from navigation.mapmatch.candidates import RoadCandidate

LOG_ZERO = -1e9  # Numerically safe representation of -inf


class TransitionModel:
    """Road-graph transition probability evaluator."""

    def __init__(
        self,
        road_graph: RoadNetworkGraph,
        beta_m: float = 5.0,
        max_speed_mps: float = 45.0,
    ) -> None:
        """Initialize transition model.

        Args:
            road_graph: The active RoadNetworkGraph instance.
            beta_m: Scale parameter for exponential difference distribution (default 5.0 m).
            max_speed_mps: Maximum physically plausible speed to reject impossible leaps.
        """
        if beta_m <= 0.0:
            raise ValueError(f"beta_m must be positive, got {beta_m}")

        self.road_graph = road_graph
        self.beta_m = float(beta_m)
        self.log_beta = math.log(self.beta_m)
        self.max_speed_mps = float(max_speed_mps)
        self._path_cache: Dict[Tuple[int, int], Optional[float]] = {}

    def get_route_distance_m(
        self,
        c_prev: RoadCandidate,
        c_curr: RoadCandidate,
        delta_t_s: Optional[float] = None,
    ) -> Optional[float]:
        """Compute network routing distance from c_prev to c_curr.

        Returns None if no directed path exists or jump is physically impossible.
        """
        # Case A: Same directed edge
        if c_prev.edge_id == c_curr.edge_id:
            d_along = c_curr.distance_along_edge_m - c_prev.distance_along_edge_m
            if d_along >= -1.0:
                return max(0.0, d_along)
            else:
                # In a directed graph, each directed edge has an encoded travel direction.
                # Backward travel along any directed edge beyond projection jitter is rejected.
                # For bidirectional roads, legitimate reverse travel must use the matching reverse directed edge.
                return None

        # Case B: Different edges
        # Remaining distance along previous edge to its target node v_prev
        d_prev_rem = max(0.0, c_prev.edge_total_length_m - c_prev.distance_along_edge_m)

        # Distance from node v_prev to node u_curr in directed graph
        node_from = c_prev.v
        node_to = c_curr.u

        if node_from == node_to:
            d_inter = 0.0
        else:
            cache_key = (node_from, node_to)
            if cache_key in self._path_cache:
                d_inter = self._path_cache[cache_key]
            else:
                try:
                    # Query shortest path with length weight
                    cutoff = 500.0  # Cutoff at 500 meters to keep lookup fast
                    d_inter = nx.dijkstra_path_length(
                        self.road_graph.graph,
                        source=node_from,
                        target=node_to,
                        weight="length_m",
                    )
                    if d_inter > cutoff:
                        d_inter = None
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    d_inter = None
                self._path_cache[cache_key] = d_inter

        if d_inter is None:
            return None

        # Distance from u_curr along current edge to c_curr
        d_curr_prog = max(0.0, c_curr.distance_along_edge_m)
        d_total = d_prev_rem + d_inter + d_curr_prog

        # Physically justified transition gate (evaluated when delta_t_s is known):
        # Kinematic travel distance + candidate projection offsets + discrete road geometry tolerance.
        # Rejects impossible multi-block leaps or disconnected U-turn loops while preserving
        # legitimate connected junction turns and highway transitions.
        if delta_t_s is not None and delta_t_s > 0.0:
            d_kinematic = self.max_speed_mps * delta_t_s
            proj_uncertainty = c_prev.distance_to_road_m + c_curr.distance_to_road_m
            geom_tol = 10.0  # Discretization and junction chord allowance

            if d_inter > d_kinematic + proj_uncertainty + geom_tol:
                return None
            if d_total > d_kinematic + proj_uncertainty + geom_tol:
                return None

        return d_total

    def compute_log_transition(
        self,
        c_prev: RoadCandidate,
        c_curr: RoadCandidate,
        prev_traj_enu: Tuple[float, float],
        curr_traj_enu: Tuple[float, float],
        delta_t_s: Optional[float] = None,
    ) -> float:
        """Compute transition log-probability ln p(c_curr | c_prev).

        Args:
            c_prev: Previous road candidate.
            c_curr: Current road candidate.
            prev_traj_enu: Previous trajectory point (east, north).
            curr_traj_enu: Current trajectory point (east, north).
            delta_t_s: Elapsed time in seconds between epochs.

        Returns:
            Log transition probability in nats, or LOG_ZERO if impossible/disconnected.
        """
        # Observed Euclidean trajectory displacement
        dx = float(curr_traj_enu[0]) - float(prev_traj_enu[0])
        dy = float(curr_traj_enu[1]) - float(prev_traj_enu[1])
        d_traj = math.hypot(dx, dy)

        # Route distance along road network
        d_graph = self.get_route_distance_m(c_prev, c_curr, delta_t_s=delta_t_s)
        if d_graph is None:
            return LOG_ZERO

        # Discrepancy metric
        delta_d = abs(d_graph - d_traj)

        # ln p = -ln(beta) - delta_d / beta
        log_p = -self.log_beta - (delta_d / self.beta_m)
        return log_p
