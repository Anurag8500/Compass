"""Road candidate search using segment-safe spatial indexing and exact projection (Phase 12).

Performs exact point-to-segment orthogonal projection over road network edges
within a configurable search radius. Adheres to strict architectural safety:
- Does NOT use segment midpoints alone (safe for long highway segments).
- Uses segment AABB spatial binning grid for fast, bounded, deterministic lookups.
- Projects into session-local ENU matching the ESKF filter state.
- ZERO ground-truth leakage: only estimated position and OSM graph are used.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from maps.extract_osm import RoadNetworkGraph, RoadSegment
from navigation.frames.local_geo import GeoReference


@dataclass(frozen=True)
class RoadCandidate:
    """A projected candidate road point along a network edge."""
    edge_id: str
    u: int
    v: int
    segment_idx: int
    projected_point_enu: Tuple[float, float]  # (east, north) [meters]
    projected_lat_lon: Tuple[float, float]  # (lat, lon) [degrees]
    distance_to_road_m: float  # Orthogonal perpendicular distance [meters]
    fraction_along_edge: float  # Normalized scalar [0.0, 1.0]
    distance_along_edge_m: float  # Metric distance from start node u [meters]
    edge_tangent_enu: Tuple[float, float]  # Unit tangent vector [tx, ty]
    edge_normal_enu: Tuple[float, float]  # Unit normal vector [-ty, tx]
    edge_azimuth_rad: float  # Road azimuth in [0, 2pi) radians
    edge_name: str
    highway: str
    edge_total_length_m: float


class CandidateSearch:
    """Segment-safe spatial index and candidate road search engine."""

    def __init__(
        self,
        road_graph: RoadNetworkGraph,
        search_radius_m: float = 35.0,
        grid_cell_size_m: float = 100.0,
    ) -> None:
        """Initialize candidate search with spatial binning over graph segments."""
        if search_radius_m <= 0.0:
            raise ValueError(f"search_radius_m must be positive, got {search_radius_m}")
        if road_graph.geo_ref is None:
            raise ValueError("RoadNetworkGraph must have an active GeoReference set before search.")

        self.road_graph = road_graph
        self.search_radius_m = search_radius_m
        self.grid_cell_size_m = grid_cell_size_m
        self._grid: Dict[Tuple[int, int], List[int]] = {}
        self._build_spatial_index()

    def _build_spatial_index(self) -> None:
        """Populate uniform grid spatial hash with road segment indices."""
        self._grid.clear()
        segments = self.road_graph.segments
        cell_size = self.grid_cell_size_m

        for seg_idx, seg in enumerate(segments):
            e_min = min(seg.p0_enu[0], seg.p1_enu[0]) - self.search_radius_m
            e_max = max(seg.p0_enu[0], seg.p1_enu[0]) + self.search_radius_m
            n_min = min(seg.p0_enu[1], seg.p1_enu[1]) - self.search_radius_m
            n_max = max(seg.p0_enu[1], seg.p1_enu[1]) + self.search_radius_m

            col_min = int(math.floor(e_min / cell_size))
            col_max = int(math.floor(e_max / cell_size))
            row_min = int(math.floor(n_min / cell_size))
            row_max = int(math.floor(n_max / cell_size))

            for col in range(col_min, col_max + 1):
                for row in range(row_min, row_max + 1):
                    key = (col, row)
                    if key not in self._grid:
                        self._grid[key] = []
                    self._grid[key].append(seg_idx)

    def search_candidates(
        self,
        query_pos_enu: Tuple[float, float],
        search_radius_m: Optional[float] = None,
        max_candidates: int = 10,
    ) -> List[RoadCandidate]:
        """Search for nearby road edge candidates for a query point in session ENU.

        Args:
            query_pos_enu: (east, north) coordinates in meters.
            search_radius_m: Optional override of search radius.
            max_candidates: Maximum number of candidates to return.

        Returns:
            Deterministically sorted list of RoadCandidate objects (sorted by distance, then edge_id).
        """
        radius = search_radius_m if search_radius_m is not None else self.search_radius_m
        qx, qy = float(query_pos_enu[0]), float(query_pos_enu[1])
        q_pt = np.array([qx, qy], dtype=np.float64)

        cell_size = self.grid_cell_size_m
        col = int(math.floor(qx / cell_size))
        row = int(math.floor(qy / cell_size))

        # Check surrounding cells if query is near cell boundary
        candidate_seg_indices: Set[int] = set()
        c_min = int(math.floor((qx - radius) / cell_size))
        c_max = int(math.floor((qx + radius) / cell_size))
        r_min = int(math.floor((qy - radius) / cell_size))
        r_max = int(math.floor((qy + radius) / cell_size))

        for c in range(c_min, c_max + 1):
            for r in range(r_min, r_max + 1):
                cell_segs = self._grid.get((c, r))
                if cell_segs:
                    candidate_seg_indices.update(cell_segs)

        if not candidate_seg_indices:
            return []

        segments = self.road_graph.segments
        geo_ref = self.road_graph.geo_ref
        assert geo_ref is not None

        # Best candidate per edge_id
        best_per_edge: Dict[str, RoadCandidate] = {}

        for seg_idx in candidate_seg_indices:
            seg = segments[seg_idx]
            p0 = seg.p0_enu
            p1 = seg.p1_enu
            v = p1 - p0
            v_sq = float(np.dot(v, v))
            if v_sq < 1e-12:
                continue

            w = q_pt - p0
            c1 = float(np.dot(w, v))
            t = max(0.0, min(1.0, c1 / v_sq))
            p_proj = p0 + t * v
            dist = float(np.linalg.norm(q_pt - p_proj))

            if dist > radius:
                continue

            edge_data = self.road_graph.graph.edges.get((seg.u, seg.v), {})
            total_len = float(edge_data.get("length_m", seg.length_m))
            if total_len <= 1e-3:
                total_len = max(seg.length_m, 1.0)

            dist_along = float(seg.cum_dist_start_m + t * seg.length_m)
            frac_along = max(0.0, min(1.0, dist_along / total_len))

            lat_proj, lon_proj, _ = geo_ref.enu_to_geodetic(p_proj[0], p_proj[1], 0.0)

            cand = RoadCandidate(
                edge_id=seg.edge_id,
                u=seg.u,
                v=seg.v,
                segment_idx=seg.segment_idx,
                projected_point_enu=(float(p_proj[0]), float(p_proj[1])),
                projected_lat_lon=(float(lat_proj), float(lon_proj)),
                distance_to_road_m=dist,
                fraction_along_edge=frac_along,
                distance_along_edge_m=dist_along,
                edge_tangent_enu=(float(seg.tangent[0]), float(seg.tangent[1])),
                edge_normal_enu=(float(seg.normal[0]), float(seg.normal[1])),
                edge_azimuth_rad=seg.azimuth_rad,
                edge_name=str(edge_data.get("name", "")),
                highway=str(edge_data.get("highway", "residential")),
                edge_total_length_m=total_len,
            )

            # Keep candidate with smallest distance for this edge
            existing = best_per_edge.get(seg.edge_id)
            if existing is None or cand.distance_to_road_m < existing.distance_to_road_m:
                best_per_edge[seg.edge_id] = cand

        # Deterministic sorting: distance ascending, then edge_id string
        sorted_candidates = sorted(
            best_per_edge.values(),
            key=lambda c: (c.distance_to_road_m, c.edge_id),
        )

        return sorted_candidates[:max_candidates]
