"""Road candidate definitions and abstract spatial search interface."""

from __future__ import annotations
import abc
from dataclasses import dataclass
from typing import List, Tuple

@dataclass(frozen=True)
class RoadCandidate:
    """A matched candidate point on a road edge."""
    edge_id: str
    projected_point_enu: Tuple[float, float, float]
    distance_to_road_m: float
    edge_azimuth_rad: float
    projected_lat_lon: Tuple[float, float]
    # Useful for tracing progress along the edge
    distance_along_edge_m: float = 0.0

class SpatialSearcher(abc.ABC):
    """Abstract interface for querying road candidates."""
    
    @abc.abstractmethod
    def search_candidates(self, pos_enu_2d: Tuple[float, float]) -> List[RoadCandidate]:
        """Find candidate road edges near the given 2D local ENU position."""
        pass
