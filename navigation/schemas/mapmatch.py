"""Map matching schemas for COMPASS.

Defines the output structure of the downstream HMM map matching engine.
Per the authoritative architecture, map matching is strictly downstream
of the ESKF/NHC/ZUPT navigation pipeline and NEVER feeds back into the
nominal filter state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple


@dataclass(frozen=True)
class MapMatchResult:
    """Downstream map matching result container.

    Attributes:
        snapped: True if the navigation estimate was successfully snapped to a road network segment.
        snapped_lat_lon: (latitude, longitude) in degrees WGS84 of the snapped road position,
            or None if no confident match was found.
        matched_road_id: OpenStreetMap way ID or road graph edge identifier (or None).
        confidence: Normalized match confidence score in [0.0, 1.0].
        heading_rad: Local road centerline azimuth in radians [0, 2pi) at the snap point (or None).
            NOTE: Intentional schema extension beyond the minimal trace struct. Provides downstream
            road orientation metadata for map-aligned heading visualization and innovation monitoring.
        distance_to_road_m: Orthogonal distance from estimated position to snapped centerline in meters (or None).
            NOTE: Intentional schema extension. Provides cross-track error measurement for map-matching
            quality auditing without feeding back into the ESKF filter state.
    """
    snapped: bool
    snapped_lat_lon: Optional[Tuple[float, float]] = None
    matched_road_id: Optional[str] = None
    confidence: float = 0.0
    heading_rad: Optional[float] = None
    distance_to_road_m: Optional[float] = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")
        if self.snapped_lat_lon is not None:
            if len(self.snapped_lat_lon) != 2:
                raise ValueError(
                    f"snapped_lat_lon must be a 2-tuple (lat, lon), got length {len(self.snapped_lat_lon)}"
                )
            lat, lon = self.snapped_lat_lon
            if not (-90.0 <= lat <= 90.0):
                raise ValueError(f"snapped latitude must be in [-90, 90], got {lat}")
            if not (-180.0 <= lon <= 180.0):
                raise ValueError(f"snapped longitude must be in [-180, 180], got {lon}")
        if self.distance_to_road_m is not None and self.distance_to_road_m < 0.0:
            raise ValueError(f"distance_to_road_m must be non-negative, got {self.distance_to_road_m}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize map match result to JSON-compatible dictionary."""
        return {
            "snapped": self.snapped,
            "snapped_lat_lon": list(self.snapped_lat_lon) if self.snapped_lat_lon is not None else None,
            "matched_road_id": self.matched_road_id,
            "confidence": self.confidence,
            "heading_rad": self.heading_rad,
            "distance_to_road_m": self.distance_to_road_m,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MapMatchResult:
        """Deserialize map match result from dictionary."""
        snapped_coord = data.get("snapped_lat_lon")
        return cls(
            snapped=bool(data["snapped"]),
            snapped_lat_lon=(float(snapped_coord[0]), float(snapped_coord[1])) if snapped_coord is not None else None,
            matched_road_id=str(data["matched_road_id"]) if data.get("matched_road_id") is not None else None,
            confidence=float(data.get("confidence", 0.0)),
            heading_rad=float(data["heading_rad"]) if data.get("heading_rad") is not None else None,
            distance_to_road_m=float(data["distance_to_road_m"]) if data.get("distance_to_road_m") is not None else None,
        )
