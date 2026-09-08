"""GNSS sensor data schemas for COMPASS.

Defines the canonical GNSS fix structure received from the GNSS manager / adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class GNSSSample:
    """Canonical GNSS fix sample.

    Attributes:
        timestamp_ns: Hardware/monotonic timestamp in nanoseconds, sourced from
            getElapsedRealtimeNanos() on Android or monotonic system clock on edge,
            ensuring synchronization against IMU clock.
        lat: WGS84 latitude in degrees [-90.0, 90.0].
        lon: WGS84 longitude in degrees [-180.0, 180.0].
        alt: Altitude in meters above mean sea level or WGS84 ellipsoid.
        speed: Ground speed in m/s (None if unavailable).
        bearing: Track angle / heading in degrees [0.0, 360.0) relative to true north (None if unavailable).
        accuracy_m: 1-sigma estimated horizontal position accuracy in meters (None if unavailable).
        sat_count: Number of satellites used in fix calculation (None if unavailable).
        trust_score: Continuous signal trust score in [0.0, 1.0], computed from
            HDOP/PDOP, SNR, innovation consistency, and carrier-to-noise ratio.
            NOTE: "Degraded GNSS" is a continuous trust condition, NOT an FSM state.
    """
    timestamp_ns: int
    lat: float
    lon: float
    alt: float
    speed: Optional[float] = None
    bearing: Optional[float] = None
    accuracy_m: Optional[float] = None
    sat_count: Optional[int] = None
    trust_score: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp_ns, int):
            raise TypeError(f"timestamp_ns must be an integer, got {type(self.timestamp_ns).__name__}")
        if self.timestamp_ns < 0:
            raise ValueError(f"timestamp_ns must be non-negative, got {self.timestamp_ns}")
        if not (-90.0 <= self.lat <= 90.0):
            raise ValueError(f"lat must be in [-90.0, 90.0], got {self.lat}")
        if not (-180.0 <= self.lon <= 180.0):
            raise ValueError(f"lon must be in [-180.0, 180.0], got {self.lon}")
        if not (0.0 <= self.trust_score <= 1.0):
            raise ValueError(f"trust_score must be in [0.0, 1.0], got {self.trust_score}")
        if self.speed is not None and self.speed < 0:
            raise ValueError(f"speed must be non-negative, got {self.speed}")
        if self.bearing is not None and not (0.0 <= self.bearing <= 360.0):
            raise ValueError(f"bearing must be in [0.0, 360.0], got {self.bearing}")
        if self.accuracy_m is not None and self.accuracy_m < 0:
            raise ValueError(f"accuracy_m must be non-negative, got {self.accuracy_m}")
        if self.sat_count is not None and self.sat_count < 0:
            raise ValueError(f"sat_count must be non-negative, got {self.sat_count}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize GNSS sample to JSON-compatible dictionary."""
        return {
            "timestamp_ns": self.timestamp_ns,
            "lat": self.lat,
            "lon": self.lon,
            "alt": self.alt,
            "speed": self.speed,
            "bearing": self.bearing,
            "accuracy_m": self.accuracy_m,
            "sat_count": self.sat_count,
            "trust_score": self.trust_score,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GNSSSample:
        """Deserialize GNSS sample from dictionary."""
        return cls(
            timestamp_ns=int(data["timestamp_ns"]),
            lat=float(data["lat"]),
            lon=float(data["lon"]),
            alt=float(data["alt"]),
            speed=float(data["speed"]) if data.get("speed") is not None else None,
            bearing=float(data["bearing"]) if data.get("bearing") is not None else None,
            accuracy_m=float(data["accuracy_m"]) if data.get("accuracy_m") is not None else None,
            sat_count=int(data["sat_count"]) if data.get("sat_count") is not None else None,
            trust_score=float(data.get("trust_score", 1.0)),
        )
