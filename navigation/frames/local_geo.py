"""Local East-North-Up (ENU) Cartesian frame and geodetic conversion (Phase 4).

Implements equirectangular local tangent-plane projections about a single,
fixed session-level reference origin (lat_ref, lon_ref, alt_ref).

Architectural Invariants:
1. Rigid Session Reference Origin:
   The origin is established at session start (e.g. first valid 3D GNSS fix)
   and remains strictly constant for the duration of the run.
   Resetting or shifting the origin mid-session is prohibited to avoid
   trajectory discontinuity and state corruption.
2. Local Cartesian Integration:
   Dead reckoning integration is performed strictly in meters in the local ENU frame:
       x = East [meters]
       y = North [meters]
       z = Up [meters]
   Integration is never performed directly in curvilinear geodetic coordinates (lat/lon).
3. Spherical Earth Radius:
   Nominal radius R_earth = 6,371,000 meters per project specification.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple, Union
import numpy as np

R_EARTH_METERS: float = 6_371_000.0


@dataclass(frozen=True)
class GeoReference:
    """Fixed session-level geodetic reference point for local ENU conversion.

    Attributes:
        lat_ref: Reference latitude in WGS84 degrees [-90.0, 90.0].
        lon_ref: Reference longitude in WGS84 degrees [-180.0, 180.0].
        alt_ref: Reference altitude in meters above mean sea level or ellipsoid.
    """
    lat_ref: float
    lon_ref: float
    alt_ref: float = 0.0

    def __post_init__(self) -> None:
        if not (-90.0 <= self.lat_ref <= 90.0):
            raise ValueError(f"lat_ref must be in [-90.0, 90.0], got {self.lat_ref}")
        if not (-180.0 <= self.lon_ref <= 180.0):
            raise ValueError(f"lon_ref must be in [-180.0, 180.0], got {self.lon_ref}")
        if not math.isfinite(self.alt_ref):
            raise ValueError(f"alt_ref must be finite, got {self.alt_ref}")

    @property
    def cos_lat_ref(self) -> float:
        """Precomputed cosine of reference latitude for longitude scaling."""
        return math.cos(math.radians(self.lat_ref))

    def geodetic_to_enu(
        self,
        lat: Union[float, np.ndarray],
        lon: Union[float, np.ndarray],
        alt: Union[float, np.ndarray],
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
        """Convert WGS84 geodetic coordinates (lat, lon, alt) to local ENU Cartesian meters.

        Equations:
            x_E = R_earth * (lon - lon_ref) * (pi / 180) * cos(lat_ref)
            y_N = R_earth * (lat - lat_ref) * (pi / 180)
            z_U = alt - alt_ref

        Returns:
            (east_m, north_m, up_m): Coordinates in local ENU frame [meters].
        """
        lat_arr = np.asarray(lat, dtype=np.float64)
        lon_arr = np.asarray(lon, dtype=np.float64)
        alt_arr = np.asarray(alt, dtype=np.float64)

        deg2rad = math.pi / 180.0
        d_lat_rad = (lat_arr - self.lat_ref) * deg2rad
        d_lon_rad = (lon_arr - self.lon_ref) * deg2rad

        east = R_EARTH_METERS * d_lon_rad * self.cos_lat_ref
        north = R_EARTH_METERS * d_lat_rad
        up = alt_arr - self.alt_ref

        if np.ndim(lat) == 0 and np.ndim(lon) == 0 and np.ndim(alt) == 0:
            return float(east), float(north), float(up)
        return east, north, up

    def enu_to_geodetic(
        self,
        east: Union[float, np.ndarray],
        north: Union[float, np.ndarray],
        up: Union[float, np.ndarray],
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
        """Convert local ENU Cartesian coordinates [meters] to WGS84 geodetic (lat, lon, alt).

        Equations:
            lat = lat_ref + (y_N / R_earth) * (180 / pi)
            lon = lon_ref + (x_E / (R_earth * cos(lat_ref))) * (180 / pi)
            alt = alt_ref + z_U

        Returns:
            (lat_deg, lon_deg, alt_m): WGS84 coordinates.
        """
        e_arr = np.asarray(east, dtype=np.float64)
        n_arr = np.asarray(north, dtype=np.float64)
        u_arr = np.asarray(up, dtype=np.float64)

        rad2deg = 180.0 / math.pi
        d_lat_deg = (n_arr / R_EARTH_METERS) * rad2deg
        d_lon_deg = (e_arr / (R_EARTH_METERS * self.cos_lat_ref)) * rad2deg

        lat_out = self.lat_ref + d_lat_deg
        lon_out = self.lon_ref + d_lon_deg
        alt_out = self.alt_ref + u_arr

        if np.ndim(east) == 0 and np.ndim(north) == 0 and np.ndim(up) == 0:
            return float(lat_out), float(lon_out), float(alt_out)
        return lat_out, lon_out, alt_out
