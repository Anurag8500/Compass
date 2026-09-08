"""Unit tests for fixed session-level local geodetic to ENU frame conversion (Phase 4)."""

from __future__ import annotations

import math
import numpy as np
import pytest

from navigation.frames.local_geo import GeoReference, R_EARTH_METERS


class TestLocalGeoFrameConversion:
    """Test suite for GeoReference local Cartesian ENU projection and round-trip fidelity."""

    def test_origin_identity(self) -> None:
        """Reference origin itself must project to [0, 0, 0] ENU."""
        ref = GeoReference(lat_ref=12.9716, lon_ref=77.5946, alt_ref=920.0)
        e, n, u = ref.geodetic_to_enu(12.9716, 77.5946, 920.0)

        assert e == pytest.approx(0.0, abs=1e-6)
        assert n == pytest.approx(0.0, abs=1e-6)
        assert u == pytest.approx(0.0, abs=1e-6)

        lat, lon, alt = ref.enu_to_geodetic(0.0, 0.0, 0.0)
        assert lat == pytest.approx(12.9716, abs=1e-9)
        assert lon == pytest.approx(77.5946, abs=1e-9)
        assert alt == pytest.approx(920.0, abs=1e-6)

    def test_cardinal_offsets(self) -> None:
        """Verify known North and East offsets match spherical equations."""
        lat0, lon0 = 20.0, 78.0
        ref = GeoReference(lat_ref=lat0, lon_ref=lon0, alt_ref=100.0)

        # 1. Exact 1000m North
        target_north_m = 1000.0
        d_lat_deg = (target_north_m / R_EARTH_METERS) * (180.0 / math.pi)
        lat_north = lat0 + d_lat_deg
        e, n, u = ref.geodetic_to_enu(lat_north, lon0, 100.0)

        assert e == pytest.approx(0.0, abs=1e-6)
        assert n == pytest.approx(target_north_m, abs=1e-6)
        assert u == pytest.approx(0.0, abs=1e-6)

        # 2. Exact 1000m East
        target_east_m = 1000.0
        d_lon_deg = (target_east_m / (R_EARTH_METERS * math.cos(math.radians(lat0)))) * (180.0 / math.pi)
        lon_east = lon0 + d_lon_deg
        e, n, u = ref.geodetic_to_enu(lat0, lon_east, 100.0)

        assert e == pytest.approx(target_east_m, abs=1e-6)
        assert n == pytest.approx(0.0, abs=1e-6)
        assert u == pytest.approx(0.0, abs=1e-6)

        # 3. Exact 50m Up
        e, n, u = ref.geodetic_to_enu(lat0, lon0, 150.0)
        assert e == pytest.approx(0.0, abs=1e-6)
        assert n == pytest.approx(0.0, abs=1e-6)
        assert u == pytest.approx(50.0, abs=1e-6)

    def test_enu_round_trip_accuracy(self) -> None:
        """Verify ENU -> Geodetic -> ENU preserves coordinates to sub-millimeter accuracy."""
        ref = GeoReference(lat_ref=28.6139, lon_ref=77.2090, alt_ref=216.0)

        test_offsets = [
            (0.0, 0.0, 0.0),
            (100.0, -50.0, 10.0),
            (2500.0, 1800.0, -25.0),
            (-5000.0, -3200.0, 100.0),
            (15000.0, -12000.0, 0.0),  # 15 km baseline
        ]

        for e_orig, n_orig, u_orig in test_offsets:
            lat, lon, alt = ref.enu_to_geodetic(e_orig, n_orig, u_orig)
            e_rt, n_rt, u_rt = ref.geodetic_to_enu(lat, lon, alt)

            assert e_rt == pytest.approx(e_orig, abs=1e-5), f"East error at {e_orig}"
            assert n_rt == pytest.approx(n_orig, abs=1e-5), f"North error at {n_orig}"
            assert u_rt == pytest.approx(u_orig, abs=1e-5), f"Up error at {u_orig}"

    def test_geodetic_round_trip_accuracy(self) -> None:
        """Verify Geodetic -> ENU -> Geodetic preserves coordinates to micro-degree precision."""
        ref = GeoReference(lat_ref=13.0827, lon_ref=80.2707, alt_ref=10.0)

        test_points = [
            (13.0827, 80.2707, 10.0),
            (13.0850, 80.2750, 15.5),
            (13.0700, 80.2600, 8.2),
            (13.1200, 80.3100, 22.0),
        ]

        for lat_orig, lon_orig, alt_orig in test_points:
            e, n, u = ref.geodetic_to_enu(lat_orig, lon_orig, alt_orig)
            lat_rt, lon_rt, alt_rt = ref.enu_to_geodetic(e, n, u)

            assert lat_rt == pytest.approx(lat_orig, abs=1e-9)
            assert lon_rt == pytest.approx(lon_orig, abs=1e-9)
            assert alt_rt == pytest.approx(alt_orig, abs=1e-5)

    def test_vectorized_operations(self) -> None:
        """Verify vectorized NumPy operations match scalar outputs."""
        ref = GeoReference(lat_ref=19.0760, lon_ref=72.8777, alt_ref=14.0)

        lats = np.array([19.0760, 19.0800, 19.0700, 19.0900])
        lons = np.array([72.8777, 72.8800, 72.8700, 72.8900])
        alts = np.array([14.0, 20.0, 10.0, 25.0])

        e_vec, n_vec, u_vec = ref.geodetic_to_enu(lats, lons, alts)
        assert len(e_vec) == 4
        assert len(n_vec) == 4
        assert len(u_vec) == 4

        for i in range(4):
            e_s, n_s, u_s = ref.geodetic_to_enu(lats[i], lons[i], alts[i])
            assert e_vec[i] == pytest.approx(e_s, abs=1e-8)
            assert n_vec[i] == pytest.approx(n_s, abs=1e-8)
            assert u_vec[i] == pytest.approx(u_s, abs=1e-8)

        lat_rt, lon_rt, alt_rt = ref.enu_to_geodetic(e_vec, n_vec, u_vec)
        assert np.allclose(lat_rt, lats, atol=1e-9)
        assert np.allclose(lon_rt, lons, atol=1e-9)
        assert np.allclose(alt_rt, alts, atol=1e-5)

    def test_invalid_reference_origin_raises(self) -> None:
        """Out-of-range coordinates must be rejected."""
        with pytest.raises(ValueError, match="lat_ref"):
            GeoReference(lat_ref=95.0, lon_ref=0.0)

        with pytest.raises(ValueError, match="lat_ref"):
            GeoReference(lat_ref=-91.0, lon_ref=0.0)

        with pytest.raises(ValueError, match="lon_ref"):
            GeoReference(lat_ref=0.0, lon_ref=185.0)

        with pytest.raises(ValueError, match="lon_ref"):
            GeoReference(lat_ref=0.0, lon_ref=-181.0)

        with pytest.raises(ValueError, match="alt_ref"):
            GeoReference(lat_ref=0.0, lon_ref=0.0, alt_ref=float("nan"))
