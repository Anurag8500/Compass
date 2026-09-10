"""Bounded-rate GNSS recovery blender for REACQUIRING state.

When the FSM transitions from DR_ONLY into REACQUIRING, the returning GNSS
fix may disagree with the ESKF's dead-reckoned position by a large amount —
accumulated drift over the outage duration. Feeding that raw disagreement
directly into the ESKF as a single update would cause an instant position
snap: a numerically large innovation that the Kalman gain applies in one
cycle, creating a visible discontinuity in the output trajectory.

This module implements the "bounded-rate blend" the Master Plan requires:
during REACQUIRING, the returning fix is passed through a rate limiter that
caps how far the blended position is allowed to differ from the ESKF's current
dead-reckoned position in a single cycle. The blended fix (not the raw fix)
is what gets handed to the ESKF update. Over consecutive cycles the blended
position converges toward the raw GNSS fix at a controlled rate.

Rate limiting is dt-aware: the maximum allowed step per cycle is:
    max_step = min(MAX_CORRECTION_PER_CYCLE_M,
                   MAX_CORRECTION_RATE_MPS * dt_s)
This means the cap scales correctly at different update frequencies rather
than being a fixed per-call constant.

Convergence is confirmed when:
  1. The horizontal distance between the blended position and the raw fix is
     below CONVERGENCE_POSITION_THRESHOLD_M, AND
  2. This condition has held for at least CONVERGENCE_MIN_FIXES consecutive
     accepted fixes.

Only after convergence is confirmed does the blender signal that the FSM
may return to GNSS_AIDED. This prevents a single lucky fix from falsely
triggering full recovery.

Plausibility gate
-----------------
A returning fix is rejected entirely (skipped, no blend step) if its
innovation against the current ESKF position exceeds
PLAUSIBILITY_REJECTION_THRESHOLD_M. This guards against GPS multipath
artifacts that produce a single wildly incorrect fix right at signal
recovery — if such a fix were blended, even at the capped rate, it would
steer the blended position in the wrong direction.

Usage
-----
    blender = RecoveryBlender()
    blended_fix, converged = blender.blend(
        raw_fix, eskf_position_local, ref_point, dt_s=0.1)
    if blended_fix is not None:
        # pass blended_fix to ESKF update
    if converged:
        # signal FSM to transition to GNSS_AIDED
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

from navigation.schemas.gnss import GNSSSample


# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

# Maximum position correction velocity during REACQUIRING (metres per second).
# The actual per-cycle step cap = min(MAX_CORRECTION_PER_CYCLE_M,
#                                     MAX_CORRECTION_RATE_MPS * dt_s).
# At 10 Hz (dt=0.1 s): cap = min(5.0, 2.0*0.1) = 0.2 m/cycle — tight.
# But we keep the absolute ceiling at 5 m so that very slow update rates
# (dt>2.5 s) do not allow runaway corrections.
MAX_CORRECTION_RATE_MPS:   float = 2.0   # metres/second — continuous rate limit
MAX_CORRECTION_PER_CYCLE_M: float = 5.0  # absolute per-cycle ceiling (metres)

# A fix is rejected entirely (plausibility gate) if its distance from the
# ESKF dead-reckoned position exceeds this threshold (meters).
# Set conservatively large — this only rejects wildly implausible fixes,
# not large-but-real drift corrections.
PLAUSIBILITY_REJECTION_THRESHOLD_M: float = 2000.0

# Blended position must be within this distance of the raw fix (meters)
# for a cycle to count toward convergence.
# Tightened from 10 m to 3 m — consistent with Anurag's 1.5 m horizontal
# tolerance scaled for lat/lon blending coordinate resolution.
CONVERGENCE_POSITION_THRESHOLD_M: float = 3.0

# Number of consecutive fixes meeting the convergence threshold required
# before signalling GNSS_AIDED recovery.
CONVERGENCE_MIN_FIXES: int = 3

# Earth radius for haversine approximation (meters)
_R_EARTH_M: float = 6_371_000.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two WGS84 points in metres."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2.0 * _R_EARTH_M * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def _enu_to_latlon(
    east_m: float, north_m: float,
    lat0: float, lon0: float,
) -> Tuple[float, float]:
    """Convert ENU offset (metres) to WGS84 lat/lon using equirectangular approximation."""
    dlat = north_m / _R_EARTH_M
    dlon = east_m / (_R_EARTH_M * math.cos(math.radians(lat0)))
    return lat0 + math.degrees(dlat), lon0 + math.degrees(dlon)


def _latlon_to_enu(
    lat: float, lon: float,
    lat0: float, lon0: float,
) -> Tuple[float, float]:
    """Convert WGS84 lat/lon to ENU offset (metres) using equirectangular approximation."""
    north_m = math.radians(lat - lat0) * _R_EARTH_M
    east_m  = math.radians(lon - lon0) * _R_EARTH_M * math.cos(math.radians(lat0))
    return east_m, north_m


# ---------------------------------------------------------------------------
# RecoveryBlender
# ---------------------------------------------------------------------------

@dataclass
class RecoveryBlender:
    """Bounded-rate GNSS fix blender for use during the REACQUIRING FSM state.

    Maintains an internal blended lat/lon that moves toward the raw GNSS fix
    at most MAX_CORRECTION_PER_CYCLE_M per call, and tracks how many
    consecutive fixes have been within CONVERGENCE_POSITION_THRESHOLD_M of
    the raw fix in order to confirm stable recovery.

    Reset by calling reset() when entering REACQUIRING from DR_ONLY, and
    again if the FSM re-enters DR_ONLY before convergence completes.

    Attributes:
        _blended_lat: Current blended latitude (degrees). None until first accepted fix.
        _blended_lon: Current blended longitude (degrees).
        _consecutive_converged: Count of consecutive fixes within convergence threshold.
    """

    _blended_lat: Optional[float] = field(default=None, repr=False)
    _blended_lon: Optional[float] = field(default=None, repr=False)
    _consecutive_converged: int = field(default=0, repr=False)

    def reset(self) -> None:
        """Reset blender state. Call when (re-)entering REACQUIRING."""
        self._blended_lat = None
        self._blended_lon = None
        self._consecutive_converged = 0

    def blend(
        self,
        raw_fix: GNSSSample,
        eskf_position_local: Tuple[float, float, float],
        reference_point: Tuple[float, float],
        dt_s: float = 0.1,
    ) -> Tuple[Optional[GNSSSample], bool]:
        """Apply one blending step toward the raw GPS fix.

        Args:
            raw_fix: The incoming GNSSSample directly from the GNSS source.
            eskf_position_local: ESKF dead-reckoned position [E, N, U] in metres
                relative to reference_point. Used only on the first call after
                reset to seed the blended position.
            reference_point: Session-level ENU origin (lat0, lon0) in degrees.
            dt_s: Time step since the previous blend call (seconds). Used to
                compute the dt-aware rate limit. Defaults to 0.1 s (10 Hz).
                Clamped to [0.01, 2.0] to guard against stale or zero dt.

        Returns:
            (blended_fix, converged) where:
              - blended_fix: A GNSSSample with lat/lon replaced by the blended
                position, or None if the fix was rejected by the plausibility gate.
              - converged: True if CONVERGENCE_MIN_FIXES consecutive fixes have
                all been within CONVERGENCE_POSITION_THRESHOLD_M of the raw fix,
                meaning the FSM may transition to GNSS_AIDED.
        """
        lat0, lon0 = reference_point

        # Seed blended position from ESKF dead-reckoned state on first call
        if self._blended_lat is None:
            east_m, north_m = eskf_position_local[0], eskf_position_local[1]
            self._blended_lat, self._blended_lon = _enu_to_latlon(east_m, north_m, lat0, lon0)

        # Plausibility gate: reject fix if it is implausibly far from blended position
        dist_to_fix = _haversine_m(
            self._blended_lat, self._blended_lon,
            raw_fix.lat, raw_fix.lon,
        )
        if dist_to_fix > PLAUSIBILITY_REJECTION_THRESHOLD_M:
            self._consecutive_converged = 0
            return None, False

        # dt-aware rate limit: how far are we allowed to move this cycle?
        eff_dt = max(0.01, min(2.0, float(dt_s)))
        max_step = min(MAX_CORRECTION_PER_CYCLE_M,
                       MAX_CORRECTION_RATE_MPS * eff_dt)

        # Move blended position toward raw fix, capped at max_step
        if dist_to_fix > 1e-6:
            scale = min(1.0, max_step / dist_to_fix)
        else:
            scale = 1.0

        # Interpolate in ENU space for numerical stability
        b_east, b_north = _latlon_to_enu(self._blended_lat, self._blended_lon, lat0, lon0)
        r_east, r_north = _latlon_to_enu(raw_fix.lat, raw_fix.lon, lat0, lon0)

        new_east  = b_east  + scale * (r_east  - b_east)
        new_north = b_north + scale * (r_north - b_north)

        self._blended_lat, self._blended_lon = _enu_to_latlon(new_east, new_north, lat0, lon0)

        # Convergence check: horizontal distance from blended to raw fix
        dist_after = _haversine_m(
            self._blended_lat, self._blended_lon,
            raw_fix.lat, raw_fix.lon,
        )
        if dist_after <= CONVERGENCE_POSITION_THRESHOLD_M:
            self._consecutive_converged += 1
        else:
            self._consecutive_converged = 0

        converged = self._consecutive_converged >= CONVERGENCE_MIN_FIXES

        # Build the blended fix — preserve all original fields except lat/lon
        blended_fix = GNSSSample(
            timestamp_ns=raw_fix.timestamp_ns,
            lat=self._blended_lat,
            lon=self._blended_lon,
            alt=raw_fix.alt,
            speed=raw_fix.speed,
            bearing=raw_fix.bearing,
            accuracy_m=raw_fix.accuracy_m,
            sat_count=raw_fix.sat_count,
            trust_score=raw_fix.trust_score,
        )

        return blended_fix, converged
