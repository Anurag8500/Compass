"""Continuous GNSS trust score computation for COMPASS navigation.

Computes a composite quality score in [0, 1] per GPS fix from up to four
evidence sources, in order of reliability:

  1. Reported horizontal accuracy (always available if fix is valid)
  2. Satellite count (available on most receivers)
  3. Fix-to-fix kinematic plausibility (available after first fix)
  4. ESKF innovation consistency — NIS from the Kalman filter's own residual
     (available after the first ESKF GNSS update)

The innovation component is the most important addition over a naive
accuracy+satellite score: if the Kalman filter is already disagreeing with
GPS fixes (high NIS), trust falls *before* the FSM grace period expires.
This catches the scenario where GPS is physically present but wrong (urban
canyon multipath, bridge reflection) rather than only the scenario where
GPS goes silent.

Weights are redistributed dynamically when evidence is missing: if satellite
count is unavailable the other three sources share that weight proportionally.
If NIS has never been recorded the innovation weight is given to plausibility.

Design principle (Master Plan Section 17)
-----------------------------------------
"Degraded GNSS" is NOT a discrete FSM state. It lives inside GNSS_AIDED as a
falling continuous trust score that inflates R continuously. Only when trust
stays below LOW_TRUST_THRESHOLD for the full grace period does the FSM
transition to DR_ONLY.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Weight constants — must sum to 1.0 when all evidence is present
# ---------------------------------------------------------------------------

ACCURACY_WEIGHT:     float = 0.35   # reported horizontal accuracy
SAT_COUNT_WEIGHT:    float = 0.15   # satellite count
PLAUSIBILITY_WEIGHT: float = 0.25   # fix-to-fix kinematic plausibility
INNOVATION_WEIGHT:   float = 0.25   # ESKF NIS consistency

# ---------------------------------------------------------------------------
# Accuracy thresholds
# ---------------------------------------------------------------------------
ACCURACY_EXCELLENT_M: float = 2.0   # accuracy <= this → score 1.0
ACCURACY_POOR_M:      float = 25.0  # accuracy >= this → score 0.0

# ---------------------------------------------------------------------------
# Satellite count thresholds
# ---------------------------------------------------------------------------
SAT_MIN: int = 4    # <= this → score 0.0
SAT_MAX: int = 10   # >= this → score 1.0

# ---------------------------------------------------------------------------
# Kinematic plausibility
# ---------------------------------------------------------------------------
MAX_PLAUSIBLE_SPEED_MPS:  float = 50.0   # ~180 km/h  → score 1.0
SPEED_EXCESS_RANGE_MPS:   float = 30.0   # 50→80 m/s range over which score decays to 0

# ---------------------------------------------------------------------------
# Innovation / NIS thresholds  (chi2_3(0.95) = 7.815 for 3-DOF position)
# ---------------------------------------------------------------------------
NIS_THRESHOLD_95: float = 7.815   # NIS <= this → score 1.0; exponential decay above

# ---------------------------------------------------------------------------
# Safety floor — prevents R inflation from going to infinity
# ---------------------------------------------------------------------------
MIN_TRUST_SCORE: float = 0.05

# ---------------------------------------------------------------------------
# Earth radius for haversine
# ---------------------------------------------------------------------------
_R_EARTH_M: float = 6_371_000.0


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two WGS84 points in metres."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi  = math.radians(lat2 - lat1)
    dlam  = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
    return 2.0 * _R_EARTH_M * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def _accuracy_component(accuracy: Optional[float]) -> float:
    """Score from reported horizontal accuracy in metres."""
    if accuracy is None or not math.isfinite(accuracy) or accuracy <= 0.0:
        # Missing accuracy is a mild negative signal, not a neutral one —
        # a receiver that can't report accuracy is probably not healthy.
        return 0.3

    if accuracy <= ACCURACY_EXCELLENT_M:
        return 1.0
    if accuracy >= ACCURACY_POOR_M:
        return 0.0

    # Linear decay from 1.0 at EXCELLENT to 0.0 at POOR
    return 1.0 - (accuracy - ACCURACY_EXCELLENT_M) / (ACCURACY_POOR_M - ACCURACY_EXCELLENT_M)


def _satellite_component(sat_count: Optional[int]) -> Optional[float]:
    """Score from satellite count. Returns None when unavailable (weight redistributed)."""
    if sat_count is None or sat_count < 0:
        return None  # field absent — handled by weight redistribution

    if sat_count >= SAT_MAX:
        return 1.0
    if sat_count <= SAT_MIN:
        return 0.0

    return (sat_count - SAT_MIN) / (SAT_MAX - SAT_MIN)


def _plausibility_component(
    current_lat: Optional[float], current_lon: Optional[float],
    current_time_ms: Optional[float],
    prev_lat: Optional[float], prev_lon: Optional[float],
    prev_time_ms: Optional[float],
) -> Optional[float]:
    """Score from fix-to-fix implied speed plausibility.

    Returns None on first fix (no previous position available), which causes
    the weight to be redistributed rather than inflating the score artificially.
    """
    if any(v is None for v in (current_lat, current_lon, current_time_ms,
                               prev_lat, prev_lon, prev_time_ms)):
        return None  # no prior fix — no plausibility evidence; redistribute weight

    dt_s = (current_time_ms - prev_time_ms) / 1000.0  # type: ignore[operator]
    if dt_s <= 0.01:
        return 0.5  # duplicate or backward timestamp — mild penalty

    dist_m = _haversine_m(prev_lat, prev_lon, current_lat, current_lon)  # type: ignore
    speed_mps = dist_m / dt_s

    if speed_mps <= MAX_PLAUSIBLE_SPEED_MPS:
        # Speed is plausible — but zero displacement (frozen position) is a mild
        # negative signal: a healthy moving vehicle should be producing *some* movement.
        # Return slightly less than 1.0 for exact zero to avoid compensating for
        # bad accuracy. Only applies when consecutive lat/lon are identical.
        if dist_m < 0.01:
            return 0.5  # frozen coordinates — mild penalty, not full trust
        return 1.0

    excess = speed_mps - MAX_PLAUSIBLE_SPEED_MPS
    if excess >= SPEED_EXCESS_RANGE_MPS:
        return 0.0

    return 1.0 - (excess / SPEED_EXCESS_RANGE_MPS)


def _innovation_component(nis: Optional[float]) -> float:
    """Score from ESKF Normalized Innovation Squared (3-DOF position gate).

    Returns 1.0 when NIS is absent (no prior filter update yet), decays
    exponentially past the 95% chi-square threshold.
    """
    if nis is None or not math.isfinite(nis) or nis < 0.0:
        return 1.0  # no evidence → neutral

    if nis <= NIS_THRESHOLD_95:
        return 1.0

    # Exponential decay past the 95% gate
    excess = nis - NIS_THRESHOLD_95
    return float(max(0.0, math.exp(-0.5 * (excess / NIS_THRESHOLD_95))))


def compute_trust_score(
    row: Dict[str, Any],
    previous_row: Optional[Dict[str, Any]],
    nis: Optional[float] = None,
) -> float:
    """Compute a continuous trust score in [0, 1] for one GPS fix.

    Args:
        row:          Current GPS data row. Expected keys:
                        'GPS ACCURACY'           (float, metres)
                        'GPS SATELLITES IN RANGE' (int)
                        'GPS LATITUDE'           (float, degrees)
                        'GPS LONGITUDE'          (float, degrees)
                        'TIME SINCE START (ms)'  (float)
        previous_row: Immediately preceding GPS row, or None on first fix.
        nis:          Most recent ESKF position NIS (Normalized Innovation
                      Squared) from the Kalman filter update. Pass None if
                      no ESKF update has been performed yet. When provided,
                      this catches scenarios where GPS is physically present
                      but the filter is consistently rejecting the fixes —
                      e.g. multipath, bridge reflections.

    Returns:
        Float in [MIN_TRUST_SCORE, 1.0].
    """
    # --- extract fields ---------------------------------------------------
    accuracy  = row.get('GPS ACCURACY')
    sat_count = row.get('GPS SATELLITES IN RANGE')
    cur_lat   = row.get('GPS LATITUDE')
    cur_lon   = row.get('GPS LONGITUDE')
    cur_t_ms  = row.get('TIME SINCE START (ms)')

    prev_lat  = previous_row.get('GPS LATITUDE')   if previous_row else None
    prev_lon  = previous_row.get('GPS LONGITUDE')  if previous_row else None
    prev_t_ms = previous_row.get('TIME SINCE START (ms)') if previous_row else None

    # --- component scores -------------------------------------------------
    acc_score   = _accuracy_component(accuracy)
    sat_score   = _satellite_component(sat_count)   # may be None
    plaus_score = _plausibility_component(cur_lat, cur_lon, cur_t_ms,
                                          prev_lat, prev_lon, prev_t_ms)  # may be None
    innov_score = _innovation_component(nis)

    # --- dynamic weight redistribution ------------------------------------
    # Absent evidence sources (None) have their weight redistributed
    # proportionally to the remaining sources.
    w_acc   = ACCURACY_WEIGHT
    w_sat   = SAT_COUNT_WEIGHT   if sat_score   is not None else 0.0
    w_plaus = PLAUSIBILITY_WEIGHT if plaus_score is not None else 0.0
    w_innov = INNOVATION_WEIGHT  if nis is not None else 0.0

    total_w = w_acc + w_sat + w_plaus + w_innov
    if total_w <= 0.0:
        return float(max(MIN_TRUST_SCORE, acc_score))  # accuracy only fallback

    w_acc   /= total_w
    w_sat   /= total_w
    w_plaus /= total_w
    w_innov /= total_w

    effective_sat   = sat_score   if sat_score   is not None else 0.0
    effective_plaus = plaus_score if plaus_score is not None else 0.0

    composite = (
        w_acc   * acc_score
        + w_sat   * effective_sat
        + w_plaus * effective_plaus
        + w_innov * innov_score
    )

    return float(max(MIN_TRUST_SCORE, min(1.0, composite)))
