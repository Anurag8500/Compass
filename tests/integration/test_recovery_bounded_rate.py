"""Integration test: RecoveryBlender enforces bounded-rate position correction.

SOURCE LABEL: SYNTHETIC — all scenarios use programmatically constructed
GNSSSample sequences. No real IO-VNBD data is involved.

What this file proves
---------------------
The Master Plan explicitly prohibits "instant snap" — a single returning GPS
fix after a long outage must NOT move the blended position by an unbounded
amount in one cycle. This is the integration-level proof that the constraint
holds under a range of drift magnitudes and edge cases.

Assertions made
---------------
1. Single-cycle correction never exceeds MAX_CORRECTION_PER_CYCLE_M for any
   drift magnitude (10 m, 50 m, 200 m, 1000 m).
2. Convergence is confirmed only after CONVERGENCE_MIN_FIXES consecutive
   fixes are within CONVERGENCE_POSITION_THRESHOLD_M — never on the first fix.
3. An implausible fix (> PLAUSIBILITY_REJECTION_THRESHOLD_M) is rejected
   entirely — blended position does not move, None is returned.
4. After convergence the blender signals True and the blended position is
   within CONVERGENCE_POSITION_THRESHOLD_M of the raw fix.
5. reset() restarts the blend from the new ESKF seed position — previous
   convergence state is not carried over.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from navigation.gnss.recovery import (
    RecoveryBlender,
    MAX_CORRECTION_PER_CYCLE_M,
    MAX_CORRECTION_RATE_MPS,
    PLAUSIBILITY_REJECTION_THRESHOLD_M,
    CONVERGENCE_POSITION_THRESHOLD_M,
    CONVERGENCE_MIN_FIXES,
    _haversine_m,
    _enu_to_latlon,
    _latlon_to_enu,
)
from navigation.schemas.gnss import GNSSSample


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REF = (51.5074, -0.1278)   # London — session ENU origin
LAT0, LON0 = REF


def _fix(lat: float, lon: float) -> GNSSSample:
    return GNSSSample(
        timestamp_ns=1_000_000_000,
        lat=lat, lon=lon, alt=50.0, trust_score=0.9,
    )


def _fix_at_enu(east_m: float, north_m: float) -> GNSSSample:
    lat, lon = _enu_to_latlon(east_m, north_m, LAT0, LON0)
    return _fix(lat, lon)


def _eskf_at_enu(east_m: float, north_m: float) -> Tuple[float, float, float]:
    return (east_m, north_m, 0.0)


def _run_until_converged(
    blender: RecoveryBlender,
    raw_fix: GNSSSample,
    eskf_pos: Tuple[float, float, float],
    max_cycles: int = 500,
    dt_s: float = 0.1,
) -> Tuple[List[float], bool]:
    """Run blender cycles; return (per-cycle corrections in metres, converged)."""
    corrections: List[float] = []
    prev_lat: Optional[float] = None
    prev_lon: Optional[float] = None
    converged = False

    # Effective per-cycle cap at the given dt
    import math as _math
    effective_cap = min(MAX_CORRECTION_PER_CYCLE_M,
                        MAX_CORRECTION_RATE_MPS * max(0.01, min(2.0, dt_s)))

    for _ in range(max_cycles):
        blended, converged = blender.blend(raw_fix, eskf_pos, REF, dt_s=dt_s)
        if blended is None:
            corrections.append(0.0)
            if converged:
                break
            continue

        if prev_lat is not None:
            dist = _haversine_m(prev_lat, prev_lon, blended.lat, blended.lon)
            corrections.append(dist)
        else:
            seed_lat, seed_lon = _enu_to_latlon(eskf_pos[0], eskf_pos[1], LAT0, LON0)
            dist = _haversine_m(seed_lat, seed_lon, blended.lat, blended.lon)
            corrections.append(dist)

        prev_lat, prev_lon = blended.lat, blended.lon
        if converged:
            break

    return corrections, converged, effective_cap


# ---------------------------------------------------------------------------
# Test 1: rate limit holds across drift magnitudes
# ---------------------------------------------------------------------------

class TestRateLimitEnforced:
    """SOURCE: SYNTHETIC — single-cycle correction must never exceed MAX_CORRECTION_PER_CYCLE_M."""

    @pytest.mark.parametrize("drift_m", [10.0, 50.0, 200.0, 1000.0])
    def test_no_single_cycle_exceeds_limit(self, drift_m: float):
        blender = RecoveryBlender()
        raw_fix = _fix_at_enu(0.0, drift_m)
        eskf_pos = _eskf_at_enu(0.0, 0.0)

        corrections, _, effective_cap = _run_until_converged(blender, raw_fix, eskf_pos)

        violations = [c for c in corrections if c > effective_cap + 0.1]
        assert not violations, (
            f"drift={drift_m}m: {len(violations)} cycle(s) exceeded "
            f"effective cap={effective_cap:.3f}m. "
            f"Max seen: {max(corrections):.3f}m"
        )

    @pytest.mark.parametrize("drift_m", [10.0, 50.0, 200.0, 1000.0])
    def test_converges_eventually(self, drift_m: float):
        blender = RecoveryBlender()
        raw_fix = _fix_at_enu(0.0, drift_m)
        eskf_pos = _eskf_at_enu(0.0, 0.0)
        # At dt=0.1s, rate=2 m/s → 0.2 m/cycle. 1000m needs ~5000 cycles.
        needed = int(drift_m / (MAX_CORRECTION_RATE_MPS * 0.1)) + 20
        _, converged, _ = _run_until_converged(blender, raw_fix, eskf_pos, max_cycles=needed)
        assert converged, f"drift={drift_m}m: blender did not converge within {needed} cycles"


# ---------------------------------------------------------------------------
# Test 2: convergence requires CONVERGENCE_MIN_FIXES — no instant snap
# ---------------------------------------------------------------------------

class TestNoInstantSnap:
    """SOURCE: SYNTHETIC — convergence must take >= CONVERGENCE_MIN_FIXES cycles."""

    def test_not_converged_on_first_fix(self):
        blender = RecoveryBlender()
        raw_fix = _fix_at_enu(0.0, 3.0)
        eskf_pos = _eskf_at_enu(0.0, 0.0)
        _, converged = blender.blend(raw_fix, eskf_pos, REF, dt_s=0.1)
        assert not converged, "Must not converge on the very first fix"

    def test_converges_exactly_at_min_fixes(self):
        blender = RecoveryBlender()
        raw_fix = _fix_at_enu(0.0, 0.0)
        eskf_pos = _eskf_at_enu(0.0, 0.0)

        results = []
        for _ in range(CONVERGENCE_MIN_FIXES + 2):
            _, converged = blender.blend(raw_fix, eskf_pos, REF, dt_s=0.1)
            results.append(converged)

        assert not any(results[:CONVERGENCE_MIN_FIXES - 1]), \
            f"Converged before {CONVERGENCE_MIN_FIXES} consecutive fixes"
        assert any(results[CONVERGENCE_MIN_FIXES - 1:]), \
            f"Did not converge after {CONVERGENCE_MIN_FIXES} consecutive fixes"

    def test_large_drift_never_instant_snap(self):
        """200m drift: first blended position must be <= effective cap from seed."""
        blender = RecoveryBlender()
        raw_fix = _fix_at_enu(0.0, 200.0)
        eskf_pos = _eskf_at_enu(0.0, 0.0)

        blended, converged = blender.blend(raw_fix, eskf_pos, REF, dt_s=0.1)

        assert blended is not None
        assert not converged

        # Effective cap at 10 Hz = min(5.0, 2.0 * 0.1) = 0.2 m
        effective_cap = min(MAX_CORRECTION_PER_CYCLE_M, MAX_CORRECTION_RATE_MPS * 0.1)

        seed_lat, seed_lon = _enu_to_latlon(0.0, 0.0, LAT0, LON0)
        first_step = _haversine_m(seed_lat, seed_lon, blended.lat, blended.lon)
        assert first_step <= effective_cap + 0.05, (
            f"First cycle moved {first_step:.4f}m — exceeds effective cap "
            f"{effective_cap:.3f}m at 10 Hz. Instant snap detected."
        )


# ---------------------------------------------------------------------------
# Test 3: plausibility gate rejects implausible fixes
# ---------------------------------------------------------------------------

class TestPlausibilityGate:
    """SOURCE: SYNTHETIC — fixes beyond PLAUSIBILITY_REJECTION_THRESHOLD_M are rejected."""

    def test_implausible_fix_returns_none(self):
        blender = RecoveryBlender()
        eskf_pos = _eskf_at_enu(0.0, 0.0)

        origin_fix = _fix_at_enu(0.0, 0.0)
        blender.blend(origin_fix, eskf_pos, REF, dt_s=0.1)

        far_fix = _fix_at_enu(0.0, PLAUSIBILITY_REJECTION_THRESHOLD_M + 100.0)
        blended, converged = blender.blend(far_fix, eskf_pos, REF, dt_s=0.1)

        assert blended is None, "Implausible fix must be rejected (None returned)"
        assert not converged

    def test_blended_position_unchanged_after_rejection(self):
        blender = RecoveryBlender()
        eskf_pos = _eskf_at_enu(0.0, 0.0)

        good_fix = _fix_at_enu(0.0, 5.0)
        blended_before, _ = blender.blend(good_fix, eskf_pos, REF, dt_s=0.1)
        assert blended_before is not None
        lat_before = blender._blended_lat
        lon_before = blender._blended_lon

        far_fix = _fix_at_enu(0.0, PLAUSIBILITY_REJECTION_THRESHOLD_M + 500.0)
        blender.blend(far_fix, eskf_pos, REF, dt_s=0.1)

        assert blender._blended_lat == lat_before
        assert blender._blended_lon == lon_before


# ---------------------------------------------------------------------------
# Test 4: convergence quality — blended position is close to raw fix
# ---------------------------------------------------------------------------

class TestConvergenceQuality:
    """SOURCE: SYNTHETIC — after convergence, blended position is within threshold of raw fix."""

    @pytest.mark.parametrize("drift_m", [20.0, 100.0, 300.0])
    def test_blended_within_threshold_at_convergence(self, drift_m: float):
        blender = RecoveryBlender()
        raw_fix = _fix_at_enu(0.0, drift_m)
        eskf_pos = _eskf_at_enu(0.0, 0.0)

        blended = None
        for _ in range(5000):
            blended, converged = blender.blend(raw_fix, eskf_pos, REF, dt_s=0.1)
            if converged:
                break

        assert blended is not None
        dist = _haversine_m(blended.lat, blended.lon, raw_fix.lat, raw_fix.lon)
        assert dist <= CONVERGENCE_POSITION_THRESHOLD_M, (
            f"drift={drift_m}m: at convergence, blended is {dist:.2f}m from raw fix "
            f"(threshold={CONVERGENCE_POSITION_THRESHOLD_M}m)"
        )


# ---------------------------------------------------------------------------
# Test 5: reset restarts blend from new ESKF seed
# ---------------------------------------------------------------------------

class TestReset:
    """SOURCE: SYNTHETIC — reset() must clear convergence state and re-seed from ESKF."""

    def test_reset_clears_convergence(self):
        blender = RecoveryBlender()
        raw_fix = _fix_at_enu(0.0, 0.0)
        eskf_pos = _eskf_at_enu(0.0, 0.0)

        # Run to convergence
        for _ in range(CONVERGENCE_MIN_FIXES + 2):
            blender.blend(raw_fix, eskf_pos, REF)

        assert blender._consecutive_converged >= CONVERGENCE_MIN_FIXES

        # Reset and verify state is cleared
        blender.reset()
        assert blender._blended_lat is None
        assert blender._consecutive_converged == 0

    def test_reset_seeds_from_new_eskf_position(self):
        blender = RecoveryBlender()
        raw_fix = _fix_at_enu(0.0, 50.0)

        blender.blend(raw_fix, _eskf_at_enu(0.0, 0.0), REF, dt_s=0.1)
        lat_session1 = blender._blended_lat

        blender.reset()
        blender.blend(raw_fix, _eskf_at_enu(0.0, 100.0), REF, dt_s=0.1)
        lat_session2 = blender._blended_lat

        assert lat_session2 != lat_session1, \
            "After reset, blender should seed from new ESKF position, not previous session's"
