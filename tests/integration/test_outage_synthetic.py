"""Integration test: OutageDetector behaviour under synthetic GPS outages.

SOURCE LABEL: SYNTHETIC — all scenarios in this file use programmatically
constructed GNSS row sequences. No real IO-VNBD data is involved.

Validates:
  1. Detector stays GNSS_AIDED during clean signal.
  2. Detector transitions to DR_ONLY after sustained low-quality rows that
     exceed the FSM grace period — simulating the PS benchmark outage
     durations (SHORT ~60 s, LONG ~1 km @ 60 km/h).
  3. Detector transitions through REACQUIRING and back to GNSS_AIDED when
     signal returns with sustained high-quality fixes.
  4. Silent dropout path: tick_no_fix() drives DR_ONLY without any bad rows,
     matching the tunnel/underpass scenario.
  5. Recovery is rate-limited: the FSM does not re-enter GNSS_AIDED
     instantly; it must hold in REACQUIRING for at least CONVERGENCE_MS.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from navigation.gnss.outage_detection import OutageDetector, EXPECTED_FIX_INTERVAL_MS
from navigation.gnss.fsm import (
    GNSSState,
    GRACE_PERIOD_MS,
    CONVERGENCE_MS,
    LOW_TRUST_THRESHOLD,
    HIGH_TRUST_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Row factories
# ---------------------------------------------------------------------------

_T = Dict[str, Any]

def _good_row(t_ms: float, lat: float = 51.5074, lon: float = -0.1278) -> _T:
    """High-quality GPS fix: low accuracy error, many satellites."""
    return {
        "GPS ACCURACY": 4.0,
        "GPS SATELLITES IN RANGE": 12,
        "GPS LATITUDE": lat,
        "GPS LONGITUDE": lon,
        "TIME SINCE START (ms)": t_ms,
    }


def _bad_row(t_ms: float, lat: float = 51.5074, lon: float = -0.1278) -> _T:
    """Low-quality GPS fix: simulates outage (sat=0, accuracy=999)."""
    return {
        "GPS ACCURACY": 999.0,
        "GPS SATELLITES IN RANGE": 0,
        "GPS LATITUDE": lat,
        "GPS LONGITUDE": lon,
        "TIME SINCE START (ms)": t_ms,
    }


def _feed_rows(detector: OutageDetector, rows: List[_T]) -> List[GNSSState]:
    """Feed a list of rows into detector; return state after each row."""
    states = []
    for i, row in enumerate(rows):
        prev = rows[i - 1] if i > 0 else None
        t_ms = float(row["TIME SINCE START (ms)"])
        result = detector.process_fix(row, prev, t_ms)
        states.append(result.state)
    return states


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

INTERVAL_MS = 100.0  # 10 Hz


def _rows_for_duration(factory, start_ms: float, duration_ms: float) -> List[_T]:
    rows = []
    t = start_ms
    while t <= start_ms + duration_ms:
        rows.append(factory(t))
        t += INTERVAL_MS
    return rows


# ---------------------------------------------------------------------------
# Test: clean signal stays GNSS_AIDED
# ---------------------------------------------------------------------------

class TestCleanSignal:

    def test_stays_gnss_aided_for_10s(self):
        """SOURCE: SYNTHETIC — clean 10-second good-fix stream stays GNSS_AIDED."""
        det = OutageDetector()
        rows = _rows_for_duration(_good_row, 0.0, 10_000.0)
        states = _feed_rows(det, rows)
        assert all(s == GNSSState.GNSS_AIDED for s in states), \
            "Clean signal should never leave GNSS_AIDED"


# ---------------------------------------------------------------------------
# Test: SHORT outage (~60 s) — PS benchmark scenario 1
# ---------------------------------------------------------------------------

class TestShortSyntheticOutage:
    """SOURCE: SYNTHETIC — 60 s outage matching the PS benchmark duration."""

    OUTAGE_MS = 60_000.0   # 60 seconds

    def setup_method(self):
        self.det = OutageDetector()
        # 2s healthy preamble
        preamble = _rows_for_duration(_good_row, 0.0, 2_000.0)
        _feed_rows(self.det, preamble)

        # Inject 60s of bad rows
        self.outage_rows = _rows_for_duration(_bad_row, 2_000.0, self.OUTAGE_MS)
        _feed_rows(self.det, self.outage_rows)

    def test_enters_dr_only_during_outage(self):
        assert self.det.state == GNSSState.DR_ONLY, \
            "Should be DR_ONLY after 60s of bad fixes"

    def test_recovers_to_gnss_aided_after_signal_returns(self):
        # Feed good rows for CONVERGENCE_MS + margin
        recovery_rows = _rows_for_duration(
            _good_row,
            2_000.0 + self.OUTAGE_MS,
            CONVERGENCE_MS + 1_000.0,
        )
        _feed_rows(self.det, recovery_rows)
        assert self.det.state == GNSSState.GNSS_AIDED, \
            "Should recover to GNSS_AIDED after sustained good fixes"

    def test_passes_through_reacquiring(self):
        """Recovery must pass through REACQUIRING — no instant snap to GNSS_AIDED."""
        t_start = 2_000.0 + self.OUTAGE_MS
        # Feed only a handful of good fixes (< CONVERGENCE_MS)
        short_recovery = _rows_for_duration(_good_row, t_start, CONVERGENCE_MS * 0.3)
        _feed_rows(self.det, short_recovery)
        # Should still be REACQUIRING, not already GNSS_AIDED
        assert self.det.state == GNSSState.REACQUIRING, \
            "Should be REACQUIRING before convergence window completes"


# ---------------------------------------------------------------------------
# Test: LONG outage (~60 s at 60 km/h = ~1 km) — PS benchmark scenario 2
# ---------------------------------------------------------------------------

class TestLongSyntheticOutage:
    """SOURCE: SYNTHETIC — 60 s outage equivalent to 1 km @ 60 km/h."""

    # 1 km @ 60 km/h = 60 s (same duration as SHORT; benchmark is about distance, not time)
    OUTAGE_MS = 60_000.0

    def test_enters_dr_only_and_recovers(self):
        det = OutageDetector()
        preamble = _rows_for_duration(_good_row, 0.0, 2_000.0)
        _feed_rows(det, preamble)

        outage = _rows_for_duration(_bad_row, 2_000.0, self.OUTAGE_MS)
        _feed_rows(det, outage)
        assert det.state == GNSSState.DR_ONLY

        recovery = _rows_for_duration(_good_row, 2_000.0 + self.OUTAGE_MS, CONVERGENCE_MS + 500.0)
        _feed_rows(det, recovery)
        assert det.state == GNSSState.GNSS_AIDED


# ---------------------------------------------------------------------------
# Test: silent dropout via tick_no_fix
# ---------------------------------------------------------------------------

class TestSilentDropout:
    """SOURCE: SYNTHETIC — GPS rows simply stop arriving (tunnel scenario)."""

    def test_silent_dropout_enters_dr_only(self):
        det = OutageDetector()
        good_row = _good_row(0.0)
        det.process_fix(good_row, None, 0.0)

        # Advance IMU clock with no fixes — gap exceeds grace period
        t = EXPECTED_FIX_INTERVAL_MS + 100.0
        while t <= GRACE_PERIOD_MS + EXPECTED_FIX_INTERVAL_MS + 500.0:
            det.tick_no_fix(t)
            t += INTERVAL_MS

        assert det.state == GNSSState.DR_ONLY, \
            "Silent dropout should trigger DR_ONLY via tick_no_fix"

    def test_short_gap_does_not_trigger(self):
        det = OutageDetector()
        det.process_fix(_good_row(0.0), None, 0.0)

        # Gap shorter than EXPECTED_FIX_INTERVAL_MS — should not start grace timer
        det.tick_no_fix(EXPECTED_FIX_INTERVAL_MS * 0.5)
        assert det.state == GNSSState.GNSS_AIDED, \
            "Short tick gap should not affect state"


# ---------------------------------------------------------------------------
# Test: outage immediately at session start (no preamble)
# ---------------------------------------------------------------------------

class TestOutageFromStart:
    """SOURCE: SYNTHETIC — outage starts with the first row of the session."""

    def test_outage_from_row_zero(self):
        det = OutageDetector()
        rows = _rows_for_duration(_bad_row, 0.0, GRACE_PERIOD_MS + 500.0)
        _feed_rows(det, rows)
        assert det.state == GNSSState.DR_ONLY


# ---------------------------------------------------------------------------
# Test: multiple outage-recovery cycles
# ---------------------------------------------------------------------------

class TestMultipleCycles:
    """SOURCE: SYNTHETIC — two sequential outage/recovery cycles."""

    def test_two_outage_recovery_cycles(self):
        det = OutageDetector()
        t = 0.0
        CYCLE_PREAMBLE = 2_000.0
        CYCLE_OUTAGE   = GRACE_PERIOD_MS + 500.0
        CYCLE_RECOVERY = CONVERGENCE_MS + 500.0

        for cycle in range(2):
            # Healthy phase
            for row in _rows_for_duration(_good_row, t, CYCLE_PREAMBLE):
                det.process_fix(row, None, float(row["TIME SINCE START (ms)"]))
            t += CYCLE_PREAMBLE

            # Outage phase
            for row in _rows_for_duration(_bad_row, t, CYCLE_OUTAGE):
                det.process_fix(row, None, float(row["TIME SINCE START (ms)"]))
            t += CYCLE_OUTAGE
            assert det.state == GNSSState.DR_ONLY, f"Cycle {cycle+1}: expected DR_ONLY"

            # Recovery phase
            for row in _rows_for_duration(_good_row, t, CYCLE_RECOVERY):
                det.process_fix(row, None, float(row["TIME SINCE START (ms)"]))
            t += CYCLE_RECOVERY
            assert det.state == GNSSState.GNSS_AIDED, f"Cycle {cycle+1}: expected GNSS_AIDED after recovery"
