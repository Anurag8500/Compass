"""Integration test: OutageFSM anti-flapping behaviour.

Verifies that OutageFSM does NOT flap (rapidly oscillate between states) when
fed noisy trust scores that straddle LOW_TRUST_THRESHOLD.  The grace period
and convergence window are the designed defence against this; this test
confirms they hold under sustained borderline noise.

Sequence
--------
30 seconds of synthetic trust scores alternating around 0.3
(pattern: 0.25, 0.35, 0.28, 0.32, 0.27, 0.33, 0.29, 0.31) at 200ms intervals.

Pass criteria
-------------
1. No GNSS_AIDED->DR_ONLY transition fires in under GRACE_PERIOD_MS.
2. No REACQUIRING->GNSS_AIDED transition fires in under CONVERGENCE_MS.
3. Total transitions over the 30-second window <= 4 (any more indicates
   oscillation that slipped past the timing guards).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from navigation.gnss.fsm import (
    OutageFSM,
    GNSSState,
    GRACE_PERIOD_MS,
    CONVERGENCE_MS,
    LOW_TRUST_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

SAMPLE_INTERVAL_MS: float = 200.0     # 5 Hz
TOTAL_DURATION_MS:  float = 30_000.0  # 30 seconds
MAX_TRANSITIONS:    int   = 4         # More than this in 30s of borderline noise = flap

NOISY_PATTERN = [0.25, 0.35, 0.28, 0.32, 0.27, 0.33, 0.29, 0.31]


def _generate_sequence() -> List[Tuple[float, float]]:
    """Return (timestamp_ms, trust_score) pairs for the full noisy window."""
    seq = []
    t, i = 0.0, 0
    while t <= TOTAL_DURATION_MS:
        seq.append((t, NOISY_PATTERN[i % len(NOISY_PATTERN)]))
        t += SAMPLE_INTERVAL_MS
        i += 1
    return seq


def _run_fsm(sequence: List[Tuple[float, float]]) -> List[Tuple[float, GNSSState, GNSSState]]:
    """Feed sequence into a fresh FSM; return list of (t_ms, from_state, to_state)."""
    fsm = OutageFSM()
    transitions: List[Tuple[float, GNSSState, GNSSState]] = []
    prev = GNSSState.GNSS_AIDED
    for t_ms, trust in sequence:
        state = fsm.update(trust, t_ms)
        if state != prev:
            transitions.append((t_ms, prev, state))
            prev = state
    return transitions


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFSMAntiFlapping:

    def setup_method(self):
        self.sequence    = _generate_sequence()
        self.transitions = _run_fsm(self.sequence)

    def test_total_transitions_within_bound(self):
        """No more than MAX_TRANSITIONS state changes in 30s of borderline noise."""
        count = len(self.transitions)
        assert count <= MAX_TRANSITIONS, (
            f"FSM made {count} transitions in {TOTAL_DURATION_MS/1000:.0f}s of "
            f"borderline noise — expected <= {MAX_TRANSITIONS}. "
            "This indicates flapping: the timing guards are not suppressing oscillation."
        )

    def test_gnss_aided_to_dr_only_respects_grace_period(self):
        """GNSS_AIDED->DR_ONLY must never fire faster than GRACE_PERIOD_MS."""
        for i, (t_ms, from_s, to_s) in enumerate(self.transitions):
            if from_s == GNSSState.GNSS_AIDED and to_s == GNSSState.DR_ONLY:
                t_entered = self.transitions[i - 1][0] if i > 0 else 0.0
                held_ms   = t_ms - t_entered
                assert held_ms >= GRACE_PERIOD_MS, (
                    f"FLAP: GNSS_AIDED->DR_ONLY at t={t_ms:.0f}ms after only "
                    f"{held_ms:.0f}ms in GNSS_AIDED (grace period = {GRACE_PERIOD_MS:.0f}ms)."
                )

    def test_reacquiring_to_gnss_aided_respects_convergence(self):
        """REACQUIRING->GNSS_AIDED must never fire faster than CONVERGENCE_MS."""
        for i, (t_ms, from_s, to_s) in enumerate(self.transitions):
            if from_s == GNSSState.REACQUIRING and to_s == GNSSState.GNSS_AIDED:
                t_entered = self.transitions[i - 1][0] if i > 0 else 0.0
                held_ms   = t_ms - t_entered
                assert held_ms >= CONVERGENCE_MS, (
                    f"FLAP: REACQUIRING->GNSS_AIDED at t={t_ms:.0f}ms after only "
                    f"{held_ms:.0f}ms in REACQUIRING (convergence = {CONVERGENCE_MS:.0f}ms)."
                )

    def test_score_range_actually_straddles_threshold(self):
        """Sanity check: the noisy pattern genuinely crosses LOW_TRUST_THRESHOLD."""
        scores = [s for _, s in self.sequence]
        assert min(scores) < LOW_TRUST_THRESHOLD, (
            "Test sequence never goes below threshold — not a valid flapping test."
        )
        assert max(scores) > LOW_TRUST_THRESHOLD, (
            "Test sequence never goes above threshold — not a valid flapping test."
        )
