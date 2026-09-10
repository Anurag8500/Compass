"""Integration test: OutageDetector behaviour on real IO-VNBD data.

SOURCE LABEL: REAL IO-VNBD DATA — rows come directly from the synchronized
IO-VNBD CSV files. No artificial masking or row modification is applied.

Phase 0 finding (documented in docs/dataset_notes.md):
    The IO-VNBD synchronized archive contains NO pre-tagged GPS outage windows.
    has_real_outages = False for all trips in the manifest.
    There is no external outage index file to load.

What this test validates instead
---------------------------------
Without pre-tagged outage windows, this file cannot measure "did the FSM
correctly detect a known outage?" That test category belongs exclusively to
test_outage_synthetic.py (SYNTHETIC label).

This file validates three things that are only testable on real data:

  1. STABILITY: the detector produces a valid, non-crashing state for every
     row of a real IO-VNBD trip. No exceptions, no NaN trust scores, no
     invalid state values.

  2. SANITY: on a clean open-sky trip where GPS quality is continuously good,
     the detector spends the large majority of time in GNSS_AIDED (>= 95% of
     rows), confirming the trust score and FSM thresholds are calibrated
     correctly for real sensor noise rather than producing spurious outages.

  3. TRUST SCORE DISTRIBUTION: trust scores computed on real rows are
     well-distributed in (0, 1) — not pathologically collapsed to 0 or 1 —
     confirming the scoring formula is meaningful on real data.

Skip behaviour
--------------
All tests are skipped (not failed) when the raw IO-VNBD data directory is
absent. This keeps the test suite green in CI environments that do not have
the dataset mounted.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from navigation.gnss.outage_detection import OutageDetector
from navigation.gnss.fsm import GNSSState


# ---------------------------------------------------------------------------
# Dataset location
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent.parent
_RAW_DATA_ROOT = _REPO_ROOT / "data" / "raw" / "io_vnbd"

# Trip to use for sanity checks (short, clean open-sky trip from the manifest)
_SANITY_TRIP_RELATIVE = (
    "Synchronised V abd S datasets/Categorised IOVNB Dataset"
    "/S (Driver A)/S1/S-S1.csv"
)
_SANITY_TRIP_PATH = _RAW_DATA_ROOT / _SANITY_TRIP_RELATIVE

_DATA_AVAILABLE = _SANITY_TRIP_PATH.exists()
_SKIP_REASON = (
    "Real IO-VNBD data not available at expected path. "
    "Mount the dataset or run synthetic tests instead."
)


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------

def _load_rows(csv_path: Path, max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load rows from an IO-VNBD CSV file, stripping whitespace from keys."""
    rows = []
    with open(csv_path, encoding="latin-1") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break
            clean = {}
            for k, v in row.items():
                key = k.strip()
                # Coerce numeric columns used by trust_score
                if key in ("GPS ACCURACY", "GPS SATELLITES IN RANGE",
                           "GPS LATITUDE", "GPS LONGITUDE", "TIME SINCE START (ms)"):
                    try:
                        clean[key] = float(v.strip())
                    except (ValueError, AttributeError):
                        clean[key] = None
                else:
                    clean[key] = v.strip() if isinstance(v, str) else v
            rows.append(clean)
    return rows


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _DATA_AVAILABLE, reason=_SKIP_REASON)
class TestRealIoVnbdOutageDetection:
    """SOURCE: REAL IO-VNBD DATA — trip S1 (Driver A, open-sky, no tagged outages)."""

    # Use first 3000 rows (~5 min at 10 Hz) for speed
    MAX_ROWS = 3000

    def setup_method(self):
        self.rows = _load_rows(_SANITY_TRIP_PATH, max_rows=self.MAX_ROWS)
        assert len(self.rows) > 0, "No rows loaded from real data file"

        self.det = OutageDetector()
        self.states: List[GNSSState] = []
        self.trust_scores: List[float] = []

        for i, row in enumerate(self.rows):
            prev = self.rows[i - 1] if i > 0 else None
            t_ms = row.get("TIME SINCE START (ms)") or 0.0
            result = self.det.process_fix(row, prev, float(t_ms))
            self.states.append(result.state)
            self.trust_scores.append(result.trust_score)

    # --- Stability ---

    def test_no_exceptions_on_real_rows(self):
        """SOURCE: REAL — detector must not throw on any real row."""
        # setup_method already ran all rows; if we get here, no exception was raised
        assert len(self.states) == len(self.rows)

    def test_all_states_are_valid_enum_values(self):
        """SOURCE: REAL — every state must be a valid GNSSState member."""
        valid = set(GNSSState)
        for i, s in enumerate(self.states):
            assert s in valid, f"Invalid state {s!r} at row {i}"

    def test_all_trust_scores_in_range(self):
        """SOURCE: REAL — trust scores must be in [0, 1] for every row."""
        for i, score in enumerate(self.trust_scores):
            assert 0.0 <= score <= 1.0, \
                f"Trust score {score:.4f} out of [0,1] at row {i}"

    # --- Sanity (calibration check) ---

    def test_gnss_aided_dominates_clean_trip(self):
        """SOURCE: REAL — open-sky trip must spend >= 95% of time in GNSS_AIDED.

        If this fails, the trust score thresholds are mis-calibrated for real
        sensor noise and will produce spurious outages in production.
        """
        gnss_aided_count = sum(1 for s in self.states if s == GNSSState.GNSS_AIDED)
        fraction = gnss_aided_count / len(self.states)
        assert fraction >= 0.95, (
            f"Only {fraction*100:.1f}% of rows are GNSS_AIDED on a clean open-sky trip "
            f"(expected >= 95%). Trust score thresholds may be too strict for real data."
        )

    # --- Trust score distribution ---

    def test_trust_scores_not_all_zero(self):
        """SOURCE: REAL — trust scores should not collapse to zero on good data."""
        mean_trust = sum(self.trust_scores) / len(self.trust_scores)
        assert mean_trust > 0.3, \
            f"Mean trust score {mean_trust:.3f} is unexpectedly low on open-sky data"

    def test_trust_scores_have_variation(self):
        """SOURCE: REAL — real sensor noise should produce some score variation."""
        min_t = min(self.trust_scores)
        max_t = max(self.trust_scores)
        assert (max_t - min_t) > 0.05, \
            "Trust scores show no variation on real data — scoring may be degenerate"
