"""Unit tests for BiasNet exclusion rules plumbing (ml/data/exclusion_rules.py).

Verifies:
1. Valid windows with no real outages are marked ELIGIBLE.
2. Windows containing invalid sensor samples are excluded.
3. Windows overlapping real GNSS outage intervals are excluded with explanatory reason.
4. Non-overlapping outage intervals do not cause false exclusions.
"""

from __future__ import annotations

import pytest
from data.pipeline.outage_index import OutageWindow
from ml.data.exclusion_rules import evaluate_biasnet_eligibility


class TestExclusionRules:
    """Test suite for BiasNet window eligibility plumbing."""

    def test_nominal_valid_window_eligible(self) -> None:
        """A valid window with no outages present is eligible for BiasNet."""
        res = evaluate_biasnet_eligibility(
            window_start_timestamp_ns=1_000_000_000,
            window_end_timestamp_ns=3_000_000_000,
            is_window_valid=True,
            real_outage_windows=None,
        )
        assert res.eligible is True
        assert res.reason == "ELIGIBLE"

    def test_invalid_sensor_samples_excluded(self) -> None:
        """A window with corrupted or unvalidated sensor samples is excluded."""
        res = evaluate_biasnet_eligibility(
            window_start_timestamp_ns=1_000_000_000,
            window_end_timestamp_ns=3_000_000_000,
            is_window_valid=False,
        )
        assert res.eligible is False
        assert res.reason == "INVALID_SENSOR_SAMPLES"

    def test_real_outage_overlap_excluded(self) -> None:
        """A window overlapping an active real outage window must be excluded."""
        outage = OutageWindow(
            start_timestamp_ns=2_000_000_000,
            end_timestamp_ns=5_000_000_000,
            is_synthetic=False,
            label="TUNNEL_A",
        )

        # Window from 1.0s to 3.0s overlaps outage (2.0s to 5.0s)
        res = evaluate_biasnet_eligibility(
            window_start_timestamp_ns=1_000_000_000,
            window_end_timestamp_ns=3_000_000_000,
            is_window_valid=True,
            real_outage_windows=[outage],
        )

        assert res.eligible is False
        assert "REAL_GNSS_OUTAGE_OVERLAP_TUNNEL_A" in res.reason

    def test_non_overlapping_outage_remains_eligible(self) -> None:
        """An outage outside the window interval does not cause exclusion."""
        outage = OutageWindow(
            start_timestamp_ns=10_000_000_000,
            end_timestamp_ns=15_000_000_000,
            is_synthetic=False,
            label="TUNNEL_FAR",
        )

        # Window from 1.0s to 3.0s ends well before 10.0s
        res = evaluate_biasnet_eligibility(
            window_start_timestamp_ns=1_000_000_000,
            window_end_timestamp_ns=3_000_000_000,
            is_window_valid=True,
            real_outage_windows=[outage],
        )

        assert res.eligible is True
        assert res.reason == "ELIGIBLE"

    def test_outage_metadata_none_invents_no_fake_exclusions(self) -> None:
        """When outage metadata is None, no fake exclusions are invented; all valid windows are eligible."""
        for start_s in [0, 50, 100, 500]:
            res = evaluate_biasnet_eligibility(
                window_start_timestamp_ns=start_s * 1_000_000_000,
                window_end_timestamp_ns=(start_s + 2) * 1_000_000_000,
                is_window_valid=True,
                real_outage_windows=None,
            )
            assert res.eligible is True
            assert res.reason == "ELIGIBLE"

