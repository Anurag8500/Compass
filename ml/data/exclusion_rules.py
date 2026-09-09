"""Shared BiasNet Exclusion Rules Plumbing.

COMPASS Phase 6 — ML Dataset Construction.
Shared infrastructure determining window eligibility for downstream Phase 8
BiasNet residual optimization labeling.

CRITICAL ARCHITECTURAL CONSTRAINTS:
1. DO NOT generate BiasNet labels in Phase 6. That is strictly deferred to Phase 8.
2. Outage Rule: Windows overlapping a real GNSS blackout interval must be
   excluded from BiasNet labeling (since BiasNet ground truth requires GNSS reference).
3. Dataset Reality: The IO-VNBD archive contains NO pre-tagged real outage CSV.
   DO NOT INVENT fake outage intervals. Report explicitly that real outage metadata
   is absent, safely yielding 0 outage exclusions until real outage metadata exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence
from data.pipeline.outage_index import OutageWindow


@dataclass(frozen=True)
class BiasNetEligibility:
    """Eligibility status and explanatory code for BiasNet label generation."""
    eligible: bool
    reason: str


def evaluate_biasnet_eligibility(
    window_start_timestamp_ns: int,
    window_end_timestamp_ns: int,
    is_window_valid: bool,
    real_outage_windows: Optional[Sequence[OutageWindow]] = None,
) -> BiasNetEligibility:
    """Determine whether an extracted feature window is eligible for BiasNet labeling.

    Args:
        window_start_timestamp_ns: Earliest timestamp in window (ns).
        window_end_timestamp_ns: Latest timestamp in window (ns).
        is_window_valid: Whether sensor samples in window pass quality validation.
        real_outage_windows: Optional list of known real GNSS outage windows.

    Returns:
        BiasNetEligibility dataclass with boolean flag and audit reason.
    """
    # Rule 1: Window sensor validity
    if not is_window_valid:
        return BiasNetEligibility(
            eligible=False,
            reason="INVALID_SENSOR_SAMPLES",
        )

    # Rule 2: Overlap with real GNSS blackout periods
    if real_outage_windows:
        for out_win in real_outage_windows:
            # Check interval overlap: max(start1, start2) <= min(end1, end2)
            overlap_start = max(window_start_timestamp_ns, out_win.start_timestamp_ns)
            overlap_end = min(window_end_timestamp_ns, out_win.end_timestamp_ns)
            if overlap_start <= overlap_end:
                return BiasNetEligibility(
                    eligible=False,
                    reason=f"REAL_GNSS_OUTAGE_OVERLAP_{out_win.label or 'UNLABELED'}",
                )

    # Eligible for BiasNet label optimization
    return BiasNetEligibility(
        eligible=True,
        reason="ELIGIBLE",
    )
