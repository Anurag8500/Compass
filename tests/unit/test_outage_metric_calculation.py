"""Unit test verifying exact outage metric definitions and slicing (Phase 11 Step 8).

Verifies that:
1. Outage start index, outage duration, and outage end index are rigorously defined.
2. 'final outage drift' corresponds exactly to the final sample INSIDE the outage
   (o_last = o_start + outage_duration_steps - 1), NOT a post-reacquisition sample.
3. Max drift and drift percentage are analytically exact on known synthetic errors.
"""

import numpy as np
import pytest


def compute_outage_metrics(
    pos_err_2d: np.ndarray,
    ref_pos_enu: np.ndarray,
    outage_start_rel_steps: int,
    outage_duration_steps: int,
) -> dict:
    """Canonical outage metric calculator mirroring run_segment_simulation."""
    o_start = outage_start_rel_steps
    o_last = min(len(pos_err_2d) - 1, o_start + outage_duration_steps - 1)
    outage_errs = pos_err_2d[o_start : o_last + 1]

    if len(outage_errs) == 0:
        return {"max_drift_m": None, "final_drift_m": None, "drift_pct": None}

    max_drift = float(np.max(outage_errs))
    final_drift = float(pos_err_2d[o_last])
    sub_diffs = np.diff(ref_pos_enu[o_start : o_last + 1, :2], axis=0)
    sub_dist = float(np.sum(np.linalg.norm(sub_diffs, axis=1)))
    drift_pct = float(final_drift / max(1e-3, sub_dist) * 100.0)

    return {
        "o_start": o_start,
        "o_last": o_last,
        "outage_sample_count": len(outage_errs),
        "max_drift_m": max_drift,
        "final_drift_m": final_drift,
        "distance_traveled_m": sub_dist,
        "drift_pct": drift_pct,
    }


def test_analytical_outage_metrics_exact():
    """Verify on a known synthetic 10-step outage interval with linear drift."""
    # 30 steps total at 10 Hz (3 seconds)
    n_steps = 30
    # Outage starts at step 10 and lasts for 10 steps (steps 10 through 19 inclusive)
    o_start = 10
    o_dur = 10
    # Step 20 is the reacquisition sample!

    # Vehicle travels East at 10 m/s: 1.0 m per step
    ref_pos = np.zeros((n_steps, 3))
    ref_pos[:, 0] = np.arange(n_steps) * 1.0  # [0, 1, 2, ..., 29] meters

    # Error profile:
    # Steps 0-9: GNSS fix error ~ 0.5 m
    # Steps 10-19: Outage drifting linearly from 1.0 m to 10.0 m (at step 19, error = 10.0 m)
    # Step 20: GNSS reacquisition brings error immediately down to 0.5 m!
    pos_err = np.full(n_steps, 0.5)
    for k in range(o_dur):
        pos_err[o_start + k] = 1.0 + k * 1.0  # step 10: 1.0, step 11: 2.0, ..., step 19: 10.0
    # Step 20 has GNSS reacquired: pos_err[20] = 0.5

    metrics = compute_outage_metrics(pos_err, ref_pos, o_start, o_dur)

    # 1. Exact start and last sample inside outage
    assert metrics["o_start"] == 10
    assert metrics["o_last"] == 19
    assert metrics["outage_sample_count"] == 10

    # 2. Final outage drift MUST be 10.0 m (at step 19), NOT 0.5 m (at step 20 post-reacquisition)
    assert np.isclose(metrics["final_drift_m"], 10.0), f"Expected 10.0m, got {metrics['final_drift_m']}"

    # 3. Max outage drift MUST be 10.0 m
    assert np.isclose(metrics["max_drift_m"], 10.0)

    # 4. Distance traveled over 10 steps (steps 10 to 19 is 9 intervals of 1.0m = 9.0m)
    assert np.isclose(metrics["distance_traveled_m"], 9.0)

    # 5. Drift percentage
    expected_pct = (10.0 / 9.0) * 100.0
    assert np.isclose(metrics["drift_pct"], expected_pct)
