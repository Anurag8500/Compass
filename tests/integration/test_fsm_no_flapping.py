"""Integration Test: Anti-Flapping and Hysteresis Under Borderline GNSS (Phase 10).

Validates:
1. Borderline / oscillating GNSS quality does NOT cause mode flapping.
2. Degraded GNSS stays strictly inside GNSS_AIDED with continuously scaled covariance R.
3. Isolated single-fix dropouts within the grace period (1.0s - 3.0s) do NOT trigger DR_ONLY.
4. Minimum dwell time (2.0s) prevents rapid state thrashing under adversarial input.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from navigation.core import NavigationCore, NavigationCoreConfig
from navigation.schemas.state import GNSSMode


def test_borderline_trust_no_flapping() -> None:
    """Inject borderline, noisy, oscillating accuracy and verify mode remains GNSS_AIDED throughout."""
    core = NavigationCore(NavigationCoreConfig(
        velocitynet_enabled=False,
        biasnet_enabled=False,
        zupt_enabled=False,
        gnss_enabled=True,
    ))

    lat0, lon0, alt0 = 52.0, -1.5, 100.0
    core.initialize(
        lat0=lat0,
        lon0=lon0,
        alt0=alt0,
        p0_enu=np.zeros(3),
        v0_enu=np.array([5.0, 0.0, 0.0]),
        q0=np.array([1.0, 0.0, 0.0, 0.0]),
        timestamp_ns=0,
    )

    # 30 seconds of driving with oscillating accuracy (from 3m to 25m back and forth)
    # This causes trust score to fluctuate between ~0.9 and ~0.25
    dt_imu = 0.1
    modes_seen = []
    r_scales = []

    for step in range(300):
        t_ns = int((step + 1) * dt_imu * 1e9)
        t_s = (step + 1) * dt_imu
        core.step_imu(np.array([0.0, 0.0, 9.81]), np.zeros(3), dt_s=dt_imu, timestamp_ns=t_ns)

        # 1 Hz GNSS fixes with oscillating accuracy
        if (step + 1) % 10 == 0:
            fix_idx = (step + 1) // 10
            # Accuracy oscillates between 2m and 30m every few seconds
            acc_h = 2.0 + 28.0 * (0.5 + 0.5 * math.sin(fix_idx * 0.8))
            p_east = 5.0 * t_s
            d_lon = p_east / (6371000.0 * math.cos(math.radians(lat0))) * (180.0 / math.pi)

            core.step_gnss_fix(
                lat=lat0,
                lon=lon0 + d_lon,
                alt=alt0,
                v_east=5.0,
                v_north=0.0,
                accuracy_h_m=acc_h,
                timestamp_ns=t_ns,
            )
            telem = core.get_gnss_telemetry()
            r_scales.append(telem["covariance_scale"])

        modes_seen.append(core.mode)

    # Mode must remain GNSS_AIDED for 100% of the duration despite severe quality swings
    assert all(m == GNSSMode.GNSS_AIDED for m in modes_seen), (
        f"Observed mode flapping under degraded GNSS: {set(modes_seen)}"
    )

    # Covariance scale must have modulated continuously (not binary 1.0 or inf)
    assert min(r_scales) < max(r_scales), "Covariance scale did not adapt continuously"
    assert max(r_scales) > 1.5, "Covariance scale failed to inflate under degraded accuracy"

    # Exactly ZERO FSM transitions should have occurred
    telem = core.get_gnss_telemetry()
    assert telem["transition_count"] == 0, f"Expected 0 transitions, got {telem['transition_count']}"


def test_isolated_missed_fixes_no_flapping() -> None:
    """Inject intermittent single-fix dropouts (e.g. 1 fix missing, gap 2.0s) and assert no transition to DR_ONLY."""
    core = NavigationCore(NavigationCoreConfig(
        velocitynet_enabled=False,
        biasnet_enabled=False,
        zupt_enabled=False,
        gnss_enabled=True,
    ))

    lat0, lon0, alt0 = 52.0, -1.5, 100.0
    core.initialize(
        lat0=lat0,
        lon0=lon0,
        alt0=alt0,
        p0_enu=np.zeros(3),
        v0_enu=np.array([5.0, 0.0, 0.0]),
        q0=np.array([1.0, 0.0, 0.0, 0.0]),
        timestamp_ns=0,
    )

    dt_imu = 0.1
    # Schedule: fixes at 1s, 2s, (skip 3s), fix at 4s, (skip 5s), fix at 6s, 7s, (skip 8s), fix at 9s, 10s
    # In each dropout, gap is exactly 2.0s (less than 3.0s timeout), so within grace period
    dropped_seconds = {3, 5, 8}

    modes_seen = []
    for step in range(100):
        t_ns = int((step + 1) * dt_imu * 1e9)
        t_s = (step + 1) * dt_imu
        core.step_imu(np.array([0.0, 0.0, 9.81]), np.zeros(3), dt_s=dt_imu, timestamp_ns=t_ns)

        if (step + 1) % 10 == 0:
            sec = int(round(t_s))
            if sec not in dropped_seconds:
                p_east = 5.0 * t_s
                d_lon = p_east / (6371000.0 * math.cos(math.radians(lat0))) * (180.0 / math.pi)
                core.step_gnss_fix(
                    lat=lat0,
                    lon=lon0 + d_lon,
                    alt=alt0,
                    v_east=5.0,
                    v_north=0.0,
                    accuracy_h_m=2.0,
                    timestamp_ns=t_ns,
                )

        modes_seen.append(core.mode)

    # Grace period prevents dropping into DR_ONLY for single-fix dropouts
    assert all(m == GNSSMode.GNSS_AIDED for m in modes_seen), (
        f"Grace period failed to absorb single-fix dropouts: {set(modes_seen)}"
    )
    telem = core.get_gnss_telemetry()
    assert telem["transition_count"] == 0


def test_dwell_time_prevents_rapid_cycling() -> None:
    """Adversarially attempt to oscillate between outage and fix arrival at 0.5 Hz.

    Assert that minimum dwell time of 2.0s prevents the FSM from cycling rapidly.
    """
    core = NavigationCore(NavigationCoreConfig(
        velocitynet_enabled=False,
        biasnet_enabled=False,
        zupt_enabled=False,
        gnss_enabled=True,
    ))

    lat0, lon0, alt0 = 52.0, -1.5, 100.0
    core.initialize(
        lat0=lat0,
        lon0=lon0,
        alt0=alt0,
        p0_enu=np.zeros(3),
        v0_enu=np.array([0.0, 0.0, 0.0]),
        q0=np.array([1.0, 0.0, 0.0, 0.0]),
        timestamp_ns=0,
    )

    # Initial fix
    core.step_gnss_fix(lat0, lon0, alt0, accuracy_h_m=2.0, timestamp_ns=int(1e9))

    # Trigger confirmed outage by running 4 seconds without GNSS
    dt_imu = 0.1
    for step in range(10, 50):
        core.step_imu(np.array([0.0, 0.0, 9.81]), np.zeros(3), dt_s=dt_imu, timestamp_ns=int(step * dt_imu * 1e9))

    assert core.mode == GNSSMode.DR_ONLY

    # Now attempt rapid fix injection at 0.5s intervals immediately after DR entry
    # Dwell time should hold mode in DR_ONLY until >= 2.0s dwell has elapsed
    t_dr_entry_ns = core.gnss_fsm.mode_entry_timestamp_ns
    t_early_fix_ns = t_dr_entry_ns + int(0.5 * 1e9)  # Only 0.5s after entry into DR_ONLY

    # Try injecting a fix early
    core.step_gnss_fix(lat0, lon0, alt0, accuracy_h_m=2.0, timestamp_ns=t_early_fix_ns)
    # Must remain in DR_ONLY because dwell time (0.5s < 2.0s) blocks premature reacquisition
    assert core.mode == GNSSMode.DR_ONLY

    # Try again at 1.5s after entry (still < 2.0s)
    t_early_fix_2_ns = t_dr_entry_ns + int(1.5 * 1e9)
    core.step_gnss_fix(lat0, lon0, alt0, accuracy_h_m=2.0, timestamp_ns=t_early_fix_2_ns)
    assert core.mode == GNSSMode.DR_ONLY

    # Finally at 2.1s (dwell >= 2.0s): permitted to transition to REACQUIRING
    t_valid_fix_ns = t_dr_entry_ns + int(2.1 * 1e9)
    core.step_gnss_fix(lat0, lon0, alt0, accuracy_h_m=2.0, timestamp_ns=t_valid_fix_ns)
    assert core.mode == GNSSMode.REACQUIRING
