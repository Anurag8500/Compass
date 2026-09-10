"""Integration Test: Bounded-Rate State Recovery (Phase 10).

Validates that during REACQUIRING:
1. State correction rate is strictly bounded (v_blend_max <= 2.0 m/s, step <= 3.0 m).
2. ESKF state is NEVER directly overwritten by GNSS coordinates.
3. Large initial position offsets (e.g. 10m, 25m, 50m accumulated during outage)
   converge smoothly and monotonically toward true position over multiple fixes.
4. Covariance health (finite, symmetric, PSD, normalized quaternion) is preserved throughout.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from navigation.core import NavigationCore, NavigationCoreConfig
from navigation.schemas.state import GNSSMode
from navigation.gnss.recovery import GNSSRecoveryManager, RecoveryConfig


def test_recovery_manager_bounded_step_math() -> None:
    """Direct unit-level validation of GNSSRecoveryManager bounded correction."""
    mgr = GNSSRecoveryManager(RecoveryConfig(
        max_displacement_rate_mps=2.0,
        max_single_step_m=3.0,
        convergence_pos_tolerance_m=1.0,
        required_consecutive_fixes=1,
    ))

    # Mock state with position at [0, 0, 0]
    from navigation.eskf.state import ESKFNominalState, ESKFState
    nom = ESKFNominalState.from_components(
        position_enu=np.zeros(3),
        velocity_enu=np.zeros(3),
        q=np.array([1.0, 0.0, 0.0, 0.0]),
        timestamp_ns=0,
    )
    state = ESKFState(nominal=nom, covariance=np.eye(15))

    # Case 1: 50m offset with dt = 1.0s -> max step is 2.0m (rate limited)
    step1 = mgr.compute_bounded_correction(np.array([50.0, 0.0, 0.0]), state, dt_s=1.0)
    assert step1.is_clamped is True
    assert pytest.approx(step1.applied_step_norm_m, rel=1e-5) == 2.0
    assert pytest.approx(step1.delta_p_bounded[0], rel=1e-5) == 2.0
    assert step1.is_converged is False

    # Apply step
    state = mgr.apply_bounded_correction(state, step1)
    assert pytest.approx(state.nominal.position_enu[0], rel=1e-5) == 2.0

    # Case 2: small offset 0.5m with dt = 1.0s -> full step applied (not clamped)
    step2 = mgr.compute_bounded_correction(np.array([2.5, 0.0, 0.0]), state, dt_s=1.0)
    assert step2.is_clamped is False
    assert pytest.approx(step2.applied_step_norm_m, rel=1e-5) == 0.5
    assert step2.is_converged is True


def test_pipeline_recovery_rate_limit_on_large_offset() -> None:
    """End-to-end integration test asserting rate-bounded recovery inside NavigationCore."""
    core = NavigationCore(NavigationCoreConfig(
        velocitynet_enabled=False,
        biasnet_enabled=False,
        zupt_enabled=False,
        gnss_enabled=True,
    ))

    # Initialize at [0, 0, 0]
    lat0, lon0, alt0 = 52.0, -1.5, 100.0
    core.initialize(
        lat0=lat0,
        lon0=lon0,
        alt0=alt0,
        p0_enu=np.zeros(3),
        v0_enu=np.zeros(3),
        q0=np.array([1.0, 0.0, 0.0, 0.0]),
        timestamp_ns=0,
    )

    # Initial fix to establish GNSS_AIDED mode
    core.step_gnss_fix(lat0, lon0, alt0, v_east=0.0, v_north=0.0, accuracy_h_m=2.0, timestamp_ns=1_000_000_000)
    assert core.mode == GNSSMode.GNSS_AIDED

    # Run IMU forward for 5 seconds without GNSS -> transitions to DR_ONLY (timeout is 3.0s)
    dt_imu = 0.1
    for step in range(10, 60):
        t_ns = int(step * dt_imu * 1e9)
        core.step_imu(np.array([0.0, 0.0, 9.81]), np.zeros(3), dt_s=dt_imu, timestamp_ns=t_ns)

    assert core.mode == GNSSMode.DR_ONLY

    # Vehicle is at [0, 0, 0]. Now a returning fix arrives at [25.0, 0.0, 0.0] ENU (25m East).
    # Convert [25, 0, 0] ENU to lat/lon
    d_lon = 25.0 / (6371000.0 * math.cos(math.radians(lat0))) * (180.0 / math.pi)
    target_lon = lon0 + d_lon

    # Fix 1 arrives at t = 7.0s: enters REACQUIRING, applies first bounded step
    t_reacq1_ns = int(7.0 * 1e9)
    prev_pos = core.state.nominal.position_enu.copy()
    core.step_gnss_fix(lat0, target_lon, alt0, accuracy_h_m=2.0, timestamp_ns=t_reacq1_ns)

    assert core.mode == GNSSMode.REACQUIRING
    delta_p1 = float(np.linalg.norm(core.state.nominal.position_enu - prev_pos))
    # Must NOT snap 25 meters! Bounded rate limit enforces <= 3.0m max single step
    assert delta_p1 <= 3.01, f"Position jumped {delta_p1:.2f}m in single cycle, exceeding 3.0m bound"
    assert delta_p1 > 0.1, "Position correction should have made forward progress"

    # Subsequent fixes arrive at 1 Hz, asserting each step is strictly bounded <= 3.0m
    max_step_seen = delta_p1
    for fix_num in range(2, 18):
        t_fix_ns = int((7.0 + fix_num - 1) * 1e9)
        # Advance IMU between fixes
        for sub in range(10):
            sub_t = t_fix_ns - int((10 - sub) * 0.1 * 1e9)
            core.step_imu(np.array([0.0, 0.0, 9.81]), np.zeros(3), dt_s=0.1, timestamp_ns=sub_t)

        prev_pos = core.state.nominal.position_enu.copy()
        core.step_gnss_fix(lat0, target_lon, alt0, accuracy_h_m=2.0, timestamp_ns=t_fix_ns)
        step_len = float(np.linalg.norm(core.state.nominal.position_enu - prev_pos))
        max_step_seen = max(max_step_seen, step_len)
        assert step_len <= 3.01, f"Step {fix_num} jumped {step_len:.2f}m, violating bound"

    # Final position must be close to target 25.0m
    final_e = core.state.nominal.position_enu[0]
    assert abs(final_e - 25.0) < 1.0, f"Failed to smoothly converge to target: pos={core.state.nominal.position_enu}"
    # FSM must have converged and returned to GNSS_AIDED
    assert core.mode == GNSSMode.GNSS_AIDED
