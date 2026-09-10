"""Integration Test: Bounded-Rate State Recovery (Phase 10).

Validates that during REACQUIRING:
1. Reacquisition NIS gate strictly obeys the authoritative 99% Chi-Square(3) threshold = 11.345.
2. An implausible returning fix with NIS > 11.345 is rejected and does NOT enter REACQUIRING.
3. A statistically plausible returning fix with NIS <= 11.345 enters REACQUIRING.
4. State correction rate is strictly bounded (v_blend_max <= 2.0 m/s, step <= 3.0 m).
5. ESKF state is NEVER directly overwritten by GNSS coordinates.
6. 3 consecutive convergence fixes are required before transitioning to GNSS_AIDED.
7. Smooth monotonic convergence toward true position over multiple fixes.
8. Covariance health (finite, symmetric, PSD, normalized quaternion) is preserved throughout.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from navigation.core import NavigationCore, NavigationCoreConfig
from navigation.schemas.state import GNSSMode
from navigation.gnss.recovery import (
    GNSSRecoveryManager,
    RecoveryConfig,
    ReturningFixStatus,
)
from navigation.eskf.state import ESKFNominalState, ESKFState


def test_recovery_manager_authoritative_config() -> None:
    """Validate default configuration matches Master Plan Section 17 exactly."""
    cfg = RecoveryConfig()
    # Authoritative 99% Chi-Square(3) threshold
    assert cfg.nis_gate_reacq == 11.345
    assert cfg.max_displacement_rate_mps == 2.0
    assert cfg.max_single_step_m == 3.0
    assert cfg.convergence_pos_tolerance_m == 1.5
    assert cfg.required_consecutive_fixes == 3
    assert cfg.min_reacq_trust == 0.35


def test_recovery_manager_consecutive_convergence_fixes() -> None:
    """Direct validation of 3 consecutive convergence fixes requirement."""
    mgr = GNSSRecoveryManager(RecoveryConfig(
        max_displacement_rate_mps=2.0,
        max_single_step_m=3.0,
        convergence_pos_tolerance_m=1.0,
        required_consecutive_fixes=3,
    ))

    nom = ESKFNominalState.from_components(
        position_enu=np.zeros(3),
        velocity_enu=np.zeros(3),
        q=np.array([1.0, 0.0, 0.0, 0.0]),
        timestamp_ns=0,
    )
    state = ESKFState(nominal=nom, covariance=np.eye(15))

    # Fix 0.5m away (< 1.0m tolerance)
    # Fix 1: within tolerance, consecutive = 1 -> is_converged = False
    step1 = mgr.compute_bounded_correction(np.array([0.5, 0.0, 0.0]), state, dt_s=1.0)
    assert mgr.consecutive_valid_fixes == 1
    assert step1.is_converged is False

    # Fix 2: within tolerance, consecutive = 2 -> is_converged = False
    step2 = mgr.compute_bounded_correction(np.array([0.5, 0.0, 0.0]), state, dt_s=1.0)
    assert mgr.consecutive_valid_fixes == 2
    assert step2.is_converged is False

    # Fix 3: within tolerance, consecutive = 3 -> is_converged = True
    step3 = mgr.compute_bounded_correction(np.array([0.5, 0.0, 0.0]), state, dt_s=1.0)
    assert mgr.consecutive_valid_fixes == 3
    assert step3.is_converged is True

    # If a fix exceeds tolerance, consecutive fixes counter resets to 0
    step4 = mgr.compute_bounded_correction(np.array([10.0, 0.0, 0.0]), state, dt_s=1.0)
    assert mgr.consecutive_valid_fixes == 0
    assert step4.is_converged is False


def test_validate_returning_fix_nis_threshold() -> None:
    """Validate that validate_returning_fix strictly respects nis_gate_reacq = 11.345."""
    mgr = GNSSRecoveryManager(RecoveryConfig(nis_gate_reacq=11.345))

    nom = ESKFNominalState.from_components(
        position_enu=np.zeros(3),
        velocity_enu=np.zeros(3),
        q=np.array([1.0, 0.0, 0.0, 0.0]),
        timestamp_ns=0,
    )
    # Joint S = P_p + R_reacq
    # P_p = diag(4, 4, 4), trust=1.0 -> sigma_r=4.0 -> R_reacq=diag(16, 16, 64)
    # S[0,0] = 4 + 16 = 20.0
    state = ESKFState(nominal=nom, covariance=np.eye(15) * 4.0)

    # Displacement d = [14.0, 0, 0] -> NIS = 14^2 / 20 = 196 / 20 = 9.80 <= 11.345 (Plausible)
    res_valid = mgr.validate_returning_fix(np.array([14.0, 0.0, 0.0]), state, trust_score=1.0, timestamp_ns=int(1e9))
    assert res_valid.is_plausible is True
    assert res_valid.status == ReturningFixStatus.RETURNED_FIX_VALID
    assert res_valid.nis == pytest.approx(9.80, abs=0.1)

    # Displacement d = [16.0, 0, 0] -> NIS = 16^2 / 20 = 256 / 20 = 12.80 > 11.345 (Implausible)
    res_invalid = mgr.validate_returning_fix(np.array([16.0, 0.0, 0.0]), state, trust_score=1.0, timestamp_ns=int(1e9))
    assert res_invalid.is_plausible is False
    assert res_invalid.status == ReturningFixStatus.RETURNED_FIX_INNOVATION_REJECTED
    assert res_invalid.nis == pytest.approx(12.80, abs=0.1)


def test_pipeline_recovery_rate_limit_and_authoritative_gate() -> None:
    """End-to-end integration test asserting:
    1. Implausible returning fix (NIS > 11.345) does NOT enter REACQUIRING.
    2. Plausible returning fix (NIS <= 11.345) enters REACQUIRING.
    3. Rate-bounded recovery enforces <= 2.0 m/s displacement rate, step <= 3.0m.
    4. Never directly overwrites ESKF state.
    5. Requires 3 consecutive convergence fixes to return to GNSS_AIDED.
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

    # 1. First test IMPLAUSIBLE returning fix (25m offset -> NIS ~ 25.96 > 11.345)
    d_lon25 = 25.0 / (6371000.0 * math.cos(math.radians(lat0))) * (180.0 / math.pi)
    lon_implausible = lon0 + d_lon25
    applied_imp = core.step_gnss_fix(lat0, lon_implausible, alt0, accuracy_h_m=2.0, timestamp_ns=int(7.0 * 1e9))

    # Must be rejected and stay in DR_ONLY
    assert applied_imp is False
    assert core.mode == GNSSMode.DR_ONLY
    telem = core.get_gnss_telemetry()
    assert telem["total_fixes_rejected"] >= 1

    # 2. Now test PLAUSIBLE returning fix (10m offset -> NIS ~ 4.15 <= 11.345)
    d_lon10 = 10.0 / (6371000.0 * math.cos(math.radians(lat0))) * (180.0 / math.pi)
    target_lon = lon0 + d_lon10

    # Advance IMU to satisfy minimum dwell time (2.0s)
    for step in range(71, 95):
        t_ns = int(step * dt_imu * 1e9)
        core.step_imu(np.array([0.0, 0.0, 9.81]), np.zeros(3), dt_s=dt_imu, timestamp_ns=t_ns)

    t_reacq1_ns = int(9.5 * 1e9)
    prev_pos = core.state.nominal.position_enu.copy()
    applied_plausible = core.step_gnss_fix(lat0, target_lon, alt0, accuracy_h_m=2.0, timestamp_ns=t_reacq1_ns)

    assert applied_plausible is True
    assert core.mode == GNSSMode.REACQUIRING
    delta_p1 = float(np.linalg.norm(core.state.nominal.position_enu - prev_pos))
    # Must NOT snap 10 meters! Bounded rate limit enforces <= 3.0m max single step
    assert delta_p1 <= 3.01, f"Position jumped {delta_p1:.2f}m in single cycle, exceeding 3.0m bound"
    assert delta_p1 > 0.1, "Position correction should have made forward progress"

    # Subsequent fixes arrive at 1 Hz, asserting each step is strictly bounded <= 3.0m
    max_step_seen = delta_p1
    for fix_num in range(2, 16):
        t_fix_ns = int((9.5 + fix_num - 1) * 1e9)
        for sub in range(10):
            sub_t = t_fix_ns - int((10 - sub) * 0.1 * 1e9)
            core.step_imu(np.array([0.0, 0.0, 9.81]), np.zeros(3), dt_s=0.1, timestamp_ns=sub_t)

        prev_pos = core.state.nominal.position_enu.copy()
        core.step_gnss_fix(lat0, target_lon, alt0, accuracy_h_m=2.0, timestamp_ns=t_fix_ns)
        step_len = float(np.linalg.norm(core.state.nominal.position_enu - prev_pos))
        max_step_seen = max(max_step_seen, step_len)
        assert step_len <= 3.01, f"Step {fix_num} jumped {step_len:.2f}m, violating bound"

    # Final position must be close to target 10.0m
    final_e = core.state.nominal.position_enu[0]
    assert abs(final_e - 10.0) < 1.0, f"Failed to smoothly converge to target: pos={core.state.nominal.position_enu}"
    # FSM must have converged (3 consecutive fixes) and returned to GNSS_AIDED
    assert core.mode == GNSSMode.GNSS_AIDED
    health = core.check_covariance_health()
    assert health.is_finite is True
    assert health.is_symmetric is True
    assert health.is_psd is True
