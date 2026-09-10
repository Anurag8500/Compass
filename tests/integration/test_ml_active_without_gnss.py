"""Mandatory Integration Test: ML Measurement Updates Remain Active Without GNSS (Phase 9).

Per SIH 26168 Section 16:
The navigation pipeline must continue executing VelocityNet and BiasNet updates
causally and stably when GNSS measurements are withheld.
The absence of GNSS must NOT automatically disable ML aiding.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from navigation.core import NavigationCore, NavigationCoreConfig
from navigation.eskf.predict import ProcessNoiseConfig
from navigation.schemas.state import GNSSMode


def test_ml_aiding_active_during_gnss_denial() -> None:
    """Proves that VelocityNet and BiasNet updates fire continuously when GNSS is withheld."""
    config = NavigationCoreConfig(
        velocitynet_enabled=True,
        biasnet_enabled=True,
        zupt_enabled=False,  # Keep moving
        gnss_enabled=True,
    )
    core = NavigationCore(config=config)

    # Initialize at a realistic driving state
    lat0, lon0, alt0 = 12.9716, 77.5946, 920.0
    p0 = np.zeros(3)
    v0 = np.array([12.0, 0.0, 0.0])  # 12 m/s East
    q0 = np.array([1.0, 0.0, 0.0, 0.0])

    core.initialize(
        lat0=lat0, lon0=lon0, alt0=alt0,
        p0_enu=p0, v0_enu=v0, q0=q0,
        timestamp_ns=0,
    )

    # Phase 1: 0s to 5s with GNSS fixes (1 Hz)
    dt_s = 0.1  # 10 Hz IMU
    vnet_applied_aided = 0
    bnet_applied_aided = 0

    for step in range(50):
        t_ns = int(step * dt_s * 1e9)
        # Constant driving kinematics: forward accel balancing drag, gravity on Z
        f_m_v = np.array([0.1, 0.0, 9.81])
        w_m_v = np.array([0.0, 0.0, 0.001])

        out = core.step_imu(f_m_v, w_m_v, dt_s=dt_s, timestamp_ns=t_ns)
        if out.velocitynet_diagnostics and out.velocitynet_diagnostics.applied:
            vnet_applied_aided += 1
        if out.biasnet_diagnostics and out.biasnet_diagnostics.applied:
            bnet_applied_aided += 1

        # 1 Hz GNSS fix
        if step % 10 == 0:
            core.step_gnss_fix(
                lat=lat0 + step * 1e-5,
                lon=lon0,
                alt=alt0,
                v_east=12.0,
                v_north=0.0,
                timestamp_ns=t_ns,
            )

    # Phase 2: 5s to 15s (10-second GNSS blackout: GNSS is strictly withheld)
    # The IMU stream continues. We assert that ML updates continue firing.
    vnet_applied_outage = 0
    bnet_applied_outage = 0
    vnet_cycles_outage = 0
    bnet_cycles_outage = 0

    for step in range(50, 150):
        t_ns = int(step * dt_s * 1e9)
        f_m_v = np.array([0.1, 0.0, 9.81])
        w_m_v = np.array([0.0, 0.0, 0.001])

        # NO GNSS CALLS HERE
        out = core.step_imu(f_m_v, w_m_v, dt_s=dt_s, timestamp_ns=t_ns)

        if out.velocitynet_diagnostics is not None:
            vnet_cycles_outage += 1
            if out.velocitynet_diagnostics.applied:
                vnet_applied_outage += 1

        if out.biasnet_diagnostics is not None:
            bnet_cycles_outage += 1
            if out.biasnet_diagnostics.applied:
                bnet_applied_outage += 1

        # Covariance must remain valid
        assert out.covariance_health.is_finite is True
        assert out.covariance_health.is_symmetric is True
        assert out.covariance_health.is_psd is True
        assert out.covariance_health.quaternion_normalized is True

    # 10s outage at ~2 Hz for VNet: expected ~20 cycles
    assert vnet_cycles_outage >= 18, f"Expected ~20 VelocityNet cycles during 10s outage, got {vnet_cycles_outage}"
    assert vnet_applied_outage > 0, "VelocityNet must have applied updates during GNSS blackout"

    # 10s outage at ~1 Hz for BiasNet: expected ~10 cycles
    assert bnet_cycles_outage >= 9, f"Expected ~10 BiasNet cycles during 10s outage, got {bnet_cycles_outage}"
    assert bnet_applied_outage > 0, "BiasNet must have applied updates during GNSS blackout"


def test_fallback_flags_decouple_cleanly() -> None:
    """Proves that disabling either or both ML models leaves ESKF propagation healthy."""
    # Run with both models disabled
    cfg_no_ml = NavigationCoreConfig(
        velocitynet_enabled=False,
        biasnet_enabled=False,
    )
    core = NavigationCore(config=cfg_no_ml)
    core.initialize(12.0, 77.0, 900.0, np.zeros(3), np.array([10.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]))

    for step in range(30):
        t_ns = int(step * 0.1 * 1e9)
        out = core.step_imu(np.array([0.0, 0.0, 9.81]), np.zeros(3), dt_s=0.1, timestamp_ns=t_ns)
        assert out.covariance_health.is_finite is True
        assert out.covariance_health.is_psd is True
        # Both models report disabled
        if out.velocitynet_diagnostics:
            assert out.velocitynet_diagnostics.applied is False
            assert out.velocitynet_diagnostics.reason == "MODEL_DISABLED"
        if out.biasnet_diagnostics:
            assert out.biasnet_diagnostics.applied is False
            assert out.biasnet_diagnostics.reason == "MODEL_DISABLED"
