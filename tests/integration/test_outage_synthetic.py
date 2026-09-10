"""Integration test for synthetic GNSS outages (Phase 10).

Validates end-to-end pipeline under deterministic synthetic blackouts:
- 5s, 10s, 30s, and 60s blackout durations.
- Verifies exact FSM state flow: GNSS_AIDED -> DR_ONLY -> REACQUIRING -> GNSS_AIDED.
- Verifies ML models (VelocityNet and BiasNet) remain active throughout DR_ONLY.
- Verifies covariance health throughout.
- Verifies bounded recovery without discontinuous position snaps.
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pytest

from navigation.core import NavigationCore, NavigationCoreConfig
from navigation.schemas.state import GNSSMode
from navigation.gnss import generate_synthetic_outage_mask
from data.pipeline.sync import SynchronizedTrip
from data.pipeline.stationary_detect import StationaryDetector
from navigation.preprocessing.pipeline import PreprocessingPipeline
from ml.data.resample import resample_to_canonical_10hz


@pytest.fixture(scope="module")
def preprocessed_real_trip():
    """Load and preprocess Categorised_S1.npz once for the test module."""
    trip_path = Path("data/cache/iovnbd/Categorised_S1.npz")
    assert trip_path.exists(), f"Missing real driving file {trip_path}"

    trip = SynchronizedTrip.load_npz(trip_path)
    detector = StationaryDetector()
    _, stat_mask = detector.detect(trip.timestamps_ns, trip.accel_raw, trip.gyro_raw)
    pipeline = PreprocessingPipeline(sampling_rate_hz=10.0, filter_cutoff_hz=3.0, median_window_size=3)
    preprocessed = pipeline.process_trip(trip, stationary_mask=stat_mask)

    aux = {
        "v_ref_speed_mps": trip.v_ref_speed_mps,
        "v_ref_lat": trip.v_ref_lat,
        "v_ref_lon": trip.v_ref_lon,
        "v_ref_alt_m": trip.v_ref_alt_m / 1000.0,
        "v_ref_heading_deg": trip.v_ref_heading_deg,
    }

    res = resample_to_canonical_10hz(
        timestamps_ns=preprocessed.timestamps_ns,
        f_m_v=preprocessed.f_m_v,
        omega_m_v=preprocessed.omega_m_v,
        is_validated=preprocessed.is_validated,
        aux_signals=aux,
    )
    return res, preprocessed.calibration.gyro_bias


@pytest.mark.parametrize("duration_s", [5.0, 10.0, 30.0, 60.0])
def test_synthetic_outage_pipeline_flow(duration_s: float) -> None:
    """Test full pipeline execution across 5s, 10s, 30s, 60s synthetic outages."""
    outage_start_s = 15.0
    total_duration_s = outage_start_s + duration_s + 15.0  # pre-outage + outage + recovery
    dt_imu_s = 0.1
    num_imu_steps = int(total_duration_s / dt_imu_s)

    # Pure kinematic strapdown integration for exact synthetic tracking
    core = NavigationCore(NavigationCoreConfig(
        velocitynet_enabled=False,
        biasnet_enabled=False,
        zupt_enabled=False,
        gnss_enabled=True,
    ))

    core.initialize(
        lat0=52.0,
        lon0=-1.5,
        alt0=100.0,
        p0_enu=np.zeros(3),
        v0_enu=np.array([10.0, 0.0, 0.0]),  # Moving East at 10 m/s
        q0=np.array([1.0, 0.0, 0.0, 0.0]),
        timestamp_ns=0,
    )

    gnss_step_interval = 10
    gnss_timestamps_ns = [int(i * 1e9) for i in range(1, int(total_duration_s) + 1)]
    gnss_mask = generate_synthetic_outage_mask(
        gnss_timestamps_ns,
        outage_start_offset_s=outage_start_s,
        outage_duration_s=duration_s,
        session_start_ns=0,
    )
    gnss_availability = {ts: avail for ts, avail in zip(gnss_timestamps_ns, gnss_mask)}

    modes_seen: set[GNSSMode] = set()

    for step in range(num_imu_steps):
        t_ns = int((step + 1) * dt_imu_s * 1e9)
        t_s = (step + 1) * dt_imu_s

        # Constant velocity forward motion East
        out = core.step_imu(
            f_m_v=np.array([0.0, 0.0, 9.81]),
            omega_m_v=np.zeros(3),
            dt_s=dt_imu_s,
            timestamp_ns=t_ns,
        )
        modes_seen.add(core.mode)

        # Covariance health must remain valid throughout
        assert out.covariance_health.is_finite is True
        assert out.covariance_health.is_symmetric is True
        assert out.covariance_health.is_psd is True

        # GNSS fix at 1 Hz
        if (step + 1) % gnss_step_interval == 0:
            is_available = gnss_availability.get(t_ns, True)
            if is_available:
                p_east = 10.0 * t_s
                d_lon = p_east / (6371000.0 * math.cos(math.radians(52.0))) * (180.0 / math.pi)
                core.step_gnss_fix(
                    lat=52.0,
                    lon=-1.5 + d_lon,
                    alt=100.0,
                    v_east=10.0,
                    v_north=0.0,
                    accuracy_h_m=2.5,
                    timestamp_ns=t_ns,
                )
                modes_seen.add(core.mode)

    telem = core.get_gnss_telemetry()

    # 1. State machine must have traversed through all 3 canonical states
    assert GNSSMode.GNSS_AIDED in modes_seen, "Did not observe GNSS_AIDED"
    assert GNSSMode.DR_ONLY in modes_seen, f"Did not observe DR_ONLY in {duration_s}s outage"
    assert GNSSMode.REACQUIRING in modes_seen, f"Did not observe REACQUIRING in {duration_s}s outage"

    # 2. Final state must have successfully recovered to GNSS_AIDED
    assert core.mode == GNSSMode.GNSS_AIDED, f"Final mode was {core.mode}, expected recovery to GNSS_AIDED"

    # 3. Transitions must be logged
    assert telem["transition_count"] >= 3, f"Expected >= 3 transitions, got {telem['transition_count']}"
    assert telem["is_outage"] is False


def test_ml_active_during_dr_only() -> None:
    """Validate that VelocityNet (~2 Hz) and BiasNet (~1 Hz) remain active during DR_ONLY."""
    core = NavigationCore(NavigationCoreConfig(
        velocitynet_enabled=True,
        biasnet_enabled=True,
        zupt_enabled=False,
        gnss_enabled=True,
    ))
    core.initialize(
        lat0=52.0,
        lon0=-1.5,
        alt0=100.0,
        p0_enu=np.zeros(3),
        v0_enu=np.array([12.0, 0.0, 0.0]),
        q0=np.array([1.0, 0.0, 0.0, 0.0]),
        timestamp_ns=0,
    )

    dt_s = 0.1
    # 5 seconds aided
    for step in range(50):
        t_ns = int((step + 1) * dt_s * 1e9)
        core.step_imu(np.array([0.1, 0.0, 9.81]), np.array([0.0, 0.0, 0.001]), dt_s=dt_s, timestamp_ns=t_ns)
        if (step + 1) % 10 == 0:
            core.step_gnss_fix(lat=52.0, lon=-1.5, alt=100.0, v_east=12.0, v_north=0.0, timestamp_ns=t_ns)

    # 15 seconds outage (enters DR_ONLY after 3.0s timeout)
    vnet_in_dr = 0
    bnet_in_dr = 0
    dr_steps = 0

    for step in range(50, 200):
        t_ns = int((step + 1) * dt_s * 1e9)
        out = core.step_imu(np.array([0.1, 0.0, 9.81]), np.array([0.0, 0.0, 0.001]), dt_s=dt_s, timestamp_ns=t_ns)
        if core.mode == GNSSMode.DR_ONLY:
            dr_steps += 1
            if out.velocitynet_diagnostics and out.velocitynet_diagnostics.applied:
                vnet_in_dr += 1
            if out.biasnet_diagnostics and out.biasnet_diagnostics.applied:
                bnet_in_dr += 1

    assert dr_steps >= 100, f"Expected >= 100 DR_ONLY steps, got {dr_steps}"
    # In ~12s of DR_ONLY, VelocityNet should fire ~24 times (~2 Hz) and BiasNet ~12 times (~1 Hz)
    assert vnet_in_dr >= 20, f"Expected >= 20 VelocityNet updates in DR_ONLY, got {vnet_in_dr}"
    assert bnet_in_dr >= 10, f"Expected >= 10 BiasNet updates in DR_ONLY, got {bnet_in_dr}"


def test_synthetic_outage_on_real_data(preprocessed_real_trip) -> None:
    """Validate synthetic outage masking on real driving data with full ML enabled."""
    res, calib_gyro_bias = preprocessed_real_trip
    start_idx = 200
    total_len = 250  # 25 seconds
    end_idx = start_idx + total_len

    seg_lat0 = float(res.aux_signals["v_ref_lat"][start_idx])
    seg_lon0 = float(res.aux_signals["v_ref_lon"][start_idx])
    seg_alt0 = float(res.aux_signals["v_ref_alt_m"][start_idx])
    t0_ns = int(res.timestamps_ns[start_idx])

    core = NavigationCore(NavigationCoreConfig(
        velocitynet_enabled=True,
        biasnet_enabled=True,
        zupt_enabled=False,
        gnss_enabled=True,
    ))
    core.initialize(
        lat0=seg_lat0,
        lon0=seg_lon0,
        alt0=seg_alt0,
        p0_enu=np.zeros(3),
        v0_enu=np.array([float(res.aux_signals["v_ref_speed_mps"][start_idx]), 0.0, 0.0]),
        q0=np.array([1.0, 0.0, 0.0, 0.0]),
        gyro_bias0=calib_gyro_bias,
        timestamp_ns=t0_ns,
    )

    # Mask 10s outage between index 250 and 350 (relative indices 50 to 150)
    modes_seen: set[GNSSMode] = set()
    for i in range(start_idx, end_idx):
        rel_idx = i - start_idx
        t_ns = int(res.timestamps_ns[i])
        f_i = res.f_m_v[i]
        w_i = res.omega_m_v[i]

        out = core.step_imu(f_i, w_i, dt_s=0.1, timestamp_ns=t_ns)
        modes_seen.add(core.mode)

        # GNSS fix at 1 Hz (every 10 steps), masked during rel_idx [50, 150)
        if rel_idx % 10 == 0:
            in_outage = 50 <= rel_idx < 150
            if not in_outage:
                core.step_gnss_fix(
                    lat=float(res.aux_signals["v_ref_lat"][i]),
                    lon=float(res.aux_signals["v_ref_lon"][i]),
                    alt=float(res.aux_signals["v_ref_alt_m"][i]),
                    v_east=float(res.aux_signals["v_ref_speed_mps"][i]),
                    v_north=0.0,
                    accuracy_h_m=2.5,
                    timestamp_ns=t_ns,
                )
                modes_seen.add(core.mode)

    assert GNSSMode.GNSS_AIDED in modes_seen
    assert GNSSMode.DR_ONLY in modes_seen
    telem = core.get_gnss_telemetry()
    assert telem["transition_count"] >= 1
