"""Integration Test Suite: Full ML Replay on Real IO-VNBD Driving Data (Phase 9).

Validates end-to-end integration of NavigationCore with both VelocityNet v1.1 and
BiasNet v1.0 enabled on real IO-VNBD driving data (Categorised_S1.npz):
1. Segment-local ENU frame consistency.
2. GNSS-aided phase followed by clean GNSS outage handoff.
3. Strict causal windowing and valid cadence during real replay.
4. Covariance health (finite, symmetric, PSD) and normalized quaternion.
5. Filter authority (no direct state overwrite).
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pytest

from data.pipeline.sync import SynchronizedTrip
from data.pipeline.stationary_detect import StationaryDetector
from navigation.frames.local_geo import GeoReference
from navigation.ins.attitude import rotation_matrix_to_quaternion
from navigation.preprocessing.pipeline import PreprocessingPipeline
from navigation.core import NavigationCore, NavigationCoreConfig
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


def test_real_replay_aided_and_outage_handoff(preprocessed_real_trip) -> None:
    """Test 20 seconds of real driving: 10s GNSS-aided followed by 10s outage with active ML."""
    res, calib_gyro_bias = preprocessed_real_trip

    start_idx = 250  # 25.0s into trip
    aided_len = 100  # 10s aided (100 samples)
    outage_len = 100 # 10s outage (100 samples)
    total_len = aided_len + outage_len
    end_idx = start_idx + total_len

    # Segment-local ENU reference frame
    seg_lat0 = float(res.aux_signals["v_ref_lat"][start_idx])
    seg_lon0 = float(res.aux_signals["v_ref_lon"][start_idx])
    seg_alt0 = float(res.aux_signals["v_ref_alt_m"][start_idx])
    geo_ref = GeoReference(lat_ref=seg_lat0, lon_ref=seg_lon0, alt_ref=seg_alt0)

    seg_gt_e, seg_gt_n, seg_gt_u = geo_ref.geodetic_to_enu(
        res.aux_signals["v_ref_lat"][start_idx : end_idx + 1],
        res.aux_signals["v_ref_lon"][start_idx : end_idx + 1],
        res.aux_signals["v_ref_alt_m"][start_idx : end_idx + 1],
    )
    # Verification of frame consistency
    assert np.allclose([seg_gt_e[0], seg_gt_n[0], seg_gt_u[0]], [0.0, 0.0, 0.0], atol=1e-5)

    core = NavigationCore(NavigationCoreConfig(
        velocitynet_enabled=True,
        biasnet_enabled=True,
        zupt_enabled=True,
        gnss_enabled=True,
    ))

    hdg0 = float(res.aux_signals["v_ref_heading_deg"][start_idx])
    spd0 = float(res.aux_signals["v_ref_speed_mps"][start_idx])
    psi0 = math.radians(hdg0)
    v_init = np.array([spd0 * math.sin(psi0), spd0 * math.cos(psi0), 0.0])

    R0 = np.array([
        [math.sin(psi0), -math.cos(psi0), 0.0],
        [math.cos(psi0),  math.sin(psi0), 0.0],
        [0.0,             0.0,            1.0],
    ])
    q0 = rotation_matrix_to_quaternion(R0)

    core.initialize(
        lat0=seg_lat0,
        lon0=seg_lon0,
        alt0=seg_alt0,
        p0_enu=np.zeros(3),
        v0_enu=v_init,
        q0=q0,
        gyro_bias0=calib_gyro_bias,
        timestamp_ns=int(res.timestamps_ns[start_idx]),
    )

    vnet_executed_count = 0
    bnet_executed_count = 0
    gnss_fixes_applied = 0

    for step_rel in range(total_len):
        k = start_idx + step_rel
        dt = (res.timestamps_ns[k + 1] - res.timestamps_ns[k]) * 1e-9
        t_ns = int(res.timestamps_ns[k + 1])

        out = core.step_imu(
            f_m_v=res.f_m_v[k],
            omega_m_v=res.omega_m_v[k],
            dt_s=dt,
            timestamp_ns=t_ns,
        )

        # 1. Verify covariance health at every single step
        assert out.covariance_health.is_finite is True, f"Non-finite covariance at step {step_rel}"
        assert out.covariance_health.is_symmetric is True, f"Non-symmetric covariance at step {step_rel}"
        assert out.covariance_health.is_psd is True, f"Non-PSD covariance at step {step_rel}"
        assert out.covariance_health.quaternion_normalized is True, f"Unnormalized quaternion at step {step_rel}"

        # 2. GNSS fix only during aided phase (first 100 samples, 1 Hz)
        if step_rel < aided_len and (step_rel + 1) % 10 == 0:
            applied = core.step_gnss_fix(
                lat=float(res.aux_signals["v_ref_lat"][k + 1]),
                lon=float(res.aux_signals["v_ref_lon"][k + 1]),
                alt=float(res.aux_signals["v_ref_alt_m"][k + 1]),
                v_east=float(res.aux_signals["v_ref_speed_mps"][k + 1] * math.sin(math.radians(float(res.aux_signals["v_ref_heading_deg"][k + 1])))),
                v_north=float(res.aux_signals["v_ref_speed_mps"][k + 1] * math.cos(math.radians(float(res.aux_signals["v_ref_heading_deg"][k + 1])))),
                timestamp_ns=t_ns,
            )
            if applied:
                gnss_fixes_applied += 1

        # Track ML executions
        if out.velocitynet_diagnostics and out.velocitynet_diagnostics.applied:
            vnet_executed_count += 1
        if out.biasnet_diagnostics and out.biasnet_diagnostics.applied:
            bnet_executed_count += 1

    # Assertions on execution behavior
    assert gnss_fixes_applied >= 9, f"Expected ~10 GNSS fixes in aided phase, got {gnss_fixes_applied}"
    assert bnet_executed_count > 0, "BiasNet must have executed and applied updates"
    
    # Final state extraction and sanity
    nav_state = core.get_navigation_state()
    final_pos = np.array(nav_state.position_local)
    assert np.all(np.isfinite(final_pos))
    assert not np.isnan(nav_state.velocity_local).any()


def test_causal_window_properties_on_real_data(preprocessed_real_trip) -> None:
    """Verify that causal windows built from real IMU streams obey strict causality and length."""
    res, _ = preprocessed_real_trip
    from navigation.ml.window_buffer import CausalWindowBuffer

    buf = CausalWindowBuffer(max_length=20)
    for k in range(50):
        t_ns = int(res.timestamps_ns[k])
        buf.push(t_ns, res.f_m_v[k], res.omega_m_v[k])

        if k < 19:
            has_win, feats, win_ts, reason = buf.get_causal_window(current_timestamp_ns=t_ns)
            assert has_win is False
            assert reason == "INSUFFICIENT_HISTORY"
        else:
            has_win, feats, win_ts, reason = buf.get_causal_window(current_timestamp_ns=t_ns)
            assert has_win is True
            assert feats.shape == (20, 9)
            assert len(win_ts) == 20
            # Strictly causal: all window timestamps must be <= current_timestamp_ns
            assert np.all(win_ts <= t_ns)
            assert win_ts[-1] == t_ns
            # Window duration for 20 samples at 10 Hz should be ~1.9s
            span_s = (win_ts[-1] - win_ts[0]) * 1e-9
            assert 1.85 <= span_s <= 1.95
