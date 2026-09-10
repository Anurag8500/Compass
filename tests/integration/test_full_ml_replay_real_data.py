"""Integration Test: Full ML Replay on Real IO-VNBD Driving Data (Phase 9).

Executes a smoke replay run using NavigationCore with both VelocityNet v1.1 and
BiasNet v1.0 enabled on real IO-VNBD driving data (Categorised_S1.npz).
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


def test_full_ml_replay_real_data_smoke() -> None:
    """Run a 5-second driving segment with full ML integration on Categorised_S1.npz."""
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

    core = NavigationCore(NavigationCoreConfig(
        velocitynet_enabled=True,
        biasnet_enabled=True,
        zupt_enabled=True,
        gnss_enabled=True,
    ))

    # Initialize at 25.0s (index 250)
    start_idx = 250
    end_idx = start_idx + 50  # 5 seconds at 10 Hz

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
        lat0=float(res.aux_signals["v_ref_lat"][start_idx]),
        lon0=float(res.aux_signals["v_ref_lon"][start_idx]),
        alt0=float(res.aux_signals["v_ref_alt_m"][start_idx]),
        p0_enu=np.zeros(3),
        v0_enu=v_init,
        q0=q0,
        gyro_bias0=preprocessed.calibration.gyro_bias,
        timestamp_ns=int(res.timestamps_ns[start_idx]),
    )

    for k in range(start_idx, end_idx):
        dt = (res.timestamps_ns[k + 1] - res.timestamps_ns[k]) * 1e-9
        t_ns = int(res.timestamps_ns[k + 1])

        out = core.step_imu(
            f_m_v=res.f_m_v[k],
            omega_m_v=res.omega_m_v[k],
            dt_s=dt,
            timestamp_ns=t_ns,
        )

        assert out.covariance_health.is_finite is True
        assert out.covariance_health.is_symmetric is True
        assert out.covariance_health.is_psd is True
        assert out.covariance_health.quaternion_normalized is True

    # Check that navigation state can be extracted cleanly
    nav_state = core.get_navigation_state()
    assert len(nav_state.position_local) == 3
    assert len(nav_state.velocity_local) == 3
    assert len(nav_state.orientation.q) == 4
