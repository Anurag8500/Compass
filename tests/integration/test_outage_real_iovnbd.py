"""Integration Test: Real IO-VNBD Data Replay & Outage Index Audit (Phase 10).

Adheres strictly to the user requirement and Master Plan Section 17:
- "Treat real IO-VNBD outage results as 'real outage' only when a verified outage index exists;
   otherwise label them real-data replay, not outage evidence."
- "Do not modify ground truth during synthetic outage injection."
- "Preserve ESKF authority and Phase 9 behavior/cadence."

Validates:
1. Real IO-VNBD dataset outage audit: verifies absence of verified external outage annotations
   in current dataset manifests, properly classifying tests as REAL_DATA_REPLAY.
2. Full real driving replay (Categorised_S1.npz) through Phase 10 supervisory pipeline.
3. Telemetry reporting distinguishing received, accepted, rejected, and outage status.
4. Covariance health throughout real driving.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from data.pipeline.sync import SynchronizedTrip
from data.pipeline.stationary_detect import StationaryDetector
from navigation.preprocessing.pipeline import PreprocessingPipeline
from navigation.core import NavigationCore, NavigationCoreConfig
from navigation.schemas.state import GNSSMode
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


def test_audit_real_iovnbd_outage_manifest() -> None:
    """Audit repository manifests to verify presence/absence of natural GNSS outage ground truth.

    Confirms that without verified natural outage ground-truth intervals, real dataset runs
    are categorized strictly as REAL_DATA_REPLAY rather than natural outage evidence.
    """
    manifest_path = Path("data/manifests/iovnbd_manifest_v1.csv")
    has_verified_outage_file = Path("data/manifests/gps_outage_index.csv").exists()

    if manifest_path.exists():
        import pandas as pd
        df = pd.read_csv(manifest_path)
        if "has_real_outages" in df.columns:
            real_outage_count = int(df["has_real_outages"].sum())
            assert real_outage_count == 0, (
                f"Found {real_outage_count} marked real outages in manifest; update classification accordingly"
            )

    # Classification assertion
    classification = "NATURAL_OUTAGE_EVIDENCE" if has_verified_outage_file else "REAL_DATA_REPLAY"
    assert classification == "REAL_DATA_REPLAY", (
        "Expected classification REAL_DATA_REPLAY in absence of verified outage index"
    )


def test_real_data_replay_phase10_supervisory(preprocessed_real_trip) -> None:
    """Execute real driving segment and assert Phase 10 supervisory telemetry and covariance health."""
    res, calib_gyro_bias = preprocessed_real_trip
    start_idx = 100
    duration_steps = 200  # 20 seconds at 10 Hz
    end_idx = start_idx + duration_steps

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

    trust_scores = []
    r_scales = []

    for i in range(start_idx, end_idx):
        t_ns = int(res.timestamps_ns[i])
        f_i = res.f_m_v[i]
        w_i = res.omega_m_v[i]

        out = core.step_imu(f_i, w_i, dt_s=0.1, timestamp_ns=t_ns)

        # Assert covariance health at every step
        assert out.covariance_health.is_finite is True
        assert out.covariance_health.is_symmetric is True
        assert out.covariance_health.is_psd is True
        assert out.covariance_health.quaternion_normalized is True

        # GNSS fix at 1 Hz (every 10 steps)
        if (i - start_idx) % 10 == 0:
            core.step_gnss_fix(
                lat=float(res.aux_signals["v_ref_lat"][i]),
                lon=float(res.aux_signals["v_ref_lon"][i]),
                alt=float(res.aux_signals["v_ref_alt_m"][i]),
                v_east=float(res.aux_signals["v_ref_speed_mps"][i]),
                v_north=0.0,
                accuracy_h_m=2.5,
                timestamp_ns=t_ns,
            )
            telem = core.get_gnss_telemetry()
            trust_scores.append(telem["last_trust_score"])
            r_scales.append(telem["covariance_scale"])

    # 1. Telemetry verification
    telem = core.get_gnss_telemetry()
    assert telem["current_mode"] == GNSSMode.GNSS_AIDED.value
    assert telem["total_fixes_received"] == 20
    assert telem["total_fixes_accepted"] >= 18
    assert telem["total_fixes_rejected"] <= 2
    assert telem["is_outage"] is False
    assert telem["transition_count"] == 0

    # 2. Continuous trust verification: trust in [0, 1] and valid covariance scaling
    assert all(0.0 <= s <= 1.0 for s in trust_scores)
    assert all(1.0 <= s <= 20.0 for s in r_scales)

    # 3. ML telemetry verification
    ml_telem = core.get_ml_telemetry()
    assert ml_telem["velocitynet"]["inference_executed"] > 0
    assert ml_telem["biasnet"]["inference_executed"] > 0
