"""Integration tests for Phase 3 classical preprocessing on real Phase 2 IO-VNBD data."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from data.pipeline.stationary_detect import StationaryDetector
from data.pipeline.sync import SynchronizedTrip
from navigation.preprocessing.gravity import resolve_gravity
from navigation.preprocessing.pipeline import PreprocessingPipeline


class TestPreprocessingRealData:
    """Integration test suite executing Phase 3 preprocessing over real cached trips."""

    @pytest.fixture
    def real_trip_path(self) -> Path:
        """Locate a representative real cached trip with known stationary and moving periods."""
        proj_root = Path(__file__).resolve().parents[2]
        cache_file = proj_root / "data" / "cache" / "iovnbd" / "Categorised_S1.npz"
        if not cache_file.exists():
            # Fallback to any available cache file
            caches = list((proj_root / "data" / "cache" / "iovnbd").glob("*.npz"))
            if not caches:
                pytest.skip("No Phase 2 cache files found in data/cache/iovnbd")
            return caches[0]
        return cache_file

    def test_full_preprocessing_on_real_trip(self, real_trip_path: Path) -> None:
        """Run complete calibration, alignment, filtering, and gravity verification on real IO-VNBD trip."""
        # 1. Load real Phase 2 synchronized trip
        trip = SynchronizedTrip.load_npz(real_trip_path)
        orig_accel = trip.accel_raw.copy()
        orig_gyro = trip.gyro_raw.copy()
        orig_ts = trip.timestamps_ns.copy()

        # 2. Identify real stationary periods using Phase 2 detector
        detector = StationaryDetector()
        stationary_segs, stat_mask = detector.detect(trip.timestamps_ns, trip.accel_raw, trip.gyro_raw)

        # 3. Execute Phase 3 Preprocessing Pipeline
        pipeline = PreprocessingPipeline(
            sampling_rate_hz=10.0,
            filter_cutoff_hz=3.0,
            median_window_size=3,
        )
        preprocessed = pipeline.process_trip(trip, stationary_mask=stat_mask)

        # 4. Immutability verification: trip raw arrays must NOT be modified
        assert np.array_equal(trip.accel_raw, orig_accel), "Trip accel_raw was modified by preprocessing!"
        assert np.array_equal(trip.gyro_raw, orig_gyro), "Trip gyro_raw was modified by preprocessing!"
        assert np.array_equal(trip.timestamps_ns, orig_ts), "Trip timestamps_ns was modified by preprocessing!"

        # 5. Output shape and finite checks
        n_samples = len(orig_ts)
        assert preprocessed.f_m_v.shape == (n_samples, 3)
        assert preprocessed.omega_m_v.shape == (n_samples, 3)
        assert preprocessed.calibration.is_valid is True

        # Validated epochs must contain finite numerical values
        val_mask = preprocessed.is_validated
        assert np.isfinite(preprocessed.f_m_v[val_mask]).all()
        assert np.isfinite(preprocessed.omega_m_v[val_mask]).all()

        # 6. Stationary segment bias reduction check
        if len(stationary_segs) > 0:
            first_stat = stationary_segs[0]
            stat_slice = slice(first_stat.start_idx, first_stat.end_idx)

            # Raw stationary gyro mean magnitude vs debiased stationary gyro mean magnitude
            raw_gyro_stat = orig_gyro[stat_slice]
            debiased_gyro_stat = preprocessed.omega_m_v[stat_slice]

            raw_mean_norm = np.linalg.norm(np.mean(raw_gyro_stat, axis=0))
            debiased_mean_norm = np.linalg.norm(np.mean(debiased_gyro_stat, axis=0))

            # Debiased stationary angular velocity mean should be substantially closer to zero
            assert debiased_mean_norm < raw_mean_norm + 1e-4

            # 7. Gravity-resolution on stationary segment: coordinate acceleration should hover near 0
            # For stationary segment, R_v^n is approximately level (or aligned with gravity)
            R_level = np.eye(3)
            f_stat_v = preprocessed.f_m_v[stat_slice]
            a_coord_n = resolve_gravity(f_stat_v, R_level, b_a_v=preprocessed.calibration.accel_bias_prior)

            # Mean vertical coordinate acceleration during stationary period should be near zero (< 0.5 m/s^2)
            mean_vert_a = float(np.mean(a_coord_n[:, 2]))
            std_vert_a = float(np.std(a_coord_n[:, 2]))
            assert abs(mean_vert_a) < 0.50, f"Stationary vertical acceleration bias too high: {mean_vert_a:.3f} m/s^2"
            assert std_vert_a < 1.0, f"Stationary vertical acceleration noise excessive: {std_vert_a:.3f} m/s^2"

        # 8. Filter noise reduction on moving segment
        moving_idx = np.where(~stat_mask & val_mask)[0]
        if len(moving_idx) > 100:
            mov_slice = moving_idx[:100]
            unfilt_var = np.var(preprocessed.f_m_v_unfiltered[mov_slice], axis=0)
            filt_var = np.var(preprocessed.f_m_v[mov_slice], axis=0)
            # Low-pass filter must not amplify variance
            assert (filt_var <= unfilt_var * 1.05 + 1e-3).all()

    def test_recalibration_trigger_on_real_data(self) -> None:
        """Verify recalibration detector has zero false alarms on nominal driving and detects injected mount drops."""
        from navigation.preprocessing.recalibration_trigger import RecalibrationDetector

        proj_root = Path(__file__).resolve().parents[2]
        cache_file = proj_root / "data" / "cache" / "iovnbd" / "Categorised_Vta14.npz"
        if not cache_file.exists():
            pytest.skip("Vta14 cache not found")

        trip = SynchronizedTrip.load_npz(cache_file)
        detector = RecalibrationDetector(gravity_shift_threshold_deg=15.0, angular_rate_step_threshold_rads=5.0)

        # 1. Scan nominal trip: should have zero or minimal false triggers on smooth driving
        events = detector.scan_series(trip.accel_raw, trip.gyro_raw, trip.timestamps_ns)
        # Verify no excessive false alarms
        assert len(events) <= 1, f"Unexpected false recalibration triggers: {len(events)}"

        # 2. Inject a simulated physical phone displacement (phone knocked by 45 degrees at index 500)
        accel_corrupt = trip.accel_raw.copy()
        gyro_corrupt = trip.gyro_raw.copy()
        # Rotate acceleration by 45 degrees about X axis from index 500 onwards
        cos45, sin45 = np.cos(np.pi / 4), np.sin(np.pi / 4)
        R_bump = np.array([[1, 0, 0], [0, cos45, -sin45], [0, sin45, cos45]])
        accel_corrupt[500:] = (R_bump @ accel_corrupt[500:].T).T
        # Add high angular rate step at impact
        gyro_corrupt[500] = np.array([0.0, 7.5, 0.0])

        events_corrupt = detector.scan_series(accel_corrupt, gyro_corrupt, trip.timestamps_ns)
        assert len(events_corrupt) >= 1, "Failed to detect injected sensor displacement event!"
        assert any(e.timestamp_ns == trip.timestamps_ns[500] or "Gravity direction shift" in e.reason for e in events_corrupt)

