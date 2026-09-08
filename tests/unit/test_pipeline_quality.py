"""Unit tests for COMPASS offline data pipeline quality evaluation.

Verifies:
1. NaN detection (FLAG_NAN_OR_NONFINITE)
2. Inf detection (FLAG_NAN_OR_NONFINITE)
3. Invalid timestamp detection (FLAG_INVALID_TIMESTAMP)
4. Non-monotonic timestamp detection (FLAG_NON_MONOTONIC_TIMESTAMP)
5. Duplicate timestamp detection (FLAG_DUPLICATE_TIMESTAMP)
6. Vector extreme acceleration detection (||f|| > 4g)
7. Vector extreme angular rate detection (||omega|| > 10 rad/s)
8. Sensor dropout detection (delta_t > 3 * nominal_delta_t)
9. Non-destructive policy: Extreme motion is preserved in validated stream
10. Validated stream policy: Non-computable records are excluded
11. Stationary detector strictly requires BOTH accel and gyro low variance
12. Parser generates canonical Phase 1 schemas (RawIMUSample, GNSSSample)
13. Deterministic pipeline evaluation across repeated executions
"""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from data.pipeline.manifest import (
    DatasetDiscoveryAudit,
    DatasetManifestBuilder,
    ManifestRow,
    audit_dataset_discovery,
    compute_downstream_readiness,
    derive_driver_map_from_categorised,
)
from data.pipeline.outage_index import OutageIndex, OutageWindow
from data.pipeline.parse import ParsedSTrip, ParsedVTrip, parse_s_file, parse_v_file
from data.pipeline.quality_tagger import (
    EXTREME_ACCEL_THRESHOLD_MS2,
    EXTREME_GYRO_THRESHOLD_RADS,
    NOMINAL_DT_NS,
    QualityTagger,
    tag_imu_samples,
)
from data.pipeline.stationary_detect import StationaryDetector
from data.pipeline.sync import (
    MAX_INTERPOLATION_GAP_NS,
    SyncDiagnostics,
    SyncMode,
    SyncPolicy,
    SyncValidationError,
    SynchronizedTrip,
    interpolate_categorical,
    interpolate_circular_deg,
    interpolate_continuous_1d,
    synchronize_s_v,
)
from navigation.schemas.gnss import GNSSSample
from navigation.schemas.imu import (
    FLAG_DUPLICATE_TIMESTAMP,
    FLAG_EXTREME_MOTION,
    FLAG_INVALID_TIMESTAMP,
    FLAG_NAN_OR_NONFINITE,
    FLAG_NON_MONOTONIC_TIMESTAMP,
    FLAG_OK,
    FLAG_SENSOR_DROPOUT,
    RawIMUSample,
    SensorSource,
)


class TestQualityTagger:
    """Unit tests for QualityTagger non-destructive evaluation."""

    def setup_method(self) -> None:
        self.tagger = QualityTagger(nominal_dt_ns=NOMINAL_DT_NS)

    def test_nan_detection(self) -> None:
        t = np.array([100_000_000, 200_000_000, 300_000_000], dtype=np.int64)
        accel = np.array([[0.0, 0.0, 9.81], [np.nan, 0.0, 9.81], [0.0, 0.0, 9.81]])
        gyro = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

        report = self.tagger.evaluate_arrays(t, accel, gyro)
        assert report.nan_or_nonfinite_count == 1
        assert report.quality_flags[1] & FLAG_NAN_OR_NONFINITE
        assert not report.is_validated[1]
        assert report.is_validated[0]
        assert report.is_validated[2]

    def test_inf_detection(self) -> None:
        t = np.array([100_000_000, 200_000_000], dtype=np.int64)
        accel = np.array([[0.0, 0.0, 9.81], [0.0, 0.0, 9.81]])
        gyro = np.array([[0.0, 0.0, 0.0], [0.0, np.inf, 0.0]])

        report = self.tagger.evaluate_arrays(t, accel, gyro)
        assert report.nan_or_nonfinite_count == 1
        assert report.quality_flags[1] & FLAG_NAN_OR_NONFINITE
        assert not report.is_validated[1]

    def test_invalid_timestamp_detection(self) -> None:
        t = np.array([-100, 0, 100_000_000], dtype=np.int64)
        accel = np.zeros((3, 3))
        accel[:, 2] = 9.81
        gyro = np.zeros((3, 3))

        report = self.tagger.evaluate_arrays(t, accel, gyro)
        # Negative timestamp is invalid; 0 is valid session origin
        assert report.invalid_timestamp_count == 1
        assert report.quality_flags[0] & FLAG_INVALID_TIMESTAMP
        assert not (report.quality_flags[1] & FLAG_INVALID_TIMESTAMP)
        assert not report.is_validated[0]
        assert report.is_validated[1]
        assert report.is_validated[2]

    def test_overlapping_flags(self) -> None:
        """One sample triggering multiple quality flags (e.g. extreme motion + dropout)."""
        t = np.array([100_000_000, 500_000_000], dtype=np.int64)  # delta_t = 400ms > 300ms -> dropout
        accel = np.array([[0.0, 0.0, 9.81], [50.0, 0.0, 0.0]])    # sample 1: ||f|| = 50 > 39.24 -> extreme motion
        gyro = np.array([[0.0, 0.0, 0.0], [0.0, 15.0, 0.0]])     # sample 1: ||omega|| = 15 > 10 -> extreme motion

        report = self.tagger.evaluate_arrays(t, accel, gyro)
        assert report.quality_flags[1] & FLAG_EXTREME_MOTION
        assert report.quality_flags[1] & FLAG_SENSOR_DROPOUT
        assert report.extreme_motion_count == 1
        assert report.dropout_count == 1
        assert report.is_validated[1]

    def test_non_monotonic_timestamp_detection(self) -> None:
        t = np.array([100_000_000, 300_000_000, 200_000_000, 400_000_000], dtype=np.int64)
        accel = np.zeros((4, 3))
        accel[:, 2] = 9.81
        gyro = np.zeros((4, 3))

        report = self.tagger.evaluate_arrays(t, accel, gyro)
        assert report.non_monotonic_count == 1
        assert report.quality_flags[2] & FLAG_NON_MONOTONIC_TIMESTAMP
        assert not report.is_validated[2]
        assert report.is_validated[0]
        assert report.is_validated[1]
        assert report.is_validated[3]

    def test_duplicate_timestamp_detection(self) -> None:
        t = np.array([100_000_000, 200_000_000, 200_000_000, 300_000_000], dtype=np.int64)
        accel = np.zeros((4, 3))
        accel[:, 2] = 9.81
        gyro = np.zeros((4, 3))

        report = self.tagger.evaluate_arrays(t, accel, gyro)
        assert report.duplicate_count == 1
        # Index 1 is kept, index 2 is flagged duplicate and omitted
        assert report.quality_flags[1] == FLAG_OK
        assert report.quality_flags[2] & FLAG_DUPLICATE_TIMESTAMP
        assert report.is_validated[1]
        assert not report.is_validated[2]

    def test_vector_extreme_accel_detection(self) -> None:
        t = np.array([100_000_000, 200_000_000], dtype=np.int64)
        # 30^2 + 30^2 = 1800, sqrt = 42.42 m/s^2 > 39.24 m/s^2 (4g)
        accel = np.array([[0.0, 0.0, 9.81], [30.0, 30.0, 0.0]])
        gyro = np.zeros((2, 3))

        report = self.tagger.evaluate_arrays(t, accel, gyro)
        assert report.extreme_accel_count == 1
        assert report.quality_flags[1] & FLAG_EXTREME_MOTION

    def test_vector_extreme_gyro_detection(self) -> None:
        t = np.array([100_000_000, 200_000_000], dtype=np.int64)
        accel = np.zeros((2, 3))
        accel[:, 2] = 9.81
        # 8^2 + 8^2 = 128, sqrt = 11.31 rad/s > 10.0 rad/s
        gyro = np.array([[0.0, 0.0, 0.0], [8.0, 8.0, 0.0]])

        report = self.tagger.evaluate_arrays(t, accel, gyro)
        assert report.extreme_gyro_count == 1
        assert report.quality_flags[1] & FLAG_EXTREME_MOTION

    def test_extreme_motion_is_preserved_in_validated_stream(self) -> None:
        """CRITICAL NON-DESTRUCTIVE RULE: Extreme motion represents real physical vehicle dynamics

        (potholes, curb impact, sudden braking). It must NEVER be dropped.
        """
        t = np.array([100_000_000, 200_000_000], dtype=np.int64)
        accel = np.array([[0.0, 0.0, 9.81], [45.0, 0.0, 0.0]])  # > 4g
        gyro = np.array([[0.0, 0.0, 0.0], [12.0, 0.0, 0.0]])   # > 10 rad/s

        report = self.tagger.evaluate_arrays(t, accel, gyro)
        assert report.quality_flags[1] & FLAG_EXTREME_MOTION
        # Must be KEPT in validated stream
        assert bool(report.is_validated[1]) is True
        assert report.validated_rows == 2
        assert report.omitted_rows == 0

    def test_dropout_detection(self) -> None:
        # Nominal is 100ms. Gap of 350ms > 3 * 100ms
        t = np.array([100_000_000, 200_000_000, 550_000_000], dtype=np.int64)
        accel = np.zeros((3, 3))
        accel[:, 2] = 9.81
        gyro = np.zeros((3, 3))

        report = self.tagger.evaluate_arrays(t, accel, gyro)
        assert report.dropout_count == 1
        assert report.quality_flags[2] & FLAG_SENSOR_DROPOUT
        # Dropouts are timing events; records are kept in validated stream
        assert bool(report.is_validated[2]) is True

    def test_evaluate_schema_samples(self) -> None:
        samples = [
            RawIMUSample(timestamp_ns=100_000_000, accel=(0.0, 0.0, 9.81), gyro=(0.0, 0.0, 0.0)),
            RawIMUSample(timestamp_ns=200_000_000, accel=(50.0, 0.0, 0.0), gyro=(0.0, 0.0, 0.0)),  # Extreme
            RawIMUSample(timestamp_ns=150_000_000, accel=(0.0, 0.0, 9.81), gyro=(0.0, 0.0, 0.0)),  # Non-monotonic
        ]
        tagged, validated, rep = tag_imu_samples(samples)
        assert len(tagged) == 3
        assert len(validated) == 2
        assert tagged[1].quality_flags & FLAG_EXTREME_MOTION
        assert tagged[2].quality_flags & FLAG_NON_MONOTONIC_TIMESTAMP
        assert validated[1].quality_flags & FLAG_EXTREME_MOTION


class TestStationaryDetector:
    """Unit tests for StationaryDetector dual-signal requirement."""

    def setup_method(self) -> None:
        self.detector = StationaryDetector(
            accel_var_threshold=0.05,
            gyro_var_threshold=0.005,
            window_samples=20,
        )

    def test_stationary_requires_both_accel_and_gyro(self) -> None:
        rng = np.random.default_rng(12345)
        n = 50
        t = np.arange(n) * 100_000_000

        # Case 1: Both calm -> stationary
        accel_calm = np.zeros((n, 3))
        accel_calm[:, 2] = 9.81 + rng.normal(0, 0.01, n)
        gyro_calm = rng.normal(0, 0.001, (n, 3))

        segs, mask = self.detector.detect(t, accel_calm, gyro_calm)
        assert len(segs) >= 1
        assert mask.any()

        # Case 2: Accel noisy, gyro calm -> NOT stationary
        accel_noisy = np.zeros((n, 3))
        accel_noisy[:, 2] = 9.81 + rng.normal(0, 1.0, n)  # High variance

        segs_a, mask_a = self.detector.detect(t, accel_noisy, gyro_calm)
        assert len(segs_a) == 0
        assert not mask_a.any()

        # Case 3: Accel calm, gyro noisy (e.g. steering or rotating vehicle) -> NOT stationary
        gyro_noisy = rng.normal(0, 0.5, (n, 3))  # High variance
        segs_g, mask_g = self.detector.detect(t, accel_calm, gyro_noisy)
        assert len(segs_g) == 0
        assert not mask_g.any()


class TestParserSynthetic:
    """Tests synthetic parsing into Phase 1 schemas."""

    def test_s_file_parser_schema_generation(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "S-Test.csv"
        # Create minimal IO-VNBD compatible S-CSV
        df = pd.DataFrame({
            "GPS LATITUDE (degrees)": [52.4025, 52.4026],
            "GPS LONGITUDE (degrees)": [-1.5034, -1.5035],
            "GPS ALTITUDE (m)": [110.0, 110.5],
            "GPS SPEED (Kmh)": [36.0, 72.0],  # 10 m/s and 20 m/s
            "GPS ACCURACY (m)": [3, 4],
            "GPS ORIENTATION (°)": [90.0, 95.0],
            "GPS SATELLITES IN RANGE": [10, 11],
            "TIME SINCE START (ms)": [1000, 1100],
            "ACCELEROMETER X (m/s²)": [0.1, 0.2],
            "ACCELEROMETER Y (m/s²)": [-0.1, -0.2],
            "ACCELEROMETER Z (m/s²)": [9.81, 9.80],
            "GYROSCOPE Yaw (rad/s)": [0.01, 0.02],
            "GYROSCOPE Pitch (rad/s)": [-0.01, -0.02],
            "GYROSCOPE Roll (rad/s)": [0.005, 0.006],
        })
        df.to_csv(csv_file, index=False)

        parsed_s = parse_s_file(csv_file)
        assert parsed_s.row_count == 2
        assert parsed_s.timestamps_ns[0] == 1_000_000_000
        assert parsed_s.timestamps_ns[1] == 1_100_000_000
        assert parsed_s.gnss_speed_mps[0] == pytest.approx(10.0)
        assert parsed_s.gnss_speed_mps[1] == pytest.approx(20.0)

        imu_samples = parsed_s.to_imu_samples()
        assert len(imu_samples) == 2
        assert isinstance(imu_samples[0], RawIMUSample)
        assert imu_samples[0].timestamp_ns == 1_000_000_000
        assert imu_samples[0].accel == (0.1, -0.1, 9.81)
        # Roll -> x, Pitch -> y, Yaw -> z
        assert imu_samples[0].gyro == (0.005, -0.01, 0.01)

        gnss_samples = parsed_s.to_gnss_samples()
        assert len(gnss_samples) == 2
        assert isinstance(gnss_samples[0], GNSSSample)
        assert gnss_samples[0].lat == 52.4025
        assert gnss_samples[0].speed == pytest.approx(10.0)

    def test_deterministic_evaluation(self) -> None:
        t = np.arange(100) * 100_000_000
        accel = np.ones((100, 3)) * 9.81
        gyro = np.zeros((100, 3))

        tagger = QualityTagger()
        rep1 = tagger.evaluate_arrays(t, accel, gyro)
        rep2 = tagger.evaluate_arrays(t, accel, gyro)

        assert rep1.total_rows == rep2.total_rows
        assert np.array_equal(rep1.quality_flags, rep2.quality_flags)
        assert np.array_equal(rep1.is_validated, rep2.is_validated)

    def test_negative_timestamp_int64_no_wrap(self, tmp_path: Path) -> None:
        """Verify negative timestamps remain negative in int64 and are tagged invalid without wrapping."""
        csv_file = tmp_path / "S-NegTime.csv"
        df = pd.DataFrame({
            "TIME SINCE START (ms)": [-100.0, 0.0, 100.0],
            "ACCELEROMETER X (m/s²)": [0.0, 0.0, 0.0],
            "ACCELEROMETER Y (m/s²)": [0.0, 0.0, 0.0],
            "ACCELEROMETER Z (m/s²)": [9.81, 9.81, 9.81],
            "GYROSCOPE Yaw (rad/s)": [0.0, 0.0, 0.0],
            "GYROSCOPE Pitch (rad/s)": [0.0, 0.0, 0.0],
            "GYROSCOPE Roll (rad/s)": [0.0, 0.0, 0.0],
        })
        df.to_csv(csv_file, index=False)

        parsed_s = parse_s_file(csv_file)
        # Verify negative timestamp did NOT wrap to large uint64 integer
        assert parsed_s.timestamps_ns[0] == -100_000_000
        assert parsed_s.timestamps_ns[1] == 0
        assert parsed_s.timestamps_ns[2] == 100_000_000

        tagger = QualityTagger()
        report = tagger.evaluate_arrays(parsed_s.timestamps_ns, parsed_s.accel_raw, parsed_s.gyro_raw)
        # -100ms should be tagged as FLAG_INVALID_TIMESTAMP; 0ms is the valid session origin
        assert report.quality_flags[0] & FLAG_INVALID_TIMESTAMP
        assert not (report.quality_flags[1] & FLAG_INVALID_TIMESTAMP)
        assert not report.is_validated[0]
        assert report.is_validated[1]
        assert report.is_validated[2]

    def test_ambiguous_timestamp_column_raises(self, tmp_path: Path) -> None:
        """Verify that ambiguous or missing timestamp column raises KeyError."""
        csv_file = tmp_path / "S-Ambiguous.csv"
        df = pd.DataFrame({
            "TIME (s)": [1.0, 2.0],
            "TIME_EXTRA": [1.0, 2.0],
            "ACCELEROMETER X (m/s²)": [0.0, 0.0],
            "ACCELEROMETER Y (m/s²)": [0.0, 0.0],
            "ACCELEROMETER Z (m/s²)": [9.81, 9.81],
            "GYROSCOPE Yaw (rad/s)": [0.0, 0.0],
            "GYROSCOPE Pitch (rad/s)": [0.0, 0.0],
            "GYROSCOPE Roll (rad/s)": [0.0, 0.0],
        })
        df.to_csv(csv_file, index=False)

        with pytest.raises(KeyError, match="Ambiguous or missing timestamp column"):
            parse_s_file(csv_file)

    def test_malformed_outage_file_raises(self, tmp_path: Path) -> None:
        """Verify that a malformed outage index file raises ValueError rather than silently returning False."""
        bad_file = tmp_path / "bad_outages.csv"
        # Missing required start_timestamp_ns and end_timestamp_ns
        bad_file.write_text("trip_id,wrong_column\nS1,12345\n", encoding="utf-8")

        with pytest.raises(ValueError, match="missing required columns"):
            OutageIndex(bad_file)


class TestSynchronization:
    """Rigorous unit tests for genuine timestamp-based S/V stream synchronization."""

    def test_sync_requires_real_interpolation(self) -> None:
        """Test with intentionally offset timestamps between S and V.

        S target times: 0.00, 0.10, 0.20 s
        V source times: 0.05, 0.15, 0.25 s with speed = [5.0, 15.0, 25.0] m/s
        Row-matching returns V row 1 (15.0) for S row 1.
        Genuine interpolation MUST return 10.0 (halfway between 5.0 and 15.0)!
        S row 0 (0.00s) is outside V range [0.05, 0.25] -> MUST be NaN (no extrapolation).
        S row 2 (0.20s) is halfway between 15.0 and 25.0 -> MUST return 20.0!
        """
        # S stream: 3 samples at 0.00, 0.10, 0.20 s (common time base)
        t_s = np.array([0, 100_000_000, 200_000_000], dtype=np.int64)
        accel_s = np.array([[0.0, 0.0, 9.81], [0.0, 0.0, 9.81], [0.0, 0.0, 9.81]])
        gyro_s = np.zeros((3, 3))
        s_trip = ParsedSTrip(
            file_path=Path("mock_s.csv"),
            row_count=3,
            timestamps_ns=t_s,
            accel_raw=accel_s,
            gyro_raw=gyro_s,
            gnss_lat=np.full(3, np.nan),
            gnss_lon=np.full(3, np.nan),
            gnss_alt=np.zeros(3),
            gnss_speed_mps=np.full(3, np.nan),
            gnss_bearing_deg=np.full(3, np.nan),
            gnss_accuracy_m=np.full(3, np.nan),
            gnss_sat_count=np.full(3, -1, dtype=np.int32),
            has_gnss=False,
        )

        # V stream: 3 samples at 0.05, 0.15, 0.25 s
        t_v = np.array([50_000_000, 150_000_000, 250_000_000], dtype=np.int64)
        v_speed = np.array([5.0, 15.0, 25.0])
        v_trip = ParsedVTrip(
            file_path=Path("mock_v.csv"),
            row_count=3,
            timestamps_ns=t_v,
            lat=np.array([10.0, 11.0, 12.0]),
            lon=np.array([20.0, 21.0, 22.0]),
            alt_m=np.array([100.0, 110.0, 120.0]),
            speed_mps=v_speed,
            heading_deg=np.array([0.0, 10.0, 20.0]),
            yaw_rate_rads=np.zeros(3),
            wheel_speeds_rads=np.zeros((3, 4)),
            can_accel_g=np.zeros((3, 2)),
            steering_angle_deg=np.zeros(3),
            handbrake=np.zeros(3, dtype=np.int32),
            gear=np.array([1, 2, 3], dtype=np.int32),
        )

        synced, report = synchronize_s_v(s_trip, v_trip, trip_id="T1", branch="Test", relative_time=False)

        assert synced.row_count == 3
        # S[0] at 0.00s is before V[0] at 0.05s -> Out of bounds, NO extrapolation
        assert np.isnan(synced.v_ref_speed_mps[0])

        # S[1] at 0.10s is midway between V(0.05)=5.0 and V(0.15)=15.0 -> MUST BE 10.0
        # (Under old row matching, it returned 15.0!)
        assert synced.v_ref_speed_mps[1] == pytest.approx(10.0)

        # S[2] at 0.20s is midway between V(0.15)=15.0 and V(0.25)=25.0 -> MUST BE 20.0
        assert synced.v_ref_speed_mps[2] == pytest.approx(20.0)

    def test_heading_circular_wrap_around_interpolation(self) -> None:
        """Verify heading azimuth circularly interpolates across 359° -> 1° to give 0°, not 180°."""
        t_source = np.array([0, 200_000_000], dtype=np.int64)
        headings = np.array([350.0, 10.0])  # Crossing 0 deg

        t_target = np.array([100_000_000], dtype=np.int64)  # Halfway
        interp_deg = interpolate_circular_deg(t_target, t_source, headings)

        # Midpoint of 350 deg (-10 deg) and +10 deg is 0 deg (or 360 deg)
        assert interp_deg[0] == pytest.approx(0.0, abs=1e-3) or interp_deg[0] == pytest.approx(360.0, abs=1e-3)
        # Linear interpolation would give (350 + 10)/2 = 180.0 deg, which is completely wrong!
        assert interp_deg[0] != pytest.approx(180.0, abs=10.0)

    def test_categorical_gear_never_fractional(self) -> None:
        """Verify discrete categorical state (gear) preserves integer values via nearest neighbor."""
        t_source = np.array([0, 200_000_000], dtype=np.int64)
        gear_source = np.array([1, 2], dtype=np.int32)

        # At t = 50ms, closer to t=0 (gear 1)
        # At t = 100ms, tie/nearest (must be integer 1 or 2, NEVER 1.5)
        # At t = 150ms, closer to t=200ms (gear 2)
        t_target = np.array([50_000_000, 100_000_000, 150_000_000], dtype=np.int64)
        gear_interp = interpolate_categorical(t_target, t_source, gear_source)

        assert gear_interp.dtype == np.int32
        assert gear_interp[0] == 1
        assert gear_interp[1] in (1, 2)
        assert gear_interp[2] == 2

    def test_no_extrapolation_outside_source_range(self) -> None:
        """Verify target timestamps outside [t_min, t_max] receive NaN (continuous) and -1 (categorical)."""
        t_source = np.array([100_000_000, 200_000_000], dtype=np.int64)
        y_source = np.array([10.0, 20.0])
        cat_source = np.array([3, 4], dtype=np.int32)

        t_target = np.array([50_000_000, 150_000_000, 250_000_000], dtype=np.int64)
        y_interp = interpolate_continuous_1d(t_target, t_source, y_source)
        cat_interp = interpolate_categorical(t_target, t_source, cat_source)

        # Before source start
        assert np.isnan(y_interp[0])
        assert cat_interp[0] == -1

        # Inside source range
        assert y_interp[1] == pytest.approx(15.0)
        assert cat_interp[1] in (3, 4)

        # After source end
        assert np.isnan(y_interp[2])
        assert cat_interp[2] == -1

    def test_source_large_gap_rejected(self) -> None:
        """Verify that gaps in source > 1.0s reject interpolation without fabricating data."""
        # 2.5 second gap between 1.0s and 3.5s
        t_source = np.array([1_000_000_000, 3_500_000_000], dtype=np.int64)
        y_source = np.array([10.0, 50.0])

        # Target falls directly inside this 2.5s gap
        t_target = np.array([2_000_000_000], dtype=np.int64)
        y_interp = interpolate_continuous_1d(t_target, t_source, y_source, max_gap_ns=MAX_INTERPOLATION_GAP_NS)

        # Must NOT linearly interpolate across 2.5s gap!
        assert np.isnan(y_interp[0])

    def test_equal_timestamps_match_exactly(self) -> None:
        """When S and V timestamps are identical, interpolated values match source exactly."""
        t_source = np.arange(5, dtype=np.int64) * 100_000_000
        y_source = np.array([1.1, 2.2, 3.3, 4.4, 5.5])

        y_interp = interpolate_continuous_1d(t_source, t_source, y_source)
        assert np.allclose(y_interp, y_source)


class TestParserHardening:
    """Tests for Step 2: Strict required vs optional fields and strict gyro mapping."""

    def test_missing_required_v_column_raises_keyerror(self, tmp_path: Path) -> None:
        """Missing required V column (e.g. Velocity) must raise KeyError identifying column and file."""
        csv_file = tmp_path / "V-MissingVel.csv"
        df = pd.DataFrame({
            "Time Since Start of Day (seconds)": [10.0, 11.0],
            "Latitude (degrees)": [52.1, 52.2],
            "Longitude (degrees)": [-1.1, -1.2],
            # Missing "Velocity (km/hr)"
            "Heading (degrees)": [180.0, 185.0],
            "Yaw Rate (deg/sec)": [0.1, 0.2],
        })
        df.to_csv(csv_file, index=False)

        with pytest.raises(KeyError, match="Velocity"):
            parse_v_file(csv_file)

    def test_missing_optional_v_columns_produce_nan_and_sentinel(self, tmp_path: Path) -> None:
        """Missing optional V columns must NOT be fabricated as zeros (produce NaN / -1)."""
        csv_file = tmp_path / "V-RequiredOnly.csv"
        df = pd.DataFrame({
            "Time Since Start of Day (seconds)": [10.0, 11.0],
            "Latitude (degrees)": [52.1, 52.2],
            "Longitude (degrees)": [-1.1, -1.2],
            "Velocity (km/hr)": [36.0, 72.0],
            "Heading (degrees)": [90.0, 95.0],
            "Yaw Rate (deg/sec)": [1.0, 2.0],
        })
        df.to_csv(csv_file, index=False)

        parsed_v = parse_v_file(csv_file)
        assert parsed_v.row_count == 2
        # Required conversions
        assert parsed_v.speed_mps[0] == pytest.approx(10.0)
        assert parsed_v.speed_mps[1] == pytest.approx(20.0)
        # Optional fields must be NaN or sentinel -1, NOT zeros!
        assert np.isnan(parsed_v.alt_m).all()
        assert np.isnan(parsed_v.wheel_speeds_rads).all()
        assert parsed_v.wheel_speeds_rads.shape == (2, 4)
        assert np.isnan(parsed_v.can_accel_g).all()
        assert parsed_v.can_accel_g.shape == (2, 2)
        assert np.isnan(parsed_v.steering_angle_deg).all()
        assert np.all(parsed_v.gear == -1)
        assert np.all(parsed_v.handbrake == -1)

    def test_missing_optional_s_gnss_alt_produces_nan(self, tmp_path: Path) -> None:
        """Missing optional GPS altitude in S-file must produce NaN, not 0.0."""
        csv_file = tmp_path / "S-NoAlt.csv"
        df = pd.DataFrame({
            "TIME SINCE START (ms)": [100.0, 200.0],
            "ACCELEROMETER X (m/s²)": [0.0, 0.0],
            "ACCELEROMETER Y (m/s²)": [0.0, 0.0],
            "ACCELEROMETER Z (m/s²)": [9.81, 9.81],
            "GYROSCOPE Roll (rad/s)": [0.0, 0.0],
            "GYROSCOPE Pitch (rad/s)": [0.0, 0.0],
            "GYROSCOPE Yaw (rad/s)": [0.0, 0.0],
            "GPS LATITUDE": [52.1, 52.2],
            "GPS LONGITUDE": [-1.1, -1.2],
            # No GPS ALTITUDE
        })
        df.to_csv(csv_file, index=False)

        parsed_s = parse_s_file(csv_file)
        assert np.isnan(parsed_s.gnss_alt).all()

    def test_strict_gyro_mapping_raises_on_generic_unless_opt_in(self, tmp_path: Path) -> None:
        """Strict IO-VNBD gyro mapping requires Roll/Pitch/Yaw unless generic gyro is explicitly allowed."""
        csv_file = tmp_path / "S-GenericGyro.csv"
        df = pd.DataFrame({
            "TIME SINCE START (ms)": [100.0, 200.0],
            "ACCELEROMETER X (m/s²)": [0.0, 0.0],
            "ACCELEROMETER Y (m/s²)": [0.0, 0.0],
            "ACCELEROMETER Z (m/s²)": [9.81, 9.81],
            "GYRO X (rad/s)": [0.01, 0.02],
            "GYRO Y (rad/s)": [0.03, 0.04],
            "GYRO Z (rad/s)": [0.05, 0.06],
        })
        df.to_csv(csv_file, index=False)

        # Default strict mode: must raise KeyError
        with pytest.raises(KeyError, match="gyr"):
            parse_s_file(csv_file, allow_generic_gyro=False)

        # Explicit opt-in: succeeds
        parsed = parse_s_file(csv_file, allow_generic_gyro=True)
        assert parsed.gyro_raw.shape == (2, 3)
        assert parsed.gyro_raw[0, 0] == pytest.approx(0.01)


class TestSyncHardening:
    """Tests for Step 3: Hardened timestamp synchronization & validation policy."""

    def test_sync_mode_relative_elapsed_vs_absolute(self) -> None:
        """Verify explicit SyncMode parameter and diagnostics."""
        t_s = np.array([100_000_000, 200_000_000, 300_000_000], dtype=np.int64)
        t_v = np.array([10_000_000_000, 10_100_000_000, 10_200_000_000], dtype=np.int64)

        s_trip = ParsedSTrip(
            file_path=Path("s.csv"), row_count=3, timestamps_ns=t_s,
            accel_raw=np.zeros((3, 3)), gyro_raw=np.zeros((3, 3)),
            gnss_lat=np.full(3, np.nan), gnss_lon=np.full(3, np.nan), gnss_alt=np.full(3, np.nan),
            gnss_speed_mps=np.full(3, np.nan), gnss_bearing_deg=np.full(3, np.nan),
            gnss_accuracy_m=np.full(3, np.nan), gnss_sat_count=np.full(3, -1, dtype=np.int32), has_gnss=False,
        )
        v_trip = ParsedVTrip(
            file_path=Path("v.csv"), row_count=3, timestamps_ns=t_v,
            lat=np.array([10.0, 10.0, 10.0]), lon=np.array([20.0, 20.0, 20.0]), alt_m=np.full(3, np.nan),
            speed_mps=np.array([10.0, 20.0, 30.0]), heading_deg=np.array([0.0, 0.0, 0.0]), yaw_rate_rads=np.zeros(3),
            wheel_speeds_rads=np.full((3, 4), np.nan), can_accel_g=np.full((3, 2), np.nan),
            steering_angle_deg=np.full(3, np.nan), handbrake=np.full(3, -1, dtype=np.int32), gear=np.full(3, -1, dtype=np.int32),
        )

        # In RELATIVE_ELAPSED mode (IO-VNBD canonical), streams are aligned by elapsed time from origin
        policy_rel = SyncPolicy(mode=SyncMode.RELATIVE_ELAPSED)
        synced_rel, rep_rel = synchronize_s_v(s_trip, v_trip, trip_id="T1", branch="Test", policy=policy_rel)
        assert synced_rel.diagnostics is not None
        assert synced_rel.diagnostics.sync_mode == "relative_elapsed"
        # Middle sample should interpolate properly in relative mode
        assert not np.isnan(synced_rel.v_ref_speed_mps[1])

        # In ABSOLUTE mode, t_s (~0.1s) and t_v (~10s) have no overlap
        policy_abs = SyncPolicy(mode=SyncMode.ABSOLUTE, min_overlap_duration_s=0.0)
        synced_abs, rep_abs = synchronize_s_v(s_trip, v_trip, trip_id="T1", branch="Test", policy=policy_abs)
        assert synced_abs.diagnostics is not None
        assert synced_abs.diagnostics.sync_mode == "absolute"
        assert synced_abs.diagnostics.valid_interpolated_count == 0

    def test_sync_unacceptable_clock_drift_rejected(self) -> None:
        """Excessive clock drift must trigger policy violation."""
        t_s = np.array([0, 10_000_000_000], dtype=np.int64)  # 10.0s duration
        t_v = np.array([0, 5_000_000_000], dtype=np.int64)   # 5.0s duration (drift = 5.0s)

        s_trip = ParsedSTrip(
            file_path=Path("s.csv"), row_count=2, timestamps_ns=t_s,
            accel_raw=np.zeros((2, 3)), gyro_raw=np.zeros((2, 3)),
            gnss_lat=np.full(2, np.nan), gnss_lon=np.full(2, np.nan), gnss_alt=np.full(2, np.nan),
            gnss_speed_mps=np.full(2, np.nan), gnss_bearing_deg=np.full(2, np.nan),
            gnss_accuracy_m=np.full(2, np.nan), gnss_sat_count=np.full(2, -1, dtype=np.int32), has_gnss=False,
        )
        v_trip = ParsedVTrip(
            file_path=Path("v.csv"), row_count=2, timestamps_ns=t_v,
            lat=np.array([10.0, 10.0]), lon=np.array([20.0, 20.0]), alt_m=np.full(2, np.nan),
            speed_mps=np.array([5.0, 5.0]), heading_deg=np.array([0.0, 0.0]), yaw_rate_rads=np.zeros(2),
            wheel_speeds_rads=np.full((2, 4), np.nan), can_accel_g=np.full((2, 2), np.nan),
            steering_angle_deg=np.full(2, np.nan), handbrake=np.full(2, -1, dtype=np.int32), gear=np.full(2, -1, dtype=np.int32),
        )

        strict_policy = SyncPolicy(max_clock_drift_s=2.0, enforce_strict_validation=True)
        with pytest.raises(SyncValidationError, match="Clock drift"):
            synchronize_s_v(s_trip, v_trip, trip_id="T_Drift", branch="Test", policy=strict_policy)

    def test_sync_invalid_source_timestamps_rejected(self) -> None:
        """Non-positive or non-finite source timestamps must raise SyncValidationError."""
        t_s = np.array([100_000_000, 200_000_000], dtype=np.int64)
        t_v = np.array([100_000_000, -1], dtype=np.int64)

        s_trip = ParsedSTrip(
            file_path=Path("s.csv"), row_count=2, timestamps_ns=t_s,
            accel_raw=np.zeros((2, 3)), gyro_raw=np.zeros((2, 3)),
            gnss_lat=np.full(2, np.nan), gnss_lon=np.full(2, np.nan), gnss_alt=np.full(2, np.nan),
            gnss_speed_mps=np.full(2, np.nan), gnss_bearing_deg=np.full(2, np.nan),
            gnss_accuracy_m=np.full(2, np.nan), gnss_sat_count=np.full(2, -1, dtype=np.int32), has_gnss=False,
        )
        v_trip = ParsedVTrip(
            file_path=Path("v.csv"), row_count=2, timestamps_ns=t_v,
            lat=np.array([10.0, 10.0]), lon=np.array([20.0, 20.0]), alt_m=np.full(2, np.nan),
            speed_mps=np.array([5.0, 5.0]), heading_deg=np.array([0.0, 0.0]), yaw_rate_rads=np.zeros(2),
            wheel_speeds_rads=np.full((2, 4), np.nan), can_accel_g=np.full((2, 2), np.nan),
            steering_angle_deg=np.full(2, np.nan), handbrake=np.full(2, -1, dtype=np.int32), gear=np.full(2, -1, dtype=np.int32),
        )

        with pytest.raises(SyncValidationError, match="negative/sentinel timestamp"):
            synchronize_s_v(s_trip, v_trip, trip_id="T_BadV", branch="Test", policy=SyncPolicy(enforce_strict_validation=True))

    def test_sync_invalid_target_timestamps_rejected(self) -> None:
        """Negative or non-finite target timestamps must raise SyncValidationError."""
        t_s = np.array([-1, -2], dtype=np.int64)
        t_v = np.array([100_000_000, 200_000_000], dtype=np.int64)

        s_trip = ParsedSTrip(
            file_path=Path("s.csv"), row_count=2, timestamps_ns=t_s,
            accel_raw=np.zeros((2, 3)), gyro_raw=np.zeros((2, 3)),
            gnss_lat=np.full(2, np.nan), gnss_lon=np.full(2, np.nan), gnss_alt=np.full(2, np.nan),
            gnss_speed_mps=np.full(2, np.nan), gnss_bearing_deg=np.full(2, np.nan),
            gnss_accuracy_m=np.full(2, np.nan), gnss_sat_count=np.full(2, -1, dtype=np.int32), has_gnss=False,
        )
        v_trip = ParsedVTrip(
            file_path=Path("v.csv"), row_count=2, timestamps_ns=t_v,
            lat=np.array([10.0, 10.0]), lon=np.array([20.0, 20.0]), alt_m=np.full(2, np.nan),
            speed_mps=np.array([5.0, 5.0]), heading_deg=np.array([0.0, 0.0]), yaw_rate_rads=np.zeros(2),
            wheel_speeds_rads=np.full((2, 4), np.nan), can_accel_g=np.full((2, 2), np.nan),
            steering_angle_deg=np.full(2, np.nan), handbrake=np.full(2, -1, dtype=np.int32), gear=np.full(2, -1, dtype=np.int32),
        )

        with pytest.raises(SyncValidationError, match="negative/sentinel timestamp"):
            synchronize_s_v(s_trip, v_trip, trip_id="T_BadS", branch="Test", policy=SyncPolicy(enforce_strict_validation=True))

    def test_sync_duplicate_source_timestamps_handled_deterministically(self) -> None:
        """Duplicate source timestamps in V are deduplicated, keeping the first occurrence."""
        t_s = np.array([100_000_000, 200_000_000, 300_000_000], dtype=np.int64)
        # V has duplicate timestamp at 200ms
        t_v = np.array([100_000_000, 200_000_000, 200_000_000, 300_000_000], dtype=np.int64)
        v_speed = np.array([10.0, 20.0, 99.0, 30.0])

        s_trip = ParsedSTrip(
            file_path=Path("s.csv"), row_count=3, timestamps_ns=t_s,
            accel_raw=np.zeros((3, 3)), gyro_raw=np.zeros((3, 3)),
            gnss_lat=np.full(3, np.nan), gnss_lon=np.full(3, np.nan), gnss_alt=np.full(3, np.nan),
            gnss_speed_mps=np.full(3, np.nan), gnss_bearing_deg=np.full(3, np.nan),
            gnss_accuracy_m=np.full(3, np.nan), gnss_sat_count=np.full(3, -1, dtype=np.int32), has_gnss=False,
        )
        v_trip = ParsedVTrip(
            file_path=Path("v.csv"), row_count=4, timestamps_ns=t_v,
            lat=np.array([1.0, 2.0, 2.0, 3.0]), lon=np.array([1.0, 2.0, 2.0, 3.0]), alt_m=np.full(4, np.nan),
            speed_mps=v_speed, heading_deg=np.array([0.0, 0.0, 0.0, 0.0]), yaw_rate_rads=np.zeros(4),
            wheel_speeds_rads=np.full((4, 4), np.nan), can_accel_g=np.full((4, 2), np.nan),
            steering_angle_deg=np.full(4, np.nan), handbrake=np.full(4, -1, dtype=np.int32), gear=np.full(4, -1, dtype=np.int32),
        )

        synced, _ = synchronize_s_v(s_trip, v_trip, trip_id="T_Dup", branch="Test")
        # At 200ms, first sample speed 20.0 must be used, not 99.0
        assert synced.v_ref_speed_mps[1] == pytest.approx(20.0)

    def test_sync_preserves_raw_timestamps_and_unwraps_working_timestamps(self, tmp_path: Path) -> None:
        """SynchronizedTrip preserves raw timestamps; working axis is non-decreasing; validated downstream samples are strictly increasing."""
        # Raw S has counter reset (3000ms -> 0ms) AND a duplicate timestamp (100ms -> 100ms)
        t_s = np.array([1_000_000_000, 2_000_000_000, 3_000_000_000, 0, 100_000_000, 100_000_000, 200_000_000], dtype=np.int64)
        t_v = np.array([0, 1_000_000_000, 2_000_000_000, 3_000_000_000, 4_000_000_000, 5_000_000_000, 6_000_000_000], dtype=np.int64)

        s_trip = ParsedSTrip(
            file_path=Path("s.csv"), row_count=7, timestamps_ns=t_s,
            accel_raw=np.zeros((7, 3)), gyro_raw=np.zeros((7, 3)),
            gnss_lat=np.full(7, np.nan), gnss_lon=np.full(7, np.nan), gnss_alt=np.full(7, np.nan),
            gnss_speed_mps=np.full(7, np.nan), gnss_bearing_deg=np.full(7, np.nan),
            gnss_accuracy_m=np.full(7, np.nan), gnss_sat_count=np.full(7, -1, dtype=np.int32), has_gnss=False,
        )
        v_trip = ParsedVTrip(
            file_path=Path("v.csv"), row_count=7, timestamps_ns=t_v,
            lat=np.zeros(7), lon=np.zeros(7), alt_m=np.full(7, np.nan),
            speed_mps=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]), heading_deg=np.zeros(7), yaw_rate_rads=np.zeros(7),
            wheel_speeds_rads=np.full((7, 4), np.nan), can_accel_g=np.full((7, 2), np.nan),
            steering_angle_deg=np.full(7, np.nan), handbrake=np.full(7, -1, dtype=np.int32), gear=np.full(7, -1, dtype=np.int32),
        )

        synced, _ = synchronize_s_v(s_trip, v_trip, trip_id="T_Reset", branch="Test")
        # 1. Raw timestamps must be preserved unmodified
        assert np.array_equal(synced.raw_timestamps_ns, t_s)
        # 2. Raw working timestamps are non-decreasing across all rows (diffs >= 0)
        raw_working_diffs = np.diff(synced.timestamps_ns)
        assert (raw_working_diffs >= 0).all(), f"Working axis must be non-decreasing: diffs = {raw_working_diffs}"
        # 3. Validated downstream samples (excluding duplicates/invalid rows) are strictly increasing (diffs > 0)
        validated_ts = synced.timestamps_ns[synced.is_validated]
        validated_diffs = np.diff(validated_ts)
        assert len(validated_ts) > 0
        assert (validated_diffs > 0).all(), f"Validated downstream timestamps must be strictly increasing: diffs = {validated_diffs}"

        # 4. Cache round-trip must preserve both raw and working timestamps
        cache_file = tmp_path / "synced_test.npz"
        synced.save_npz(cache_file)
        loaded = SynchronizedTrip.load_npz(cache_file)
        assert np.array_equal(loaded.raw_timestamps_ns, t_s)
        assert np.array_equal(loaded.timestamps_ns, synced.timestamps_ns)

    def test_sync_v_duplicate_timestamps_deduplicated_before_unwrapping(self) -> None:
        """V duplicate timestamps must be deduplicated before timeline construction, keeping first occurrence."""
        t_s = np.array([0, 100_000_000, 200_000_000], dtype=np.int64)
        # V has duplicate at 100ms with conflicting speeds
        t_v = np.array([0, 100_000_000, 100_000_000, 200_000_000], dtype=np.int64)
        v_speed = np.array([10.0, 25.0, 999.0, 40.0])

        s_trip = ParsedSTrip(
            file_path=Path("s.csv"), row_count=3, timestamps_ns=t_s,
            accel_raw=np.zeros((3, 3)), gyro_raw=np.zeros((3, 3)),
            gnss_lat=np.full(3, np.nan), gnss_lon=np.full(3, np.nan), gnss_alt=np.full(3, np.nan),
            gnss_speed_mps=np.full(3, np.nan), gnss_bearing_deg=np.full(3, np.nan),
            gnss_accuracy_m=np.full(3, np.nan), gnss_sat_count=np.full(3, -1, dtype=np.int32), has_gnss=False,
        )
        v_trip = ParsedVTrip(
            file_path=Path("v.csv"), row_count=4, timestamps_ns=t_v,
            lat=np.array([1.0, 2.0, 2.0, 3.0]), lon=np.array([1.0, 2.0, 2.0, 3.0]), alt_m=np.full(4, np.nan),
            speed_mps=v_speed, heading_deg=np.zeros(4), yaw_rate_rads=np.zeros(4),
            wheel_speeds_rads=np.full((4, 4), np.nan), can_accel_g=np.full((4, 2), np.nan),
            steering_angle_deg=np.full(4, np.nan), handbrake=np.full(4, -1, dtype=np.int32), gear=np.full(4, -1, dtype=np.int32),
        )

        synced, _ = synchronize_s_v(s_trip, v_trip, trip_id="T_DupV", branch="Test")
        # Target sample 1 is at 100ms. Must interpolate to exactly 25.0, NOT 999.0 and NOT shifted forward
        assert synced.v_ref_speed_mps[1] == pytest.approx(25.0)

    def test_sync_v_duplicate_timestamps_regression_conflicting_values_not_shifted(self) -> None:
        """Regression: duplicate V timestamps must not be unwrapped/shifted into subsequent time slots.
        
        If unwrapping was run before deduplication, the duplicate at 100ms would be shifted to 200ms
        with speed 999.0, corrupting the midpoint at 150ms and the endpoint at 200ms.
        Deduplicating BEFORE timeline construction keeps (100ms, 20.0), discards (100ms, 999.0),
        and leaves (200ms, 30.0) correctly aligned so that midpoint 150ms interpolates to exactly 25.0.
        """
        t_s = np.array([0, 100_000_000, 150_000_000, 200_000_000], dtype=np.int64)
        t_v = np.array([0, 100_000_000, 100_000_000, 200_000_000], dtype=np.int64)
        v_speed = np.array([10.0, 20.0, 999.0, 30.0])

        s_trip = ParsedSTrip(
            file_path=Path("s.csv"), row_count=4, timestamps_ns=t_s,
            accel_raw=np.zeros((4, 3)), gyro_raw=np.zeros((4, 3)),
            gnss_lat=np.full(4, np.nan), gnss_lon=np.full(4, np.nan), gnss_alt=np.full(4, np.nan),
            gnss_speed_mps=np.full(4, np.nan), gnss_bearing_deg=np.full(4, np.nan),
            gnss_accuracy_m=np.full(4, np.nan), gnss_sat_count=np.full(4, -1, dtype=np.int32), has_gnss=False,
        )
        v_trip = ParsedVTrip(
            file_path=Path("v.csv"), row_count=4, timestamps_ns=t_v,
            lat=np.zeros(4), lon=np.zeros(4), alt_m=np.full(4, np.nan),
            speed_mps=v_speed, heading_deg=np.zeros(4), yaw_rate_rads=np.zeros(4),
            wheel_speeds_rads=np.full((4, 4), np.nan), can_accel_g=np.full((4, 2), np.nan),
            steering_angle_deg=np.full(4, np.nan), handbrake=np.full(4, -1, dtype=np.int32), gear=np.full(4, -1, dtype=np.int32),
        )

        synced, _ = synchronize_s_v(s_trip, v_trip, trip_id="T_DupV_Regress", branch="Test")
        # Target 0 (0ms) -> 10.0
        assert synced.v_ref_speed_mps[0] == pytest.approx(10.0)
        # Target 1 (100ms) -> 20.0 (first occurrence preserved, 999.0 rejected)
        assert synced.v_ref_speed_mps[1] == pytest.approx(20.0)
        # Target 2 (150ms) -> 25.0 (linear interpolation between 20.0 and 30.0; would be ~509.5 if unwrapped first)
        assert synced.v_ref_speed_mps[2] == pytest.approx(25.0)
        # Target 3 (200ms) -> 30.0 (would be 999.0 if duplicate was shifted forward into 200ms slot)
        assert synced.v_ref_speed_mps[3] == pytest.approx(30.0)

    def test_sync_v_backward_timestamps_rejected(self) -> None:
        """Genuine backward timestamps in V must raise SyncValidationError."""
        t_s = np.array([0, 100_000_000, 200_000_000], dtype=np.int64)
        t_v = np.array([0, 200_000_000, 100_000_000, 300_000_000], dtype=np.int64)  # Jumps back 200ms -> 100ms

        s_trip = ParsedSTrip(
            file_path=Path("s.csv"), row_count=3, timestamps_ns=t_s,
            accel_raw=np.zeros((3, 3)), gyro_raw=np.zeros((3, 3)),
            gnss_lat=np.full(3, np.nan), gnss_lon=np.full(3, np.nan), gnss_alt=np.full(3, np.nan),
            gnss_speed_mps=np.full(3, np.nan), gnss_bearing_deg=np.full(3, np.nan),
            gnss_accuracy_m=np.full(3, np.nan), gnss_sat_count=np.full(3, -1, dtype=np.int32), has_gnss=False,
        )
        v_trip = ParsedVTrip(
            file_path=Path("v.csv"), row_count=4, timestamps_ns=t_v,
            lat=np.zeros(4), lon=np.zeros(4), alt_m=np.full(4, np.nan),
            speed_mps=np.array([1.0, 2.0, 3.0, 4.0]), heading_deg=np.zeros(4), yaw_rate_rads=np.zeros(4),
            wheel_speeds_rads=np.full((4, 4), np.nan), can_accel_g=np.full((4, 2), np.nan),
            steering_angle_deg=np.full(4, np.nan), handbrake=np.full(4, -1, dtype=np.int32), gear=np.full(4, -1, dtype=np.int32),
        )

        with pytest.raises(SyncValidationError, match="non-monotonic/backward"):
            synchronize_s_v(s_trip, v_trip, trip_id="T_V_Back", branch="Test")

    def test_cache_diagnostics_roundtrip_exact(self, tmp_path: Path) -> None:
        """All SyncDiagnostics fields must survive save_npz -> load_npz identically."""
        t_s = np.array([0, 100_000_000, 200_000_000], dtype=np.int64)
        t_v = np.array([10_000_000_000, 10_100_000_000, 10_200_000_000], dtype=np.int64)

        s_trip = ParsedSTrip(
            file_path=Path("s.csv"), row_count=3, timestamps_ns=t_s,
            accel_raw=np.zeros((3, 3)), gyro_raw=np.zeros((3, 3)),
            gnss_lat=np.full(3, np.nan), gnss_lon=np.full(3, np.nan), gnss_alt=np.full(3, np.nan),
            gnss_speed_mps=np.full(3, np.nan), gnss_bearing_deg=np.full(3, np.nan),
            gnss_accuracy_m=np.full(3, np.nan), gnss_sat_count=np.full(3, -1, dtype=np.int32), has_gnss=False,
        )
        v_trip = ParsedVTrip(
            file_path=Path("v.csv"), row_count=3, timestamps_ns=t_v,
            lat=np.zeros(3), lon=np.zeros(3), alt_m=np.full(3, np.nan),
            speed_mps=np.array([5.0, 5.0, 5.0]), heading_deg=np.zeros(3), yaw_rate_rads=np.zeros(3),
            wheel_speeds_rads=np.full((3, 4), np.nan), can_accel_g=np.full((3, 2), np.nan),
            steering_angle_deg=np.full(3, np.nan), handbrake=np.full(3, -1, dtype=np.int32), gear=np.full(3, -1, dtype=np.int32),
        )

        synced, _ = synchronize_s_v(s_trip, v_trip, trip_id="T_Cache", branch="Test")
        assert synced.diagnostics is not None

        cache_file = tmp_path / "test_diag.npz"
        synced.save_npz(cache_file)
        loaded = SynchronizedTrip.load_npz(cache_file)

        assert loaded.diagnostics is not None
        assert loaded.diagnostics.clock_origin_offset_s == synced.diagnostics.clock_origin_offset_s
        assert loaded.diagnostics.relative_elapsed_offset_s == synced.diagnostics.relative_elapsed_offset_s
        assert loaded.diagnostics.clock_drift_s == pytest.approx(synced.diagnostics.clock_drift_s)
        assert loaded.diagnostics.overlap_duration_s == pytest.approx(synced.diagnostics.overlap_duration_s)
        assert loaded.diagnostics.source_sample_count == synced.diagnostics.source_sample_count
        assert loaded.diagnostics.target_sample_count == synced.diagnostics.target_sample_count
        assert loaded.diagnostics.valid_interpolated_count == synced.diagnostics.valid_interpolated_count
        assert loaded.diagnostics.out_of_bounds_count == synced.diagnostics.out_of_bounds_count
        assert loaded.diagnostics.gap_violation_count == synced.diagnostics.gap_violation_count
        assert loaded.diagnostics.valid_coverage_ratio == pytest.approx(synced.diagnostics.valid_coverage_ratio)
        assert loaded.diagnostics.sync_mode == synced.diagnostics.sync_mode
        assert loaded.diagnostics.status == synced.diagnostics.status
        assert loaded.diagnostics.sync_passed == synced.diagnostics.sync_passed


class TestStationaryDetectorHardening:
    """Tests for Step 5: Input validation and finite-data handling in StationaryDetector."""

    def test_stationary_detector_shape_mismatch_raises(self) -> None:
        """Shape mismatches must raise ValueError."""
        detector = StationaryDetector(window_samples=5)
        t = np.arange(10) * 100_000_000
        bad_accel = np.zeros((8, 3))  # Length 8 vs 10
        gyro = np.zeros((10, 3))

        with pytest.raises(ValueError, match="shape mismatch"):
            detector.detect(t, bad_accel, gyro)

    def test_stationary_detector_nan_inf_does_not_create_false_stationary(self) -> None:
        """Windows containing NaNs or Infs must NEVER be detected as stationary."""
        detector = StationaryDetector(window_samples=5)
        t = np.arange(20) * 100_000_000
        # Constant zero force except a NaN in the middle
        accel = np.zeros((20, 3))
        accel[:, 2] = 9.81
        accel[8, 0] = np.nan
        gyro = np.zeros((20, 3))

        segs, mask = detector.detect(t, accel, gyro)
        # Sample index 8 must NOT be stationary
        assert not mask[8]


class TestOutageIndexHardening:
    """Tests for Step 11: Outage CSV validation and normalization."""

    def test_outage_invalid_time_range_raises(self, tmp_path: Path) -> None:
        """Outage CSV where start_timestamp >= end_timestamp must raise ValueError."""
        csv_file = tmp_path / "bad_range.csv"
        csv_file.write_text("trip_id,start_timestamp_ns,end_timestamp_ns\nS1,5000,4000\n", encoding="utf-8")

        with pytest.raises(ValueError, match="Invalid outage time range"):
            OutageIndex(csv_file)

    def test_outage_trip_id_normalization(self, tmp_path: Path) -> None:
        """Trip ID lookups must be normalized (case-insensitive, whitespace-trimmed)."""
        csv_file = tmp_path / "norm_outages.csv"
        csv_file.write_text("trip_id,start_timestamp_ns,end_timestamp_ns,label\n s3b ,1000,2000,test\n", encoding="utf-8")

        idx = OutageIndex(csv_file)
        assert idx.has_real_outages_for_trip("S3B")
        assert idx.has_real_outages_for_trip("s3b")
        assert len(idx.get_outages("S3b")) == 1


class TestManifestHardening:
    """Tests for Step 6 & 7: Driver mapping and sync diagnostics in manifest."""

    def test_driver_derivation_from_categorised_path(self, tmp_path: Path) -> None:
        """Driver ID must be derived deterministically from Categorised directory structure."""
        cat_dir = tmp_path / "Synchronised V abd S datasets" / "Categorised IOVNB Dataset" / "M (Driver B)"
        cat_dir.mkdir(parents=True)
        (cat_dir / "S-M.csv").write_text("dummy", encoding="utf-8")

        driver_map = derive_driver_map_from_categorised(tmp_path)
        assert "m" in driver_map
        assert driver_map["m"] == "Driver B"

    def test_manifest_row_contains_sync_diagnostics(self) -> None:
        """ManifestRow must contain all required synchronization audit fields."""
        row = ManifestRow(
            trip_id="T1", branch="B", driver_id="Driver A",
            s_filename="s.csv", s_relative_path="s.csv",
            v_filename="v.csv", v_relative_path="v.csv",
            raw_s_rows=10, raw_v_rows=10, synced_rows=10, validated_rows=10, omitted_rows=0,
            flag_ok_count=10, measured_rate_hz=10.0, duration_s=1.0,
            has_stationary_segment=False, stationary_segment_count=0, total_stationary_duration_s=0.0,
            has_real_outages_for_trip=False, has_real_outages_in_dataset=False, has_real_outages=False,
            outage_windows_json="[]", extreme_accel_count=0, extreme_gyro_count=0,
            extreme_motion_count=0, dropout_count=0, nan_or_nonfinite_count=0,
            cached_npz_path="cache.npz",
            sync_mode="relative_elapsed",
            sync_status="PASSED",
            sync_clock_origin_offset_s=0.0,
            sync_clock_drift_s=0.0,
            sync_overlap_duration_s=1.0,
            sync_valid_interpolated_count=10,
            sync_out_of_bounds_count=0,
            sync_gap_violation_count=0,
            sync_valid_coverage_ratio=1.0,
        )
        d = row.to_dict()
        assert "sync_mode" in d
        assert "sync_status" in d
        assert "sync_clock_drift_s" in d
        assert "sync_overlap_duration_s" in d
        assert "sync_valid_coverage_ratio" in d
        assert "sync_passed" in d
        assert "downstream_ready" in d
        assert d["sync_passed"] is True
        assert d["downstream_ready"] is True

    def test_dataset_discovery_audit(self, tmp_path: Path) -> None:
        """Programmatic discovery audit must identify matched pairs, unmatched files, and malformed names."""
        data_dir = tmp_path / "test_raw"
        cat_dir = data_dir / "Categorised IOVNB Dataset" / "M (Driver B)"
        cat_dir.mkdir(parents=True)
        (cat_dir / "S-M.csv").write_text("dummy", encoding="utf-8")
        (cat_dir / "V-M.csv").write_text("dummy", encoding="utf-8")

        # S without V
        s_only_dir = data_dir / "Categorised IOVNB Dataset" / "S_Solo"
        s_only_dir.mkdir(parents=True)
        (s_only_dir / "S-Solo.csv").write_text("dummy", encoding="utf-8")

        # V without S
        v_only_dir = data_dir / "Uncategorised IOVNB Dataset"
        v_only_dir.mkdir(parents=True)
        (v_only_dir / "V-Ghost.csv").write_text("dummy", encoding="utf-8")

        # Malformed CSV
        (data_dir / "random_notes.csv").write_text("dummy", encoding="utf-8")

        audit = audit_dataset_discovery(data_dir)
        assert audit.matched_pairs_count == 1
        assert audit.unmatched_s_count == 1
        assert audit.unmatched_v_count == 1
        assert len(audit.malformed_csv_files) == 1
        assert not audit.is_fully_paired

    def test_downstream_readiness_semantics(self) -> None:
        """Downstream readiness requires sync_passed, validated_rows >= 100, and non-empty inputs."""
        assert compute_downstream_readiness(sync_passed=True, validated_rows=500, raw_s_rows=500, raw_v_rows=500) is True
        assert compute_downstream_readiness(sync_passed=False, validated_rows=500, raw_s_rows=500, raw_v_rows=500) is False
        assert compute_downstream_readiness(sync_passed=True, validated_rows=50, raw_s_rows=500, raw_v_rows=500) is False
        assert compute_downstream_readiness(sync_passed=True, validated_rows=500, raw_s_rows=0, raw_v_rows=500) is False

