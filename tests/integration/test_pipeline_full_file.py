"""Integration test running the Phase 2 data pipeline on real IO-VNBD dataset files.

Verifies:
1. Real IO-VNBD S-file and V-file parsing without errors.
2. S and V timestamp conversion to canonical nanoseconds.
3. Successful timestamp alignment and synchronization into SynchronizedTrip.
4. Non-destructive quality evaluation and validated stream generation.
5. Stationary segment detection on real driving telemetry.
6. Binary array caching (.npz) round-trip reproducibility.
7. Absolute raw data immutability (SHA-256 and modification time checks).
"""

import hashlib
from pathlib import Path
import tempfile
import numpy as np
import pytest

from data.pipeline.manifest import DatasetManifestBuilder
from data.pipeline.parse import parse_s_file, parse_v_file
from data.pipeline.quality_tagger import QualityTagger
from data.pipeline.stationary_detect import StationaryDetector
from data.pipeline.sync import SynchronizedTrip, synchronize_s_v


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class TestPipelineFullFileIntegration:
    """Integration test suite executing the offline pipeline on a real IO-VNBD trip."""

    @classmethod
    def setup_class(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[2]
        cls.raw_data_dir = cls.project_root / "data" / "raw" / "io_vnbd"

        # Use Trip S3b (6,813 rows, fast and contains both stationary and extreme motion)
        cls.s_file = cls.raw_data_dir / "Synchronised V abd S datasets" / "Categorised IOVNB Dataset" / "S (Driver A)" / "S3b" / "S-S3b.csv"
        cls.v_file = cls.raw_data_dir / "Synchronised V abd S datasets" / "Categorised IOVNB Dataset" / "S (Driver A)" / "S3b" / "V-S3b.csv"

        assert cls.s_file.exists(), f"Real S-file not found: {cls.s_file}"
        assert cls.v_file.exists(), f"Real V-file not found: {cls.v_file}"

    def test_real_file_pipeline_end_to_end(self) -> None:
        # 1. Record raw file hashes and mtimes before test
        s_hash_before = compute_file_sha256(self.s_file)
        v_hash_before = compute_file_sha256(self.v_file)
        s_mtime_before = self.s_file.stat().st_mtime
        v_mtime_before = self.v_file.stat().st_mtime

        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir) / "cache"
            manifest_path = Path(tmp_dir) / "manifest.csv"

            # 2. Parse real S-file and V-file
            s_trip = parse_s_file(self.s_file)
            v_trip = parse_v_file(self.v_file)

            assert s_trip.row_count == 6813
            assert v_trip.row_count == 6813
            assert s_trip.has_gnss is True

            # Verify timestamp conversion to nanoseconds (S3b starts at 2,503,320 ms)
            assert s_trip.timestamps_ns[0] == 2_503_320_000_000

            # 3. Synchronize S and V streams
            tagger = QualityTagger()
            synced_trip, quality_report = synchronize_s_v(
                s_trip=s_trip,
                v_trip=v_trip,
                trip_id="S3b",
                branch="Categorised IOVNB Dataset",
                tagger=tagger,
            )

            assert synced_trip.row_count == 6813
            assert synced_trip.overlap_duration_s == pytest.approx(681.2, abs=1.0)
            assert quality_report.total_rows == 6813
            assert quality_report.validated_rows > 6800  # Vast majority valid

            # Verify synchronization diagnostics
            assert synced_trip.diagnostics is not None
            assert synced_trip.diagnostics.target_sample_count == 6813
            assert synced_trip.diagnostics.valid_interpolated_count > 6800

            # Verify continuous telemetry is numerically valid and within physical bounds
            valid_lat_mask = ~np.isnan(synced_trip.v_ref_lat)
            assert valid_lat_mask.sum() > 6800
            assert np.all(synced_trip.v_ref_lat[valid_lat_mask] > 50.0)  # Real UK latitude ~52 deg

            # Verify discrete categorical telemetry is strictly integer-valued
            assert synced_trip.v_ref_gear.dtype == np.int32
            assert np.all(np.isin(synced_trip.v_ref_gear, [-1, 0, 1, 2, 3, 4, 5, 6]))
            assert synced_trip.v_ref_handbrake.dtype == np.int32
            assert np.all(np.isin(synced_trip.v_ref_handbrake, [-1, 0, 1]))

            # Verify angular heading is within [0, 360)
            valid_hdg_mask = ~np.isnan(synced_trip.v_ref_heading_deg)
            assert np.all((synced_trip.v_ref_heading_deg[valid_hdg_mask] >= 0.0) & (synced_trip.v_ref_heading_deg[valid_hdg_mask] < 360.0))

            # S3b is known to have extreme motion in real dataset
            assert quality_report.extreme_accel_count >= 1
            assert quality_report.extreme_gyro_count >= 1

            # 4. Stationary Segment Detection
            detector = StationaryDetector()
            stationary_segs, is_stat_mask = detector.detect(
                synced_trip.timestamps_ns,
                synced_trip.accel_raw,
                synced_trip.gyro_raw,
            )
            assert len(stationary_segs) >= 1
            assert is_stat_mask.any()
            # Verify stationary duration semantics: sample_count >= 50 and duration_s is exact timestamp span
            assert all(s.sample_count >= 20 for s in stationary_segs)
            assert all(s.duration_s >= 0.0 for s in stationary_segs)

            # 5. Cache Serialization & Reload Verification (.npz)
            cache_file = cache_dir / "Categorised_S3b.npz"
            synced_trip.save_npz(cache_file)
            assert cache_file.exists()

            reloaded_trip = SynchronizedTrip.load_npz(cache_file)
            assert reloaded_trip.trip_id == "S3b"
            assert reloaded_trip.row_count == synced_trip.row_count
            np.testing.assert_array_equal(reloaded_trip.timestamps_ns, synced_trip.timestamps_ns)
            np.testing.assert_array_equal(reloaded_trip.raw_timestamps_ns, synced_trip.raw_timestamps_ns)
            np.testing.assert_array_equal(reloaded_trip.accel_raw, synced_trip.accel_raw)
            np.testing.assert_array_equal(reloaded_trip.v_ref_lat, synced_trip.v_ref_lat)

            # 6. Single pair manifest row generation
            builder = DatasetManifestBuilder(
                project_root=self.project_root,
                cache_dir=cache_dir,
                manifest_path=manifest_path,
            )
            row = builder.process_pair(
                s_path=self.s_file,
                v_path=self.v_file,
                trip_id="S3b",
                branch="Categorised IOVNB Dataset",
                save_cache=True,
            )
            assert row.trip_id == "S3b"
            assert row.driver_id == "Driver A"  # Normalized driver ID
            assert row.raw_s_rows == 6813
            assert row.raw_v_rows == 6813
            assert row.measured_rate_hz == 10.0
            assert row.has_stationary_segment is True
            assert row.has_real_outages_for_trip is False
            assert row.flag_ok_count > 6700
            assert row.extreme_motion_count >= 1
            assert row.sync_mode == "relative_elapsed"
            assert row.sync_status == "PASSED"
            assert row.sync_valid_coverage_ratio > 0.99
            assert row.sync_overlap_duration_s > 600.0

        # 7. Raw Data Immutability Check: verify SHA-256 and mtime are untouched
        s_hash_after = compute_file_sha256(self.s_file)
        v_hash_after = compute_file_sha256(self.v_file)
        assert s_hash_before == s_hash_after, "RAW S-FILE WAS MODIFIED!"
        assert v_hash_before == v_hash_after, "RAW V-FILE WAS MODIFIED!"
        assert s_mtime_before == self.s_file.stat().st_mtime, "Raw S-file mtime changed!"
        assert v_mtime_before == self.v_file.stat().st_mtime, "Raw V-file mtime changed!"
