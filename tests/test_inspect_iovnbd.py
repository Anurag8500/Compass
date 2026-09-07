"""test_inspect_iovnbd.py - Comprehensive Unit Tests for Phase 0 Inspection Tooling.

Verifies:
  1. Sampling-rate calculation (10 Hz from 0.1s dt and 100ms dt)
  2. Duplicate timestamp detection
  3. Non-monotonic timestamp detection
  4. Vector acceleration extreme-motion detection (|f| = sqrt(ax^2+ay^2+az^2) > 39.24 m/s^2)
  5. Vector gyro extreme-motion detection (|omega| = sqrt(gx^2+gy^2+gz^2) > 10.0 rad/s)
  6. Stationary detection requiring BOTH acceleration AND gyroscope low variance
  7. Controlled sensor-column detection (rejection of false positives)
  8. Folder-aware S/V pairing
  9. Prevention of cross-branch S/V pairing (Synchronized vs Unsynchronized)
  10. Graceful empty/missing dataset directory behavior
  11. Read-only data immutability verification
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# Add repository root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.inspect_iovnbd import (
    CandidateStationarySegment,
    FileReport,
    TimestampStats,
    analyze_sv_pairing,
    analyze_timestamps,
    classify_dataset_file,
    discover_candidate_stationary,
    find_sensor_columns,
    find_timestamp_column,
    inspect_csv_file,
    inspect_dataset,
)


class TestInspectIOVNBD(unittest.TestCase):
    def setUp(self):
        # Set fixed random seed for deterministic tests
        np.random.seed(42)

    # -----------------------------------------------------------------------
    # 1. Sampling Rate Calculation (Seconds vs Milliseconds)
    # -----------------------------------------------------------------------
    def test_sampling_rate_seconds_and_milliseconds(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 1A: Timestamp in seconds (dt = 0.1s -> 10.0 Hz)
            t_sec = np.arange(0.0, 10.0, 0.1)
            f_sec = Path(tmp_dir) / "test_sec.csv"
            pd.DataFrame({"Time Since Start of Day (seconds)": t_sec}).to_csv(f_sec, index=False)

            stats_sec = analyze_timestamps(f_sec, "Time Since Start of Day (seconds)")
            self.assertIsNotNone(stats_sec)
            self.assertEqual(stats_sec.unit, "seconds")
            self.assertAlmostEqual(stats_sec.measured_rate_hz, 10.0, delta=0.01)
            self.assertAlmostEqual(stats_sec.median_delta_s, 0.1, delta=0.001)

            # 1B: Timestamp in milliseconds (dt = 100ms -> 10.0 Hz)
            t_ms = np.arange(1000, 11000, 100)
            f_ms = Path(tmp_dir) / "test_ms.csv"
            pd.DataFrame({"TIME SINCE START (ms)": t_ms}).to_csv(f_ms, index=False)

            stats_ms = analyze_timestamps(f_ms, "TIME SINCE START (ms)")
            self.assertIsNotNone(stats_ms)
            self.assertEqual(stats_ms.unit, "milliseconds")
            self.assertAlmostEqual(stats_ms.measured_rate_hz, 10.0, delta=0.01)
            self.assertAlmostEqual(stats_ms.median_delta_s, 0.1, delta=0.001)

    # -----------------------------------------------------------------------
    # 2 & 3. Duplicate and Non-Monotonic Timestamp Detection
    # -----------------------------------------------------------------------
    def test_duplicate_and_non_monotonic_timestamps(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Sequence with 2 duplicates (1.2, 1.2 and 1.5, 1.5) and 1 non-monotonic step (1.4 -> 1.3)
            t_vals = [1.0, 1.1, 1.2, 1.2, 1.4, 1.3, 1.5, 1.5, 1.6]
            f_path = Path(tmp_dir) / "anomaly_time.csv"
            pd.DataFrame({"time": t_vals}).to_csv(f_path, index=False)

            stats = analyze_timestamps(f_path, "time")
            self.assertIsNotNone(stats)
            self.assertEqual(stats.duplicate_count, 2)
            self.assertEqual(stats.non_monotonic_count, 1)

    # -----------------------------------------------------------------------
    # 4. Vector Acceleration Extreme-Motion Detection (|f| > 39.24 m/s^2)
    # -----------------------------------------------------------------------
    def test_vector_acceleration_extreme_motion(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            f_path = Path(tmp_dir) / "accel_extreme.csv"
            # Row 0: Normal gravity [0, 0, 9.81] -> |f| = 9.81 (not extreme)
            # Row 1: High component [35, 0, 0] -> |f| = 35.0 (not extreme, < 39.24)
            # Row 2: Vector extreme: [25, 25, 25] -> ax, ay, az are each 25 (< 39.24),
            #        but magnitude is sqrt(3*625) = 43.3 > 39.24!
            # Row 3: Super extreme: [30, 30, 30] -> |f| = 51.96 > 39.24
            df = pd.DataFrame({
                "time": [0.0, 0.1, 0.2, 0.3],
                "ACCELEROMETER X (m/s2)": [0.0, 35.0, 25.0, 30.0],
                "ACCELEROMETER Y (m/s2)": [0.0, 0.0, 25.0, 30.0],
                "ACCELEROMETER Z (m/s2)": [9.81, 0.0, 25.0, 30.0],
                "GYROSCOPE Roll (rad/s)": [0.0, 0.0, 0.0, 0.0],
                "GYROSCOPE Pitch (rad/s)": [0.0, 0.0, 0.0, 0.0],
                "GYROSCOPE Yaw (rad/s)": [0.0, 0.0, 0.0, 0.0],
            })
            df.to_csv(f_path, index=False)

            report = inspect_csv_file(f_path)
            self.assertIsNotNone(report.quality_stats)
            # Exactly rows 2 and 3 cross the 4g (39.24 m/s^2) vector threshold
            self.assertEqual(report.quality_stats.extreme_accel_count, 2)
            self.assertEqual(report.quality_stats.extreme_gyro_count, 0)

    # -----------------------------------------------------------------------
    # 5. Vector Gyroscope Extreme-Motion Detection (|omega| > 10.0 rad/s)
    # -----------------------------------------------------------------------
    def test_vector_gyro_extreme_motion(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            f_path = Path(tmp_dir) / "gyro_extreme.csv"
            # Row 0: Normal rotation [0.01, 0.01, 0.01] -> |omega| = 0.017 (not extreme)
            # Row 1: Single axis 8.0 -> |omega| = 8.0 (not extreme, < 10)
            # Row 2: Vector extreme: [6.0, 6.0, 6.0] -> each axis is 6.0 (< 10),
            #        but magnitude is sqrt(3*36) = 10.39 > 10.0 rad/s!
            df = pd.DataFrame({
                "time": [0.0, 0.1, 0.2],
                "accel_x": [0.0, 0.0, 0.0],
                "accel_y": [0.0, 0.0, 0.0],
                "accel_z": [9.81, 9.81, 9.81],
                "gyro_x": [0.01, 8.0, 6.0],
                "gyro_y": [0.01, 0.0, 6.0],
                "gyro_z": [0.01, 0.0, 6.0],
            })
            df.to_csv(f_path, index=False)

            report = inspect_csv_file(f_path)
            self.assertIsNotNone(report.quality_stats)
            # Exactly row 2 crosses 10 rad/s vector threshold
            self.assertEqual(report.quality_stats.extreme_gyro_count, 1)

    # -----------------------------------------------------------------------
    # 6. Stationary Detection Requires BOTH Accel AND Gyro Low Variance
    # -----------------------------------------------------------------------
    def test_stationary_detection_requires_both_signals(self):
        n = 100
        t = np.arange(0, 10.0, 0.1)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Case A: Both accel and gyro are calm -> Stationary SHOULD be detected
            f_both_calm = Path(tmp_dir) / "both_calm.csv"
            pd.DataFrame({
                "time": t,
                "ACCELEROMETER X": np.random.normal(0, 0.005, n),
                "ACCELEROMETER Y": np.random.normal(0, 0.005, n),
                "ACCELEROMETER Z": np.random.normal(9.81, 0.005, n),
                "GYROSCOPE Roll": np.random.normal(0, 0.0005, n),
                "GYROSCOPE Pitch": np.random.normal(0, 0.0005, n),
                "GYROSCOPE Yaw": np.random.normal(0, 0.0005, n),
            }).to_csv(f_both_calm, index=False)

            rep_a = inspect_csv_file(f_both_calm)
            self.assertGreater(len(rep_a.stationary_segments), 0)
            self.assertGreater(rep_a.stationary_segments[0].gyro_variance, 0.0)
            self.assertLess(rep_a.stationary_segments[0].gyro_variance, 0.005)

            # Case B: Accel calm, but Gyro NOISY (e.g. spinning turntable) -> NOT stationary
            f_gyro_noisy = Path(tmp_dir) / "gyro_noisy.csv"
            pd.DataFrame({
                "time": t,
                "ACCELEROMETER X": np.random.normal(0, 0.005, n),
                "ACCELEROMETER Y": np.random.normal(0, 0.005, n),
                "ACCELEROMETER Z": np.random.normal(9.81, 0.005, n),
                "GYROSCOPE Roll": np.random.normal(0, 0.2, n),  # High variance > 0.005
                "GYROSCOPE Pitch": np.random.normal(0, 0.2, n),
                "GYROSCOPE Yaw": np.random.normal(0, 0.2, n),
            }).to_csv(f_gyro_noisy, index=False)

            rep_b = inspect_csv_file(f_gyro_noisy)
            self.assertEqual(len(rep_b.stationary_segments), 0, "No stationary segments if gyro is noisy")

            # Case C: Gyro calm, but Accel NOISY (e.g. engine vibration) -> NOT stationary
            f_accel_noisy = Path(tmp_dir) / "accel_noisy.csv"
            pd.DataFrame({
                "time": t,
                "ACCELEROMETER X": np.random.normal(0, 0.5, n),  # High variance > 0.05
                "ACCELEROMETER Y": np.random.normal(0, 0.5, n),
                "ACCELEROMETER Z": np.random.normal(9.81, 0.5, n),
                "GYROSCOPE Roll": np.random.normal(0, 0.0005, n),
                "GYROSCOPE Pitch": np.random.normal(0, 0.0005, n),
                "GYROSCOPE Yaw": np.random.normal(0, 0.0005, n),
            }).to_csv(f_accel_noisy, index=False)

            rep_c = inspect_csv_file(f_accel_noisy)
            self.assertEqual(len(rep_c.stationary_segments), 0, "No stationary segments if accel is noisy")

    # -----------------------------------------------------------------------
    # 7. Controlled Sensor-Column Detection (Eliminate False Positives)
    # -----------------------------------------------------------------------
    def test_sensor_column_detection_controlled(self):
        # Real IO-VNBD mixed columns
        cols = [
            "GPS LATITUDE (degrees)",
            "GPS LONGITUDE (degrees)",
            "GPS ALTITUDE (m)",
            "GPS SPEED (Kmh)",
            "TIME SINCE START (ms)",
            "ACCELEROMETER X (m/s2)",
            "ACCELEROMETER Y (m/s2)",
            "ACCELEROMETER Z (m/s2)",
            "GRAVITY X (m/s2)",
            "GRAVITY Y (m/s2)",
            "GRAVITY Z (m/s2)",
            "GYROSCOPE Yaw (rad/s)",
            "GYROSCOPE Pitch (rad/s)",
            "GYROSCOPE Roll (rad/s)",
            "MAGNETIC FIELD X (uT)",
            "ORIENTATION (Yaw) (deg)",
            "Indicated Longitudinal Acceleration (g)",
            "Accelerator Pedal Position",
            "Air Temperature (degrees)",
        ]
        sensors = find_sensor_columns(cols)

        # Must correctly identify only phone IMU axes
        self.assertEqual(
            sensors["accel"],
            ["ACCELEROMETER X (m/s2)", "ACCELEROMETER Y (m/s2)", "ACCELEROMETER Z (m/s2)"]
        )
        self.assertEqual(
            sensors["gyro"],
            ["GYROSCOPE Roll (rad/s)", "GYROSCOPE Pitch (rad/s)", "GYROSCOPE Yaw (rad/s)"]
        )

        # Must NOT classify false positives:
        self.assertNotIn("GPS LATITUDE (degrees)", sensors["accel"])
        self.assertNotIn("GRAVITY X (m/s2)", sensors["accel"])
        self.assertNotIn("Indicated Longitudinal Acceleration (g)", sensors["accel"])
        self.assertNotIn("Accelerator Pedal Position", sensors["accel"])
        self.assertNotIn("ORIENTATION (Yaw) (deg)", sensors["gyro"])
        self.assertNotIn("Air Temperature (degrees)", sensors["gyro"])

    # -----------------------------------------------------------------------
    # 8 & 9. Folder-Aware S/V Pairing & Cross-Branch Isolation
    # -----------------------------------------------------------------------
    def test_folder_aware_sv_pairing(self):
        # Create reports across multiple branches
        reports = [
            # Branch 1: Categorised IOVNB Dataset
            FileReport(
                relative_path="Synchronised V abd S datasets/Categorised IOVNB Dataset/Driver A/S1/S-S1.csv",
                filename="S-S1.csv",
                extension=".csv",
                size_bytes=1000,
                category="S- file (Smartphone)",
                subdirectory_type="Synchronised (Categorised)",
                branch_name="Categorised IOVNB Dataset",
                row_count=500,
            ),
            FileReport(
                relative_path="Synchronised V abd S datasets/Categorised IOVNB Dataset/Driver A/S1/V-S1.csv",
                filename="V-S1.csv",
                extension=".csv",
                size_bytes=1200,
                category="V- file (Vehicle CAN/VBOX)",
                subdirectory_type="Synchronised (Categorised)",
                branch_name="Categorised IOVNB Dataset",
                row_count=500,
            ),
            # Branch 2: Uncategorised IOVNB Dataset
            FileReport(
                relative_path="Synchronised V abd S datasets/Uncategorised IOVNB Dataset/S-Dataset/S-S1.csv",
                filename="S-S1.csv",
                extension=".csv",
                size_bytes=1000,
                category="S- file (Smartphone)",
                subdirectory_type="Synchronised (Uncategorised)",
                branch_name="Uncategorised IOVNB Dataset",
                row_count=500,
            ),
            FileReport(
                relative_path="Synchronised V abd S datasets/Uncategorised IOVNB Dataset/V-Dataset/V-S1.csv",
                filename="V-S1.csv",
                extension=".csv",
                size_bytes=1200,
                category="V- file (Vehicle CAN/VBOX)",
                subdirectory_type="Synchronised (Uncategorised)",
                branch_name="Uncategorised IOVNB Dataset",
                row_count=500,
            ),
            # Branch 3: Unsynchronised V and S Dataset (Solo S- file, no V file)
            FileReport(
                relative_path="Unsynchronised V and S Dataset/S-Solo.csv",
                filename="S-Solo.csv",
                extension=".csv",
                size_bytes=800,
                category="S- file (Smartphone)",
                subdirectory_type="Unsynchronised V and S",
                branch_name="Unsynchronised V and S Dataset",
                row_count=300,
            ),
        ]

        pairs, unmatched_s, unmatched_v = analyze_sv_pairing(reports)

        # There must be exactly 2 pairs: 1 in Categorised, 1 in Uncategorised
        self.assertEqual(len(pairs), 2)
        branches = {p.branch for p in pairs}
        self.assertIn("Categorised IOVNB Dataset", branches)
        self.assertIn("Uncategorised IOVNB Dataset", branches)

        # Cross-branch check: S-Solo from Unsynchronised must NOT pair with anything in Synchronised
        self.assertEqual(len(unmatched_s), 1)
        self.assertIn("[Unsynchronised V and S Dataset]", unmatched_s[0])
        self.assertEqual(len(unmatched_v), 0)

    # -----------------------------------------------------------------------
    # 10. Empty or Missing Dataset Directory Behavior
    # -----------------------------------------------------------------------
    def test_empty_or_missing_dataset_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            empty_dir = Path(tmp_dir) / "non_existent_data_dir"
            file_reports, pairs, unmatched_s, unmatched_v = inspect_dataset(empty_dir)

            # Must return empty structures gracefully without crashing
            self.assertEqual(len(file_reports), 0)
            self.assertEqual(len(pairs), 0)
            self.assertEqual(len(unmatched_s), 0)
            self.assertEqual(len(unmatched_v), 0)

    # -----------------------------------------------------------------------
    # 11. Read-Only Immutability Verification
    # -----------------------------------------------------------------------
    def test_raw_data_read_only_immutability(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "S-Sample.csv"
            original_content = "time,accel_x,accel_y,accel_z\n0.0,0.0,0.0,9.81\n0.1,0.0,0.0,9.81\n"
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write(original_content)

            stat_before = csv_path.stat()

            # Run inspection
            report = inspect_csv_file(csv_path)
            self.assertEqual(report.row_count, 2)

            stat_after = csv_path.stat()

            # File must not be modified
            with open(csv_path, "r", encoding="utf-8") as f:
                content_after = f.read()

            self.assertEqual(original_content, content_after)
            self.assertEqual(stat_before.st_size, stat_after.st_size)
            self.assertEqual(stat_before.st_mtime, stat_after.st_mtime)


if __name__ == "__main__":
    unittest.main()
