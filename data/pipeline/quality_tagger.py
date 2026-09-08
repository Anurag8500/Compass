"""Non-destructive quality evaluation and validated stream tagging for COMPASS.

Implements the non-destructive quality tagging rule:
- Raw IMU data is immutable.
- Every evaluated record receives bitmask quality flags.
- Truly corrupt/non-computable rows (NaN, +Inf, -Inf, non-monotonic, duplicate,
  or non-positive timestamps) are omitted ONLY from the downstream validated stream.
- Physical dynamics (extreme motion ||f|| > 4g or ||omega|| > 10 rad/s) and timing gaps
  (dropouts delta_t > 3 * nominal_delta_t) are tagged with bitflags but KEPT in the
  validated stream so estimators can adjust measurement variances rather than losing
  physical motion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from navigation.schemas.imu import (
    FLAG_DUPLICATE_TIMESTAMP,
    FLAG_EXTREME_MOTION,
    FLAG_INVALID_TIMESTAMP,
    FLAG_NAN_OR_NONFINITE,
    FLAG_NON_MONOTONIC_TIMESTAMP,
    FLAG_OK,
    FLAG_SENSOR_DROPOUT,
    RawIMUSample,
)

# Physical and timing thresholds established by authoritative architecture
EXTREME_ACCEL_THRESHOLD_MS2: float = 39.24    # 4g in m/s^2 (physical shock threshold)
EXTREME_GYRO_THRESHOLD_RADS: float = 10.0     # 10 rad/s (~573 deg/s sharp rotation)
NOMINAL_RATE_HZ: float = 10.0                 # Canonical IO-VNBD native/ML interface rate
NOMINAL_DT_NS: int = 100_000_000              # 100 ms in nanoseconds
DROPOUT_RATIO_THRESHOLD: float = 3.0          # delta_t > 3 * nominal_dt constitutes a dropout


@dataclass
class QualityReport:
    """Statistical summary and boolean masks resulting from quality evaluation."""
    total_rows: int
    validated_rows: int
    omitted_rows: int
    flag_ok_count: int
    nan_or_nonfinite_count: int
    invalid_timestamp_count: int
    non_monotonic_count: int
    duplicate_count: int
    extreme_accel_count: int
    extreme_gyro_count: int
    extreme_motion_count: int
    dropout_count: int
    quality_flags: np.ndarray        # 1D uint32 array of bitmask flags
    is_validated: np.ndarray         # 1D bool array: True if kept in validated stream

    @property
    def omission_rate_percent(self) -> float:
        """Percentage of total rows omitted from integration."""
        return (self.omitted_rows / self.total_rows * 100.0) if self.total_rows > 0 else 0.0


class QualityTagger:
    """Non-destructive sensor quality tagger operating over numpy arrays or schema objects."""

    def __init__(
        self,
        nominal_dt_ns: int = NOMINAL_DT_NS,
        extreme_accel_threshold: float = EXTREME_ACCEL_THRESHOLD_MS2,
        extreme_gyro_threshold: float = EXTREME_GYRO_THRESHOLD_RADS,
        dropout_threshold_ratio: float = DROPOUT_RATIO_THRESHOLD,
    ) -> None:
        self.nominal_dt_ns = nominal_dt_ns
        self.extreme_accel_threshold = extreme_accel_threshold
        self.extreme_gyro_threshold = extreme_gyro_threshold
        self.dropout_threshold_ratio = dropout_threshold_ratio
        self.dropout_dt_ns = int(nominal_dt_ns * dropout_threshold_ratio)

    @classmethod
    def from_sampling_rate(
        cls,
        rate_hz: float,
        extreme_accel_threshold: float = EXTREME_ACCEL_THRESHOLD_MS2,
        extreme_gyro_threshold: float = EXTREME_GYRO_THRESHOLD_RADS,
        dropout_threshold_ratio: float = DROPOUT_RATIO_THRESHOLD,
    ) -> QualityTagger:
        """Construct a QualityTagger configured for a specific native sampling rate."""
        if rate_hz <= 0:
            raise ValueError(f"Sampling rate must be positive, got {rate_hz}")
        nominal_dt_ns = int(round(1e9 / rate_hz))
        return cls(
            nominal_dt_ns=nominal_dt_ns,
            extreme_accel_threshold=extreme_accel_threshold,
            extreme_gyro_threshold=extreme_gyro_threshold,
            dropout_threshold_ratio=dropout_threshold_ratio,
        )

    @staticmethod
    def derive_nominal_dt_ns(timestamps_ns: np.ndarray, fallback_dt_ns: int = NOMINAL_DT_NS) -> int:
        """Derive median delta_t from an empirical timestamp array."""
        t_arr = np.asarray(timestamps_ns, dtype=np.int64)
        if len(t_arr) < 2:
            return fallback_dt_ns
        diffs = np.diff(t_arr)
        pos_diffs = diffs[diffs > 0]
        if len(pos_diffs) == 0:
            return fallback_dt_ns
        med = float(np.median(pos_diffs))
        return int(med) if med > 0 else fallback_dt_ns

    def evaluate_arrays(
        self,
        timestamps_ns: np.ndarray,
        accel: np.ndarray,
        gyro: np.ndarray,
        nominal_dt_ns: Optional[int] = None,
    ) -> QualityReport:
        """Evaluate arrays of raw sensor samples and return quality flags and validated mask.

        Args:
            timestamps_ns: 1D array of nanosecond timestamps (int64).
            accel: Nx3 array of specific forces [ax, ay, az] in m/s^2.
            gyro: Nx3 array of angular velocities [gx, gy, gz] in rad/s.
            nominal_dt_ns: Optional override for nominal sample period in ns.

        Returns:
            QualityReport with bitflags, validated stream mask, and exact row counters.
        """
        n = len(timestamps_ns)
        effective_nominal_dt = nominal_dt_ns or self.nominal_dt_ns
        effective_dropout_dt = int(effective_nominal_dt * self.dropout_threshold_ratio)

        if timestamps_ns.ndim != 1:
            raise ValueError(f"timestamps_ns must be 1D, got shape {timestamps_ns.shape}")
        if accel.ndim != 2 or accel.shape != (n, 3):
            raise ValueError(f"accel array shape mismatch: expected ({n}, 3), got {accel.shape}")
        if gyro.ndim != 2 or gyro.shape != (n, 3):
            raise ValueError(f"gyro array shape mismatch: expected ({n}, 3), got {gyro.shape}")

        if n == 0:
            return QualityReport(
                total_rows=0,
                validated_rows=0,
                omitted_rows=0,
                flag_ok_count=0,
                nan_or_nonfinite_count=0,
                invalid_timestamp_count=0,
                non_monotonic_count=0,
                duplicate_count=0,
                extreme_accel_count=0,
                extreme_gyro_count=0,
                extreme_motion_count=0,
                dropout_count=0,
                quality_flags=np.zeros(0, dtype=np.uint32),
                is_validated=np.zeros(0, dtype=bool),
            )

        flags = np.zeros(n, dtype=np.uint32)

        # 1. NaN or non-finite check on accel, gyro, or timestamp
        accel_finite = np.isfinite(accel).all(axis=1)
        gyro_finite = np.isfinite(gyro).all(axis=1)
        t_finite = np.isfinite(timestamps_ns)
        nan_mask = ~(accel_finite & gyro_finite & t_finite)
        flags[nan_mask] |= FLAG_NAN_OR_NONFINITE

        # 2. Invalid timestamp (negative elapsed timestamp or sentinel < 0; t == 0 is valid session origin)
        invalid_t_mask = timestamps_ns < 0
        flags[invalid_t_mask] |= FLAG_INVALID_TIMESTAMP

        # 3 & 4. Timing progression (non-monotonic and duplicates)
        if n > 1:
            diffs = np.diff(timestamps_ns)
            # Prepend positive nominal dt so first sample doesn't trigger delta check
            diffs_padded = np.concatenate([[effective_nominal_dt], diffs])

            # Non-monotonic: t_k < t_{k-1}
            non_mono_mask = diffs_padded < 0
            flags[non_mono_mask] |= FLAG_NON_MONOTONIC_TIMESTAMP

            # Duplicate: t_k == t_{k-1}
            dup_mask = (diffs_padded == 0) & (np.arange(n) > 0)
            flags[dup_mask] |= FLAG_DUPLICATE_TIMESTAMP

            # Dropout: delta_t > dropout_dt_ns
            dropout_mask = diffs_padded > effective_dropout_dt
            flags[dropout_mask] |= FLAG_SENSOR_DROPOUT
        else:
            non_mono_mask = np.zeros(n, dtype=bool)
            dup_mask = np.zeros(n, dtype=bool)
            dropout_mask = np.zeros(n, dtype=bool)

        # 5. Extreme Motion: ||f|| > 4g or ||omega|| > 10 rad/s
        # Only compute on finite samples
        valid_finite = ~nan_mask
        extreme_accel_mask = np.zeros(n, dtype=bool)
        extreme_gyro_mask = np.zeros(n, dtype=bool)

        if valid_finite.any():
            accel_mag = np.linalg.norm(accel[valid_finite], axis=1)
            gyro_mag = np.linalg.norm(gyro[valid_finite], axis=1)

            ext_a_subset = accel_mag > self.extreme_accel_threshold
            ext_g_subset = gyro_mag > self.extreme_gyro_threshold

            extreme_accel_mask[valid_finite] = ext_a_subset
            extreme_gyro_mask[valid_finite] = ext_g_subset

            flags[extreme_accel_mask | extreme_gyro_mask] |= FLAG_EXTREME_MOTION

        # 6. Validated Stream Policy:
        # Exclude: NaN/Inf, Non-Monotonic, Duplicate (first instance kept), Invalid Timestamp
        # Keep: Extreme motion (real dynamics), Sensor dropouts (timing gaps)
        omitted_mask = (
            (flags & FLAG_NAN_OR_NONFINITE != 0)
            | (flags & FLAG_NON_MONOTONIC_TIMESTAMP != 0)
            | (flags & FLAG_DUPLICATE_TIMESTAMP != 0)
            | (flags & FLAG_INVALID_TIMESTAMP != 0)
        )
        is_validated = ~omitted_mask

        flag_ok_mask = (flags == FLAG_OK)
        extreme_motion_mask = (flags & FLAG_EXTREME_MOTION != 0)

        return QualityReport(
            total_rows=n,
            validated_rows=int(np.sum(is_validated)),
            omitted_rows=int(np.sum(omitted_mask)),
            flag_ok_count=int(np.sum(flag_ok_mask)),
            nan_or_nonfinite_count=int(np.sum(nan_mask)),
            invalid_timestamp_count=int(np.sum(invalid_t_mask)),
            non_monotonic_count=int(np.sum(non_mono_mask)),
            duplicate_count=int(np.sum(dup_mask)),
            extreme_accel_count=int(np.sum(extreme_accel_mask)),
            extreme_gyro_count=int(np.sum(extreme_gyro_mask)),
            extreme_motion_count=int(np.sum(extreme_motion_mask)),
            dropout_count=int(np.sum(dropout_mask)),
            quality_flags=flags,
            is_validated=is_validated,
        )

    def evaluate_samples(
        self,
        samples: List[RawIMUSample],
        nominal_dt_ns: Optional[int] = None,
    ) -> Tuple[List[RawIMUSample], List[RawIMUSample], QualityReport]:
        """Evaluate a list of RawIMUSample objects.

        Returns:
            (tagged_raw_samples, validated_samples, quality_report)
        """
        n = len(samples)
        if n == 0:
            rep = self.evaluate_arrays(np.zeros(0, dtype=np.int64), np.zeros((0, 3)), np.zeros((0, 3)), nominal_dt_ns=nominal_dt_ns)
            return [], [], rep

        timestamps = np.array([s.timestamp_ns for s in samples], dtype=np.int64)
        accel = np.array([s.accel for s in samples], dtype=float)
        gyro = np.array([s.gyro for s in samples], dtype=float)

        report = self.evaluate_arrays(timestamps, accel, gyro, nominal_dt_ns=nominal_dt_ns)

        tagged_samples: List[RawIMUSample] = []
        validated_samples: List[RawIMUSample] = []

        for i, s in enumerate(samples):
            f = int(report.quality_flags[i])
            tagged = RawIMUSample(
                timestamp_ns=s.timestamp_ns,
                accel=s.accel,
                gyro=s.gyro,
                quality_flags=f,
                source=s.source,
                sensor_id=s.sensor_id,
            )
            tagged_samples.append(tagged)
            if report.is_validated[i]:
                validated_samples.append(tagged)

        return tagged_samples, validated_samples, report


def tag_imu_samples(
    samples: List[RawIMUSample],
    nominal_dt_ns: int = NOMINAL_DT_NS,
) -> Tuple[List[RawIMUSample], List[RawIMUSample], QualityReport]:
    """Convenience function for non-destructive quality tagging."""
    tagger = QualityTagger(nominal_dt_ns=nominal_dt_ns)
    return tagger.evaluate_samples(samples)
