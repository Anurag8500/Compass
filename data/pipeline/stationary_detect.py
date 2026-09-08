"""Stationary segment detection for COMPASS.

Implements the dual-signal low-variance stationary detector established in Phase 0.
A vehicle is considered stationary ONLY when both acceleration and gyroscope
magnitudes exhibit sustained low variance:
    rolling_var(||f||) < accel_var_threshold (0.05 m^2/s^4)
    AND
    rolling_var(||omega||) < gyro_var_threshold (0.005 rad^2/s^2)
over a minimum window duration (default 5.0s / 50 samples @ 10 Hz).

Stationary detection produces annotations (time ranges and boolean masks);
it NEVER deletes or modifies sensor samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pandas as pd

DEFAULT_ACCEL_VAR_THRESHOLD: float = 0.05     # m^2/s^4
DEFAULT_GYRO_VAR_THRESHOLD: float = 0.005     # rad^2/s^2
DEFAULT_WINDOW_SAMPLES: int = 50              # 50 samples @ 10 Hz = 5.0 seconds


@dataclass(frozen=True)
class StationarySegment:
    """Discrete stationary time interval annotation.

    Timing Semantics:
    - Detection criterion is based strictly on consecutive sample count: sample_count >= window_samples
      (default: 50 consecutive samples).
    - duration_s reports the exact elapsed timestamp span: (end_timestamp_ns - start_timestamp_ns) / 1e9.
      Note that for N=50 samples at nominal 10 Hz (0.10s interval), the timestamp span from sample 0 to
      sample 49 covers (50 - 1) * 0.10s = 4.90s, while the discrete observation covers 50 sample periods.
    """
    start_timestamp_ns: int
    end_timestamp_ns: int
    duration_s: float
    start_idx: int
    end_idx: int
    sample_count: int
    mean_accel_var: float
    mean_gyro_var: float


class StationaryDetector:
    """Detects stationary periods from 3-axis accelerometer and gyroscope streams."""

    def __init__(
        self,
        accel_var_threshold: float = DEFAULT_ACCEL_VAR_THRESHOLD,
        gyro_var_threshold: float = DEFAULT_GYRO_VAR_THRESHOLD,
        window_samples: int = DEFAULT_WINDOW_SAMPLES,
    ) -> None:
        self.accel_var_threshold = accel_var_threshold
        self.gyro_var_threshold = gyro_var_threshold
        self.window_samples = window_samples

    def detect(
        self,
        timestamps_ns: np.ndarray,
        accel: np.ndarray,
        gyro: np.ndarray,
    ) -> Tuple[List[StationarySegment], np.ndarray]:
        """Detect stationary segments from synchronized/aligned motion arrays.

        Args:
            timestamps_ns: 1D array of nanosecond timestamps.
            accel: Nx3 array of specific force (m/s^2).
            gyro: Nx3 array of angular velocity (rad/s).

        Returns:
            (segments, is_stationary_mask)
            - segments: List of detected StationarySegment objects.
            - is_stationary_mask: 1D boolean array indicating whether each sample is stationary.
        """
        n = len(timestamps_ns)
        if timestamps_ns.ndim != 1:
            raise ValueError(f"timestamps_ns must be 1D, got shape {timestamps_ns.shape}")
        if accel.ndim != 2 or accel.shape != (n, 3):
            raise ValueError(f"accel array shape mismatch: expected ({n}, 3), got {accel.shape}")
        if gyro.ndim != 2 or gyro.shape != (n, 3):
            raise ValueError(f"gyro array shape mismatch: expected ({n}, 3), got {gyro.shape}")

        if n < self.window_samples:
            return [], np.zeros(n, dtype=bool)

        # 1. Compute vector magnitudes and finite mask
        finite_samples = np.isfinite(accel).all(axis=1) & np.isfinite(gyro).all(axis=1)
        accel_mag = np.linalg.norm(accel, axis=1)
        gyro_mag = np.linalg.norm(gyro, axis=1)

        # 2. Rolling variances using pandas Series
        s_a = pd.Series(accel_mag)
        s_g = pd.Series(gyro_mag)

        roll_var_a = s_a.rolling(window=self.window_samples).var().to_numpy()
        roll_var_g = s_g.rolling(window=self.window_samples).var().to_numpy()

        # Handle NaNs from initial rolling window and non-finite samples (set variance to high sentinel)
        roll_var_a = np.nan_to_num(roll_var_a, nan=999.0, posinf=999.0, neginf=999.0)
        roll_var_g = np.nan_to_num(roll_var_g, nan=999.0, posinf=999.0, neginf=999.0)

        # Ensure all samples in the rolling window are strictly finite before declaring stationary
        all_finite_window = pd.Series(finite_samples).rolling(window=self.window_samples).min().to_numpy() == 1.0

        # 3. Dual condition: BOTH accel and gyro variance must be below threshold AND all samples finite
        is_stationary_point = (
            (roll_var_a < self.accel_var_threshold)
            & (roll_var_g < self.gyro_var_threshold)
            & all_finite_window
        )

        # 4. Extract continuous segments
        segments: List[StationarySegment] = []
        is_stationary_mask = np.zeros(n, dtype=bool)

        in_segment = False
        start_idx = 0

        for i in range(n):
            val = is_stationary_point[i]
            if val and not in_segment:
                in_segment = True
                # The rolling window ends at i, so stationary motion started at (i - window_samples + 1)
                start_idx = max(0, i - self.window_samples + 1)
            elif not val and in_segment:
                in_segment = False
                end_idx = i
                if (end_idx - start_idx) >= self.window_samples:
                    t_start = int(timestamps_ns[start_idx])
                    t_end = int(timestamps_ns[end_idx - 1])
                    dur_s = max(0.0, (t_end - t_start) / 1e9)
                    var_a = float(np.mean(roll_var_a[start_idx:end_idx]))
                    var_g = float(np.mean(roll_var_g[start_idx:end_idx]))

                    segments.append(
                        StationarySegment(
                            start_timestamp_ns=t_start,
                            end_timestamp_ns=t_end,
                            duration_s=dur_s,
                            start_idx=start_idx,
                            end_idx=end_idx,
                            sample_count=end_idx - start_idx,
                            mean_accel_var=var_a,
                            mean_gyro_var=var_g,
                        )
                    )
                    is_stationary_mask[start_idx:end_idx] = True

        # Flush trailing segment if file ended while stationary
        if in_segment:
            end_idx = n
            if (end_idx - start_idx) >= self.window_samples:
                t_start = int(timestamps_ns[start_idx])
                t_end = int(timestamps_ns[end_idx - 1])
                dur_s = max(0.0, (t_end - t_start) / 1e9)
                var_a = float(np.mean(roll_var_a[start_idx:end_idx]))
                var_g = float(np.mean(roll_var_g[start_idx:end_idx]))

                segments.append(
                    StationarySegment(
                        start_timestamp_ns=t_start,
                        end_timestamp_ns=t_end,
                        duration_s=dur_s,
                        start_idx=start_idx,
                        end_idx=end_idx,
                        sample_count=end_idx - start_idx,
                        mean_accel_var=var_a,
                        mean_gyro_var=var_g,
                    )
                )
                is_stationary_mask[start_idx:end_idx] = True

        return segments, is_stationary_mask
