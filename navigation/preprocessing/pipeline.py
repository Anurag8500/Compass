"""Unified preprocessing pipeline entry point for COMPASS (Phase 3).

Coordinates the complete classical preprocessing transform chain:
    Phase 2 Synchronized Trip
               |
    1. Stationary Calibration (b_g, tilt angles)
               |
    2. Device -> Vehicle Mounting Alignment (R_b^v)
               |
    3. Dual-Stage Denoising Filter (Median + Butterworth)
               |
    4. Recalibration Discontinuity Monitoring
               |
    Vehicle-Frame Specific Force f_m^v & Angular Rate omega_m^v
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from data.pipeline.sync import SynchronizedTrip
from navigation.preprocessing.alignment import MountingAlignment, estimate_mounting_alignment
from navigation.preprocessing.calibration import CalibrationProfile, calibrate_stationary_window
from navigation.preprocessing.filtering import IMUFilter
from navigation.preprocessing.recalibration_trigger import RecalibrationDetector, RecalibrationEvent


@dataclass(frozen=True)
class PreprocessedTrip:
    """Complete preprocessed vehicle-frame output of Phase 3.

    Attributes:
        trip_id: Unique trip identifier.
        timestamps_ns: (N,) int64 working timestamp grid.
        f_m_v: (N, 3) float64 filtered specific force in vehicle frame (m/s^2).
        omega_m_v: (N, 3) float64 filtered angular velocity in vehicle frame (rad/s).
        f_m_v_unfiltered: (N, 3) float64 aligned unfiltered specific force (m/s^2).
        omega_m_v_unfiltered: (N, 3) float64 aligned unfiltered angular velocity (rad/s).
        calibration: Estimated stationary calibration profile.
        alignment: Estimated mounting alignment transformation.
        recalibration_events: List of detected sensor shift/discontinuity events.
        is_validated: (N,) bool integration mask from Phase 2.
        quality_flags: (N,) uint32 quality flag mask.
    """
    trip_id: str
    timestamps_ns: np.ndarray
    f_m_v: np.ndarray
    omega_m_v: np.ndarray
    f_m_v_unfiltered: np.ndarray
    omega_m_v_unfiltered: np.ndarray
    calibration: CalibrationProfile
    alignment: MountingAlignment
    recalibration_events: List[RecalibrationEvent]
    is_validated: np.ndarray
    quality_flags: np.ndarray


class PreprocessingPipeline:
    """Encapsulates the Phase 3 classical preprocessing chain."""

    def __init__(
        self,
        sampling_rate_hz: float = 10.0,
        filter_cutoff_hz: float = 3.0,
        median_window_size: int = 3,
        min_speed_for_yaw_mps: float = 3.0,
    ) -> None:
        self.sampling_rate_hz = sampling_rate_hz
        self.filter = IMUFilter(
            sampling_rate_hz=sampling_rate_hz,
            cutoff_hz=filter_cutoff_hz,
            median_window_size=median_window_size,
        )
        self.recal_detector = RecalibrationDetector()
        self.min_speed_for_yaw_mps = min_speed_for_yaw_mps

    def process_trip(
        self,
        trip: SynchronizedTrip,
        stationary_mask: Optional[np.ndarray] = None,
        prior_calibration: Optional[CalibrationProfile] = None,
    ) -> PreprocessedTrip:
        """Process a synchronized trip into calibrated, aligned, and filtered vehicle-frame streams."""
        accel_raw = trip.accel_raw  # (N, 3) device frame
        gyro_raw = trip.gyro_raw    # (N, 3) device frame
        ts = trip.timestamps_ns     # (N,) int64
        n_samples = len(ts)

        # 1. Determine stationary calibration window (startup stationary window)
        if stationary_mask is not None and np.sum(stationary_mask) >= 20:
            stat_indices = np.where(stationary_mask)[0]
            # Find the first contiguous stationary segment with at least 20 samples
            splits = np.where(np.diff(stat_indices) > 1)[0]
            blocks = np.split(stat_indices, splits + 1)
            chosen_block = None
            for b in blocks:
                if len(b) >= 20:
                    chosen_block = b
                    break
            if chosen_block is None:
                chosen_block = stat_indices[:min(50, len(stat_indices))]

            cal_accel = accel_raw[chosen_block]
            cal_gyro = gyro_raw[chosen_block]
            cal_ts = ts[chosen_block]
        else:
            # Default to first 50 samples (5.0s at 10 Hz)
            cal_len = min(50, n_samples)
            cal_accel = accel_raw[:cal_len]
            cal_gyro = gyro_raw[:cal_len]
            cal_ts = ts[:cal_len]

        calibration = calibrate_stationary_window(
            accel=cal_accel,
            gyro=cal_gyro,
            timestamps_ns=cal_ts,
            prior_profile=prior_calibration,
        )

        # 2. Estimate mounting alignment R_b^v
        alignment = estimate_mounting_alignment(
            stationary_accel=cal_accel,
            moving_gnss_speed_mps=trip.s_gnss_speed_mps,
            moving_gnss_bearing_deg=trip.s_gnss_bearing_deg,
            moving_gyro=gyro_raw,
            timestamps_ns=ts,
            gyro_bias=calibration.gyro_bias,
            min_speed_mps=self.min_speed_for_yaw_mps,
            moving_accel=accel_raw,
        )

        # 3. Transform to vehicle frame
        # f_m^v = R_b^v (f_m^b - b_a_prior)
        # omega_m^v = R_b^v (omega_m^b - b_g)
        f_v_unfilt = alignment.transform_specific_force(accel_raw, b_a_prior=calibration.accel_bias_prior)
        omega_v_unfilt = alignment.transform_angular_velocity(gyro_raw, b_g=calibration.gyro_bias)

        # 4. Filter vehicle-frame specific force and angular velocity
        f_v_filt = self.filter.filter_series(f_v_unfilt)
        omega_v_filt = self.filter.filter_series(omega_v_unfilt)

        # 5. Scan for recalibration events
        events = self.recal_detector.scan_series(
            f_v_filt,
            omega_v_filt,
            ts,
            stationary_mask=stationary_mask,
        )

        return PreprocessedTrip(
            trip_id=trip.trip_id,
            timestamps_ns=ts.copy(),
            f_m_v=f_v_filt,
            omega_m_v=omega_v_filt,
            f_m_v_unfiltered=f_v_unfilt,
            omega_m_v_unfiltered=omega_v_unfilt,
            calibration=calibration,
            alignment=alignment,
            recalibration_events=events,
            is_validated=trip.is_validated.copy(),
            quality_flags=trip.quality_flags.copy(),
        )
