"""Device-to-vehicle mounting alignment for COMPASS (Phase 3).

Implements fixed mounting orientation estimation (R_b^v) transforming
device-frame IMU measurements into canonical vehicle frame coordinates:
    f_m^v = R_b^v (f_m^b - b_a_prior)
    omega_m^v = R_b^v (omega_m^b - b_g)

Coordinate Frames:
- Device/Body Frame (b): Physical casing of smartphone or external sensor.
- Vehicle Frame (v):
    X_v: Forward longitudinal direction.
    Y_v: Lateral direction (perpendicular to forward).
    Z_v: Upward vertical direction (level at rest yields f_z^v ≈ +9.81 m/s^2).
- Navigation Frame (n): Local East-North-Up (ENU) tangent plane.

Alignment Strategy:
1. Roll and pitch (tilt): Estimated from specific force gravity reaction during stationary calibration.
2. Yaw (azimuth): Estimated by comparing gyro-integrated heading against GNSS track heading
   during straight-line vehicle motion above configurable minimum speed threshold (e.g. >= 3.0 m/s).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple
import numpy as np

from navigation.preprocessing.calibration import CalibrationProfile
from navigation.schemas.imu import AlignedIMUSample, RawIMUSample


def rotation_matrix_from_vectors(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    """Compute the minimum-angle 3x3 rotation matrix R that rotates v_from onto v_to: R @ v_from = v_to."""
    a = v_from / np.linalg.norm(v_from)
    b = v_to / np.linalg.norm(v_to)
    dot = float(np.dot(a, b))

    if dot > 0.99999999:
        return np.eye(3, dtype=np.float64)
    if dot < -0.99999999:
        # 180-degree rotation: pick any orthogonal axis
        ortho = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, ortho)
        axis /= np.linalg.norm(axis)
        # Rodrigues for 180 degrees
        K = np.array([
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ])
        return np.eye(3) + 2.0 * (K @ K)

    v = np.cross(a, b)
    s = np.linalg.norm(v)
    K = np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])
    return np.eye(3) + K + (K @ K) * ((1.0 - dot) / (s * s))


@dataclass(frozen=True)
class MountingAlignment:
    """Fixed mounting transformation from device body frame to vehicle frame.

    Attributes:
        R_b_v: 3x3 direction cosine matrix satisfying v_v = R_b_v @ v_b.
        roll_deg: Estimated mounting roll angle in degrees.
        pitch_deg: Estimated mounting pitch angle in degrees.
        yaw_deg: Estimated mounting yaw angle in degrees.
        is_yaw_aligned: True if yaw has been resolved from vehicle motion & GNSS track.
    """
    R_b_v: np.ndarray
    roll_deg: float
    pitch_deg: float
    yaw_deg: float
    is_yaw_aligned: bool = False

    def transform_specific_force(
        self,
        accel_b: np.ndarray | Tuple[float, float, float],
        b_a_prior: Optional[Tuple[float, float, float] | np.ndarray] = None,
    ) -> np.ndarray:
        """Transform specific force from device frame to vehicle frame.

        f_m^v = R_b^v (f_m^b - b_a_prior)
        """
        arr = np.asarray(accel_b, dtype=np.float64)
        if b_a_prior is not None:
            arr = arr - np.asarray(b_a_prior, dtype=np.float64)
        if arr.ndim == 1:
            return self.R_b_v @ arr
        return (self.R_b_v @ arr.T).T

    def transform_angular_velocity(
        self,
        gyro_b: np.ndarray | Tuple[float, float, float],
        b_g: Optional[Tuple[float, float, float] | np.ndarray] = None,
    ) -> np.ndarray:
        """Transform angular velocity from device frame to vehicle frame.

        omega_m^v = R_b^v (omega_m^b - b_g)
        """
        arr = np.asarray(gyro_b, dtype=np.float64)
        if b_g is not None:
            arr = arr - np.asarray(b_g, dtype=np.float64)
        if arr.ndim == 1:
            return self.R_b_v @ arr
        return (self.R_b_v @ arr.T).T

    def align_sample(
        self,
        raw_sample: RawIMUSample,
        calibration: Optional[CalibrationProfile] = None,
    ) -> AlignedIMUSample:
        """Convert a single RawIMUSample into a calibrated AlignedIMUSample in vehicle frame."""
        b_g = calibration.gyro_bias if calibration is not None else (0.0, 0.0, 0.0)
        b_a = calibration.accel_bias_prior if calibration is not None else (0.0, 0.0, 0.0)

        f_v = self.transform_specific_force(raw_sample.accel, b_a_prior=b_a)
        omega_v = self.transform_angular_velocity(raw_sample.gyro, b_g=b_g)

        is_usable = bool(raw_sample.quality_flags & 0x07 == 0)

        return AlignedIMUSample(
            timestamp_ns=raw_sample.timestamp_ns,
            accel_vehicle=(float(f_v[0]), float(f_v[1]), float(f_v[2])),
            gyro_vehicle=(float(omega_v[0]), float(omega_v[1]), float(omega_v[2])),
            quality_flags=raw_sample.quality_flags,
            is_usable_for_integration=is_usable,
        )


def estimate_mounting_alignment(
    stationary_accel: np.ndarray,
    moving_gnss_speed_mps: Optional[np.ndarray] = None,
    moving_gnss_bearing_deg: Optional[np.ndarray] = None,
    moving_gyro: Optional[np.ndarray] = None,
    timestamps_ns: Optional[np.ndarray] = None,
    gyro_bias: Optional[Tuple[float, float, float] | np.ndarray] = None,
    min_speed_mps: float = 3.0,
) -> MountingAlignment:
    """Estimate fixed device-to-vehicle mounting rotation R_b^v from stationary gravity and GNSS course.

    Args:
        stationary_accel: (N, 3) stationary specific force measurements at rest (m/s^2).
        moving_gnss_speed_mps: Optional (M,) GNSS ground speed during moving intervals (m/s).
        moving_gnss_bearing_deg: Optional (M,) GNSS track bearing (0-360 deg, clockwise from True North).
        moving_gyro: Optional (M, 3) gyro measurements matching moving interval (rad/s).
        timestamps_ns: Optional (M,) timestamps for gyro integration.
        gyro_bias: Optional 3-axis gyro bias to subtract prior to heading comparison.
        min_speed_mps: Minimum speed threshold to evaluate GNSS heading (default: 3.0 m/s).

    Returns:
        MountingAlignment holding R_b^v and Euler angles.
    """
    f_stat = np.asarray(stationary_accel, dtype=np.float64)
    if f_stat.ndim != 2 or f_stat.shape[1] != 3 or f_stat.shape[0] == 0:
        raise ValueError(f"stationary_accel must be non-empty (N, 3), got {f_stat.shape}")

    # 1. Tilt alignment: Align measured support reaction force to vehicle vertical +Z = [0, 0, 1]
    f_mean = np.mean(f_stat, axis=0)
    norm_f = np.linalg.norm(f_mean)
    if norm_f < 1e-4:
        R_tilt = np.eye(3, dtype=np.float64)
    else:
        # Rotates measured gravity reaction vector in body frame onto vehicle [0, 0, 1]
        v_target_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        R_tilt = rotation_matrix_from_vectors(f_mean, v_target_up)

    # Compute tilt angles in degrees
    # If phone is flat screen up: f_mean = [0, 0, 9.81], R_tilt = I, pitch=0, roll=0.
    f_unit = f_mean / (norm_f if norm_f > 1e-4 else 1.0)
    pitch_deg = math.degrees(math.atan2(f_unit[0], math.sqrt(f_unit[1] * f_unit[1] + f_unit[2] * f_unit[2])))
    roll_deg = math.degrees(math.atan2(-f_unit[1], f_unit[2]))

    # 2. Yaw alignment from GNSS track heading vs device sensed direction
    yaw_deg = 0.0
    is_yaw_aligned = False

    if (
        moving_gnss_speed_mps is not None
        and moving_gnss_bearing_deg is not None
        and moving_gyro is not None
        and timestamps_ns is not None
    ):
        speeds = np.asarray(moving_gnss_speed_mps, dtype=np.float64)
        bearings = np.asarray(moving_gnss_bearing_deg, dtype=np.float64)
        gyros = np.asarray(moving_gyro, dtype=np.float64)
        ts = np.asarray(timestamps_ns, dtype=np.int64)

        # Find valid moving intervals above speed threshold with finite bearing
        valid_motion_mask = (speeds >= min_speed_mps) & np.isfinite(bearings) & (bearings >= 0.0)

        if np.sum(valid_motion_mask) >= 10:
            # Tilt-correct the gyros into the leveled horizontal plane
            b_g = np.asarray(gyro_bias, dtype=np.float64) if gyro_bias is not None else np.zeros(3)
            gyros_debiased = gyros - b_g
            gyros_level = (R_tilt @ gyros_debiased.T).T

            # In leveled vehicle frame, z-axis is vertical yaw rate: omega_z
            # Integrate heading relative to first moving sample
            dt_s = np.diff(ts, prepend=ts[0]) / 1e9
            dt_s[dt_s < 0] = 0.0
            dt_s[dt_s > 1.0] = 0.1  # clamp gaps

            # Heading change from gyro integration (rad)
            gyro_heading_rad = np.cumsum(gyros_level[:, 2] * dt_s)
            gyro_heading_deg = np.degrees(gyro_heading_rad)

            # Circular difference between GNSS bearing and gyro integrated heading
            # GNSS bearing is clockwise from True North (Navigation frame).
            # Over reasonably straight segments, difference resolves mounting azimuth offset.
            valid_idx = np.where(valid_motion_mask)[0]
            if len(valid_idx) >= 10:
                gnss_sub = bearings[valid_idx]
                gyro_sub = gyro_heading_deg[valid_idx]
                angle_diffs = (gnss_sub - gyro_sub + 180.0) % 360.0 - 180.0
                # Median circular offset
                sin_sum = np.sum(np.sin(np.radians(angle_diffs)))
                cos_sum = np.sum(np.cos(np.radians(angle_diffs)))
                yaw_deg = float(np.degrees(np.arctan2(sin_sum, cos_sum)))
                is_yaw_aligned = True

    # 3. Form full R_b^v: R_b^v = R_z(yaw) @ R_tilt
    # Note: R_z(yaw) represents rotation about vehicle vertical axis
    yaw_rad = math.radians(yaw_deg)
    cos_y = math.cos(yaw_rad)
    sin_y = math.sin(yaw_rad)
    R_yaw = np.array([
        [cos_y, sin_y, 0.0],
        [-sin_y, cos_y, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    R_b_v = R_yaw @ R_tilt

    return MountingAlignment(
        R_b_v=R_b_v,
        roll_deg=roll_deg,
        pitch_deg=pitch_deg,
        yaw_deg=yaw_deg,
        is_yaw_aligned=is_yaw_aligned,
    )
