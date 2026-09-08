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

Alignment Strategy & Observability Principles:
1. Roll and pitch (tilt): Estimated from specific force gravity reaction during stationary calibration.
   The rest support reaction vector is rotated onto vehicle vertical +Z_v = [0, 0, 1]^T.
2. Yaw (azimuth): Gyro integration provides orientation CHANGE only; it does not provide
   absolute mounting yaw by itself. Mounting yaw is resolved only when there is a valid
   observable azimuth constraint:
       a. An explicitly supplied/reference mounting yaw (`reference_yaw_deg`), OR
       b. Qualifying straight-line longitudinal acceleration correlation:
          During straight-line forward acceleration (v >= min_speed, dv/dt >= min_accel,
          |omega_z| <= max_yaw_rate), vehicle acceleration acts along +X_v. Leveled horizontal
          specific force correlates with this forward direction to resolve mounting yaw.
3. Unresolved Yaw Semantics: If available motion data do not provide sufficient observability
   (e.g., cruising, low speed, turning, or high circular dispersion across acceleration epochs),
   mounting yaw remains strictly UNRESOLVED (is_yaw_aligned=False, yaw_deg=0.0).
   The implementation NEVER fabricates an arbitrary or unobserved mounting yaw.
   In this state, +Z_v is guaranteed vertical/gravity-aligned, but horizontal axes (X_v, Y_v)
   remain provisional; downstream navigation fusion (Phase 5 ESKF) must resolve azimuth.
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


def rotation_matrix_to_euler_deg(R: np.ndarray) -> Tuple[float, float, float]:
    """Extract Z-Y-X Euler angles (roll, pitch, yaw) in degrees from 3x3 rotation matrix R.

    Convention: R = R_z(yaw) @ R_y(pitch) @ R_x(roll)
    """
    R_arr = np.asarray(R, dtype=np.float64)
    sin_p = -float(R_arr[2, 0])
    sin_p = max(-1.0, min(1.0, sin_p))
    pitch_rad = math.asin(sin_p)

    # Check for gimbal lock (|pitch| ≈ 90 deg)
    if abs(R_arr[2, 0]) < 0.999999:
        roll_rad = math.atan2(float(R_arr[2, 1]), float(R_arr[2, 2]))
        yaw_rad = math.atan2(float(R_arr[1, 0]), float(R_arr[0, 0]))
    else:
        roll_rad = 0.0
        yaw_rad = math.atan2(-float(R_arr[0, 1]), float(R_arr[1, 1]))

    return math.degrees(roll_rad), math.degrees(pitch_rad), math.degrees(yaw_rad)


@dataclass(frozen=True)
class MountingAlignment:
    """Fixed mounting transformation from device body frame to vehicle frame.

    Transforms vectors as:
        v_v = R_b_v @ v_b
        f_m^v = R_b_v (f_m^b - b_a_prior)
        omega_m^v = R_b_v (omega_m^b - b_g)

    CRITICAL YAW OBSERVABILITY & FRAME SEMANTICS:
    - When is_yaw_aligned=True:
        R_b^v resolves both vertical tilt and horizontal mounting azimuth.
        X_v = vehicle forward, Y_v = vehicle left/lateral, Z_v = vehicle up.
    - When is_yaw_aligned=False:
        R_b^v is strictly tilt/gravity-aligned. The +Z_v axis is definitively vertical
        (aligned with local gravity reaction at rest), but the horizontal mounting azimuth
        remains provisional/unresolved.
        In this case, X_v and Y_v must NOT be treated as definitively vehicle-forward/lateral;
        downstream navigation aiding (ESKF GNSS heading, Non-Holonomic Constraints)
        must resolve the remaining horizontal azimuth degree of freedom.

    Attributes:
        R_b_v: 3x3 direction cosine matrix satisfying v_v = R_b_v @ v_b.
        roll_deg: Estimated mounting roll angle in degrees.
        pitch_deg: Estimated mounting pitch angle in degrees.
        yaw_deg: Estimated mounting yaw angle in degrees (0.0 if unresolved).
        is_yaw_aligned: True if yaw has been resolved from valid observations.
        alignment_status: Diagnostic status string indicating calibration observability:
            - RESOLVED_REFERENCE_AZIMUTH: explicitly supplied mounting yaw.
            - RESOLVED_ACCELERATION_CORRELATION: derived from straight-line forward acceleration.
            - UNRESOLVED_INSUFFICIENT_OBSERVABILITY: insufficient straight-line acceleration epochs.
            - UNRESOLVED_HIGH_CIRCULAR_DISPERSION: acceleration correlation lacked angular consistency.
    """
    R_b_v: np.ndarray
    roll_deg: float
    pitch_deg: float
    yaw_deg: float
    is_yaw_aligned: bool = False
    alignment_status: str = "NOMINAL"

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
    moving_accel: Optional[np.ndarray] = None,
    reference_yaw_deg: Optional[float] = None,
    min_accel_mps2: float = 0.40,
    max_yaw_rate_rads: float = 0.05,
    min_valid_epochs: int = 10,
    max_circular_dispersion_deg: float = 15.0,
) -> MountingAlignment:
    """Estimate fixed device-to-vehicle mounting rotation R_b^v.

    TILT ESTIMATION (Observability: Gravitational reaction at rest):
    The support reaction vector at rest is aligned onto vehicle +Z = [0, 0, 1].

    YAW ESTIMATION (Observability: Straight-line vehicle acceleration):
    Gyro integration gives orientation change, NOT absolute yaw relative to vehicle.
    Mounting yaw is observable ONLY when:
    1. A reference yaw is explicitly supplied (`reference_yaw_deg`), OR
    2. Straight-line longitudinal acceleration epochs can be correlated:
       During straight-line forward acceleration (v >= min_speed, dv/dt >= min_accel,
       |omega_z| <= max_yaw_rate), the vehicle acceleration vector acts along vehicle +X.
       The horizontal specific force in the tilt-leveled device frame correlates with this
       forward direction, resolving mounting yaw with confidence gating.
    CRITICAL YAW OBSERVABILITY & FRAME SEMANTICS:
    - When is_yaw_aligned=True:
        R_b^v resolves both vertical tilt and horizontal mounting azimuth.
        X_v = vehicle forward, Y_v = vehicle left/lateral, Z_v = vehicle up.
    - When is_yaw_aligned=False:
        R_b^v provides only gravity/tilt alignment. The +Z_v axis is definitively vertical
        (aligned with local gravity reaction at rest), but horizontal azimuth remains
        provisional/unresolved. In this state, X_v and Y_v must NOT be treated as
        definitively vehicle-forward/lateral; downstream navigation fusion (Phase 5 ESKF)
        is responsible for resolving the remaining horizontal heading degree of freedom.
    If observability criteria are not met, yaw is NOT fabricated; the algorithm
    explicitly returns is_yaw_aligned=False and yaw_deg=0.0 (where 0.0 is a sentinel
    reporting value and NOT a measured mounting azimuth).

    Args:
        stationary_accel: (N, 3) stationary specific force measurements at rest (m/s^2).
        moving_gnss_speed_mps: Optional (M,) GNSS ground speed during moving intervals (m/s).
        moving_gnss_bearing_deg: Optional (M,) GNSS track bearing (0-360 deg).
        moving_gyro: Optional (M, 3) gyro measurements matching moving interval (rad/s).
        timestamps_ns: Optional (M,) timestamps.
        gyro_bias: Optional 3-axis gyro bias.
        min_speed_mps: Minimum speed threshold to evaluate motion (default: 3.0 m/s).
        moving_accel: Optional (M, 3) accelerometer measurements matching moving interval.
        reference_yaw_deg: Optional explicitly known/measured mounting yaw angle (deg).
        min_accel_mps2: Minimum longitudinal acceleration for forward correlation (m/s^2).
        max_yaw_rate_rads: Maximum yaw rate threshold for straight-line gating (rad/s).
        min_valid_epochs: Minimum number of qualifying epochs required to declare yaw resolved.
        max_circular_dispersion_deg: Maximum allowable circular standard deviation of yaw candidates (deg).

    Returns:
        MountingAlignment holding R_b^v, Euler angles, and resolution status.
    """
    f_stat = np.asarray(stationary_accel, dtype=np.float64)
    if f_stat.ndim != 2 or f_stat.shape[1] != 3 or f_stat.shape[0] == 0:
        raise ValueError(f"stationary_accel must be non-empty (N, 3), got {f_stat.shape}")

    if not np.isfinite(f_stat).all():
        raise ValueError("stationary_accel contains non-finite values (NaN/Inf)")

    # 1. Tilt alignment: Align measured support reaction force to vehicle vertical +Z = [0, 0, 1]
    f_mean = np.mean(f_stat, axis=0)
    norm_f = float(np.linalg.norm(f_mean))
    if norm_f < 1e-4:
        R_tilt = np.eye(3, dtype=np.float64)
    else:
        v_target_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        R_tilt = rotation_matrix_from_vectors(f_mean, v_target_up)

    # 2. Yaw alignment
    yaw_deg = 0.0
    is_yaw_aligned = False
    status = "UNRESOLVED_INSUFFICIENT_OBSERVABILITY"

    # Option A: Explicit reference yaw provided
    if reference_yaw_deg is not None and np.isfinite(reference_yaw_deg):
        yaw_deg = float(reference_yaw_deg)
        is_yaw_aligned = True
        status = "RESOLVED_REFERENCE_AZIMUTH"

    # Option B: Longitudinal acceleration correlation during straight-line vehicle motion
    elif (
        moving_gnss_speed_mps is not None
        and moving_accel is not None
        and moving_gyro is not None
        and timestamps_ns is not None
    ):
        speeds = np.asarray(moving_gnss_speed_mps, dtype=np.float64)
        accels = np.asarray(moving_accel, dtype=np.float64)
        gyros = np.asarray(moving_gyro, dtype=np.float64)
        ts = np.asarray(timestamps_ns, dtype=np.int64)

        n_mov = len(speeds)
        if n_mov == len(accels) == len(gyros) == len(ts) and n_mov >= min_valid_epochs:
            dt_s = np.diff(ts, prepend=ts[0]) / 1e9
            dt_s[dt_s <= 0] = 0.1
            dt_s[dt_s > 1.0] = 0.1

            cum_time = np.cumsum(dt_s)
            if cum_time[-1] > cum_time[0]:
                gnss_accel = np.gradient(speeds, cum_time)
            else:
                gnss_accel = np.zeros_like(speeds)

            # Level gyros and accels into intermediate horizontal plane
            b_g = np.asarray(gyro_bias, dtype=np.float64) if gyro_bias is not None else np.zeros(3)
            gyros_level = (R_tilt @ (gyros - b_g).T).T
            accels_level = (R_tilt @ accels.T).T

            yaw_rate = np.abs(gyros_level[:, 2])
            h_accel_norm = np.linalg.norm(accels_level[:, :2], axis=1)

            qualifying_mask = (
                np.isfinite(speeds)
                & (speeds >= min_speed_mps)
                & np.isfinite(gnss_accel)
                & (gnss_accel >= min_accel_mps2)
                & (yaw_rate <= max_yaw_rate_rads)
                & (h_accel_norm >= 0.30)
            )

            # If GNSS bearing is available, verify heading is stable
            if moving_gnss_bearing_deg is not None:
                bearings = np.asarray(moving_gnss_bearing_deg, dtype=np.float64)
                if len(bearings) == n_mov:
                    bearing_diff = np.abs(np.diff(bearings, prepend=bearings[0]))
                    bearing_diff = np.minimum(bearing_diff, 360.0 - bearing_diff)
                    bearing_rate_deg_s = bearing_diff / dt_s
                    qualifying_mask &= (bearing_rate_deg_s <= 3.0)

            qualifying_idx = np.where(qualifying_mask)[0]

            if len(qualifying_idx) >= min_valid_epochs:
                f_lx = accels_level[qualifying_idx, 0]
                f_ly = accels_level[qualifying_idx, 1]

                candidate_yaws = np.arctan2(-f_ly, f_lx)

                sin_sum = float(np.sum(np.sin(candidate_yaws)))
                cos_sum = float(np.sum(np.cos(candidate_yaws)))
                mean_yaw_rad = math.atan2(sin_sum, cos_sum)

                R_bar = math.sqrt(sin_sum * sin_sum + cos_sum * cos_sum) / len(qualifying_idx)
                if R_bar >= 0.999999:
                    circular_std_deg = 0.0
                elif R_bar > 1e-4:
                    circular_std_deg = math.degrees(math.sqrt(-2.0 * math.log(R_bar)))
                else:
                    circular_std_deg = 180.0

                if circular_std_deg <= max_circular_dispersion_deg:
                    yaw_deg = math.degrees(mean_yaw_rad)
                    is_yaw_aligned = True
                    status = f"RESOLVED_ACCELERATION_CORRELATION (N={len(qualifying_idx)}, std={circular_std_deg:.1f}deg)"
                else:
                    status = f"UNRESOLVED_HIGH_CIRCULAR_DISPERSION (std={circular_std_deg:.1f}deg > {max_circular_dispersion_deg}deg)"
            else:
                status = f"UNRESOLVED_INSUFFICIENT_OBSERVABILITY (found {len(qualifying_idx)} < {min_valid_epochs} epochs)"

    # 3. Form authoritative R_b^v: R_b^v = R_z(yaw) @ R_tilt
    # Standard active rotation matrix about Z_v:
    # R_z(yaw) = [[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]]
    yaw_rad = math.radians(yaw_deg)
    cos_y = math.cos(yaw_rad)
    sin_y = math.sin(yaw_rad)
    R_yaw = np.array([
        [cos_y, -sin_y, 0.0],
        [sin_y, cos_y, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    R_b_v = R_yaw @ R_tilt

    # Extract diagnostic Euler angles directly from authoritative R_b^v
    roll_deg, pitch_deg, reported_yaw_deg = rotation_matrix_to_euler_deg(R_b_v)

    final_yaw_deg = reported_yaw_deg if is_yaw_aligned else 0.0

    return MountingAlignment(
        R_b_v=R_b_v,
        roll_deg=roll_deg,
        pitch_deg=pitch_deg,
        yaw_deg=final_yaw_deg,
        is_yaw_aligned=is_yaw_aligned,
        alignment_status=status,
    )
