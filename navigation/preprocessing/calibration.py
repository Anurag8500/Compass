"""Stationary-window IMU calibration for COMPASS (Phase 3).

Implements initial sensor calibration from stationary epochs:
1. Gyroscope bias estimation:
     b_g = mean(gyro)
   since true angular velocity is approximately zero at rest.
2. Initial roll and pitch estimation:
   Computed from the measured specific-force gravity reaction vector at rest.
3. Accelerometer bias prior:
   Uses the nominal prior b_a = [0, 0, 0]^T.

CRITICAL PHYSICAL LIMITATION:
A single arbitrary static pose CANNOT uniquely separate full 3D accelerometer
bias from physical tilt (5 unknowns: 2 tilt angles + 3 bias components, with
only 3 measurement equations). Nominal zero prior is assigned; dynamic refinement
is performed downstream by the ESKF during vehicle motion when external aiding
(GNSS, NHC, ZUPT) is available.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class CalibrationProfile:
    """Immutable calibration parameters determined from stationary observation.

    Attributes:
        gyro_bias: 3-axis estimated angular velocity bias [b_gx, b_gy, b_gz] in rad/s.
        accel_bias_prior: Initial nominal 3-axis specific force bias [b_ax, b_ay, b_az] in m/s^2.
            Defaults strictly to [0.0, 0.0, 0.0] for initial calibration.
        initial_roll_rad: Estimated initial bank/roll angle relative to horizontal plane (rad).
        initial_pitch_rad: Estimated initial elevation/pitch angle relative to horizontal plane (rad).
        sample_count: Number of stationary samples averaged.
        duration_s: Duration of the stationary window in seconds.
        is_valid: True if window met variance and sample count thresholds.
        gyro_std: Standard deviation of gyro samples during stationary epoch (rad/s).
        accel_norm: Mean Euclidean norm of specific force during stationary epoch (m/s^2).
    """
    gyro_bias: Tuple[float, float, float]
    initial_roll_rad: float
    initial_pitch_rad: float
    sample_count: int
    duration_s: float
    is_valid: bool
    accel_bias_prior: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    gyro_std: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    accel_norm: float = 9.80665

    def to_dict(self) -> Dict[str, Any]:
        """Serialize calibration profile to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CalibrationProfile:
        """Deserialize calibration profile from dictionary."""
        return cls(
            gyro_bias=(float(data["gyro_bias"][0]), float(data["gyro_bias"][1]), float(data["gyro_bias"][2])),
            initial_roll_rad=float(data["initial_roll_rad"]),
            initial_pitch_rad=float(data["initial_pitch_rad"]),
            sample_count=int(data["sample_count"]),
            duration_s=float(data["duration_s"]),
            is_valid=bool(data["is_valid"]),
            accel_bias_prior=(
                float(data.get("accel_bias_prior", [0.0, 0.0, 0.0])[0]),
                float(data.get("accel_bias_prior", [0.0, 0.0, 0.0])[1]),
                float(data.get("accel_bias_prior", [0.0, 0.0, 0.0])[2]),
            ),
            gyro_std=(
                float(data.get("gyro_std", [0.0, 0.0, 0.0])[0]),
                float(data.get("gyro_std", [0.0, 0.0, 0.0])[1]),
                float(data.get("gyro_std", [0.0, 0.0, 0.0])[2]),
            ),
            accel_norm=float(data.get("accel_norm", 9.80665)),
        )


def calibrate_stationary_window(
    accel: np.ndarray,
    gyro: np.ndarray,
    timestamps_ns: Optional[np.ndarray] = None,
    min_samples: int = 20,
    prior_profile: Optional[CalibrationProfile] = None,
) -> CalibrationProfile:
    """Calibrate gyro bias and initial attitude tilt from a stationary IMU sample window.

    Args:
        accel: (N, 3) measured specific force in device body frame (m/s^2).
        gyro: (N, 3) measured angular velocity in device frame (rad/s).
        timestamps_ns: Optional (N,) monotonic timestamps in nanoseconds.
        min_samples: Minimum required stationary samples (default: 20).
        prior_profile: Optional calibration profile from a previous session.
            Treated strictly as an initial prior with uncertainty, never as
            an unquestionable replacement for fresh stationary observation.

    Returns:
        CalibrationProfile with estimated biases and tilt angles.

    Raises:
        ValueError: If input arrays have mismatched or invalid dimensions.
    """
    accel_arr = np.asarray(accel, dtype=np.float64)
    gyro_arr = np.asarray(gyro, dtype=np.float64)

    if accel_arr.ndim != 2 or accel_arr.shape[1] != 3:
        raise ValueError(f"accel must have shape (N, 3), got {accel_arr.shape}")
    if gyro_arr.ndim != 2 or gyro_arr.shape[1] != 3:
        raise ValueError(f"gyro must have shape (N, 3), got {gyro_arr.shape}")
    if accel_arr.shape[0] != gyro_arr.shape[0]:
        raise ValueError(f"Sample count mismatch: accel={accel_arr.shape[0]} vs gyro={gyro_arr.shape[0]}")

    n_samples = accel_arr.shape[0]

    # Handle insufficient samples fallback
    if n_samples < min_samples or not np.isfinite(accel_arr).all() or not np.isfinite(gyro_arr).all():
        if prior_profile is not None and prior_profile.is_valid:
            # Degraded fallback to prior profile
            return CalibrationProfile(
                gyro_bias=prior_profile.gyro_bias,
                initial_roll_rad=prior_profile.initial_roll_rad,
                initial_pitch_rad=prior_profile.initial_pitch_rad,
                sample_count=n_samples,
                duration_s=0.0,
                is_valid=False,
                accel_bias_prior=prior_profile.accel_bias_prior,
                gyro_std=prior_profile.gyro_std,
                accel_norm=prior_profile.accel_norm,
            )
        return CalibrationProfile(
            gyro_bias=(0.0, 0.0, 0.0),
            initial_roll_rad=0.0,
            initial_pitch_rad=0.0,
            sample_count=n_samples,
            duration_s=0.0,
            is_valid=False,
            accel_bias_prior=(0.0, 0.0, 0.0),
            gyro_std=(0.0, 0.0, 0.0),
            accel_norm=0.0,
        )

    # 1. Gyroscope bias estimation: b_g = mean(omega)
    gyro_bias_vec = np.mean(gyro_arr, axis=0)
    gyro_std_vec = np.std(gyro_arr, axis=0)

    # 2. Specific force gravity reaction direction at rest
    f_mean = np.mean(accel_arr, axis=0)
    f_norm = float(np.linalg.norm(f_mean))

    if f_norm < 1e-4:
        roll = 0.0
        pitch = 0.0
    else:
        u_g = f_mean / f_norm
        # Pitch: tilt nose-up
        pitch = math.atan2(u_g[0], math.sqrt(u_g[1] * u_g[1] + u_g[2] * u_g[2]))
        # Roll: bank right-down
        roll = math.atan2(-u_g[1], u_g[2])

    # Duration calculation and timestamp validation
    duration_s = 0.0
    if timestamps_ns is not None:
        ts_arr = np.asarray(timestamps_ns, dtype=np.int64)
        if len(ts_arr) != n_samples:
            raise ValueError(f"timestamps_ns length ({len(ts_arr)}) must match sample count ({n_samples})")
        if not np.isfinite(ts_arr).all():
            raise ValueError("timestamps_ns contains non-finite values")
        if np.any(ts_arr < 0):
            raise ValueError("timestamps_ns cannot contain negative values")
        if n_samples > 1 and np.any(np.diff(ts_arr) < 0):
            raise ValueError("timestamps_ns must be monotonic non-decreasing")
        duration_s = float(ts_arr[-1] - ts_arr[0]) / 1e9

    return CalibrationProfile(
        gyro_bias=(float(gyro_bias_vec[0]), float(gyro_bias_vec[1]), float(gyro_bias_vec[2])),
        initial_roll_rad=float(roll),
        initial_pitch_rad=float(pitch),
        sample_count=n_samples,
        duration_s=duration_s,
        is_valid=True,
        accel_bias_prior=(0.0, 0.0, 0.0),
        gyro_std=(float(gyro_std_vec[0]), float(gyro_std_vec[1]), float(gyro_std_vec[2])),
        accel_norm=f_norm,
    )
