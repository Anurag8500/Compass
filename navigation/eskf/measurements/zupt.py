"""Classical Gated Zero Velocity Update (ZUPT) for ESKF (Phase 5).

Provides:
- ClassicalZUPTDetector: A zero-ML standstill detector operating over a sliding
  window of calibrated inertial readings (angular velocity and specific force).
- ZUPTMeasurementModel: Constructs the zero-velocity observation vector,
  measurement Jacobian H_zupt (observing delta_v^n), and noise covariance R_zupt,
  and applies the generic ESKF update with innovation gating.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Deque, Optional, Sequence, Tuple
import numpy as np

from navigation.eskf.gating import MahalanobisGating
from navigation.eskf.state import ESKFState
from navigation.eskf.update import UpdateDiagnostics, eskf_update


STANDARD_GRAVITY: float = 9.80665


@dataclass(frozen=True)
class ZUPTDetectorConfig:
    """Thresholds and parameters for the classical zero-ML standstill detector.

    Attributes:
        window_size: Number of samples in sliding window (default 8, ~0.8s at 10 Hz).
        omega_max_rads: Maximum norm of angular velocity [rad/s] (default 0.05).
        accel_dev_max_mps2: Maximum deviation of mean specific force norm from g [m/s^2] (default 0.25).
        accel_var_max_mps4: Maximum variance of specific force norm [(m/s^2)^2] (default 0.015).
        speed_max_mps: Maximum auxiliary GNSS Doppler/speed [m/s] if provided (default 0.1).
        g_ref: Reference gravitational acceleration [m/s^2] (default 9.80665).
    """
    window_size: int = 8
    omega_max_rads: float = 0.05
    accel_dev_max_mps2: float = 0.25
    accel_var_max_mps4: float = 0.015
    speed_max_mps: float = 0.1
    g_ref: float = STANDARD_GRAVITY


@dataclass(frozen=True)
class ZUPTDetectorDiagnostics:
    """Detailed diagnostics emitted per detection evaluation.

    Attributes:
        is_stationary: Whether all standstill criteria are satisfied.
        window_len: Number of samples currently in the window.
        omega_norm_mean: Mean angular velocity norm across window [rad/s].
        accel_norm_dev: Deviation of mean specific force norm from g [m/s^2].
        accel_norm_var: Variance of specific force norm across window [(m/s^2)^2].
        speed_mps: Auxiliary speed checked against threshold [m/s] if provided.
    """
    is_stationary: bool
    window_len: int
    omega_norm_mean: float
    accel_norm_dev: float
    accel_norm_var: float
    speed_mps: Optional[float] = None


class ClassicalZUPTDetector:
    """Sliding-window classical detector for vehicle standstill.

    Uses physics-based invariants (quiescent specific force magnitude near g,
    near-zero specific force variance, near-zero angular velocity) without
    any machine-learning dependency.
    """

    def __init__(self, config: Optional[ZUPTDetectorConfig] = None) -> None:
        self.config = ZUPTDetectorConfig() if config is None else config
        self._omega_buffer: Deque[np.ndarray] = deque(maxlen=self.config.window_size)
        self._accel_buffer: Deque[np.ndarray] = deque(maxlen=self.config.window_size)

    def reset(self) -> None:
        """Clear internal buffers."""
        self._omega_buffer.clear()
        self._accel_buffer.clear()

    def push(
        self,
        omega_v: np.ndarray,
        f_v: np.ndarray,
        speed_mps: Optional[float] = None,
    ) -> ZUPTDetectorDiagnostics:
        """Push a new sample and evaluate standstill on current buffer.

        Args:
            omega_v: Angular velocity vector [wx, wy, wz] in vehicle frame [rad/s].
            f_v: Specific force vector [fx, fy, fz] in vehicle frame [m/s^2].
            speed_mps: Optional GNSS speed [m/s] for auxiliary validation.

        Returns:
            ZUPTDetectorDiagnostics with check results.
        """
        w = np.asarray(omega_v, dtype=np.float64).reshape(3)
        f = np.asarray(f_v, dtype=np.float64).reshape(3)

        self._omega_buffer.append(w)
        self._accel_buffer.append(f)

        if len(self._accel_buffer) < self.config.window_size:
            return ZUPTDetectorDiagnostics(
                is_stationary=False,
                window_len=len(self._accel_buffer),
                omega_norm_mean=float("nan"),
                accel_norm_dev=float("nan"),
                accel_norm_var=float("nan"),
                speed_mps=speed_mps,
            )

        return self.evaluate_window(
            omega_window=np.array(self._omega_buffer),
            accel_window=np.array(self._accel_buffer),
            speed_mps=speed_mps,
        )

    def evaluate_window(
        self,
        omega_window: np.ndarray,
        accel_window: np.ndarray,
        speed_mps: Optional[float] = None,
    ) -> ZUPTDetectorDiagnostics:
        """Evaluate standstill criteria over a static window of readings.

        Args:
            omega_window: Array of shape (N, 3) vehicle angular velocity [rad/s].
            accel_window: Array of shape (N, 3) vehicle specific force [m/s^2].
            speed_mps: Optional auxiliary speed [m/s].

        Returns:
            ZUPTDetectorDiagnostics.
        """
        w_arr = np.asarray(omega_window, dtype=np.float64)
        f_arr = np.asarray(accel_window, dtype=np.float64)

        n = len(f_arr)
        if n == 0 or len(w_arr) != n:
            return ZUPTDetectorDiagnostics(
                is_stationary=False,
                window_len=n,
                omega_norm_mean=float("nan"),
                accel_norm_dev=float("nan"),
                accel_norm_var=float("nan"),
                speed_mps=speed_mps,
            )

        # 1. Angular velocity norm check
        omega_norms = np.linalg.norm(w_arr, axis=1)
        mean_omega_norm = float(np.mean(omega_norms))
        max_omega_norm = float(np.max(omega_norms))

        # 2. Specific force norm checks
        accel_norms = np.linalg.norm(f_arr, axis=1)
        mean_accel_norm = float(np.mean(accel_norms))
        accel_dev = abs(mean_accel_norm - self.config.g_ref)
        accel_var = float(np.var(accel_norms))

        # Conditions
        cond_omega = max_omega_norm < self.config.omega_max_rads
        cond_accel_dev = accel_dev < self.config.accel_dev_max_mps2
        cond_accel_var = accel_var < self.config.accel_var_max_mps4

        cond_speed = True
        if speed_mps is not None and math.isfinite(speed_mps):
            cond_speed = abs(speed_mps) < self.config.speed_max_mps

        is_stat = bool(cond_omega and cond_accel_dev and cond_accel_var and cond_speed)

        return ZUPTDetectorDiagnostics(
            is_stationary=is_stat,
            window_len=n,
            omega_norm_mean=mean_omega_norm,
            accel_norm_dev=accel_dev,
            accel_norm_var=accel_var,
            speed_mps=speed_mps,
        )


@dataclass(frozen=True)
class ZUPTMeasurementConfig:
    """Configuration for ESKF ZUPT measurement update.

    Attributes:
        velocity_noise_sigma: 1-sigma pseudo-measurement noise [m/s] (default 0.03 m/s).
    """
    velocity_noise_sigma: float = 0.03


class ZUPTMeasurementModel:
    """Constructs zero-velocity pseudo-measurement for ESKF update."""

    def __init__(
        self,
        config: Optional[ZUPTMeasurementConfig] = None,
        gating: Optional[MahalanobisGating] = None,
    ) -> None:
        """Initialize ZUPT measurement model.

        Args:
            config: Optional ZUPT measurement noise configuration.
            gating: Optional MahalanobisGating instance (default 99% chi2 gate).
        """
        self.config = ZUPTMeasurementConfig() if config is None else config
        self.gating = MahalanobisGating(confidence_level=0.99) if gating is None else gating

    def create_measurement(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Construct zero-velocity measurement z, Jacobian H, and noise covariance R.

        Returns:
            Tuple of:
            - z: (3,) zero vector [0, 0, 0] [m/s]
            - H: (3, 15) Jacobian observing velocity error delta_v^n
            - R: (3, 3) diagonal noise covariance matrix
        """
        z = np.zeros(3, dtype=np.float64)

        H = np.zeros((3, 15), dtype=np.float64)
        H[0:3, 3:6] = np.eye(3, dtype=np.float64)

        sigma_sq = self.config.velocity_noise_sigma ** 2
        R = (np.eye(3, dtype=np.float64) * sigma_sq)

        return z, H, R

    def update(
        self,
        state: ESKFState,
        timestamp_ns: Optional[int] = None,
    ) -> Tuple[ESKFState, UpdateDiagnostics]:
        """Apply zero-velocity measurement update to ESKF state with innovation gating.

        Args:
            state: Current ESKF state before update.
            timestamp_ns: Optional timestamp override for updated state.

        Returns:
            Tuple of (updated_state, diagnostics).
        """
        z, H, R = self.create_measurement()
        h_val = state.velocity_enu

        return eskf_update(
            state=state,
            z=z,
            h_val=h_val,
            H=H,
            R=R,
            gating=self.gating,
            timestamp_ns=timestamp_ns,
        )
