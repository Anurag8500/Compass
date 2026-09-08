"""Conservative recalibration trigger detector for COMPASS (Phase 3).

Detects physical sensor orientation discontinuities or mounting dislodgments
(e.g., phone falling out of mount, knocked by driver, or slipping)
that are inconsistent with plausible road vehicle kinematics.

Design Requirements:
- Threshold-based, configurable, deterministic.
- False positives are minimized by requiring:
  1. An angular velocity shock step (|Delta omega| > threshold, e.g. > 5.0 rad/s), OR
  2. A persistent gravity direction shift (> threshold) sustained over a resting
     or low-dynamics epoch (to avoid triggering on normal road grades or braking), OR
  3. A confirmed stationary epoch gravity shift relative to initial calibration.
- Non-destructive: flags a RecalibrationEvent rather than silently modifying
  active navigation state or calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class RecalibrationEvent:
    """Event triggered when sensor orientation discontinuity is detected."""
    triggered: bool
    timestamp_ns: int
    reason: str
    metric_value: float
    threshold: float


class RecalibrationDetector:
    """Conservative detector for physical sensor orientation jumps."""

    def __init__(
        self,
        gravity_shift_threshold_deg: float = 35.0,
        angular_rate_step_threshold_rads: float = 5.0,
        persistence_samples: int = 20,
        stationary_persistence_samples: int = 5,
        cooldown_samples: int = 50,
    ) -> None:
        """Initialize RecalibrationDetector.

        Args:
            gravity_shift_threshold_deg: Persistent gravity direction shift threshold (degrees).
            angular_rate_step_threshold_rads: Max instantaneous angular velocity step (rad/s).
            persistence_samples: Consecutive samples the shift must persist before triggering in motion.
            stationary_persistence_samples: Required persistence samples when vehicle is confirmed stationary.
            cooldown_samples: Number of samples to wait before triggering another event.
        """
        self.shift_threshold_rad = math.radians(gravity_shift_threshold_deg)
        self.gyro_step_threshold = float(angular_rate_step_threshold_rads)
        self.persistence_samples = int(persistence_samples)
        self.stationary_persistence_samples = int(stationary_persistence_samples)
        self.cooldown_samples = int(cooldown_samples)

        self._last_gyro: Optional[np.ndarray] = None
        self._baseline_gravity_unit: Optional[np.ndarray] = None
        self._persistent_shift_counter: int = 0
        self._cooldown_counter: int = 0

    def reset(self, keep_baseline: bool = False) -> None:
        """Reset internal detector state."""
        self._last_gyro = None
        if not keep_baseline:
            self._baseline_gravity_unit = None
        self._persistent_shift_counter = 0
        self._cooldown_counter = 0

    def set_baseline_gravity(self, gravity_vec: np.ndarray | Tuple[float, float, float]) -> None:
        """Set baseline gravity unit vector from stationary calibration."""
        g_arr = np.asarray(gravity_vec, dtype=np.float64)
        norm_g = np.linalg.norm(g_arr)
        if norm_g > 1e-3:
            self._baseline_gravity_unit = g_arr / norm_g

    def evaluate_sample(
        self,
        accel: np.ndarray | Tuple[float, float, float],
        gyro: np.ndarray | Tuple[float, float, float],
        timestamp_ns: int,
        is_stationary: bool = False,
    ) -> RecalibrationEvent:
        """Evaluate a single sample for physical orientation discontinuity.

        Args:
            accel: Specific force vector (m/s^2).
            gyro: Angular velocity vector (rad/s).
            timestamp_ns: Monotonic epoch timestamp in nanoseconds.
            is_stationary: Optional flag indicating vehicle is known stationary.

        Returns:
            RecalibrationEvent detailing whether an event was triggered.
        """
        a_vec = np.asarray(accel, dtype=np.float64)
        g_vec = np.asarray(gyro, dtype=np.float64)

        if not np.isfinite(a_vec).all() or not np.isfinite(g_vec).all():
            return RecalibrationEvent(
                triggered=False,
                timestamp_ns=timestamp_ns,
                reason="Non-finite input rejected",
                metric_value=0.0,
                threshold=self.gyro_step_threshold,
            )

        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1

        # 1. Angular rate shock step check: |omega_k - omega_{k-1}| > threshold
        if self._last_gyro is not None:
            gyro_diff = float(np.linalg.norm(g_vec - self._last_gyro))
            if gyro_diff > self.gyro_step_threshold and self._cooldown_counter == 0:
                self._last_gyro = g_vec.copy()
                self._cooldown_counter = self.cooldown_samples
                return RecalibrationEvent(
                    triggered=True,
                    timestamp_ns=timestamp_ns,
                    reason=f"Angular velocity shock step ({gyro_diff:.2f} rad/s > {self.gyro_step_threshold:.2f} rad/s)",
                    metric_value=gyro_diff,
                    threshold=self.gyro_step_threshold,
                )
        self._last_gyro = g_vec.copy()

        # 2. Gravity vector orientation shift check
        a_norm = float(np.linalg.norm(a_vec))
        g_norm = 9.80665
        g_rate = float(np.linalg.norm(g_vec))

        # Vehicle must be in low dynamics (or explicitly stationary) to evaluate gravity direction
        is_steady = (abs(a_norm - g_norm) < 1.5) and (g_rate < 0.20 or is_stationary)

        # Persistence threshold depends on stationary confirmation
        required_persistence = (
            self.stationary_persistence_samples if is_stationary else self.persistence_samples
        )

        if is_steady and a_norm > 1e-3:
            current_g_unit = a_vec / a_norm
            if self._baseline_gravity_unit is None:
                self._baseline_gravity_unit = current_g_unit.copy()
            else:
                dot = float(np.clip(np.dot(self._baseline_gravity_unit, current_g_unit), -1.0, 1.0))
                angle_shift = math.acos(dot)

                if angle_shift > self.shift_threshold_rad:
                    self._persistent_shift_counter += 1
                    if self._persistent_shift_counter >= required_persistence and self._cooldown_counter == 0:
                        shift_deg = math.degrees(angle_shift)
                        thresh_deg = math.degrees(self.shift_threshold_rad)
                        self._cooldown_counter = self.cooldown_samples
                        # Update baseline to new pose to avoid repeated triggering
                        self._baseline_gravity_unit = current_g_unit.copy()
                        self._persistent_shift_counter = 0
                        reason_prefix = "Stationary-confirmed" if is_stationary else "Persistent"
                        return RecalibrationEvent(
                            triggered=True,
                            timestamp_ns=timestamp_ns,
                            reason=f"{reason_prefix} gravity direction shift ({shift_deg:.1f} deg > {thresh_deg:.1f} deg)",
                            metric_value=shift_deg,
                            threshold=thresh_deg,
                        )
                else:
                    self._persistent_shift_counter = max(0, self._persistent_shift_counter - 1)
        else:
            # During violent/high-dynamic motion, do not accumulate; decay rapidly
            self._persistent_shift_counter = max(0, self._persistent_shift_counter - 2)

        return RecalibrationEvent(
            triggered=False,
            timestamp_ns=timestamp_ns,
            reason="Nominal",
            metric_value=0.0,
            threshold=math.degrees(self.shift_threshold_rad),
        )

    def scan_series(
        self,
        accel_series: np.ndarray,
        gyro_series: np.ndarray,
        timestamps_ns: np.ndarray,
        stationary_mask: Optional[np.ndarray] = None,
    ) -> List[RecalibrationEvent]:
        """Scan a complete time series and return all detected recalibration events."""
        self.reset(keep_baseline=(self._baseline_gravity_unit is not None))
        events: List[RecalibrationEvent] = []
        n = accel_series.shape[0]
        for i in range(n):
            is_stat = bool(stationary_mask[i]) if stationary_mask is not None else False
            ev = self.evaluate_sample(
                accel_series[i],
                gyro_series[i],
                int(timestamps_ns[i]),
                is_stationary=is_stat,
            )
            if ev.triggered:
                events.append(ev)
        return events
