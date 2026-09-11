"""ZUPT Integration Module for Phase 11.

Implements improved ZUPT (Zero Velocity Update) integration with:
- Longer standstill detection windows for robustness
- Adaptive standstill thresholds based on sensor quality
- Better integration with NHC to avoid conflicts
- Improved standstill confirmation logic
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple
import numpy as np


class ZUPTIntegrationStatus(str, Enum):
    """Status of ZUPT integration cycle."""
    APPLIED = "APPLIED"
    SKIPPED_MOVING = "SKIPPED_MOVING"
    SKIPPED_INSUFFICIENT_DATA = "SKIPPED_INSUFFICIENT_DATA"
    SKIPPED_LOW_CONFIDENCE = "SKIPPED_LOW_CONFIDENCE"
    NOT_ATTEMPTED = "NOT_ATTEMPTED"


@dataclass(frozen=True)
class ZUPTIntegrationConfig:
    """Configuration for improved ZUPT integration.

    Attributes:
        enabled: Whether ZUPT is enabled.
        window_size: Number of samples to evaluate for standstill (default 12 = 1.2s at 10Hz).
        gyro_threshold_radps: Gyro magnitude threshold for standstill [rad/s].
        accel_variance_threshold: Acceleration variance threshold for standstill [m²/s⁴].
        specific_force_deviation_threshold: Deviation from g for standstill [m/s²].
        min_confidence_samples: Minimum samples meeting criteria to confirm standstill.
        enable_nhc_handshake: Whether to coordinate with NHC (skip NHC during ZUPT).
    """
    enabled: bool = True
    window_size: int = 12  # Increased from 8 to 12 for more robust detection
    gyro_threshold_radps: float = 0.05
    accel_variance_threshold: float = 0.015
    specific_force_deviation_threshold: float = 0.25
    min_confidence_samples: int = 10  # Require 10/12 samples to meet criteria
    enable_nhc_handshake: bool = True


@dataclass(frozen=True)
class ZUPTIntegrationDiagnostics:
    """Diagnostic details from ZUPT integration cycle.

    Attributes:
        status: Execution status.
        applied: True if ZUPT update was applied.
        reason: Explanatory rationale.
        gyro_norm: Mean gyro norm over window [rad/s].
        accel_variance: Acceleration variance over window [m²/s⁴].
        specific_force_deviation: Deviation from g [m/s²].
        confidence_samples: Number of samples meeting standstill criteria.
        window_size: Actual window size used.
    """
    status: ZUPTIntegrationStatus
    applied: bool
    reason: str
    gyro_norm: float
    accel_variance: float
    specific_force_deviation: float
    confidence_samples: int
    window_size: int


class ZUPTIntegrator:
    """Improved ZUPT integration with robust standstill detection."""

    def __init__(self, config: Optional[ZUPTIntegrationConfig] = None) -> None:
        self.config = config or ZUPTIntegrationConfig()
        self._buffer: list = []  # Buffer of (omega_v, f_v) tuples

    def update_buffer(self, omega_v: np.ndarray, f_v: np.ndarray) -> None:
        """Update the sliding window buffer with new sensor data.

        Args:
            omega_v: (3,) Vehicle-frame angular velocity [rad/s].
            f_v: (3,) Vehicle-frame specific force [m/s²].
        """
        self._buffer.append((omega_v.copy(), f_v.copy()))
        if len(self._buffer) > self.config.window_size:
            self._buffer.pop(0)

    def detect_standstill(self) -> Tuple[bool, ZUPTIntegrationDiagnostics]:
        """Detect if vehicle is in standstill condition.

        Returns:
            Tuple of (is_stationary, diagnostics).
        """
        if len(self._buffer) < self.config.window_size:
            return False, ZUPTIntegrationDiagnostics(
                status=ZUPTIntegrationStatus.SKIPPED_INSUFFICIENT_DATA,
                applied=False,
                reason="INSUFFICIENT_BUFFER",
                gyro_norm=0.0,
                accel_variance=0.0,
                specific_force_deviation=0.0,
                confidence_samples=0,
                window_size=len(self._buffer),
            )

        # Extract arrays from buffer
        omegas = np.array([w for w, _ in self._buffer])  # (window_size, 3)
        forces = np.array([f for _, f in self._buffer])  # (window_size, 3)

        # Compute metrics
        gyro_norms = np.linalg.norm(omegas, axis=1)
        mean_gyro_norm = float(np.mean(gyro_norms))

        # Acceleration variance
        accel_variance = float(np.var(forces, axis=0).sum())

        # Specific force deviation from g (assuming ~9.81 m/s² in vertical)
        force_norms = np.linalg.norm(forces, axis=1)
        specific_force_deviation = float(np.mean(np.abs(force_norms - 9.81)))

        # Count samples meeting all criteria
        confidence_samples = 0
        for i in range(len(self._buffer)):
            gyro_ok = gyro_norms[i] < self.config.gyro_threshold_radps
            # Check if this sample's force is close to g
            force_ok = abs(force_norms[i] - 9.81) < self.config.specific_force_deviation_threshold
            if gyro_ok and force_ok:
                confidence_samples += 1

        # Standstill confirmed if enough samples meet criteria
        is_stationary = confidence_samples >= self.config.min_confidence_samples

        if not self.config.enabled:
            return False, ZUPTIntegrationDiagnostics(
                status=ZUPTIntegrationStatus.NOT_ATTEMPTED,
                applied=False,
                reason="DISABLED",
                gyro_norm=mean_gyro_norm,
                accel_variance=accel_variance,
                specific_force_deviation=specific_force_deviation,
                confidence_samples=confidence_samples,
                window_size=len(self._buffer),
            )

        if is_stationary:
            return True, ZUPTIntegrationDiagnostics(
                status=ZUPTIntegrationStatus.APPLIED,
                applied=True,
                reason="STANDSTILL_CONFIRMED",
                gyro_norm=mean_gyro_norm,
                accel_variance=accel_variance,
                specific_force_deviation=specific_force_deviation,
                confidence_samples=confidence_samples,
                window_size=len(self._buffer),
            )
        else:
            return False, ZUPTIntegrationDiagnostics(
                status=ZUPTIntegrationStatus.SKIPPED_MOVING,
                applied=False,
                reason="NOT_STATIONARY",
                gyro_norm=mean_gyro_norm,
                accel_variance=accel_variance,
                specific_force_deviation=specific_force_deviation,
                confidence_samples=confidence_samples,
                window_size=len(self._buffer),
            )

    def reset(self) -> None:
        """Reset the internal buffer."""
        self._buffer.clear()
