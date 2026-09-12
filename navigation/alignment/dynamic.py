"""Causal Dynamic Mounting Alignment and Confidence Estimation (Phase 11).

Estimates the fixed horizontal device-to-vehicle mounting rotation R_b^v
from observable straight-line forward vehicle acceleration during GNSS-aided
operation, without future ground-truth leakage or offline optimization.

States:
- ALIGNMENT_UNKNOWN: Insufficient evidence observed (< 5 qualifying epochs).
- ALIGNMENT_LOW_CONFIDENCE: Partial evidence (5-14 epochs or high dispersion).
- ALIGNMENT_CONFIDENT: Statistically sound evidence (>= 15 epochs, R_bar >= 0.85).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Optional, Tuple
import numpy as np


class AlignmentConfidence(str, Enum):
    """Alignment confidence tiers for modulating downstream kinematic constraints."""
    UNKNOWN = "ALIGNMENT_UNKNOWN"
    LOW_CONFIDENCE = "ALIGNMENT_LOW_CONFIDENCE"
    CONFIDENT = "ALIGNMENT_CONFIDENT"


@dataclass
class DynamicAlignmentState:
    """State of runtime mounting alignment estimator."""
    confidence: AlignmentConfidence = AlignmentConfidence.UNKNOWN
    mounting_yaw_deg: float = 0.0
    accumulated_epochs: int = 0
    mean_resultant_length: float = 0.0
    circular_dispersion_deg: float = 180.0
    is_frozen: bool = False


class DynamicMountingAligner:
    """Causal, runtime estimator of device-to-vehicle mounting orientation."""

    def __init__(
        self,
        min_speed_mps: float = 4.0,
        min_accel_mps2: float = 0.35,
        max_yaw_rate_rads: float = 0.05,
        min_epochs_confident: int = 15,
        min_resultant_length: float = 0.80,
    ) -> None:
        self.min_speed_mps = min_speed_mps
        self.min_accel_mps2 = min_accel_mps2
        self.max_yaw_rate_rads = max_yaw_rate_rads
        self.min_epochs_confident = min_epochs_confident
        self.min_resultant_length = min_resultant_length

        self.reset()

    def reset(self) -> None:
        """Reset internal accumulator and confidence."""
        self._sum_fx: float = 0.0
        self._sum_fy: float = 0.0
        self._sum_norm: float = 0.0
        self._epoch_count: int = 0
        self.state = DynamicAlignmentState()

    def set_prealigned(self, mounting_yaw_deg: float = 0.0) -> None:
        """Explicitly set pre-aligned status for pre-aligned devices or synthetic benchmarks."""
        self.state = DynamicAlignmentState(
            confidence=AlignmentConfidence.CONFIDENT,
            mounting_yaw_deg=float(mounting_yaw_deg),
            accumulated_epochs=self.min_epochs_confident,
            mean_resultant_length=1.0,
            circular_dispersion_deg=0.0,
            is_frozen=False,
        )

    def freeze(self) -> None:
        """Freeze alignment during GNSS outage or degraded navigation."""
        self.state.is_frozen = True

    def unfreeze(self) -> None:
        """Unfreeze alignment when healthy GNSS is restored."""
        self.state.is_frozen = False

    def update(
        self,
        f_level: np.ndarray,
        omega_level: np.ndarray,
        gnss_speed_mps: Optional[float],
        gnss_accel_mps2: Optional[float],
        is_gnss_trusted: bool,
    ) -> DynamicAlignmentState:
        """Process a single 10-Hz sample to causally update mounting alignment.

        Args:
            f_level: 3-vector specific force in tilt-leveled frame (gravity aligned with +Z).
            omega_level: 3-vector angular velocity in tilt-leveled frame.
            gnss_speed_mps: Current GNSS speed (None if unavailable/untrusted).
            gnss_accel_mps2: Current longitudinal acceleration derived from GNSS velocity.
            is_gnss_trusted: True if GNSS fix is confident and active.

        Returns:
            Updated DynamicAlignmentState.
        """
        if self.state.is_frozen or not is_gnss_trusted:
            return self.state

        if gnss_speed_mps is None or gnss_accel_mps2 is None:
            return self.state

        # Check observability criteria:
        # 1. High speed (vehicle traveling on road)
        # 2. Straight line (low yaw rate)
        # 3. Definite forward acceleration
        # 4. Measurable horizontal specific force
        yaw_rate = abs(float(omega_level[2]))
        h_norm = float(math.sqrt(f_level[0]**2 + f_level[1]**2))

        is_qualifying = (
            (gnss_speed_mps >= self.min_speed_mps)
            and (yaw_rate <= self.max_yaw_rate_rads)
            and (gnss_accel_mps2 >= self.min_accel_mps2)
            and (h_norm >= 0.25)
        )

        if not is_qualifying:
            return self.state

        # Accumulate vector evidence
        self._sum_fx += float(f_level[0])
        self._sum_fy += float(f_level[1])
        self._sum_norm += h_norm
        self._epoch_count += 1

        # Statistical vector metrics
        res_len = math.sqrt(self._sum_fx**2 + self._sum_fy**2)
        r_bar = res_len / max(1e-6, self._sum_norm)
        # When device is rotated by +psi, f_b = [cos(psi), -sin(psi), 0]
        # Therefore psi = atan2(-fy, fx)
        mean_yaw_rad = math.atan2(-self._sum_fy, self._sum_fx)

        if r_bar >= 0.999999:
            disp_deg = 0.0
        elif r_bar > 1e-4:
            disp_deg = math.degrees(math.sqrt(max(0.0, -2.0 * math.log(r_bar))))
        else:
            disp_deg = 180.0

        # Confidence state transition
        if self._epoch_count >= self.min_epochs_confident and r_bar >= self.min_resultant_length:
            confidence = AlignmentConfidence.CONFIDENT
            yaw_deg = math.degrees(mean_yaw_rad)
        elif self._epoch_count >= 5:
            confidence = AlignmentConfidence.LOW_CONFIDENCE
            yaw_deg = math.degrees(mean_yaw_rad)
        else:
            confidence = AlignmentConfidence.UNKNOWN
            yaw_deg = 0.0

        self.state = DynamicAlignmentState(
            confidence=confidence,
            mounting_yaw_deg=yaw_deg,
            accumulated_epochs=self._epoch_count,
            mean_resultant_length=r_bar,
            circular_dispersion_deg=disp_deg,
            is_frozen=False,
        )
        return self.state

    def get_rotation_matrix(self) -> np.ndarray:
        """Return 3x3 horizontal rotation matrix R_z(mounting_yaw).

        Satisfies v_v = R_b_v @ v_b, rotating device body forward/lateral axes into vehicle FLU.
        """
        if self.state.confidence != AlignmentConfidence.CONFIDENT:
            return np.eye(3, dtype=np.float64)

        psi = math.radians(self.state.mounting_yaw_deg)
        c, s = math.cos(psi), math.sin(psi)
        return np.array([
            [c,  -s, 0.0],
            [s,   c, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
