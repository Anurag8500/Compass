"""ML inference prediction schemas for COMPASS.

Defines the output containers for the two canonical ML models:
1. VelocityNet: Forward speed estimate + heteroscedastic uncertainty.
2. BiasNet: 6-vector IMU bias residual + heteroscedastic uncertainty.

These outputs are NOT applied as state overrides; they are ingested by
the ESKF as uncertainty-aware pseudo-measurements via innovation gating.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, List


class MLModelType(str, Enum):
    """The two authoritative COMPASS ML models."""
    VELOCITY_NET = "VELOCITY_NET"
    BIAS_NET = "BIAS_NET"


@dataclass(frozen=True)
class MLPrediction:
    """Canonical container for ML model inference predictions.

    Attributes:
        model: Identifies the emitting network (VELOCITY_NET or BIAS_NET).
        value: Predicted mean value(s):
            - VelocityNet: 1-element list [forward_speed_m_s]
            - BiasNet: 6-element list [delta_b_ax, delta_b_ay, delta_b_az, delta_b_gx, delta_b_gy, delta_b_gz]
        log_variance: Predicted log-variance s = log(sigma^2) for each predicted quantity
            (heteroscedastic uncertainty estimation). Measurement variance R = exp(s).
        window_end_timestamp_ns: Nanosecond timestamp of the latest IMU sample
            included in the feature window that generated this prediction.
    """
    model: MLModelType
    value: List[float]
    log_variance: List[float]
    window_end_timestamp_ns: int

    def __post_init__(self) -> None:
        if len(self.value) != len(self.log_variance):
            raise ValueError(
                f"value length ({len(self.value)}) must match log_variance length ({len(self.log_variance)})"
            )
        if self.model == MLModelType.VELOCITY_NET and len(self.value) != 1:
            raise ValueError(
                f"VELOCITY_NET prediction must have 1 element (forward speed), got {len(self.value)}"
            )
        if self.model == MLModelType.BIAS_NET and len(self.value) != 6:
            raise ValueError(
                f"BIAS_NET prediction must have 6 elements (3 accel + 3 gyro bias residuals), got {len(self.value)}"
            )
        if not isinstance(self.window_end_timestamp_ns, int):
            raise TypeError(
                f"window_end_timestamp_ns must be an integer, got {type(self.window_end_timestamp_ns).__name__}"
            )
        if self.window_end_timestamp_ns < 0:
            raise ValueError(f"window_end_timestamp_ns must be non-negative, got {self.window_end_timestamp_ns}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize ML prediction to JSON-compatible dictionary."""
        return {
            "model": self.model.value,
            "value": list(self.value),
            "log_variance": list(self.log_variance),
            "window_end_timestamp_ns": self.window_end_timestamp_ns,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MLPrediction:
        """Deserialize ML prediction from dictionary."""
        return cls(
            model=MLModelType(data["model"]),
            value=[float(v) for v in data["value"]],
            log_variance=[float(v) for v in data["log_variance"]],
            window_end_timestamp_ns=int(data["window_end_timestamp_ns"]),
        )
