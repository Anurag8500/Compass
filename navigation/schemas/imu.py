"""IMU sensor data schemas for COMPASS.

Defines the canonical data structures for raw device-frame IMU samples,
aligned vehicle-frame IMU samples, and ML feature windows.

Coordinate Frame Architecture:
- RawIMUSample: Body/Device Frame (b). Unaligned, raw sensor coordinate system.
- AlignedIMUSample: Vehicle Frame (v). Specific force f^v and angular velocity omega^v
  after mounting alignment R_b^v and bias compensation.
- FeatureWindow: 20x9 matrix of vehicle-frame motion features at canonical 10 Hz rate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, List, Tuple


class SensorSource(str, Enum):
    """Source of the IMU sensor measurement."""
    PHONE = "PHONE"
    EXTERNAL = "EXTERNAL"


# Non-destructive quality bitmask flags (Master Plan Section 13, Implementation Plan Phase 2)
FLAG_OK: int = 0x00
FLAG_NAN_OR_NONFINITE: int = 0x01       # Accel/gyro contains NaN, +Inf, or -Inf
FLAG_INVALID_TIMESTAMP: int = 0x02      # Non-positive or out-of-range epoch timestamp
FLAG_NON_MONOTONIC_TIMESTAMP: int = 0x04  # Timestamp t_k <= t_{k-1}
FLAG_DUPLICATE_TIMESTAMP: int = 0x08    # Repeated timestamp for identical sensor ID
FLAG_EXTREME_MOTION: int = 0x10         # ||f|| > 4g (39.24 m/s^2) or ||omega|| > 10 rad/s (real dynamics, NOT corrupt)
FLAG_SENSOR_DROPOUT: int = 0x20         # Gap delta_t > 3 * delta_t_nominal


@dataclass(frozen=True)
class RawIMUSample:
    """Raw IMU measurement in the RAW DEVICE FRAME.

    Coordinate Frame:
        Device Body Frame (b). The axes (x, y, z) are aligned with the physical
        casing of the smartphone or external IMU sensor module.
        No mounting orientation, frame alignment, or vehicle transformation
        is applied at this stage.

    Units:
        accel: Specific force in m/s^2 (includes reaction to physical gravity).
        gyro: Angular velocity in rad/s.
        timestamp_ns: Monotonic or epoch timestamp in nanoseconds.
    """
    timestamp_ns: int
    accel: Tuple[float, float, float]
    gyro: Tuple[float, float, float]
    quality_flags: int = FLAG_OK
    source: SensorSource = SensorSource.PHONE
    sensor_id: str = "phone_internal"

    def __post_init__(self) -> None:
        if len(self.accel) != 3:
            raise ValueError(f"accel must be a 3-tuple (x, y, z), got length {len(self.accel)}")
        if len(self.gyro) != 3:
            raise ValueError(f"gyro must be a 3-tuple (x, y, z), got length {len(self.gyro)}")
        if not isinstance(self.timestamp_ns, int):
            raise TypeError(f"timestamp_ns must be an integer, got {type(self.timestamp_ns).__name__}")
        if self.timestamp_ns < 0:
            raise ValueError(f"timestamp_ns must be non-negative, got {self.timestamp_ns}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize sample to JSON-compatible dictionary."""
        return {
            "timestamp_ns": self.timestamp_ns,
            "accel": list(self.accel),
            "gyro": list(self.gyro),
            "quality_flags": self.quality_flags,
            "source": self.source.value,
            "sensor_id": self.sensor_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RawIMUSample:
        """Deserialize sample from dictionary."""
        return cls(
            timestamp_ns=int(data["timestamp_ns"]),
            accel=(float(data["accel"][0]), float(data["accel"][1]), float(data["accel"][2])),
            gyro=(float(data["gyro"][0]), float(data["gyro"][1]), float(data["gyro"][2])),
            quality_flags=int(data.get("quality_flags", FLAG_OK)),
            source=SensorSource(data.get("source", SensorSource.PHONE.value)),
            sensor_id=str(data.get("sensor_id", "phone_internal")),
        )


@dataclass(frozen=True)
class AlignedIMUSample:
    """Calibrated and aligned IMU measurement in the VEHICLE FRAME.

    Coordinate Frame:
        Vehicle Frame (v).
        Axes convention: Forward (x), Right (y), Down (z) or ENU-vehicle.
        accel_vehicle: Specific force f^v = R_b^v * (f_raw^b - b_a), m/s^2.
        gyro_vehicle: Angular velocity omega^v = R_b^v * (omega_raw^b - b_g), rad/s.
        Physical gravity is NOT removed here; strapdown INS propagation in ESKF
        resolves gravity internally via vehicle attitude R_v^n.

    Flags:
        is_usable_for_integration: False only if genuinely non-computable
        (NaN/Inf, non-monotonic timestamp). True for extreme motion samples.
    """
    timestamp_ns: int
    accel_vehicle: Tuple[float, float, float]
    gyro_vehicle: Tuple[float, float, float]
    quality_flags: int = FLAG_OK
    is_usable_for_integration: bool = True

    def __post_init__(self) -> None:
        if len(self.accel_vehicle) != 3:
            raise ValueError(f"accel_vehicle must be a 3-tuple, got length {len(self.accel_vehicle)}")
        if len(self.gyro_vehicle) != 3:
            raise ValueError(f"gyro_vehicle must be a 3-tuple, got length {len(self.gyro_vehicle)}")
        if not isinstance(self.timestamp_ns, int):
            raise TypeError(f"timestamp_ns must be an integer, got {type(self.timestamp_ns).__name__}")
        if self.timestamp_ns < 0:
            raise ValueError(f"timestamp_ns must be non-negative, got {self.timestamp_ns}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize sample to JSON-compatible dictionary."""
        return {
            "timestamp_ns": self.timestamp_ns,
            "accel_vehicle": list(self.accel_vehicle),
            "gyro_vehicle": list(self.gyro_vehicle),
            "quality_flags": self.quality_flags,
            "is_usable_for_integration": self.is_usable_for_integration,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AlignedIMUSample:
        """Deserialize sample from dictionary."""
        return cls(
            timestamp_ns=int(data["timestamp_ns"]),
            accel_vehicle=(
                float(data["accel_vehicle"][0]),
                float(data["accel_vehicle"][1]),
                float(data["accel_vehicle"][2]),
            ),
            gyro_vehicle=(
                float(data["gyro_vehicle"][0]),
                float(data["gyro_vehicle"][1]),
                float(data["gyro_vehicle"][2]),
            ),
            quality_flags=int(data.get("quality_flags", FLAG_OK)),
            is_usable_for_integration=bool(data.get("is_usable_for_integration", True)),
        )


@dataclass(frozen=True)
class FeatureWindow:
    """Canonical 20x9 ML input feature window sampled at 10 Hz over 2.0 seconds.

    Matrix dimensions:
        Rows: 20 time steps (spaced by 100 ms = 10 Hz).
        Columns: 9 vehicle-frame kinematic features in canonical order:
            0: f_x^v       (forward specific force, m/s^2)
            1: f_y^v       (lateral/right specific force, m/s^2)
            2: f_z^v       (vertical/down specific force, m/s^2)
            3: omega_x^v   (roll rate, rad/s)
            4: omega_y^v   (pitch rate, rad/s)
            5: omega_z^v   (yaw rate, rad/s)
            6: ||f^v||     (specific force vector Euclidean norm, m/s^2)
            7: ||f_dot^v|| (specific force derivative norm, m/s^3)
            8: ||omega^v|| (angular velocity vector Euclidean norm, rad/s)

    Invariants:
        - Exactly 20 rows and 9 columns when is_valid is True.
        - Input features are strictly in the VEHICLE FRAME.
    """
    window_end_timestamp_ns: int
    samples: List[List[float]]
    is_valid: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.window_end_timestamp_ns, int):
            raise TypeError(
                f"window_end_timestamp_ns must be an integer, got {type(self.window_end_timestamp_ns).__name__}"
            )
        if self.window_end_timestamp_ns < 0:
            raise ValueError(f"window_end_timestamp_ns must be non-negative, got {self.window_end_timestamp_ns}")

        if self.is_valid:
            if len(self.samples) != 20:
                raise ValueError(f"FeatureWindow must contain exactly 20 sample rows, got {len(self.samples)}")
            for idx, row in enumerate(self.samples):
                if len(row) != 9:
                    raise ValueError(
                        f"FeatureWindow row {idx} must contain exactly 9 feature columns, got {len(row)}"
                    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize window to JSON-compatible dictionary."""
        return {
            "window_end_timestamp_ns": self.window_end_timestamp_ns,
            "samples": [list(row) for row in self.samples],
            "is_valid": self.is_valid,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureWindow:
        """Deserialize window from dictionary."""
        return cls(
            window_end_timestamp_ns=int(data["window_end_timestamp_ns"]),
            samples=[[float(v) for v in row] for row in data["samples"]],
            is_valid=bool(data.get("is_valid", True)),
        )
