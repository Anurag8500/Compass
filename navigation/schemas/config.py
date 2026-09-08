"""Configuration and external sensor packet schemas for COMPASS.

Defines the external sensor packet interface (for edge / FOG / non-phone IMUs)
and the model_config.json schema that governs feature normalization, windowing,
and channel ordering for VelocityNet and BiasNet.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Canonical channel ordering for the 20x9 vehicle-frame feature representation
CANONICAL_CHANNELS: Tuple[str, ...] = (
    "f_x_v",          # Forward specific force (m/s^2)
    "f_y_v",          # Lateral/right specific force (m/s^2)
    "f_z_v",          # Down/vertical specific force (m/s^2)
    "omega_x_v",      # Roll angular velocity (rad/s)
    "omega_y_v",      # Pitch angular velocity (rad/s)
    "omega_z_v",      # Yaw angular velocity (rad/s)
    "norm_f_v",       # Specific force vector magnitude (m/s^2)
    "norm_f_dot_v",   # Specific force jerk / derivative magnitude (m/s^3)
    "norm_omega_v",   # Angular velocity vector magnitude (rad/s)
)


@dataclass(frozen=True)
class ExternalSensorPacket:
    """Standardized packet schema for non-phone external sensors (edge / FOG / MEMS).

    Coordinate Frame:
        External Sensor Frame. Raw, uncalibrated, unaligned sensor readings
        prior to mounting alignment R_b^v and calibration.

    Attributes:
        seq: Monotonically increasing packet sequence number.
        t_host_ns: Monotonic timestamp upon packet arrival at host system (ns).
        t_sensor_ns: Internal clock timestamp reported by sensor hardware (ns).
        accel: Raw 3-axis specific force (x, y, z) in m/s^2.
        gyro: Raw 3-axis angular velocity (x, y, z) in rad/s.
        mag: Optional 3-axis magnetic flux density (x, y, z) in microteslas (uT).
        declared_rate_hz: Manufacturer declared nominal sampling frequency (e.g. 100.0, 200.0 Hz).
        sensor_id: Hardware identifier or calibration lookup key (e.g. "wt901_01", "fog_isro_01").
    """
    seq: int
    t_host_ns: int
    t_sensor_ns: int
    accel: Tuple[float, float, float]
    gyro: Tuple[float, float, float]
    declared_rate_hz: float
    sensor_id: str
    mag: Optional[Tuple[float, float, float]] = None

    def __post_init__(self) -> None:
        if len(self.accel) != 3:
            raise ValueError(f"accel must be a 3-tuple, got length {len(self.accel)}")
        if len(self.gyro) != 3:
            raise ValueError(f"gyro must be a 3-tuple, got length {len(self.gyro)}")
        if self.mag is not None and len(self.mag) != 3:
            raise ValueError(f"mag must be a 3-tuple if provided, got length {len(self.mag)}")
        if self.declared_rate_hz <= 0.0:
            raise ValueError(f"declared_rate_hz must be positive, got {self.declared_rate_hz}")
        if self.seq < 0:
            raise ValueError(f"seq must be non-negative, got {self.seq}")
        if self.t_host_ns < 0:
            raise ValueError(f"t_host_ns must be non-negative, got {self.t_host_ns}")
        if self.t_sensor_ns < 0:
            raise ValueError(f"t_sensor_ns must be non-negative, got {self.t_sensor_ns}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize external packet to JSON-compatible dictionary."""
        return {
            "seq": self.seq,
            "t_host_ns": self.t_host_ns,
            "t_sensor_ns": self.t_sensor_ns,
            "accel": list(self.accel),
            "gyro": list(self.gyro),
            "mag": list(self.mag) if self.mag is not None else None,
            "declared_rate_hz": self.declared_rate_hz,
            "sensor_id": self.sensor_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExternalSensorPacket:
        """Deserialize external packet from dictionary."""
        mag_raw = data.get("mag")
        return cls(
            seq=int(data["seq"]),
            t_host_ns=int(data["t_host_ns"]),
            t_sensor_ns=int(data["t_sensor_ns"]),
            accel=(float(data["accel"][0]), float(data["accel"][1]), float(data["accel"][2])),
            gyro=(float(data["gyro"][0]), float(data["gyro"][1]), float(data["gyro"][2])),
            mag=(float(mag_raw[0]), float(mag_raw[1]), float(mag_raw[2])) if mag_raw is not None else None,
            declared_rate_hz=float(data["declared_rate_hz"]),
            sensor_id=str(data["sensor_id"]),
        )


@dataclass(frozen=True)
class ModelConfig:
    """Schema for model_config.json defining normalization parameters, windowing, and channel ordering.

    Serves as the single source of truth shared between offline PyTorch training,
    ONNX runtime on edge, and LiteRT on Android.

    Feature Window & Execution Cadence Semantics:
        - window_size: Exactly 20 samples. Fixed by canonical architecture.
        - window_duration_s: Exactly 2.0 seconds. At the canonical 10 Hz feature decimation
          rate (delta_t = 0.1 s), 2.0 s / 0.1 s = 20 samples.
        - stride_duration_s: The sliding step duration between consecutive feature windows
          fed to the neural network during training/inference. For VelocityNet, the sliding
          stride is typically 0.5 s (~2 Hz execution cadence); for BiasNet, typically 1.0 s
          (~1 Hz execution cadence).
        - NOTE ON RATES: Do NOT confuse stride_duration_s with:
          1. Raw IMU native ingestion rate (up to ~200 Hz on edge / FOG).
          2. Canonical decimation rate for feature samples (strictly 10 Hz).
          3. Final navigation state output rate (10 Hz on mobile, higher on edge).
    """
    model_name: str
    normalization_means: List[float]
    normalization_stds: List[float]
    filter_coefficients: Dict[str, Any]
    window_size: int = 20
    window_duration_s: float = 2.0
    stride_duration_s: float = 0.5
    channel_order: List[str] = field(default_factory=lambda: list(CANONICAL_CHANNELS))

    def __post_init__(self) -> None:
        num_channels = len(self.channel_order)
        if len(self.normalization_means) != num_channels:
            raise ValueError(
                f"normalization_means length ({len(self.normalization_means)}) must match "
                f"channel_order length ({num_channels})"
            )
        if len(self.normalization_stds) != num_channels:
            raise ValueError(
                f"normalization_stds length ({len(self.normalization_stds)}) must match "
                f"channel_order length ({num_channels})"
            )
        for idx, std in enumerate(self.normalization_stds):
            if std <= 0.0:
                raise ValueError(
                    f"normalization_stds[{idx}] must be strictly positive to prevent division by zero, got {std}"
                )
        if self.window_size != 20:
            raise ValueError(f"window_size must be 20 for canonical 10 Hz / 2.0 s ML windows, got {self.window_size}")
        if self.window_duration_s <= 0.0:
            raise ValueError(f"window_duration_s must be positive, got {self.window_duration_s}")
        if self.stride_duration_s <= 0.0:
            raise ValueError(f"stride_duration_s must be positive, got {self.stride_duration_s}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to a dictionary."""
        return {
            "model_name": self.model_name,
            "normalization_means": list(self.normalization_means),
            "normalization_stds": list(self.normalization_stds),
            "filter_coefficients": dict(self.filter_coefficients),
            "window_size": self.window_size,
            "window_duration_s": self.window_duration_s,
            "stride_duration_s": self.stride_duration_s,
            "channel_order": list(self.channel_order),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        """Deserialize configuration from a dictionary."""
        return cls(
            model_name=str(data["model_name"]),
            normalization_means=[float(m) for m in data["normalization_means"]],
            normalization_stds=[float(s) for s in data["normalization_stds"]],
            filter_coefficients=dict(data.get("filter_coefficients", {})),
            window_size=int(data.get("window_size", 20)),
            window_duration_s=float(data.get("window_duration_s", 2.0)),
            stride_duration_s=float(data.get("stride_duration_s", 0.5)),
            channel_order=[str(c) for c in data.get("channel_order", list(CANONICAL_CHANNELS))],
        )

    def to_json(self, indent: int = 2) -> str:
        """Serialize to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> ModelConfig:
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def to_file(self, path: Path | str) -> None:
        """Save configuration directly to a JSON file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def from_file(cls, path: Path | str) -> ModelConfig:
        """Load configuration directly from a JSON file."""
        with open(Path(path), "r", encoding="utf-8") as f:
            return cls.from_json(f.read())
