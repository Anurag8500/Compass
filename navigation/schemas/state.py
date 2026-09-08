"""Navigation and orientation state schemas for COMPASS.

Defines the canonical estimated navigation state, attitude representation,
and GNSS FSM operational modes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Tuple


class GNSSMode(str, Enum):
    """Authoritative GNSS Finite State Machine (FSM) states.

    The COMPASS architecture strictly enforces exactly three discrete FSM states:
    - GNSS_AIDED: Full GNSS availability, active position/velocity correction in ESKF.
    - DR_ONLY: Dead-reckoning mode during GNSS outage (tunnel, parking, foliage, jamming).
    - REACQUIRING: GNSS signal has returned; validating convergence before full aiding.

    Continuous signal degradation is represented by GNSSSample.trust_score, NOT a 4th state.
    """
    GNSS_AIDED = "GNSS_AIDED"
    DR_ONLY = "DR_ONLY"
    REACQUIRING = "REACQUIRING"


@dataclass(frozen=True)
class OrientationState:
    """Orientation and gyro bias state representation.

    Attributes:
        q: Attitude quaternion (w, x, y, z) representing the rotation from the
           vehicle frame to the local navigation frame (R_v^n).
           Convention: Hamilton quaternion, scalar first: [q_w, q_x, q_y, q_z].
           Note: Quaternions are validated to ensure non-zero norm, but are NOT
           mutated or normalized inside this schema to preserve raw data container semantics.
        gyro_bias: Estimated 3-axis gyroscope bias (b_gx, b_gy, b_gz) in rad/s
           resolved in the vehicle frame.
    """
    q: Tuple[float, float, float, float]
    gyro_bias: Tuple[float, float, float]

    def __post_init__(self) -> None:
        if len(self.q) != 4:
            raise ValueError(f"Quaternion q must be a 4-tuple (w, x, y, z), got length {len(self.q)}")
        if len(self.gyro_bias) != 3:
            raise ValueError(f"gyro_bias must be a 3-tuple (x, y, z), got length {len(self.gyro_bias)}")
        qw, qx, qy, qz = self.q
        norm_sq = qw * qw + qx * qx + qy * qy + qz * qz
        if norm_sq < 1e-12:
            raise ValueError(
                f"Quaternion q cannot have zero or near-zero norm (squared norm {norm_sq:.2e}). "
                "Degenerate quaternions cannot represent attitude rotation."
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize orientation state to JSON-compatible dictionary."""
        return {
            "q": list(self.q),
            "gyro_bias": list(self.gyro_bias),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OrientationState:
        """Deserialize orientation state from dictionary."""
        return cls(
            q=(float(data["q"][0]), float(data["q"][1]), float(data["q"][2]), float(data["q"][3])),
            gyro_bias=(float(data["gyro_bias"][0]), float(data["gyro_bias"][1]), float(data["gyro_bias"][2])),
        )


@dataclass(frozen=True)
class NavigationState:
    """Authoritative nominal navigation state with ESKF error-state covariance.

    CRITICAL ARCHITECTURAL INVARIANT:
    The reference_point (lat0, lon0) defines the origin of the Cartesian East-North-Up
    (ENU) tangent plane for the ENTIRE navigation session. It is fixed upon session start
    (e.g., first confident GNSS fix) and MUST NOT be silently reset or shifted upon every
    GNSS update. Resetting the origin during a session would invalidate integrated trajectory
    history and corrupt the ESKF error-state covariance P.

    Attributes:
        position_local: Local Cartesian position [p_E, p_N, p_U] in meters
            (East-North-Up), relative to the fixed session reference_point (lat0, lon0).
        velocity_local: Velocity vector [v_E, v_N, v_U] in m/s in ENU navigation frame.
        orientation: Current attitude quaternion R_v^n and estimated gyro bias.
        accel_bias: Estimated 3-axis accelerometer bias [b_ax, b_ay, b_az] in m/s^2
            in the vehicle frame.
        covariance: 15x15 error-state covariance matrix P in R^(15x15) for:
            [delta_p(3), delta_v(3), delta_theta(3), delta_b_a(3), delta_b_g(3)].
        reference_point: Fixed local tangent-plane origin (lat0, lon0) in degrees WGS84 for the session.
        mode: Operational GNSS FSM mode (GNSS_AIDED, DR_ONLY, or REACQUIRING).
        timestamp_ns: Nanosecond epoch timestamp corresponding to the current state estimate.
    """
    position_local: Tuple[float, float, float]
    velocity_local: Tuple[float, float, float]
    orientation: OrientationState
    accel_bias: Tuple[float, float, float]
    covariance: List[List[float]]
    reference_point: Tuple[float, float]
    mode: GNSSMode
    timestamp_ns: int

    def __post_init__(self) -> None:
        if len(self.position_local) != 3:
            raise ValueError(f"position_local must be a 3-tuple (E, N, U), got length {len(self.position_local)}")
        if len(self.velocity_local) != 3:
            raise ValueError(f"velocity_local must be a 3-tuple (v_E, v_N, v_U), got length {len(self.velocity_local)}")
        if len(self.accel_bias) != 3:
            raise ValueError(f"accel_bias must be a 3-tuple, got length {len(self.accel_bias)}")
        if len(self.reference_point) != 2:
            raise ValueError(f"reference_point must be a 2-tuple (lat0, lon0), got length {len(self.reference_point)}")
        lat0, lon0 = self.reference_point
        if not (-90.0 <= lat0 <= 90.0):
            raise ValueError(f"reference_point lat0 must be in [-90, 90], got {lat0}")
        if not (-180.0 <= lon0 <= 180.0):
            raise ValueError(f"reference_point lon0 must be in [-180, 180], got {lon0}")
        if not isinstance(self.timestamp_ns, int):
            raise TypeError(f"timestamp_ns must be an integer, got {type(self.timestamp_ns).__name__}")
        if self.timestamp_ns < 0:
            raise ValueError(f"timestamp_ns must be non-negative, got {self.timestamp_ns}")

        # Validate 15x15 covariance matrix structure
        if len(self.covariance) != 15:
            raise ValueError(f"ESKF covariance matrix must be 15x15, got {len(self.covariance)} rows")
        for i, row in enumerate(self.covariance):
            if len(row) != 15:
                raise ValueError(f"ESKF covariance row {i} must have length 15, got {len(row)}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize navigation state to JSON-compatible dictionary."""
        return {
            "position_local": list(self.position_local),
            "velocity_local": list(self.velocity_local),
            "orientation": self.orientation.to_dict(),
            "accel_bias": list(self.accel_bias),
            "covariance": [list(row) for row in self.covariance],
            "reference_point": list(self.reference_point),
            "mode": self.mode.value,
            "timestamp_ns": self.timestamp_ns,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NavigationState:
        """Deserialize navigation state from dictionary."""
        return cls(
            position_local=(
                float(data["position_local"][0]),
                float(data["position_local"][1]),
                float(data["position_local"][2]),
            ),
            velocity_local=(
                float(data["velocity_local"][0]),
                float(data["velocity_local"][1]),
                float(data["velocity_local"][2]),
            ),
            orientation=OrientationState.from_dict(data["orientation"]),
            accel_bias=(
                float(data["accel_bias"][0]),
                float(data["accel_bias"][1]),
                float(data["accel_bias"][2]),
            ),
            covariance=[[float(val) for val in row] for row in data["covariance"]],
            reference_point=(float(data["reference_point"][0]), float(data["reference_point"][1])),
            mode=GNSSMode(data["mode"]),
            timestamp_ns=int(data["timestamp_ns"]),
        )
