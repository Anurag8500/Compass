"""Inertial Navigation System (INS) strapdown mechanization package for COMPASS."""

from navigation.ins.attitude import (
    delta_quaternion,
    propagate_attitude,
    quaternion_conjugate,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_norm,
    quaternion_normalize,
    quaternion_to_euler_deg,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)
from navigation.ins.propagation import (
    INSState,
    INSTrajectory,
    StrapdownINS,
)

__all__ = [
    "delta_quaternion",
    "propagate_attitude",
    "quaternion_conjugate",
    "quaternion_inverse",
    "quaternion_multiply",
    "quaternion_norm",
    "quaternion_normalize",
    "quaternion_to_euler_deg",
    "quaternion_to_rotation_matrix",
    "rotation_matrix_to_quaternion",
    "INSState",
    "INSTrajectory",
    "StrapdownINS",
]
