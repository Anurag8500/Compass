"""Classical preprocessing package for COMPASS (Phase 3).

Modules:
- calibration: Stationary-window gyro bias and initial roll/pitch tilt estimation.
- alignment: Device-to-vehicle mounting alignment (R_b^v).
- filtering: Dual-stage median spike rejection and 4th-order Butterworth low-pass filter.
- gravity: Pure mathematical gravity resolution test and validation utility.
- recalibration_trigger: Orientation jump and physical sensor displacement detector.
- bootstrap_attitude: Offline test-only complementary filter attitude estimator.
- pipeline: Unified PreprocessingPipeline coordinator.
"""

from navigation.preprocessing.alignment import MountingAlignment, estimate_mounting_alignment
from navigation.preprocessing.bootstrap_attitude import (
    BootstrapAttitudeEstimator,
    quaternion_from_axis_angle,
    quaternion_multiply,
    quaternion_to_euler_rad,
    quaternion_to_rotation_matrix,
)
from navigation.preprocessing.calibration import CalibrationProfile, calibrate_stationary_window
from navigation.preprocessing.filtering import IMUFilter, apply_median_filter, compute_butterworth_4th_coeffs
from navigation.preprocessing.gravity import STANDARD_GRAVITY_MPS2, resolve_gravity, verify_stationary_gravity_resolution
from navigation.preprocessing.pipeline import PreprocessedTrip, PreprocessingPipeline
from navigation.preprocessing.recalibration_trigger import RecalibrationDetector, RecalibrationEvent

__all__ = [
    "CalibrationProfile",
    "calibrate_stationary_window",
    "MountingAlignment",
    "estimate_mounting_alignment",
    "IMUFilter",
    "apply_median_filter",
    "compute_butterworth_4th_coeffs",
    "STANDARD_GRAVITY_MPS2",
    "resolve_gravity",
    "verify_stationary_gravity_resolution",
    "RecalibrationDetector",
    "RecalibrationEvent",
    "BootstrapAttitudeEstimator",
    "quaternion_multiply",
    "quaternion_to_rotation_matrix",
    "quaternion_from_axis_angle",
    "quaternion_to_euler_rad",
    "PreprocessedTrip",
    "PreprocessingPipeline",
]
