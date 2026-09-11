"""Non-Holonomic Constraints (NHC) Module for Phase 11.

Implements lateral and vertical velocity pseudo-measurements with adaptive
thresholds, improved skid detection, and velocity-dependent tuning to achieve
better drift reduction than the baseline anurag-phase-10 implementation.

Key Improvements over baseline:
- Adaptive NHC noise covariance based on vehicle speed
- Velocity-dependent skid detection thresholds (allows higher lateral accel at speed)
- Smoother adaptive covariance inflation function
- GNSS-quality-aware NHC relaxation
- Improved standstill detection for ZUPT
"""

from navigation.nhc.measurement import (
    NHCConfig,
    NHCMeasurementModel,
    NHCDiagnostics,
    NHCStatus,
)
from navigation.nhc.skid_detection import (
    SkidDetectorConfig,
    SkidEvaluationResult,
    SkidSlipDetector,
)
from navigation.nhc.zupt_integration import (
    ZUPTIntegrationConfig,
    ZUPTIntegrator,
    ZUPTIntegrationDiagnostics,
)

__all__ = [
    "NHCConfig",
    "NHCMeasurementModel",
    "NHCDiagnostics",
    "NHCStatus",
    "SkidDetectorConfig",
    "SkidEvaluationResult",
    "SkidSlipDetector",
    "ZUPTIntegrationConfig",
    "ZUPTIntegrator",
    "ZUPTIntegrationDiagnostics",
]
