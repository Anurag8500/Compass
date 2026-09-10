"""GNSS Supervisory Subsystem for COMPASS (Phase 10).

Exposes:
- Continuous trust score calculation and covariance scaling (trust_score.py)
- Time-based outage detection and persistent rejection tracking (outage_detection.py)
- Three-state GNSS Finite State Machine (fsm.py)
- Reacquisition validation and bounded-rate recovery (recovery.py)
"""

from navigation.gnss.trust_score import (
    GNSSQualityResult,
    GNSSTrustScoreCalculator,
    TrustScoreConfig,
    scale_gnss_covariance,
)
from navigation.gnss.outage_detection import (
    GNSSOutageDetector,
    OutageCondition,
    OutageDetectorConfig,
    OutageStatus,
    generate_synthetic_outage_mask,
)
from navigation.gnss.fsm import (
    FSMConfig,
    GNSSModeFSM,
    GNSSModeTransition,
)
from navigation.gnss.recovery import (
    BoundedCorrectionStep,
    GNSSRecoveryManager,
    RecoveryConfig,
    RecoveryValidationResult,
    ReturningFixStatus,
)

__all__ = [
    "GNSSQualityResult",
    "GNSSTrustScoreCalculator",
    "TrustScoreConfig",
    "scale_gnss_covariance",
    "GNSSOutageDetector",
    "OutageCondition",
    "OutageDetectorConfig",
    "OutageStatus",
    "FSMConfig",
    "GNSSModeFSM",
    "GNSSModeTransition",
    "BoundedCorrectionStep",
    "GNSSRecoveryManager",
    "RecoveryConfig",
    "RecoveryValidationResult",
    "ReturningFixStatus",
]
