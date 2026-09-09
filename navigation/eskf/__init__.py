"""Error-State Kalman Filter (ESKF) Package for C.O.M.P.A.S.S. (Phase 5).

Provides:
- State definitions: 16-state nominal manifold, 15-state error manifold, 15x15 covariance.
- Prediction: Discrete nominal strapdown propagation, analytical F_d, discrete Q_d.
- Generic Gated Update: Innovation calculation, Mahalanobis gating, Joseph-form covariance update.
- Measurement Models: GNSS (position and velocity in local ENU frame) and Classical Gated ZUPT.
"""

from navigation.eskf.state import (
    ESKFNominalState,
    ESKFState,
    inject_error,
    skew,
)
from navigation.eskf.predict import (
    ProcessNoiseConfig,
    compute_discrete_F,
    compute_discrete_Q,
    predict_eskf,
)
from navigation.eskf.gating import (
    GatingDiagnostics,
    MahalanobisGating,
)
from navigation.eskf.update import (
    UpdateDiagnostics,
    eskf_update,
)
from navigation.eskf.measurements.gnss import (
    GNSSMeasurementModel,
    GNSSUpdateConfig,
)
from navigation.eskf.measurements.zupt import (
    ClassicalZUPTDetector,
    ZUPTDetectorConfig,
    ZUPTDetectorDiagnostics,
    ZUPTMeasurementConfig,
    ZUPTMeasurementModel,
)

__all__ = [
    # State
    "ESKFNominalState",
    "ESKFState",
    "inject_error",
    "skew",
    # Prediction
    "ProcessNoiseConfig",
    "compute_discrete_F",
    "compute_discrete_Q",
    "predict_eskf",
    # Gating & Update
    "GatingDiagnostics",
    "MahalanobisGating",
    "UpdateDiagnostics",
    "eskf_update",
    # Measurements
    "GNSSMeasurementModel",
    "GNSSUpdateConfig",
    "ClassicalZUPTDetector",
    "ZUPTDetectorConfig",
    "ZUPTDetectorDiagnostics",
    "ZUPTMeasurementConfig",
    "ZUPTMeasurementModel",
]
