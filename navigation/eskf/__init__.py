"""Error-State Kalman Filter (ESKF) Package for C.O.M.P.A.S.S. (Phase 5 & Phase 9).

Provides:
- State definitions: 16-state nominal manifold, 15-state error manifold, 15x15 covariance.
- Prediction: Discrete nominal strapdown propagation, analytical F_d, discrete Q_d.
- Generic Gated Update: Innovation calculation, Mahalanobis gating, Joseph-form covariance update.
- Measurement Models: GNSS, Classical Gated ZUPT, VelocityNet, BiasNet.
- Cadence Scheduling: Time-aware 2 Hz / 1 Hz ML update scheduling.
"""

from navigation.eskf.state import (
    ESKFNominalState,
    ESKFState,
    compute_reset_jacobian,
    inject_error,
    reset_covariance,
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
from navigation.eskf.scheduling import (
    CadenceConfig,
    MLCadenceScheduler,
)
from navigation.eskf.measurements.gnss import (
    GNSSMeasurementModel,
    GNSSUpdateConfig,
    course_to_enu_velocity,
    course_to_horizontal_velocity,
)
from navigation.eskf.measurements.zupt import (
    ClassicalZUPTDetector,
    ZUPTDetectorConfig,
    ZUPTDetectorDiagnostics,
    ZUPTMeasurementConfig,
    ZUPTMeasurementModel,
)
from navigation.eskf.measurements.velocitynet import (
    CausalEMA,
    VelocityNetAdapterDiagnostics,
    VelocityNetConfig,
    VelocityNetMeasurementModel,
)
from navigation.eskf.measurements.biasnet import (
    BiasNetAdapterDiagnostics,
    BiasNetConfig,
    BiasNetMeasurementModel,
    PHASE8_R_BIAS_DIAG,
)

__all__ = [
    # State
    "ESKFNominalState",
    "ESKFState",
    "compute_reset_jacobian",
    "inject_error",
    "reset_covariance",
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
    # Scheduling
    "CadenceConfig",
    "MLCadenceScheduler",
    # Measurements
    "GNSSMeasurementModel",
    "GNSSUpdateConfig",
    "course_to_enu_velocity",
    "course_to_horizontal_velocity",
    "ClassicalZUPTDetector",
    "ZUPTDetectorConfig",
    "ZUPTDetectorDiagnostics",
    "ZUPTMeasurementConfig",
    "ZUPTMeasurementModel",
    "CausalEMA",
    "VelocityNetAdapterDiagnostics",
    "VelocityNetConfig",
    "VelocityNetMeasurementModel",
    "BiasNetAdapterDiagnostics",
    "BiasNetConfig",
    "BiasNetMeasurementModel",
    "PHASE8_R_BIAS_DIAG",
]
