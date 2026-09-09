"""ML data package for C.O.M.P.A.S.S.

Provides shared, deterministic, leakage-audited dataset construction for
VelocityNet and BiasNet:
- Canonical 10 Hz resampling / decimation
- 9-channel vehicle-frame feature generation
- Strictly causal (20, 9) windowing
- Driver/file-level splitting
- Training-only normalization
- VelocityNet causal window-end labels
- BiasNet outage-exclusion plumbing
"""

from ml.data.resample import resample_to_canonical_10hz
from ml.data.features import compute_canonical_features
from ml.data.windowing import extract_causal_windows, ExtractedWindow
from ml.data.split import DriverFileSplit, create_driver_file_split
from ml.data.normalization import FeatureNormalizer
from ml.data.velocitynet_labels import extract_velocitynet_labels
from ml.data.exclusion_rules import BiasNetEligibility, evaluate_biasnet_eligibility

__all__ = [
    "resample_to_canonical_10hz",
    "compute_canonical_features",
    "extract_causal_windows",
    "ExtractedWindow",
    "DriverFileSplit",
    "create_driver_file_split",
    "FeatureNormalizer",
    "extract_velocitynet_labels",
    "BiasNetEligibility",
    "evaluate_biasnet_eligibility",
]
