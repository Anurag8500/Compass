"""Canonical data schemas and interfaces for COMPASS (SIH 2026 Problem Statement 26168).

Single source of truth for all data structures passing between ingestion,
preprocessing, classical strapdown propagation, ESKF fusion, ML feature windows,
ML model inference, and downstream map matching.

NO numerical navigation algorithms exist in this package.
"""

from navigation.schemas.config import (
    CANONICAL_CHANNELS,
    ExternalSensorPacket,
    ModelConfig,
)
from navigation.schemas.gnss import GNSSSample
from navigation.schemas.imu import (
    FLAG_DUPLICATE_TIMESTAMP,
    FLAG_EXTREME_MOTION,
    FLAG_INVALID_TIMESTAMP,
    FLAG_NAN_OR_NONFINITE,
    FLAG_NON_MONOTONIC_TIMESTAMP,
    FLAG_OK,
    FLAG_SENSOR_DROPOUT,
    AlignedIMUSample,
    FeatureWindow,
    RawIMUSample,
    SensorSource,
)
from navigation.schemas.mapmatch import MapMatchResult
from navigation.schemas.ml import MLModelType, MLPrediction
from navigation.schemas.state import (
    GNSSMode,
    NavigationState,
    OrientationState,
)

__all__ = [
    # IMU Schemas & Bitflags
    "RawIMUSample",
    "AlignedIMUSample",
    "FeatureWindow",
    "SensorSource",
    "FLAG_OK",
    "FLAG_NAN_OR_NONFINITE",
    "FLAG_INVALID_TIMESTAMP",
    "FLAG_NON_MONOTONIC_TIMESTAMP",
    "FLAG_DUPLICATE_TIMESTAMP",
    "FLAG_EXTREME_MOTION",
    "FLAG_SENSOR_DROPOUT",
    # GNSS Schemas
    "GNSSSample",
    # State & Attitude Schemas
    "GNSSMode",
    "OrientationState",
    "NavigationState",
    # ML Prediction Schemas
    "MLModelType",
    "MLPrediction",
    # Map Match Schemas
    "MapMatchResult",
    # Config & External Packet Schemas
    "ExternalSensorPacket",
    "ModelConfig",
    "CANONICAL_CHANNELS",
]
