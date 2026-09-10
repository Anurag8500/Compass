"""Authoritative ML Model Runner for C.O.M.P.A.S.S. Navigation Core (Phase 9).

This module manages the execution of frozen, exported ONNX neural models
(VelocityNet v1.1 and BiasNet v1.0) strictly as measurement hypotheses.

Architectural Authority Invariants:
1. Pure Functional Transformation: Models produce candidate measurement structures (z, R).
   They NEVER mutate, inject, or directly overwrite the ESKF state or covariance.
2. Frozen Standardization: Applies exact training-split normalization loaded from
   data/ml_dataset_v1/normalization.json. No in-situ re-normalization.
3. Strict Causal Input Validation: Input tensors must be strictly causal (20 samples @ 10 Hz <= t).
   Non-finite, out-of-bounds, or temporally discontinuous windows are rejected with
   deterministic diagnostic reason codes and zero state corruption.
4. Numerical Safety: Outputs are bounded by model-specific physical safeguards:
   - VelocityNet: log_variance clamped to [-10.0, 10.0]; variance floor Rv_min = 1.0 m^2/s^2.
   - BiasNet: delta_ba clamped to [-2.0, 2.0] m/s^2; delta_bg clamped to [-0.15, 0.15] rad/s.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import onnxruntime as ort

from ml.data.normalization import FeatureNormalizer

# Numerical and physical sanity thresholds
ACCEL_MAGNITUDE_MAX_MPS2: float = 100.0   # Extreme physical crash/sensor detachment boundary
GYRO_MAGNITUDE_MAX_RADS: float = 30.0     # Extreme angular rate boundary
MAX_TIMESTAMP_GAP_S: float = 0.5          # Maximum tolerable inter-sample gap at 10 Hz nominal


@dataclass(frozen=True)
class ModelRunnerConfig:
    """Configuration for ML model runtime paths and bounds."""
    velocitynet_onnx_path: str = "models/velocitynet_v1_1.onnx"
    biasnet_onnx_path: str = "models/biasnet_v1.onnx"
    normalization_json_path: str = "data/ml_dataset_v1/normalization.json"
    
    # Uncertainty clamps for VelocityNet
    r_v_min_mps2: float = 1.0             # Conservative variance floor (1.0 m/s std)
    r_v_max_mps2: float = 25.0            # Conservative variance ceiling (5.0 m/s std)
    
    # In-graph / runtime physical clamps for BiasNet
    bias_accel_clamp_mps2: float = 2.0    # Hard physical clamp on delta_ba
    bias_gyro_clamp_rads: float = 0.15    # Hard physical clamp on delta_bg


@dataclass(frozen=True)
class VelocityNetOutput:
    """Structured output from VelocityNet inference."""
    speed_mps: float
    log_variance: float
    variance: float
    valid: bool
    reason: Optional[str] = None


@dataclass(frozen=True)
class BiasNetOutput:
    """Structured output from BiasNet inference."""
    delta_accel_bias: np.ndarray          # Shape (3,) [dba_x, dba_y, dba_z]
    delta_gyro_bias: np.ndarray           # Shape (3,) [dbg_x, dbg_y, dbg_z]
    delta_bias_vector: np.ndarray         # Shape (6,) concatenation
    valid: bool
    reason: Optional[str] = None


class MLWindowValidator:
    """Validates causal motion history windows before neural model evaluation."""

    @staticmethod
    def validate_features(
        raw_features: np.ndarray,
        timestamps_ns: Optional[np.ndarray] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Validate an unnormalized (20, 9) feature array.

        Args:
            raw_features: (20, 9) or (1, 20, 9) array of canonical features.
            timestamps_ns: Optional (20,) int64 array of sample timestamps.

        Returns:
            Tuple of (is_valid, reason_code).
        """
        arr = np.asarray(raw_features, dtype=np.float64)
        if arr.ndim == 3:
            if arr.shape[0] != 1:
                return False, "INVALID_BATCH_SIZE"
            arr = arr[0]

        if arr.shape != (20, 9):
            return False, f"INVALID_SHAPE_{arr.shape}"

        # 1. Finite check
        if not np.all(np.isfinite(arr)):
            return False, "NON_FINITE_INPUT"

        # 2. Physical range check on raw kinematics
        # Channels: 0:3 f_v, 3:6 omega_v, 6 norm_f, 7 norm_f_dot, 8 norm_omega
        if np.any(np.abs(arr[:, 0:3]) > ACCEL_MAGNITUDE_MAX_MPS2):
            return False, "OOD_ACCEL_FEATURES"
        if np.any(np.abs(arr[:, 3:6]) > GYRO_MAGNITUDE_MAX_RADS):
            return False, "OOD_GYRO_FEATURES"

        # 3. Timestamp continuity check (if timestamps provided)
        if timestamps_ns is not None:
            ts = np.asarray(timestamps_ns, dtype=np.int64).reshape(-1)
            if len(ts) != 20:
                return False, f"INVALID_TIMESTAMPS_LEN_{len(ts)}"
            diffs_s = np.diff(ts) * 1e-9
            if np.any(diffs_s <= 0.0):
                return False, "NON_MONOTONIC_TIMESTAMPS"
            if np.any(diffs_s > MAX_TIMESTAMP_GAP_S):
                return False, "TIMESTAMP_GAP"

        return True, None


class ONNXModelRunner:
    """Production ONNX Runtime runner for frozen VelocityNet and BiasNet models."""

    def __init__(self, config: Optional[ModelRunnerConfig] = None) -> None:
        self.config = config or ModelRunnerConfig()
        
        # Load frozen normalizer
        norm_file = Path(self.config.normalization_json_path)
        if not norm_file.exists():
            raise FileNotFoundError(f"Normalization parameters not found at {norm_file}")
        self.normalizer = FeatureNormalizer.load_json(norm_file)

        # Initialize ONNX inference sessions
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        vnet_path = Path(self.config.velocitynet_onnx_path)
        if not vnet_path.exists():
            raise FileNotFoundError(f"VelocityNet ONNX not found at {vnet_path}")
        self.vnet_session = ort.InferenceSession(str(vnet_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.vnet_input_name = self.vnet_session.get_inputs()[0].name

        bnet_path = Path(self.config.biasnet_onnx_path)
        if not bnet_path.exists():
            raise FileNotFoundError(f"BiasNet ONNX not found at {bnet_path}")
        self.bnet_session = ort.InferenceSession(str(bnet_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.bnet_input_name = self.bnet_session.get_inputs()[0].name

    def normalize(self, raw_window_20x9: np.ndarray) -> np.ndarray:
        """Apply frozen channel-wise standardization to a (20, 9) feature window."""
        return self.normalizer.transform(raw_window_20x9).astype(np.float32)

    def run_velocitynet(
        self,
        raw_window_20x9: np.ndarray,
        timestamps_ns: Optional[np.ndarray] = None,
    ) -> VelocityNetOutput:
        """Execute VelocityNet v1.1 on a raw 20x9 feature window.

        Args:
            raw_window_20x9: (20, 9) unnormalized canonical feature array.
            timestamps_ns: Optional (20,) timestamps in nanoseconds.

        Returns:
            VelocityNetOutput with candidate speed, log_variance, and bounded variance.
        """
        is_valid, reason = MLWindowValidator.validate_features(raw_window_20x9, timestamps_ns)
        if not is_valid:
            return VelocityNetOutput(
                speed_mps=0.0,
                log_variance=0.0,
                variance=self.config.r_v_max_mps2,
                valid=False,
                reason=reason,
            )

        norm_feats = self.normalize(raw_window_20x9)
        input_tensor = norm_feats[np.newaxis, :, :]  # Shape (1, 20, 9)

        try:
            outputs = self.vnet_session.run(None, {self.vnet_input_name: input_tensor})
            speed_val = float(outputs[0].reshape(-1)[0])
            log_var_val = float(outputs[1].reshape(-1)[0])
        except Exception as e:
            return VelocityNetOutput(
                speed_mps=0.0,
                log_variance=0.0,
                variance=self.config.r_v_max_mps2,
                valid=False,
                reason=f"INFERENCE_EXCEPTION_{type(e).__name__}",
            )

        # Output sanity checks
        if not (math.isfinite(speed_val) and math.isfinite(log_var_val)):
            return VelocityNetOutput(
                speed_mps=0.0,
                log_variance=0.0,
                variance=self.config.r_v_max_mps2,
                valid=False,
                reason="NONFINITE_MODEL_OUTPUT",
            )

        # Exact transformation: log_var -> variance -> R_v
        # 1. Respect model's [-10.0, 10.0] clamp
        clamped_log_var = max(-10.0, min(10.0, log_var_val))
        raw_variance = math.exp(clamped_log_var)
        # 2. Apply conservative safety bounds [R_v_min, R_v_max]
        safe_variance = max(self.config.r_v_min_mps2, min(self.config.r_v_max_mps2, raw_variance))

        return VelocityNetOutput(
            speed_mps=speed_val,
            log_variance=clamped_log_var,
            variance=safe_variance,
            valid=True,
            reason=None,
        )

    def run_biasnet(
        self,
        raw_window_20x9: np.ndarray,
        timestamps_ns: Optional[np.ndarray] = None,
    ) -> BiasNetOutput:
        """Execute BiasNet v1.0 on a raw 20x9 feature window.

        Args:
            raw_window_20x9: (20, 9) unnormalized canonical feature array.
            timestamps_ns: Optional (20,) timestamps in nanoseconds.

        Returns:
            BiasNetOutput with candidate bias correction vectors.
        """
        zero_ba = np.zeros(3, dtype=np.float64)
        zero_bg = np.zeros(3, dtype=np.float64)
        zero_vec = np.zeros(6, dtype=np.float64)

        is_valid, reason = MLWindowValidator.validate_features(raw_window_20x9, timestamps_ns)
        if not is_valid:
            return BiasNetOutput(
                delta_accel_bias=zero_ba,
                delta_gyro_bias=zero_bg,
                delta_bias_vector=zero_vec,
                valid=False,
                reason=reason,
            )

        norm_feats = self.normalize(raw_window_20x9)
        input_tensor = norm_feats[np.newaxis, :, :]  # Shape (1, 20, 9)

        try:
            outputs = self.bnet_session.run(None, {self.bnet_input_name: input_tensor})
            raw_pred = np.asarray(outputs[0], dtype=np.float64).reshape(-1)
        except Exception as e:
            return BiasNetOutput(
                delta_accel_bias=zero_ba,
                delta_gyro_bias=zero_bg,
                delta_bias_vector=zero_vec,
                valid=False,
                reason=f"INFERENCE_EXCEPTION_{type(e).__name__}",
            )

        if len(raw_pred) != 6 or not np.all(np.isfinite(raw_pred)):
            return BiasNetOutput(
                delta_accel_bias=zero_ba,
                delta_gyro_bias=zero_bg,
                delta_bias_vector=zero_vec,
                valid=False,
                reason="NONFINITE_MODEL_OUTPUT",
            )

        # Enforce physical clamps
        ba_clamp = self.config.bias_accel_clamp_mps2
        bg_clamp = self.config.bias_gyro_clamp_rads
        dba = np.clip(raw_pred[0:3], -ba_clamp, ba_clamp)
        dbg = np.clip(raw_pred[3:6], -bg_clamp, bg_clamp)
        dvec = np.concatenate([dba, dbg])

        return BiasNetOutput(
            delta_accel_bias=dba,
            delta_gyro_bias=dbg,
            delta_bias_vector=dvec,
            valid=True,
            reason=None,
        )
