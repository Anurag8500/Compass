"""Training-Only Feature Normalization for COMPASS ML Models.

COMPASS Phase 6 — ML Dataset Construction.
Governs standardization of the canonical (20, 9) feature tensors.

CRITICAL LEAKAGE INVARIANT:
    Normalization parameters (means, stds) MUST BE COMPUTED
    STRICTLY AND EXCLUSIVELY FROM THE TRAINING SPLIT.
Never use validation windows, test windows, or full-dataset statistics.

Invariants:
1. Normalization is performed independently per channel across all time steps.
2. Standard deviations smaller than 1e-6 are clamped to 1.0 to prevent division by zero.
3. Fully compatible with Phase 1 ModelConfig schema and ONNX/LiteRT deployment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np

from navigation.schemas.config import CANONICAL_CHANNELS, ModelConfig

NUM_CHANNELS: int = 9
MIN_STD_EPSILON: float = 1e-6


@dataclass
class FeatureNormalizer:
    """Channel-wise standardizer for (20, 9) feature tensors."""
    means: np.ndarray             # (9,) float64 channel means
    stds: np.ndarray              # (9,) float64 channel standard deviations
    channel_names: List[str]      # Canonical channel names
    sample_count: int = 0         # Number of training samples/windows used for fitting

    def __post_init__(self) -> None:
        self.means = np.asarray(self.means, dtype=np.float64)
        self.stds = np.asarray(self.stds, dtype=np.float64)
        if len(self.means) != NUM_CHANNELS:
            raise ValueError(f"means length ({len(self.means)}) must be {NUM_CHANNELS}")
        if len(self.stds) != NUM_CHANNELS:
            raise ValueError(f"stds length ({len(self.stds)}) must be {NUM_CHANNELS}")
        if len(self.channel_names) != NUM_CHANNELS:
            raise ValueError(f"channel_names length ({len(self.channel_names)}) must be {NUM_CHANNELS}")

        # Ensure no negative or zero stds
        if np.any(self.stds <= 0.0):
            raise ValueError(f"All standard deviations must be strictly positive, got {self.stds}")

    @classmethod
    def fit(
        cls,
        train_tensors: Union[np.ndarray, Sequence[np.ndarray]],
        channel_names: Optional[List[str]] = None,
    ) -> FeatureNormalizer:
        """Compute normalization parameters strictly across training tensors.

        Args:
            train_tensors: Array of shape (N, 20, 9) or (N, 9) containing training samples only.
            channel_names: Optional channel names. Defaults to CANONICAL_CHANNELS.

        Returns:
            Fitted FeatureNormalizer instance.
        """
        channels = list(channel_names) if channel_names else list(CANONICAL_CHANNELS)
        arr = np.asarray(train_tensors, dtype=np.float64)

        if arr.ndim == 3:
            # Shape (N, 20, 9) -> flatten time dimension to (N*20, 9)
            flat = arr.reshape(-1, arr.shape[-1])
        elif arr.ndim == 2:
            # Shape (N, 9)
            flat = arr
        else:
            raise ValueError(f"train_tensors must be 2D or 3D array, got shape {arr.shape}")

        if flat.shape[1] != NUM_CHANNELS:
            raise ValueError(f"Expected {NUM_CHANNELS} channels, got {flat.shape[1]}")
        if flat.shape[0] == 0:
            raise ValueError("Cannot fit normalizer on empty training data")

        # Validate finiteness
        if not np.all(np.isfinite(flat)):
            raise ValueError("Training data contains non-finite values (NaN/Inf)")

        # Compute unbiased mean and standard deviation
        means = np.mean(flat, axis=0)
        stds = np.std(flat, axis=0, ddof=1) if flat.shape[0] > 1 else np.ones(NUM_CHANNELS)

        # Robust protection against near-zero std (e.g. constant channel)
        stds = np.where(stds < MIN_STD_EPSILON, 1.0, stds)

        return cls(
            means=means,
            stds=stds,
            channel_names=channels,
            sample_count=flat.shape[0],
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Standardize feature array: X_norm = (X - mean) / std.

        Args:
            X: Array of shape (..., 9).

        Returns:
            Standardized array with same shape as X.
        """
        arr = np.asarray(X, dtype=np.float64)
        if arr.shape[-1] != NUM_CHANNELS:
            raise ValueError(f"Last dimension must be {NUM_CHANNELS}, got {arr.shape[-1]}")
        return (arr - self.means) / self.stds

    def inverse_transform(self, X_norm: np.ndarray) -> np.ndarray:
        """Revert standardized feature array: X = X_norm * std + mean.

        Args:
            X_norm: Array of shape (..., 9).

        Returns:
            Unnormalized array with same shape as X_norm.
        """
        arr = np.asarray(X_norm, dtype=np.float64)
        if arr.shape[-1] != NUM_CHANNELS:
            raise ValueError(f"Last dimension must be {NUM_CHANNELS}, got {arr.shape[-1]}")
        return arr * self.stds + self.means

    def to_model_config(
        self,
        model_name: str = "VelocityNet",
        filter_coefficients: Optional[Dict[str, Any]] = None,
    ) -> ModelConfig:
        """Convert to Phase 1 ModelConfig schema."""
        return ModelConfig(
            model_name=model_name,
            normalization_means=[float(m) for m in self.means],
            normalization_stds=[float(s) for s in self.stds],
            filter_coefficients=filter_coefficients or {},
            channel_order=list(self.channel_names),
        )

    @classmethod
    def from_model_config(cls, config: ModelConfig) -> FeatureNormalizer:
        """Instantiate from Phase 1 ModelConfig schema."""
        return cls(
            means=np.array(config.normalization_means, dtype=np.float64),
            stds=np.array(config.normalization_stds, dtype=np.float64),
            channel_names=list(config.channel_order),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize normalizer to dictionary."""
        return {
            "means": [float(m) for m in self.means],
            "stds": [float(s) for s in self.stds],
            "channel_names": list(self.channel_names),
            "sample_count": self.sample_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FeatureNormalizer:
        """Deserialize normalizer from dictionary."""
        return cls(
            means=np.array(data["means"], dtype=np.float64),
            stds=np.array(data["stds"], dtype=np.float64),
            channel_names=list(data["channel_names"]),
            sample_count=int(data.get("sample_count", 0)),
        )

    def save_json(self, output_path: Path | str) -> None:
        """Save normalizer to a JSON file."""
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, json_path: Path | str) -> FeatureNormalizer:
        """Load normalizer from a JSON file."""
        with open(Path(json_path), "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
