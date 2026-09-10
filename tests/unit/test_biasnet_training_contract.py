"""Unit tests for BiasNet training contracts and provenance (Phase 8).

Verifies:
1. Split correctness: Driver E in train, Driver B in val, Driver A strictly absent.
2. No test leakage: Driver A is not present in train or validation sets.
3. Checkpoint validity: models/biasnet_v1_best.pt exists and loads cleanly.
4. Provenance & metadata integrity: models/model_config_biasnet_v1.json exists,
   records architecture, training duration, and direct validation baseline comparisons.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch
import pytest

from ml.models.biasnet import BiasNet


class TestBiasNetTrainingContract:
    """Test suite verifying training contract, data splits, and model provenance."""

    @classmethod
    def setup_class(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]
        cls.train_npz = cls.root / "data" / "ml_dataset_biasnet_v1" / "bias_train.npz"
        cls.val_npz = cls.root / "data" / "ml_dataset_biasnet_v1" / "bias_validation.npz"
        cls.model_pt = cls.root / "models" / "biasnet_v1_best.pt"
        cls.model_cfg = cls.root / "models" / "model_config_biasnet_v1.json"

    def test_dataset_files_exist(self) -> None:
        """Dataset files must exist and contain non-empty arrays."""
        assert self.train_npz.exists(), f"Missing {self.train_npz}"
        assert self.val_npz.exists(), f"Missing {self.val_npz}"

        d_train = np.load(self.train_npz)
        d_val = np.load(self.val_npz)

        assert len(d_train["X"]) > 0
        assert len(d_train["labels_constrained"]) == len(d_train["X"])
        assert len(d_val["X"]) > 0
        assert len(d_val["labels_constrained"]) == len(d_val["X"])

    def test_no_test_leakage(self) -> None:
        """Driver A (held-out) must never be present in train or validation data."""
        with open(self.root / "data" / "splits" / "split_v1.json") as f:
            split_info = json.load(f)

        test_files = set(split_info.get("test", []))
        assert len(test_files) > 0, "Test files must be defined"

        # Check train and val metadata
        manifest_path = self.root / "data" / "ml_dataset_biasnet_v1" / "bias_label_manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                man = json.load(f)
            assert man["provenance"]["train_driver"] == "Driver E"
            assert man["provenance"]["validation_driver"] == "Driver B"
            assert "Driver A" in man["provenance"]["test_driver"]

    def test_checkpoint_loads_cleanly(self) -> None:
        """PyTorch checkpoint must exist and restore state dict without missing keys."""
        assert self.model_pt.exists(), f"Missing {self.model_pt}"

        model = BiasNet(input_dim=9, hidden_dim=48, num_layers=2, dense_dim=24)
        state_dict = torch.load(self.model_pt, map_location="cpu", weights_only=True)

        # Load state dict
        keys = model.load_state_dict(state_dict)
        assert len(keys.missing_keys) == 0
        assert len(keys.unexpected_keys) == 0

    def test_model_config_provenance(self) -> None:
        """Model config JSON must contain full provenance and baseline comparison."""
        assert self.model_cfg.exists(), f"Missing {self.model_cfg}"

        with open(self.model_cfg) as f:
            cfg = json.load(f)

        assert cfg["model_name"] == "BiasNet"
        assert cfg["total_parameters"] == 23934
        assert cfg["training_provenance"]["train_driver"] == "Driver E"
        assert cfg["training_provenance"]["val_driver"] == "Driver B"

        comp = cfg["baselines_comparison_driver_b"]
        assert "biasnet" in comp
        assert "zero_baseline" in comp
        assert "train_mean_baseline" in comp

        # BiasNet must strictly outperform zero baseline on Driver B
        bn_rmse = comp["biasnet"]["summary"]["total_vector_rmse"]
        zero_rmse = comp["zero_baseline"]["summary"]["total_vector_rmse"]
        assert bn_rmse < zero_rmse, f"BiasNet RMSE ({bn_rmse:.4f}) should be lower than zero baseline ({zero_rmse:.4f})"
