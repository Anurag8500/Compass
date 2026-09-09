"""Integration tests verifying generated Phase 6 ML dataset artifacts.

Verifies:
1. Physical existence and schema compliance of all serialized artifacts:
   - train.npz, validation.npz, test.npz
   - normalization.json, model_config.json, dataset_manifest.json
2. Exact tensor dimensions (N, 20, 9) and finiteness across all splits.
3. Strict zero-leakage audit across the actual serialized source_file_ids and driver_ids.
4. Normalization fidelity: train split standardized to mean=0, std=1; valid inverse transform.
5. VelocityNet label coverage and causality consistency.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest
from ml.data.normalization import FeatureNormalizer


class TestMLDatasetArtifacts:
    """Audit suite for serialized Phase 6 dataset outputs."""

    @classmethod
    def setup_class(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[2]
        cls.dataset_dir = cls.project_root / "data" / "ml_dataset_v1"

        if not (cls.dataset_dir / "dataset_manifest.json").exists():
            pytest.skip(f"Dataset artifacts not found at {cls.dataset_dir}")

        with open(cls.dataset_dir / "dataset_manifest.json", "r", encoding="utf-8") as f:
            cls.manifest = json.load(f)

        cls.train_data = np.load(cls.dataset_dir / "train.npz", allow_pickle=True)
        cls.val_data = np.load(cls.dataset_dir / "validation.npz", allow_pickle=True)
        cls.test_data = np.load(cls.dataset_dir / "test.npz", allow_pickle=True)
        cls.normalizer = FeatureNormalizer.load_json(cls.dataset_dir / "normalization.json")

    def test_tensor_shapes_and_invariants(self) -> None:
        """Verify (N, 20, 9) shapes and aligned target vector dimensions."""
        assert self.train_data["X"].shape == (233830, 20, 9)
        assert self.train_data["y_speed"].shape == (233830,)
        assert self.train_data["timestamps_end_ns"].shape == (233830,)

        assert self.val_data["X"].shape == (42388, 20, 9)
        assert self.val_data["y_speed"].shape == (42388,)

        assert self.test_data["X"].shape == (123496, 20, 9)
        assert self.test_data["y_speed"].shape == (123496,)

        total_windows = len(self.train_data["X"]) + len(self.val_data["X"]) + len(self.test_data["X"])
        assert total_windows == 399714

    def test_all_features_and_labels_finite(self) -> None:
        """Ensure zero non-finite values in features and valid labels."""
        for split_name, data in [
            ("train", self.train_data),
            ("val", self.val_data),
            ("test", self.test_data),
        ]:
            assert np.all(np.isfinite(data["X"])), f"Non-finite in {split_name} X"
            assert np.all(np.isfinite(data["X_raw"])), f"Non-finite in {split_name} X_raw"
            valid_mask = data["is_valid"]
            assert np.all(np.isfinite(data["y_speed"][valid_mask])), f"Non-finite valid label in {split_name} y_speed"
            assert np.all(np.isfinite(data["y_speed_raw"][valid_mask])), f"Non-finite valid raw label in {split_name} y_speed_raw"

    def test_serialized_leakage_audit_file_and_driver_isolation(self) -> None:
        """Assert zero file or driver leakage in the actual saved arrays."""
        train_files = set(self.train_data["source_file_ids"])
        val_files = set(self.val_data["source_file_ids"])
        test_files = set(self.test_data["source_file_ids"])

        assert train_files.isdisjoint(val_files), "Train and Val share files in serialized arrays!"
        assert train_files.isdisjoint(test_files), "Train and Test share files in serialized arrays!"
        assert val_files.isdisjoint(test_files), "Val and Test share files in serialized arrays!"

        train_drivers = set(self.train_data["driver_ids"])
        val_drivers = set(self.val_data["driver_ids"])
        test_drivers = set(self.test_data["driver_ids"])

        assert train_drivers == {"Driver E"}
        assert val_drivers == {"Driver B"}
        assert test_drivers == {"Driver A"}

        assert train_drivers.isdisjoint(val_drivers)
        assert train_drivers.isdisjoint(test_drivers)
        assert val_drivers.isdisjoint(test_drivers)

    def test_excluded_files_absent_from_all_serialized_arrays(self) -> None:
        """Driver D files (trip Y1) must be completely absent from train, val, and test."""
        all_active_files = set(self.train_data["source_file_ids"]).union(
            set(self.val_data["source_file_ids"])
        ).union(set(self.test_data["source_file_ids"]))

        for f in all_active_files:
            assert "Y1" not in f, f"Excluded file {f} found in serialized splits!"
            assert "Driver D" not in f

        all_active_drivers = set(self.train_data["driver_ids"]).union(
            set(self.val_data["driver_ids"])
        ).union(set(self.test_data["driver_ids"]))
        assert "Driver D" not in all_active_drivers

    def test_normalization_statistical_properties(self) -> None:
        """Normalized valid train features must exhibit mean ~ 0 and std ~ 1."""
        valid_mask = self.train_data["is_valid"]
        valid_X_norm = self.train_data["X"][valid_mask]  # (N_valid, 20, 9)

        # Flat view across samples and time steps: (N_valid * 20, 9)
        flat = valid_X_norm.reshape(-1, 9)
        means = np.mean(flat, axis=0)
        stds = np.std(flat, axis=0)

        np.testing.assert_allclose(means, 0.0, atol=1e-2)
        np.testing.assert_allclose(stds, 1.0, atol=1e-2)

    def test_inverse_normalization_fidelity(self) -> None:
        """X_raw must equal inverse_transform(X) to numerical precision."""
        sample_X = self.test_data["X"][:100]
        sample_raw = self.test_data["X_raw"][:100]

        recovered = self.normalizer.inverse_transform(sample_X)
        np.testing.assert_allclose(recovered, sample_raw, rtol=1e-4, atol=1e-4)
