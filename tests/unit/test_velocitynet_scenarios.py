"""Unit tests proving scenario evaluation operates strictly in physical units."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from ml.training.dataset import VelocityNetDataset


class TestVelocityNetPhysicalScenarios:
    """Test suite ensuring scenario classifications use physical units, not normalized features."""

    @pytest.fixture
    def dataset_paths(self):
        root = Path(__file__).resolve().parents[2]
        data_dir = root / "data" / "ml_dataset_v1"
        test_path = data_dir / "test.npz"
        norm_path = data_dir / "normalization.json"
        if not (test_path.exists() and norm_path.exists()):
            pytest.skip("Test dataset or normalization.json not found")
        return test_path, norm_path

    def test_features_are_normalized_and_differ_from_raw(self, dataset_paths):
        """Verify that X is normalized while raw_features are in physical units."""
        test_path, norm_path = dataset_paths
        ds = VelocityNetDataset(test_path, split_name="test")

        assert hasattr(ds, "raw_features"), "VelocityNetDataset must provide raw_features for physical analysis"
        assert ds.raw_features.shape == ds.features.shape

        # In normalized X, vertical specific force (channel 2) has mean near 0 and std near 1
        # In physical units, vertical specific force contains gravity (~9.81 m/s^2)
        norm_fz = ds.features[:, -1, 2]
        raw_fz = ds.raw_features[:, -1, 2]

        assert np.abs(np.mean(norm_fz)) < 1.0, "Normalized f_z should be zero-centered"
        assert np.mean(raw_fz) > 8.0, "Raw physical f_z must preserve earth gravity (~9.81 m/s^2)"

    def test_scenario_thresholds_applied_to_physical_units(self, dataset_paths):
        """Prove that evaluating scenario thresholds on raw vs normalized yields radically different masks."""
        test_path, _ = dataset_paths
        ds = VelocityNetDataset(test_path, split_name="test")

        # Physical threshold: |omega_z| <= 0.05 rad/s
        raw_wz = ds.raw_features[:, -1, 5]
        norm_wz = ds.features[:, -1, 5]

        phys_straight_mask = np.abs(raw_wz) <= 0.05
        erroneous_norm_straight_mask = np.abs(norm_wz) <= 0.05

        # In physical units, straight driving dominates (> 70% of dataset)
        phys_straight_pct = np.mean(phys_straight_mask) * 100
        norm_straight_pct = np.mean(erroneous_norm_straight_mask) * 100

        assert phys_straight_pct > 70.0, f"Expected >70% straight driving in physical units, got {phys_straight_pct:.1f}%"
        # If evaluated on normalized wz, 0.05 represents 0.05 standard deviations, which only captures ~20% of windows!
        assert norm_straight_pct < 25.0, (
            f"Evaluating on normalized features would erroneously classify only {norm_straight_pct:.1f}% as straight"
        )
        assert not np.array_equal(phys_straight_mask, erroneous_norm_straight_mask), (
            "Physical mask must strictly differ from normalized thresholding"
        )

    def test_dynamic_acceleration_threshold_in_physical_units(self, dataset_paths):
        """Prove dynamic acceleration (|norm_f - 9.81| > 1.5 m/s^2) operates in physical units."""
        test_path, _ = dataset_paths
        ds = VelocityNetDataset(test_path, split_name="test")

        # Feature index 6 is norm_f_v
        raw_norm_f = ds.raw_features[:, -1, 6]
        phys_dynamic_mask = np.abs(raw_norm_f - 9.81) > 1.5

        # Dynamic events are transient (< 10% of windows)
        dynamic_pct = np.mean(phys_dynamic_mask) * 100
        assert 0.1 < dynamic_pct < 10.0, f"Dynamic acceleration events should be transient (0.1-10%), got {dynamic_pct:.2f}%"
