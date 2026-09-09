"""Unit tests for VelocityNetDataset and DataLoader mechanics."""

from pathlib import Path
import pytest
import torch
from ml.training.dataset import VelocityNetDataset, create_dataloader


class TestVelocityNetDataset:
    """Test suite for Phase 6 dataset loading, batching, and integrity."""

    @classmethod
    def setup_class(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[2]
        cls.val_path = cls.project_root / "data" / "ml_dataset_v1" / "validation.npz"
        if not cls.val_path.exists():
            pytest.skip("Phase 6 validation.npz artifact not found")

    def test_validation_dataset_loads_correctly(self) -> None:
        """Verify validation dataset initializes with valid window count."""
        ds = VelocityNetDataset(self.val_path, split_name="validation")
        # Validation has 21,080 valid windows
        assert len(ds) == 21080

        feat, target = ds[0]
        assert feat.shape == (20, 9)
        assert feat.dtype == torch.float32
        assert isinstance(target.item(), float)
        assert torch.isfinite(feat).all()
        assert torch.isfinite(target)

    def test_metadata_retrieval(self) -> None:
        """Verify metadata dictionary is correctly populated."""
        ds = VelocityNetDataset(self.val_path, split_name="validation")
        meta = ds.get_metadata(0)
        assert "timestamp_start_ns" in meta
        assert "timestamp_end_ns" in meta
        assert "source_file_id" in meta
        assert "driver_id" in meta
        assert "speed_mps" in meta
        assert meta["driver_id"] == "Driver B"

    def test_dataloader_batch_generation(self) -> None:
        """Verify DataLoader batches features into (batch_size, 20, 9)."""
        ds = VelocityNetDataset(self.val_path, split_name="validation")
        loader = create_dataloader(ds, batch_size=32, shuffle=False)

        batch_x, batch_y = next(iter(loader))
        assert batch_x.shape == (32, 20, 9)
        assert batch_y.shape == (32,)
        assert batch_x.dtype == torch.float32
        assert batch_y.dtype == torch.float32

    def test_train_and_test_split_integrity(self) -> None:
        """Verify train and held-out test splits have expected counts and driver IDs."""
        train_path = self.project_root / "data" / "ml_dataset_v1" / "train.npz"
        test_path = self.project_root / "data" / "ml_dataset_v1" / "test.npz"

        if train_path.exists():
            train_ds = VelocityNetDataset(train_path, split_name="train")
            assert len(train_ds) == 226928
            assert train_ds.get_metadata(0)["driver_id"] == "Driver E"

            # Check that static train mean is computed dynamically and is finite
            import numpy as np
            train_mean = float(np.mean(train_ds.targets))
            assert 10.0 < train_mean < 25.0, f"Unexpected train mean: {train_mean}"

        if test_path.exists():
            test_ds = VelocityNetDataset(test_path, split_name="test")
            assert len(test_ds) == 123464
            assert test_ds.get_metadata(0)["driver_id"] == "Driver A"

    def test_baseline_causality_properties(self) -> None:
        """Verify distinction between operational causal baseline and non-causal oracle."""
        import numpy as np
        from ml.training.train_velocitynet import compute_metrics

        test_path = self.project_root / "data" / "ml_dataset_v1" / "test.npz"
        train_path = self.project_root / "data" / "ml_dataset_v1" / "train.npz"
        if not (test_path.exists() and train_path.exists()):
            pytest.skip("Dataset splits not found")

        train_ds = VelocityNetDataset(train_path, split_name="train")
        test_ds = VelocityNetDataset(test_path, split_name="test")

        # Operational Causal Baseline uses training distribution mean only
        train_mean = float(np.mean(train_ds.targets))
        causal_preds = np.full_like(test_ds.targets, train_mean)
        causal_metrics = compute_metrics(causal_preds, test_ds.targets)
        assert causal_metrics["rmse"] > 8.0  # Around 9.305 m/s

        # Non-causal oracle uses ground-truth speed from test targets (infeasible without GNSS)
        oracle_preds = np.roll(test_ds.targets, 1)
        oracle_preds[0] = test_ds.targets[0]
        oracle_metrics = compute_metrics(oracle_preds, test_ds.targets)
        assert oracle_metrics["rmse"] < 1.0  # Lag-1 autocorrelation ~ 0.401 m/s
        # Verify that causal and non-causal are fundamentally different categories
        assert causal_metrics["rmse"] > 5.0 * oracle_metrics["rmse"]

