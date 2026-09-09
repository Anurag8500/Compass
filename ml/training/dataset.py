"""PyTorch Dataset and DataLoader for VelocityNet training and evaluation.

Adheres strictly to the Phase 6 data engineering contracts:
- Consumes pre-normalized (20, 9) tensors and window-end labels
- Filters by is_valid == True; rejects non-finite entries
- Preserves driver/file split isolation (no split mixing)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class VelocityNetDataset(Dataset):
    """PyTorch Dataset for VelocityNet training and evaluation."""

    def __init__(
        self,
        npz_path: Union[str, Path],
        split_name: str = "custom",
        use_raw_speed: bool = False,
    ) -> None:
        """Initialize dataset from Phase 6 serialized .npz archive.

        Args:
            npz_path: Path to train.npz, validation.npz, or test.npz.
            split_name: Human-readable split name for logging.
            use_raw_speed: If True, uses y_speed_raw instead of causally smoothed y_speed.
        """
        self.npz_path = Path(npz_path)
        self.split_name = split_name
        self.use_raw_speed = use_raw_speed

        if not self.npz_path.exists():
            raise FileNotFoundError(f"Dataset archive not found: {self.npz_path}")

        # Load archive with allow_pickle=True to read string metadata arrays
        data = np.load(self.npz_path, allow_pickle=True)

        is_valid_mask = data["is_valid"].astype(bool)
        raw_x = data["X"]
        raw_y = data["y_speed_raw"] if use_raw_speed else data["y_speed"]

        # Filter strictly by valid mask
        valid_indices = np.where(is_valid_mask)[0]
        self.features = raw_x[valid_indices].astype(np.float32)
        self.targets = raw_y[valid_indices].astype(np.float32)
        self.timestamps_end_ns = data["timestamps_end_ns"][valid_indices].astype(np.int64)
        self.timestamps_start_ns = data["timestamps_start_ns"][valid_indices].astype(np.int64)

        if "source_file_ids" in data:
            self.source_file_ids = data["source_file_ids"][valid_indices]
        else:
            self.source_file_ids = np.array(["unknown"] * len(valid_indices), dtype=object)

        if "driver_ids" in data:
            self.driver_ids = data["driver_ids"][valid_indices]
        else:
            self.driver_ids = np.array(["unknown"] * len(valid_indices), dtype=object)

        # Sanity check finiteness
        if not np.all(np.isfinite(self.features)):
            raise ValueError(f"Non-finite values detected in valid features of {self.npz_path}")
        if not np.all(np.isfinite(self.targets)):
            raise ValueError(f"Non-finite values detected in valid targets of {self.npz_path}")

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = torch.from_numpy(self.features[idx])  # shape (20, 9)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return feat, target

    def get_metadata(self, idx: int) -> Dict[str, Union[int, str, float]]:
        """Return diagnostic metadata for a specific window index."""
        return {
            "timestamp_start_ns": int(self.timestamps_start_ns[idx]),
            "timestamp_end_ns": int(self.timestamps_end_ns[idx]),
            "source_file_id": str(self.source_file_ids[idx]),
            "driver_id": str(self.driver_ids[idx]),
            "speed_mps": float(self.targets[idx]),
        }


def create_dataloader(
    dataset: VelocityNetDataset,
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 0,
    seed: int = 42,
) -> DataLoader:
    """Create a DataLoader with deterministic generator."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator if shuffle else None,
        pin_memory=False,
    )
