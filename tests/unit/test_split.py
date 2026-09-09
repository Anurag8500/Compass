"""Unit tests for driver/file-level dataset partitioning (ml/data/split.py).

Verifies:
1. Complete assignment of all 142 downstream-ready IO-VNBD trips.
2. Exact assignment: Driver E -> Train, Driver B -> Validation, Driver A -> Test.
3. Strict disjoint file and driver sets.
4. Hard failure on injected file or driver cross-contamination.
5. JSON persistence round-trip fidelity.
"""

from __future__ import annotations

from pathlib import Path
import pytest
from ml.data.split import DriverFileSplit, create_driver_file_split


class TestSplit:
    """Test suite for driver/file-level splitting."""

    @pytest.fixture
    def split(self) -> DriverFileSplit:
        return create_driver_file_split()

    def test_complete_and_deterministic_file_counts(self, split: DriverFileSplit) -> None:
        """Verify exact counts matching IO-VNBD manifest inventory."""
        assert len(split.train_files) == 128
        assert len(split.validation_files) == 2
        assert len(split.test_files) == 12
        assert len(split.excluded_files) == 2
        total = len(split.train_files) + len(split.validation_files) + len(split.test_files) + len(split.excluded_files)
        assert total == 144

    def test_disjoint_files_and_drivers(self, split: DriverFileSplit) -> None:
        """Assert zero overlap between train, val, and test across files and drivers."""
        s_train = set(split.train_files)
        s_val = set(split.validation_files)
        s_test = set(split.test_files)

        assert s_train.isdisjoint(s_val)
        assert s_train.isdisjoint(s_test)
        assert s_val.isdisjoint(s_test)

        d_train = {split.driver_mapping[f] for f in split.train_files}
        d_val = {split.driver_mapping[f] for f in split.validation_files}
        d_test = {split.driver_mapping[f] for f in split.test_files}

        assert d_train == {"Driver E"}
        assert d_val == {"Driver B"}
        assert d_test == {"Driver A"}

        assert d_train.isdisjoint(d_val)
        assert d_train.isdisjoint(d_test)
        assert d_val.isdisjoint(d_test)

    def test_hard_rejection_on_injected_file_leakage(self) -> None:
        """Injecting a file into both Train and Validation must raise a ValueError."""
        with pytest.raises(ValueError, match="share files"):
            DriverFileSplit(
                version="test_leak",
                train_files=["file_01.npz", "file_shared.npz"],
                validation_files=["file_shared.npz", "file_02.npz"],
                test_files=["file_03.npz"],
                excluded_files=[],
                driver_mapping={"file_01.npz": "D1", "file_shared.npz": "D1", "file_02.npz": "D2", "file_03.npz": "D3"},
                split_by_driver={"D1": "train", "D2": "val", "D3": "test"},
            )

    def test_hard_rejection_on_injected_driver_leakage(self) -> None:
        """Injecting the same driver across Train and Test must raise a ValueError."""
        with pytest.raises(ValueError, match="share drivers"):
            DriverFileSplit(
                version="test_leak_driver",
                train_files=["file_01.npz"],
                validation_files=["file_02.npz"],
                test_files=["file_03.npz"],
                excluded_files=[],
                driver_mapping={"file_01.npz": "Driver_Leak", "file_02.npz": "Driver_2", "file_03.npz": "Driver_Leak"},
                split_by_driver={"Driver_Leak": "train", "Driver_2": "val"},
            )

    def test_json_roundtrip(self, tmp_path: Path, split: DriverFileSplit) -> None:
        """Split must serialize to and deserialize from JSON identically."""
        json_file = tmp_path / "test_split.json"
        split.to_json(json_file)
        reloaded = DriverFileSplit.from_json(json_file)

        assert reloaded.version == split.version
        assert reloaded.train_files == split.train_files
        assert reloaded.validation_files == split.validation_files
        assert reloaded.test_files == split.test_files
        assert reloaded.excluded_files == split.excluded_files

    def test_type_hints_evaluable(self) -> None:
        """Type hints must evaluate without NameError (e.g. typing.get_type_hints)."""
        import typing
        hints = typing.get_type_hints(create_driver_file_split)
        assert "output_json_path" in hints
