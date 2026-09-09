"""Mandatory Leakage Audit Test Suite (tests/unit/test_leakage_audit.py).

COMPASS Phase 6 — ML Dataset Construction.
THIS IS THE SINGLE MOST IMPORTANT TEST SUITE IN PHASE 6.

Requirements:
1. No file appears in multiple splits.
2. No driver appears in multiple splits.
3. Every generated window traces back to exactly one source file and one driver.
4. No train window shares source data or timestamps with validation/test.
5. HARD FAILURE on any leakage violation — zero warnings tolerated.
"""

from __future__ import annotations

from pathlib import Path
import pytest
from ml.data.split import DriverFileSplit, create_driver_file_split


class TestLeakageAudit:
    """Rigorous mathematical audit for zero data leakage across splits."""

    @pytest.fixture
    def split(self) -> DriverFileSplit:
        return create_driver_file_split()

    def test_file_level_leakage_audit_strictly_disjoint(self, split: DriverFileSplit) -> None:
        """Assert TRAIN, VAL, and TEST file sets have zero intersection."""
        train_set = set(split.train_files)
        val_set = set(split.validation_files)
        test_set = set(split.test_files)

        # 1. Train vs Validation
        train_val_overlap = train_set.intersection(val_set)
        assert len(train_val_overlap) == 0, (
            f"LEAKAGE VIOLATION: Files found in both TRAIN and VAL: {train_val_overlap}"
        )

        # 2. Train vs Test
        train_test_overlap = train_set.intersection(test_set)
        assert len(train_test_overlap) == 0, (
            f"LEAKAGE VIOLATION: Files found in both TRAIN and TEST: {train_test_overlap}"
        )

        # 3. Val vs Test
        val_test_overlap = val_set.intersection(test_set)
        assert len(val_test_overlap) == 0, (
            f"LEAKAGE VIOLATION: Files found in both VAL and TEST: {val_test_overlap}"
        )

    def test_driver_level_leakage_audit_strictly_disjoint(self, split: DriverFileSplit) -> None:
        """Assert TRAIN, VAL, and TEST driver assignments have zero intersection."""
        train_drivers = {split.driver_mapping[f] for f in split.train_files}
        val_drivers = {split.driver_mapping[f] for f in split.validation_files}
        test_drivers = {split.driver_mapping[f] for f in split.test_files}

        # 1. Train vs Validation
        driver_tv_overlap = train_drivers.intersection(val_drivers)
        assert len(driver_tv_overlap) == 0, (
            f"LEAKAGE VIOLATION: Drivers found in both TRAIN and VAL: {driver_tv_overlap}"
        )

        # 2. Train vs Test
        driver_tt_overlap = train_drivers.intersection(test_drivers)
        assert len(driver_tt_overlap) == 0, (
            f"LEAKAGE VIOLATION: Drivers found in both TRAIN and TEST: {driver_tt_overlap}"
        )

        # 3. Val vs Test
        driver_vt_overlap = val_drivers.intersection(test_drivers)
        assert len(driver_vt_overlap) == 0, (
            f"LEAKAGE VIOLATION: Drivers found in both VAL and TEST: {driver_vt_overlap}"
        )

    def test_held_out_test_driver_integrity(self, split: DriverFileSplit) -> None:
        """At least one entire driver must be completely held out for TEST."""
        test_drivers = {split.driver_mapping[f] for f in split.test_files}
        assert len(test_drivers) >= 1
        assert "Driver A" in test_drivers

        # Ensure no Driver A file exists in train or validation
        for f in split.train_files + split.validation_files:
            driver = split.driver_mapping.get(f)
            assert driver != "Driver A", f"Driver A file {f} leaked into train/val!"

    def test_synthetic_cross_contamination_triggers_hard_failure(self) -> None:
        """Verifies that the audit code will reliably fail loudly if leakage occurs."""
        bad_split_data = {
            "version": "corrupted_split",
            "train": ["trip_1.npz", "trip_leak.npz"],
            "validation": ["trip_2.npz"],
            "test": ["trip_leak.npz"],  # Injected leakage!
            "excluded": [],
            "driver_mapping": {"trip_1.npz": "D1", "trip_leak.npz": "D1", "trip_2.npz": "D2"},
            "split_by_driver": {"D1": "train", "D2": "val"},
        }

        with pytest.raises(ValueError, match="share files"):
            DriverFileSplit.from_dict(bad_split_data)
