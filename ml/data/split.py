"""Driver- and File-Level Dataset Partitioning.

COMPASS Phase 6 — ML Dataset Construction.
Enforces zero window-level, file-level, and driver-level leakage across splits.

Partition Architecture:
    - TRAIN: Driver E (64 unique trips, 128 synchronized files across both branches)
    - VALIDATION: Driver B (1 unique trip M, 2 synchronized files)
    - TEST: Driver A (6 unique trips S1, S2, S3a, S3b, S3c, S4, 12 synchronized files)
    - EXCLUDED: Driver D (trip Y1, flagged for clock drift policy threshold in Phase 2)

CRITICAL LEAKAGE INVARIANT:
    TRAIN ∩ VALIDATION = empty
    TRAIN ∩ TEST       = empty
    VALIDATION ∩ TEST  = empty
Both at the source-file level AND at the driver level.
An entire independent driver (Driver A) is strictly held out for final evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import pandas as pd


@dataclass(frozen=True)
class DriverFileSplit:
    """Immutable container for driver/file-level split assignments."""
    version: str
    train_files: List[str]
    validation_files: List[str]
    test_files: List[str]
    excluded_files: List[str]
    driver_mapping: Dict[str, str]  # file_stem / trip_id -> driver_id
    split_by_driver: Dict[str, str] # driver_id -> split_name

    def __post_init__(self) -> None:
        # Assert strict mutual exclusivity at file level
        s_train = set(self.train_files)
        s_val = set(self.validation_files)
        s_test = set(self.test_files)
        s_excl = set(self.excluded_files)

        if s_train.intersection(s_val):
            overlap = s_train.intersection(s_val)
            raise ValueError(f"TRAIN and VALIDATION share files: {overlap}")
        if s_train.intersection(s_test):
            overlap = s_train.intersection(s_test)
            raise ValueError(f"TRAIN and TEST share files: {overlap}")
        if s_val.intersection(s_test):
            overlap = s_val.intersection(s_test)
            raise ValueError(f"VALIDATION and TEST share files: {overlap}")

        # Assert strict mutual exclusivity at driver level for active splits
        d_train = {self.driver_mapping[f] for f in self.train_files if f in self.driver_mapping}
        d_val = {self.driver_mapping[f] for f in self.validation_files if f in self.driver_mapping}
        d_test = {self.driver_mapping[f] for f in self.test_files if f in self.driver_mapping}

        if d_train.intersection(d_val):
            overlap = d_train.intersection(d_val)
            raise ValueError(f"TRAIN and VALIDATION share drivers: {overlap}")
        if d_train.intersection(d_test):
            overlap = d_train.intersection(d_test)
            raise ValueError(f"TRAIN and TEST share drivers: {overlap}")
        if d_val.intersection(d_test):
            overlap = d_val.intersection(d_test)
            raise ValueError(f"VALIDATION and TEST share drivers: {overlap}")

    def get_split(self, file_identifier: str) -> str:
        """Query which split a given file or trip belongs to."""
        clean_id = Path(file_identifier).stem
        for f in self.train_files:
            if Path(f).stem == clean_id or f == file_identifier:
                return "train"
        for f in self.validation_files:
            if Path(f).stem == clean_id or f == file_identifier:
                return "validation"
        for f in self.test_files:
            if Path(f).stem == clean_id or f == file_identifier:
                return "test"
        for f in self.excluded_files:
            if Path(f).stem == clean_id or f == file_identifier:
                return "excluded"
        return "unknown"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize split definition to a JSON-compatible dict."""
        return {
            "version": self.version,
            "train": list(self.train_files),
            "validation": list(self.validation_files),
            "test": list(self.test_files),
            "excluded": list(self.excluded_files),
            "driver_mapping": dict(self.driver_mapping),
            "split_by_driver": dict(self.split_by_driver),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DriverFileSplit:
        """Deserialize split definition from dict."""
        return cls(
            version=str(data["version"]),
            train_files=list(data["train"]),
            validation_files=list(data["validation"]),
            test_files=list(data["test"]),
            excluded_files=list(data.get("excluded", [])),
            driver_mapping=dict(data.get("driver_mapping", {})),
            split_by_driver=dict(data.get("split_by_driver", {})),
        )

    def to_json(self, output_path: Path | str) -> None:
        """Write split to an authoritative JSON file."""
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, json_path: Path | str) -> DriverFileSplit:
        """Load split from JSON file."""
        with open(Path(json_path), "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def create_driver_file_split(
    manifest_csv_path: Path | str = "data/manifests/iovnbd_manifest_v1.csv",
    output_json_path: Optional[Path | str] = "data/splits/split_v1.json",
    version: str = "split_v1",
) -> DriverFileSplit:
    """Construct deterministic driver/file-level split from authoritative manifest.

    Args:
        manifest_csv_path: Path to Phase 2 dataset manifest CSV.
        output_json_path: Optional path to save the serialized split JSON.
        version: Version string for the split specification.

    Returns:
        DriverFileSplit object verified for 100% leakage isolation.
    """
    manifest_path = Path(manifest_csv_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest CSV not found at {manifest_path}")

    df = pd.read_csv(manifest_path)

    # Required columns
    for col in ["driver_id", "cached_npz_path", "downstream_ready"]:
        if col not in df.columns:
            raise ValueError(f"Manifest missing required column: {col}")

    # Canonical driver assignment
    driver_split_map = {
        "Driver E": "train",
        "Driver B": "validation",
        "Driver A": "test",
    }

    train_files: List[str] = []
    val_files: List[str] = []
    test_files: List[str] = []
    excl_files: List[str] = []
    driver_mapping: Dict[str, str] = {}

    for _, row in df.iterrows():
        npz_rel = str(row["cached_npz_path"]).replace("\\", "/")
        driver = str(row["driver_id"]).strip()
        ready = bool(row["downstream_ready"])

        file_stem = Path(npz_rel).stem
        driver_mapping[file_stem] = driver
        driver_mapping[npz_rel] = driver

        if not ready:
            excl_files.append(npz_rel)
            continue

        target_split = driver_split_map.get(driver)
        if target_split == "train":
            train_files.append(npz_rel)
        elif target_split == "validation":
            val_files.append(npz_rel)
        elif target_split == "test":
            test_files.append(npz_rel)
        else:
            excl_files.append(npz_rel)

    # Sort for absolute determinism
    train_files.sort()
    val_files.sort()
    test_files.sort()
    excl_files.sort()

    split = DriverFileSplit(
        version=version,
        train_files=train_files,
        validation_files=val_files,
        test_files=test_files,
        excluded_files=excl_files,
        driver_mapping=driver_mapping,
        split_by_driver=driver_split_map,
    )

    if output_json_path:
        split.to_json(output_json_path)

    return split
