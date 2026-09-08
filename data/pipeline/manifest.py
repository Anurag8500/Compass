"""Dataset manifest builder and registry for COMPASS.

Builds the authoritative versioned dataset manifest:
    data/manifests/iovnbd_manifest_v1.csv
linking raw CSV files, quality metrics, stationary annotations,
outage indexing, and cached synchronized binary arrays (.npz).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

import re
from data.pipeline.outage_index import OutageIndex, OutageWindow
from data.pipeline.parse import parse_s_file, parse_v_file
from data.pipeline.quality_tagger import QualityTagger
from data.pipeline.stationary_detect import StationaryDetector
from data.pipeline.sync import SynchronizedTrip, synchronize_s_v


def extract_base_trip_id(filename: str) -> str:
    """Extracts trip identifier from S- or V- filename (e.g. S-S1.csv -> S1, S-Vta12.csv -> Vta12)."""
    clean_name = re.sub(r"^[SsVv]-", "", filename)
    clean_name = re.sub(r"\.csv$", "", clean_name, flags=re.IGNORECASE)
    return clean_name.strip()


def discover_dataset_pairs(data_dir: Path) -> List[Tuple[Path, Path, str, str]]:
    """Discovers folder-aware matched S/V pairs across Categorised and Uncategorised branches.

    Returns:
        List of (s_path, v_path, trip_id, branch_name) sorted deterministically.
    """
    data_dir = Path(data_dir)
    csv_files = list(data_dir.rglob("*.csv"))

    categorised_s: Dict[str, Path] = {}
    categorised_v: Dict[str, Path] = {}
    uncategorised_s: Dict[str, Path] = {}
    uncategorised_v: Dict[str, Path] = {}

    for f in csv_files:
        name_lower = f.name.lower()
        parts = [p.lower() for p in f.parts]
        is_s = name_lower.startswith("s-")
        is_v = name_lower.startswith("v-")

        if not (is_s or is_v):
            continue

        trip_id = extract_base_trip_id(f.name)
        trip_id_key = trip_id.lower()

        if any("categorised" in p and "uncategorised" not in p for p in parts):
            parent_key = f.parent.name.lower()
            key = f"{parent_key}/{trip_id_key}"
            if is_s:
                categorised_s[key] = f
            else:
                categorised_v[key] = f
        elif any("uncategorised" in p for p in parts):
            key = trip_id_key
            if is_s:
                uncategorised_s[key] = f
            else:
                uncategorised_v[key] = f

    matched: List[Tuple[Path, Path, str, str]] = []

    for key, s_path in categorised_s.items():
        if key in categorised_v:
            v_path = categorised_v[key]
            trip_id = extract_base_trip_id(s_path.name)
            matched.append((s_path, v_path, trip_id, "Categorised IOVNB Dataset"))

    for key, s_path in uncategorised_s.items():
        if key in uncategorised_v:
            v_path = uncategorised_v[key]
            trip_id = extract_base_trip_id(s_path.name)
            matched.append((s_path, v_path, trip_id, "Uncategorised IOVNB Dataset"))

    matched.sort(key=lambda x: (x[3], x[2]))
    return matched


def derive_driver_map_from_categorised(data_dir: Path) -> Dict[str, str]:
    """Deterministically derive trip_id -> driver_id mapping from Categorised folder names.

    E.g. files under 'M (Driver B)/' yield 'm' -> 'Driver B'.
    Files under 'S (Driver A)/' yield 's1' -> 'Driver A', etc.
    """
    driver_map: Dict[str, str] = {}
    for csv_file in Path(data_dir).rglob("*.csv"):
        parts = [p.lower() for p in csv_file.parts]
        if any("categorised" in p and "uncategorised" not in p for p in parts):
            match = re.search(r"Driver\s+([A-Za-z0-9]+)", str(csv_file), re.IGNORECASE)
            if match:
                driver_str = f"Driver {match.group(1).upper()}"
                trip_id = extract_base_trip_id(csv_file.name).lower()
                driver_map[trip_id] = driver_str
    return driver_map


@dataclass
class ManifestRow:
    """Single row in the dataset manifest with full synchronization audit fields."""
    trip_id: str
    branch: str
    driver_id: str
    s_filename: str
    s_relative_path: str
    v_filename: str
    v_relative_path: str
    raw_s_rows: int
    raw_v_rows: int
    synced_rows: int
    validated_rows: int
    omitted_rows: int
    flag_ok_count: int
    measured_rate_hz: float
    duration_s: float
    has_stationary_segment: bool
    stationary_segment_count: int
    total_stationary_duration_s: float
    has_real_outages_for_trip: bool
    has_real_outages_in_dataset: bool
    has_real_outages: bool                   # Kept for backward compatibility
    outage_windows_json: str
    extreme_accel_count: int
    extreme_gyro_count: int
    extreme_motion_count: int
    dropout_count: int
    nan_or_nonfinite_count: int
    cached_npz_path: str
    non_monotonic_count: int = 0
    duplicate_count: int = 0
    invalid_timestamp_count: int = 0
    sync_mode: str = "relative_elapsed"
    sync_status: str = "PASSED"
    sync_clock_origin_offset_s: float = 0.0
    sync_clock_drift_s: float = 0.0
    sync_overlap_duration_s: float = 0.0
    sync_valid_interpolated_count: int = 0
    sync_out_of_bounds_count: int = 0
    sync_gap_violation_count: int = 0
    sync_valid_coverage_ratio: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DatasetManifestBuilder:
    """Builds, verifies, and exports the IO-VNBD dataset manifest."""

    def __init__(
        self,
        project_root: Optional[Path | str] = None,
        cache_dir: Optional[Path | str] = None,
        manifest_path: Optional[Path | str] = None,
    ) -> None:
        self.project_root = Path(project_root or Path.cwd()).resolve()
        self.cache_dir = Path(cache_dir or (self.project_root / "data" / "cache" / "iovnbd")).resolve()
        self.manifest_path = Path(manifest_path or (self.project_root / "data" / "manifests" / "iovnbd_manifest_v1.csv")).resolve()

        self.tagger = QualityTagger()
        self.stationary_detector = StationaryDetector()
        self.outage_index = OutageIndex()
        self.trip_driver_map: Dict[str, str] = {}

    def _extract_driver_id(self, file_path: Path, trip_id: str = "") -> str:
        """Extract and normalize driver identifier from directory path or resolved Categorised mapping."""
        # 1. Direct path check (e.g. Categorised branch folder name contains 'Driver X')
        match = re.search(r"Driver\s+([A-Za-z0-9]+)", str(file_path), re.IGNORECASE)
        if match:
            return f"Driver {match.group(1).upper()}"

        # 2. Canonical data-derived lookup from discovered Categorised branch
        tid_key = trip_id.strip().lower()
        if tid_key in self.trip_driver_map:
            return self.trip_driver_map[tid_key]

        # 3. Dynamic lookup if self.trip_driver_map is empty but raw dataset is accessible
        raw_dir = self.project_root / "data" / "raw" / "io_vnbd"
        if not self.trip_driver_map and raw_dir.exists():
            self.trip_driver_map = derive_driver_map_from_categorised(raw_dir)
            if tid_key in self.trip_driver_map:
                return self.trip_driver_map[tid_key]

        # 4. Unknown fallback if not present in dataset directory hierarchy
        return "Unknown"

    def process_pair(
        self,
        s_path: Path,
        v_path: Path,
        trip_id: str,
        branch: str,
        save_cache: bool = True,
    ) -> ManifestRow:
        """Process a single S/V pair and optionally serialize cached array."""
        driver_id = self._extract_driver_id(s_path, trip_id=trip_id)

        # 1. Parse raw files
        s_trip = parse_s_file(s_path)
        v_trip = parse_v_file(v_path)

        # 2. Synchronize & Quality Tag
        synced_trip, quality_rep = synchronize_s_v(
            s_trip=s_trip,
            v_trip=v_trip,
            trip_id=trip_id,
            branch=branch,
            tagger=self.tagger,
        )

        # 3. Detect stationary segments on synchronized IMU stream
        stationary_segs, _ = self.stationary_detector.detect(
            synced_trip.timestamps_ns,
            synced_trip.accel_raw,
            synced_trip.gyro_raw,
        )
        has_stationary = len(stationary_segs) > 0
        total_stationary_dur = sum(s.duration_s for s in stationary_segs)

        # 4. Outage query
        outages = self.outage_index.get_outages(trip_id)
        has_real_for_trip = self.outage_index.has_real_outages_for_trip(trip_id)
        has_real_in_dataset = self.outage_index.has_real_outages

        outages_json = json.dumps([
            {
                "start_ns": o.start_timestamp_ns,
                "end_ns": o.end_timestamp_ns,
                "duration_s": o.duration_s,
                "is_synthetic": o.is_synthetic,
                "label": o.label,
            }
            for o in outages
        ])

        # 5. Determine cached path (normalized branch name + trip_id)
        branch_safe = "Categorised" if "Categorised" in branch else "Uncategorised"
        cache_filename = f"{branch_safe}_{trip_id}.npz"
        cache_full_path = self.cache_dir / cache_filename

        if save_cache:
            synced_trip.save_npz(cache_full_path)

        # Make paths repository-relative for portability (fallback to as_posix across drives)
        try:
            s_rel = s_path.relative_to(self.project_root).as_posix()
        except ValueError:
            s_rel = s_path.as_posix()

        try:
            v_rel = v_path.relative_to(self.project_root).as_posix()
        except ValueError:
            v_rel = v_path.as_posix()

        try:
            cache_rel = cache_full_path.relative_to(self.project_root).as_posix()
        except ValueError:
            cache_rel = cache_full_path.as_posix()

        # Measured sampling rate: median delta
        dt_ns = np.diff(synced_trip.timestamps_ns)
        med_dt_s = float(np.median(dt_ns)) / 1e9 if len(dt_ns) > 0 else 0.1
        rate_hz = round(1.0 / med_dt_s, 2) if med_dt_s > 0 else 10.0

        # Synchronization diagnostics
        diag = synced_trip.diagnostics
        sync_mode = diag.sync_mode if diag else "relative_elapsed"
        sync_status = diag.status if diag else "PASSED"
        sync_clock_origin_offset_s = round(diag.clock_origin_offset_s, 3) if diag else 0.0
        sync_clock_drift_s = round(diag.clock_drift_s, 3) if diag else 0.0
        sync_overlap_duration_s = round(diag.overlap_duration_s, 2) if diag else round(synced_trip.overlap_duration_s, 2)
        sync_valid_count = diag.valid_interpolated_count if diag else synced_trip.row_count
        sync_oob_count = diag.out_of_bounds_count if diag else 0
        sync_gap_count = diag.gap_violation_count if diag else 0
        sync_coverage = round(diag.valid_coverage_ratio, 4) if diag else 1.0

        return ManifestRow(
            trip_id=trip_id,
            branch=branch,
            driver_id=driver_id,
            s_filename=s_path.name,
            s_relative_path=s_rel,
            v_filename=v_path.name,
            v_relative_path=v_rel,
            raw_s_rows=s_trip.row_count,
            raw_v_rows=v_trip.row_count,
            synced_rows=synced_trip.row_count,
            validated_rows=quality_rep.validated_rows,
            omitted_rows=quality_rep.omitted_rows,
            flag_ok_count=quality_rep.flag_ok_count,
            measured_rate_hz=rate_hz,
            duration_s=round(synced_trip.overlap_duration_s, 2),
            has_stationary_segment=has_stationary,
            stationary_segment_count=len(stationary_segs),
            total_stationary_duration_s=round(total_stationary_dur, 2),
            has_real_outages_for_trip=has_real_for_trip,
            has_real_outages_in_dataset=has_real_in_dataset,
            has_real_outages=has_real_for_trip,
            outage_windows_json=outages_json,
            extreme_accel_count=quality_rep.extreme_accel_count,
            extreme_gyro_count=quality_rep.extreme_gyro_count,
            extreme_motion_count=quality_rep.extreme_motion_count,
            dropout_count=quality_rep.dropout_count,
            nan_or_nonfinite_count=quality_rep.nan_or_nonfinite_count,
            non_monotonic_count=quality_rep.non_monotonic_count,
            duplicate_count=quality_rep.duplicate_count,
            invalid_timestamp_count=quality_rep.invalid_timestamp_count,
            cached_npz_path=cache_rel,
            sync_mode=sync_mode,
            sync_status=sync_status,
            sync_clock_origin_offset_s=sync_clock_origin_offset_s,
            sync_clock_drift_s=sync_clock_drift_s,
            sync_overlap_duration_s=sync_overlap_duration_s,
            sync_valid_interpolated_count=sync_valid_count,
            sync_out_of_bounds_count=sync_oob_count,
            sync_gap_violation_count=sync_gap_count,
            sync_valid_coverage_ratio=sync_coverage,
        )

    def build_full_manifest(
        self,
        raw_data_dir: Optional[Path | str] = None,
        save_cache: bool = True,
    ) -> pd.DataFrame:
        """Walk full IO-VNBD dataset, process all 144 pairs, and save manifest CSV."""
        data_dir = Path(raw_data_dir or (self.project_root / "data" / "raw" / "io_vnbd")).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

        # Discover all matched S/V pairs reliably
        pairs = discover_dataset_pairs(data_dir)
        self.trip_driver_map = derive_driver_map_from_categorised(data_dir)

        rows: List[ManifestRow] = []

        for s_path, v_path, trip_id, branch in pairs:
            row = self.process_pair(
                s_path=s_path,
                v_path=v_path,
                trip_id=trip_id,
                branch=branch,
                save_cache=save_cache,
            )
            rows.append(row)

        df = pd.DataFrame([r.to_dict() for r in rows])
        df.to_csv(self.manifest_path, index=False)
        return df


def build_manifest(project_root: Optional[Path | str] = None) -> pd.DataFrame:
    """Convenience function to build the full dataset manifest."""
    builder = DatasetManifestBuilder(project_root=project_root)
    return builder.build_full_manifest()
