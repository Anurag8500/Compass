#!/usr/bin/env python3
"""inspect_iovnbd.py - Read-Only Inspection Tool for IO-VNBD Dataset (Phase 0).

Authoritative inspection script for COMPASS (SIH 26168 - ISRO).
Recursively audits raw dataset files located in data/raw/io_vnbd/ without modifying,
deleting, or altering any raw sensor data.

Key Capabilities:
  - File inventory & byte sizes
  - Dataset classification (S- files, V- files, GPS outage index, subdirectories)
  - CSV schemas (exact column headers, dtypes, row counts)
  - Real measured sampling rates (measured_rate_hz = 1 / median(diff(timestamp)))
  - Timestamp integrity (duplicates, non-monotonicity, bounds)
  - Vector magnitude extreme motion (|f| > 4g = 39.24 m/s^2, |omega| > 10 rad/s)
  - Dual-signal candidate stationary detection (both accel and gyro low variance)
  - Controlled sensor column matching (eliminates false positives)
  - Folder-aware S- / V- file pairing verification
  - Chunked memory handling for large CSV files

Outputs:
  - docs/iovnbd_inspection_report.md
  - docs/iovnbd_inspection_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Physical Constants & Configuration Thresholds
# ---------------------------------------------------------------------------
GRAVITY_CONSTANT: float = 9.81  # m/s^2
EXTREME_ACCEL_THRESHOLD_MS2: float = 4.0 * GRAVITY_CONSTANT  # 39.24 m/s^2 (|f| > 4g)
EXTREME_GYRO_THRESHOLD_RADS: float = 10.0  # rad/s (|omega| > 10 rad/s)

STATIONARY_WINDOW_SAMPLES: int = 50  # ~5.0 seconds at canonical 10 Hz
STATIONARY_ACCEL_VAR_THRESHOLD: float = 0.05  # m^2/s^4
STATIONARY_GYRO_VAR_THRESHOLD: float = 0.005  # rad^2/s^2
CSV_CHUNK_SIZE: int = 5000  # rows per chunk for streaming audits


# ---------------------------------------------------------------------------
# Data Models for Inspection Results
# ---------------------------------------------------------------------------

@dataclass
class TimestampStats:
    column_name: str
    unit: str  # "seconds" or "milliseconds"
    min_timestamp: float
    max_timestamp: float
    duration_s: float
    median_delta_s: float
    mean_delta_s: float
    std_delta_s: float
    measured_rate_hz: float
    duplicate_count: int
    non_monotonic_count: int


@dataclass
class QualityStats:
    total_rows: int
    nan_count: int
    inf_count: int
    extreme_accel_count: int  # Vector magnitude |f| > 39.24 m/s^2
    extreme_gyro_count: int   # Vector magnitude |omega| > 10.0 rad/s
    notes: List[str] = field(default_factory=list)


@dataclass
class CandidateStationarySegment:
    start_index: int
    end_index: int
    start_time: float
    end_time: float
    duration_s: float
    accel_variance: float
    gyro_variance: float


@dataclass
class FileReport:
    relative_path: str
    filename: str
    extension: str
    size_bytes: int
    category: str  # "S- file (Smartphone)", "V- file (Vehicle CAN/VBOX)", etc.
    subdirectory_type: str  # "Synchronised (Categorised)", "Synchronised (Uncategorised)", "Unsynchronised", "Other"
    branch_name: str  # Isolated branch identifier for folder-aware pairing
    row_count: Optional[int] = None
    column_names: List[str] = field(default_factory=list)
    dtypes: Dict[str, str] = field(default_factory=dict)
    timestamp_stats: Optional[TimestampStats] = None
    quality_stats: Optional[QualityStats] = None
    stationary_segments: List[CandidateStationarySegment] = field(default_factory=list)
    error_message: Optional[str] = None


@dataclass
class SVPairReport:
    branch: str
    base_id: str
    s_file: str
    v_file: str
    s_rel_path: str
    v_rel_path: str
    s_rows: int
    v_rows: int
    s_rate_hz: float
    v_rate_hz: float
    notes: str


# ---------------------------------------------------------------------------
# Classification & Identification Helpers
# ---------------------------------------------------------------------------

def classify_dataset_file(file_path: Path | str) -> Tuple[str, str, str]:
    """Classifies a dataset file based on filename, parent directories, and dataset branch.

    Returns:
        (category, subdirectory_type, branch_name)
    """
    file_path = Path(file_path)
    name = file_path.name.lower()
    parent_parts = [p.lower() for p in file_path.parts]
    parent_str = "/".join(parent_parts)

    # Subdirectory and Branch classification (check uncategorised before categorised!)
    if "uncategorised" in parent_str or "uncategorized" in parent_str:
        subdir_type = "Synchronised (Uncategorised)"
        branch_name = "Uncategorised IOVNB Dataset"
    elif "categorised" in parent_str or "categorized" in parent_str:
        subdir_type = "Synchronised (Categorised)"
        branch_name = "Categorised IOVNB Dataset"
    elif any("unsynchronis" in p or "unsynchroniz" in p for p in parent_parts):
        subdir_type = "Unsynchronised V and S"
        branch_name = "Unsynchronised V and S Dataset"
    elif any("synchronis" in p or "synchroniz" in p for p in parent_parts):
        subdir_type = "Synchronised V and S"
        branch_name = "Synchronised V and S"
    else:
        subdir_type = "Root / Other"
        branch_name = "Other"

    # Category
    ext = file_path.suffix.lower()
    if ext in [".jpg", ".jpeg", ".png"]:
        category = "Trip Photo"
    elif ext == ".zip":
        category = "Dataset Archive"
    elif "outage" in name:
        category = "GPS outage/index file"
    elif (file_path.name.startswith("S-") or file_path.name.startswith("s-")) and ext == ".csv":
        category = "S- file (Smartphone)"
    elif (file_path.name.startswith("V-") or file_path.name.startswith("v-")) and ext == ".csv":
        category = "V- file (Vehicle CAN/VBOX)"
    elif ext == ".csv":
        category = "Supporting CSV"
    else:
        category = "Other"

    return category, subdir_type, branch_name


def find_timestamp_column(columns: List[str]) -> Optional[str]:
    """Identifies the most probable timestamp column using priority matching."""
    cleaned = {c: re.sub(r"[^\w\s]", "", c.strip().lower()) for c in columns}

    # Priority 1: Exact well-known dataset columns
    for orig, norm in cleaned.items():
        if norm in ["time since start ms", "time since start of day seconds"]:
            return orig

    # Priority 2: Generic strong names
    for orig, norm in cleaned.items():
        if norm in ["time", "timestamp", "t", "time s", "time sec", "seconds", "epoch", "time ms"]:
            return orig

    # Priority 3: Contains time/timestamp (avoiding 'air temperature', 'coolant temperature')
    for orig, norm in cleaned.items():
        if ("time" in norm or "timestamp" in norm) and "temp" not in norm:
            return orig

    return None


def find_sensor_columns(columns: List[str]) -> Dict[str, List[str]]:
    """Identifies 3-axis accelerometer and gyroscope columns using controlled regex patterns.

    Avoids false positives from:
      - Latitude / Longitude (e.g. 'Latitude (degrees)')
      - Gravity vector (e.g. 'GRAVITY X')
      - Vehicle CAN longitudinal/lateral acceleration ('Indicated Longitudinal Acceleration')
      - Orientation angles ('ORIENTATION (Yaw)')
      - Magnetometer ('MAGNETIC FIELD X')
      - Temperature ('Air Temperature')
      - Pedal position ('Accelerator Pedal Position')

    Returns:
        {"accel": [col_x, col_y, col_z], "gyro": [col_x, col_y, col_z]}
        (Empty list if 3 distinct axes are not identified)
    """
    cleaned = {c: c.strip() for c in columns}

    accel_candidates: Dict[str, str] = {}
    gyro_candidates: Dict[str, str] = {}

    for orig, col in cleaned.items():
        col_lower = col.lower()

        # Reject obvious non-sensor columns
        if any(ign in col_lower for ign in ["gravity", "magnetic", "pedal", "temperature", "temp", "indicated", "orientation"]):
            continue

        # Accelerometer Matching:
        # Must explicitly match accelerometer X, Y, or Z (e.g. 'ACCELEROMETER X (m/s²)', 'accel_x', 'ax')
        if "accel" in col_lower or re.match(r"^a[xyz]$", col_lower) or re.match(r"^accel[_\s]?[xyz]$", col_lower):
            if re.search(r"\bx\b|[_\s]x\b|^ax$", col_lower):
                accel_candidates["x"] = orig
            elif re.search(r"\by\b|[_\s]y\b|^ay$", col_lower):
                accel_candidates["y"] = orig
            elif re.search(r"\bz\b|[_\s]z\b|^az$", col_lower):
                accel_candidates["z"] = orig

        # Gyroscope Matching:
        # IO-VNBD uses 'GYROSCOPE Yaw (rad/s)', 'GYROSCOPE Pitch (rad/s)', 'GYROSCOPE Roll (rad/s)'
        # Generic forms use 'GYRO_X', 'gyro_y', 'gx', 'gy', 'gz'
        if "gyro" in col_lower or re.match(r"^g[xyz]$", col_lower) or re.match(r"^gyro[_\s]?[xyz]$", col_lower):
            # X axis / Roll
            if re.search(r"\broll\b|\bx\b|[_\s]x\b|^gx$", col_lower):
                gyro_candidates["x"] = orig
            # Y axis / Pitch
            elif re.search(r"\bpitch\b|\by\b|[_\s]y\b|^gy$", col_lower):
                gyro_candidates["y"] = orig
            # Z axis / Yaw
            elif re.search(r"\byaw\b|\bz\b|[_\s]z\b|^gz$", col_lower):
                gyro_candidates["z"] = orig

    accel_cols: List[str] = []
    if "x" in accel_candidates and "y" in accel_candidates and "z" in accel_candidates:
        accel_cols = [accel_candidates["x"], accel_candidates["y"], accel_candidates["z"]]

    gyro_cols: List[str] = []
    if "x" in gyro_candidates and "y" in gyro_candidates and "z" in gyro_candidates:
        gyro_cols = [gyro_candidates["x"], gyro_candidates["y"], gyro_candidates["z"]]

    return {"accel": accel_cols, "gyro": gyro_cols}


# ---------------------------------------------------------------------------
# Inspection Routines (Read-Only & Chunked)
# ---------------------------------------------------------------------------

def inspect_csv_file(file_path: Path | str) -> FileReport:
    """Safely and non-destructively inspects a CSV file using chunked streaming."""
    file_path = Path(file_path)
    category, subdir_type, branch_name = classify_dataset_file(file_path)
    size_bytes = file_path.stat().st_size
    rel_path = str(file_path.as_posix())

    report = FileReport(
        relative_path=rel_path,
        filename=file_path.name,
        extension=file_path.suffix,
        size_bytes=size_bytes,
        category=category,
        subdirectory_type=subdir_type,
        branch_name=branch_name,
    )

    if size_bytes == 0:
        report.row_count = 0
        report.error_message = "File is empty (0 bytes)"
        return report

    try:
        # Read header only (using latin-1 to handle any degree / micro / squared symbols safely)
        df_head = pd.read_csv(file_path, nrows=5, encoding="latin-1", skipinitialspace=True)
        report.column_names = [c.strip() for c in df_head.columns]

        t_col = find_timestamp_column(report.column_names)
        sensor_cols = find_sensor_columns(report.column_names)
        has_3axis_accel = len(sensor_cols["accel"]) == 3
        has_3axis_gyro = len(sensor_cols["gyro"]) == 3

        # Stream file in chunks for memory-safe row counting, NaN/Inf, and extreme-motion calculation
        total_rows = 0
        nan_count = 0
        inf_count = 0
        extreme_accel_count = 0
        extreme_gyro_count = 0

        report.dtypes = {col: str(dtype) for col, dtype in df_head.dtypes.items()}

        chunk_iter = pd.read_csv(
            file_path,
            chunksize=CSV_CHUNK_SIZE,
            encoding="latin-1",
            skipinitialspace=True,
        )

        for chunk in chunk_iter:
            chunk.columns = [c.strip() for c in chunk.columns]
            chunk_len = len(chunk)
            total_rows += chunk_len

            # NaN check
            nan_count += int(chunk.isna().sum().sum())

            # Inf check
            numeric_chunk = chunk.select_dtypes(include=[np.number])
            if not numeric_chunk.empty:
                inf_count += int(np.isinf(numeric_chunk.to_numpy()).sum())

            # 3.1 Extreme-Motion: Vector Magnitude |f| = sqrt(ax^2 + ay^2 + az^2) > 39.24 m/s^2
            if has_3axis_accel:
                ax_col, ay_col, az_col = sensor_cols["accel"]
                if (
                    pd.api.types.is_numeric_dtype(chunk[ax_col])
                    and pd.api.types.is_numeric_dtype(chunk[ay_col])
                    and pd.api.types.is_numeric_dtype(chunk[az_col])
                ):
                    ax = chunk[ax_col].to_numpy(dtype=float)
                    ay = chunk[ay_col].to_numpy(dtype=float)
                    az = chunk[az_col].to_numpy(dtype=float)
                    accel_mag = np.sqrt(ax * ax + ay * ay + az * az)
                    extreme_accel_count += int(np.sum(accel_mag > EXTREME_ACCEL_THRESHOLD_MS2))

            # 3.1 Extreme-Motion: Vector Magnitude |omega| = sqrt(gx^2 + gy^2 + gz^2) > 10.0 rad/s
            if has_3axis_gyro:
                gx_col, gy_col, gz_col = sensor_cols["gyro"]
                if (
                    pd.api.types.is_numeric_dtype(chunk[gx_col])
                    and pd.api.types.is_numeric_dtype(chunk[gy_col])
                    and pd.api.types.is_numeric_dtype(chunk[gz_col])
                ):
                    gx = chunk[gx_col].to_numpy(dtype=float)
                    gy = chunk[gy_col].to_numpy(dtype=float)
                    gz = chunk[gz_col].to_numpy(dtype=float)
                    gyro_mag = np.sqrt(gx * gx + gy * gy + gz * gz)
                    extreme_gyro_count += int(np.sum(gyro_mag > EXTREME_GYRO_THRESHOLD_RADS))

        report.row_count = total_rows
        report.quality_stats = QualityStats(
            total_rows=total_rows,
            nan_count=nan_count,
            inf_count=inf_count,
            extreme_accel_count=extreme_accel_count,
            extreme_gyro_count=extreme_gyro_count,
        )

        if total_rows == 0:
            report.error_message = "CSV contains only headers or no data rows"
            return report

        # 1. Timestamp Analysis (loading ONLY timestamp column to minimize RAM)
        if t_col:
            report.timestamp_stats = analyze_timestamps(file_path, t_col)

        # 2. Candidate Stationary Detection (loading ONLY sensor columns)
        if has_3axis_accel and has_3axis_gyro and total_rows >= STATIONARY_WINDOW_SAMPLES:
            report.stationary_segments = discover_candidate_stationary(
                file_path=file_path,
                accel_cols=sensor_cols["accel"],
                gyro_cols=sensor_cols["gyro"],
                time_col=t_col,
                time_stats=report.timestamp_stats,
            )

    except Exception as exc:
        report.error_message = f"Error reading CSV: {str(exc)}"

    return report


def analyze_timestamps(file_path: Path, t_col: str) -> Optional[TimestampStats]:
    """Inspects timestamp column with low memory overhead."""
    try:
        t_col_clean = t_col.strip().lower()
        df_t = pd.read_csv(
            file_path,
            usecols=lambda c: c.strip().lower() == t_col_clean,
            encoding="latin-1",
            skipinitialspace=True,
        )
        if df_t.empty:
            return None

        df_t.columns = [c.strip() for c in df_t.columns]
        matched_col = df_t.columns[0]

        if not pd.api.types.is_numeric_dtype(df_t[matched_col]):
            return None

        t_series = df_t[matched_col].dropna().to_numpy(dtype=float)
        if len(t_series) < 2:
            return None

        diffs = np.diff(t_series)
        raw_med_dt = float(np.median(diffs))
        raw_mean_dt = float(np.mean(diffs))
        raw_std_dt = float(np.std(diffs))

        # Determine time unit (seconds vs milliseconds)
        col_lower = matched_col.lower()
        if "(ms)" in col_lower or "_ms" in col_lower or "ms" in col_lower:
            time_unit = "milliseconds"
            scale_to_s = 0.001
        elif "(seconds)" in col_lower or "(s)" in col_lower or "_s" in col_lower:
            time_unit = "seconds"
            scale_to_s = 1.0
        elif raw_med_dt >= 10.0:  # e.g. ~100 ms delta
            time_unit = "milliseconds"
            scale_to_s = 0.001
        else:
            time_unit = "seconds"
            scale_to_s = 1.0

        med_dt_s = raw_med_dt * scale_to_s
        mean_dt_s = raw_mean_dt * scale_to_s
        std_dt_s = raw_std_dt * scale_to_s

        measured_rate_hz = 1.0 / med_dt_s if med_dt_s > 0 else 0.0
        dup_count = int(np.sum(diffs == 0))
        non_mono_count = int(np.sum(diffs < 0))

        min_t_s = float(t_series[0]) * scale_to_s
        max_t_s = float(t_series[-1]) * scale_to_s

        return TimestampStats(
            column_name=matched_col,
            unit=time_unit,
            min_timestamp=float(t_series[0]),
            max_timestamp=float(t_series[-1]),
            duration_s=max_t_s - min_t_s,
            median_delta_s=med_dt_s,
            mean_delta_s=mean_dt_s,
            std_delta_s=std_dt_s,
            measured_rate_hz=measured_rate_hz,
            duplicate_count=dup_count,
            non_monotonic_count=non_mono_count,
        )
    except Exception:
        return None


def discover_candidate_stationary(
    file_path: Path,
    accel_cols: List[str],
    gyro_cols: List[str],
    time_col: Optional[str],
    time_stats: Optional[TimestampStats],
    window_samples: int = STATIONARY_WINDOW_SAMPLES,
    accel_var_threshold: float = STATIONARY_ACCEL_VAR_THRESHOLD,
    gyro_var_threshold: float = STATIONARY_GYRO_VAR_THRESHOLD,
) -> List[CandidateStationarySegment]:
    """3.2 Stationary Detection: Discovery heuristic checking BOTH acceleration AND gyroscope low variance.

    A segment is considered candidate stationary ONLY when:
      rolling_var(accel_magnitude) < accel_var_threshold (0.05 m^2/s^4)
      AND
      rolling_var(gyro_magnitude) < gyro_var_threshold (0.005 rad^2/s^2)
    """
    segments: List[CandidateStationarySegment] = []
    cols_to_load = list(accel_cols) + list(gyro_cols)
    if time_col and time_col not in cols_to_load:
        cols_to_load.append(time_col)

    cols_clean_set = {c.strip().lower() for c in cols_to_load}

    try:
        df = pd.read_csv(
            file_path,
            usecols=lambda c: c.strip().lower() in cols_clean_set,
            encoding="latin-1",
            skipinitialspace=True,
        )
        df.columns = [c.strip() for c in df.columns]
        col_lookup = {c.lower(): c for c in df.columns}

        # Resolve exact stripped names
        ax_col = col_lookup[accel_cols[0].strip().lower()]
        ay_col = col_lookup[accel_cols[1].strip().lower()]
        az_col = col_lookup[accel_cols[2].strip().lower()]

        gx_col = col_lookup[gyro_cols[0].strip().lower()]
        gy_col = col_lookup[gyro_cols[1].strip().lower()]
        gz = col_lookup[gyro_cols[2].strip().lower()]

        t_col_resolved = col_lookup.get(time_col.strip().lower()) if time_col else None

        if len(df) < window_samples:
            return segments

        # Vector acceleration magnitude
        ax = df[ax_col].to_numpy(dtype=float)
        ay = df[ay_col].to_numpy(dtype=float)
        az = df[az_col].to_numpy(dtype=float)
        accel_mag = pd.Series(np.sqrt(ax * ax + ay * ay + az * az))

        # Vector gyroscope magnitude
        gx = df[gx_col].to_numpy(dtype=float)
        gy = df[gy_col].to_numpy(dtype=float)
        gz_arr = df[gz].to_numpy(dtype=float)
        gyro_mag = pd.Series(np.sqrt(gx * gx + gy * gy + gz_arr * gz_arr))

        rolling_var_a = accel_mag.rolling(window=window_samples).var()
        rolling_var_g = gyro_mag.rolling(window=window_samples).var()

        # Both signals must be calm
        is_stationary = (rolling_var_a < accel_var_threshold) & (rolling_var_g < gyro_var_threshold)

        scale_to_s = 0.001 if (time_stats and time_stats.unit == "milliseconds") else 1.0

        in_segment = False
        start_idx = 0

        for i, val in enumerate(is_stationary):
            if val and not in_segment:
                in_segment = True
                start_idx = i - window_samples + 1
            elif not val and in_segment:
                in_segment = False
                end_idx = i
                seg_len = end_idx - start_idx
                if seg_len >= window_samples:
                    t_start = (float(df[t_col_resolved].iloc[start_idx]) * scale_to_s) if t_col_resolved else float(start_idx)
                    t_end = (float(df[t_col_resolved].iloc[end_idx - 1]) * scale_to_s) if t_col_resolved else float(end_idx - 1)
                    var_a = float(rolling_var_a.iloc[start_idx:end_idx].mean())
                    var_g = float(rolling_var_g.iloc[start_idx:end_idx].mean())

                    segments.append(
                        CandidateStationarySegment(
                            start_index=start_idx,
                            end_index=end_idx,
                            start_time=t_start,
                            end_time=t_end,
                            duration_s=t_end - t_start if t_col_resolved else float(seg_len),
                            accel_variance=var_a,
                            gyro_variance=var_g,
                        )
                    )
                    if len(segments) >= 5:
                        break

        # Flush final segment if file ended while stationary
        if in_segment and len(segments) < 5:
            end_idx = len(df)
            seg_len = end_idx - start_idx
            if seg_len >= window_samples:
                t_start = (float(df[t_col_resolved].iloc[start_idx]) * scale_to_s) if t_col_resolved else float(start_idx)
                t_end = (float(df[t_col_resolved].iloc[end_idx - 1]) * scale_to_s) if t_col_resolved else float(end_idx - 1)
                var_a = float(rolling_var_a.iloc[start_idx:end_idx].mean())
                var_g = float(rolling_var_g.iloc[start_idx:end_idx].mean())

                segments.append(
                    CandidateStationarySegment(
                        start_index=start_idx,
                        end_index=end_idx,
                        start_time=t_start,
                        end_time=t_end,
                        duration_s=t_end - t_start if t_col_resolved else float(seg_len),
                        accel_variance=var_a,
                        gyro_variance=var_g,
                    )
                )

    except Exception:
        pass

    return segments


# ---------------------------------------------------------------------------
# Folder-Aware S/V Pairing Analysis
# ---------------------------------------------------------------------------

def extract_base_trip_id(filename: str) -> str:
    """Extracts trip identifier from S- or V- filename (e.g. S-S1.csv -> S1, S-Vta12.csv -> vta12)."""
    clean_name = re.sub(r"^[SsVv]-", "", filename)
    clean_name = re.sub(r"\.csv$", "", clean_name, flags=re.IGNORECASE)
    return clean_name.strip()


def analyze_sv_pairing(file_reports: List[FileReport]) -> Tuple[List[SVPairReport], List[str], List[str]]:
    """3.4 Folder-Aware S/V Pairing.

    Pairs S- and V- files strictly within the same dataset branch:
      - Categorised IOVNB Dataset
      - Uncategorised IOVNB Dataset
      - Unsynchronised V and S Dataset
    Never crosses branches. Normalizes casing and preserves relative paths.
    """
    branches: Dict[str, List[FileReport]] = {}
    for rep in file_reports:
        branches.setdefault(rep.branch_name, []).append(rep)

    all_pairs: List[SVPairReport] = []
    all_unmatched_s: List[str] = []
    all_unmatched_v: List[str] = []

    for branch_name, reports in branches.items():
        s_files: Dict[str, FileReport] = {}
        v_files: Dict[str, FileReport] = {}

        for rep in reports:
            if rep.category.startswith("S- file"):
                trip_id = extract_base_trip_id(rep.filename).lower()
                norm_path = rep.relative_path.replace("\\", "/")
                parent_dir = norm_path.rsplit("/", 1)[0].lower()
                key = f"{parent_dir}/{trip_id}" if "Categorised" in branch_name else trip_id
                s_files[key] = rep
            elif rep.category.startswith("V- file"):
                trip_id = extract_base_trip_id(rep.filename).lower()
                norm_path = rep.relative_path.replace("\\", "/")
                parent_dir = norm_path.rsplit("/", 1)[0].lower()
                key = f"{parent_dir}/{trip_id}" if "Categorised" in branch_name else trip_id
                v_files[key] = rep

        for key, s_rep in s_files.items():
            if key in v_files:
                v_rep = v_files[key]
                trip_id = extract_base_trip_id(s_rep.filename)
                all_pairs.append(
                    SVPairReport(
                        branch=branch_name,
                        base_id=trip_id,
                        s_file=s_rep.filename,
                        v_file=v_rep.filename,
                        s_rel_path=s_rep.relative_path,
                        v_rel_path=v_rep.relative_path,
                        s_rows=s_rep.row_count or 0,
                        v_rows=v_rep.row_count or 0,
                        s_rate_hz=s_rep.timestamp_stats.measured_rate_hz if s_rep.timestamp_stats else 0.0,
                        v_rate_hz=v_rep.timestamp_stats.measured_rate_hz if v_rep.timestamp_stats else 0.0,
                        notes=f"Branch: {branch_name}",
                    )
                )
            else:
                all_unmatched_s.append(f"[{branch_name}] {s_rep.relative_path}")

        for key, v_rep in v_files.items():
            if key not in s_files:
                all_unmatched_v.append(f"[{branch_name}] {v_rep.relative_path}")

    return all_pairs, all_unmatched_s, all_unmatched_v


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

def generate_markdown_report(
    data_dir: Path,
    file_reports: List[FileReport],
    pairs: List[SVPairReport],
    unmatched_s: List[str],
    unmatched_v: List[str],
) -> str:
    """Generates the comprehensive Markdown inspection report."""
    lines: List[str] = []
    lines.append("# IO-VNBD Dataset Inspection Report (Phase 0)")
    lines.append(f"**Target Directory**: `{data_dir.as_posix()}`  ")
    lines.append(f"**Total Files Discovered**: {len(file_reports)}  ")
    lines.append("")

    if not file_reports:
        lines.append("## Status: No Dataset Files Present")
        lines.append("")
        lines.append("> [!WARNING]")
        lines.append("> The directory `data/raw/io_vnbd/` is currently empty or does not exist.")
        lines.append("> Raw dataset discovery cannot be completed until the IO-VNBD dataset is placed in this directory.")
        return "\n".join(lines)

    # 1. Summary by Category & Branch
    cat_counts: Dict[str, int] = {}
    branch_counts: Dict[str, int] = {}
    for r in file_reports:
        cat_counts[r.category] = cat_counts.get(r.category, 0) + 1
        branch_counts[r.branch_name] = branch_counts.get(r.branch_name, 0) + 1

    lines.append("## 1. Dataset Overview & Inventory")
    lines.append("### Breakdown by Branch")
    lines.append("| Dataset Branch | File Count |")
    lines.append("|---|---|")
    for b_name, count in sorted(branch_counts.items()):
        lines.append(f"| {b_name} | {count} |")
    lines.append("")

    lines.append("### Breakdown by File Category")
    lines.append("| Category | File Count |")
    lines.append("|---|---|")
    for cat, count in sorted(cat_counts.items()):
        lines.append(f"| {cat} | {count} |")
    lines.append("")

    # 2. S/V Pairing Analysis
    lines.append("## 2. Folder-Aware S / V Pairing Analysis")
    lines.append(f"- **Total Matched Pairs**: {len(pairs)}")
    lines.append(f"- **Unmatched S- Files**: {len(unmatched_s)}")
    lines.append(f"- **Unmatched V- Files**: {len(unmatched_v)}")
    lines.append("")

    pairs_by_branch: Dict[str, int] = {}
    for p in pairs:
        pairs_by_branch[p.branch] = pairs_by_branch.get(p.branch, 0) + 1

    lines.append("### Pairs per Branch")
    lines.append("| Branch | Matched S/V Pairs |")
    lines.append("|---|---|")
    for b_name, count in sorted(pairs_by_branch.items()):
        lines.append(f"| {b_name} | {count} pairs |")
    lines.append("")

    lines.append("### Matched Pairs Sample (First 15)")
    lines.append("| Branch | Trip ID | S- File (Rows, Rate) | V- File (Rows, Rate) | Row Δ |")
    lines.append("|---|---|---|---|---|")
    for p in pairs[:15]:
        row_diff = p.s_rows - p.v_rows
        lines.append(
            f"| {p.branch} | `{p.base_id}` | `{p.s_file}` ({p.s_rows:,}, {p.s_rate_hz:.2f}Hz) | "
            f"`{p.v_file}` ({p.v_rows:,}, {p.v_rate_hz:.2f}Hz) | {row_diff:+d} |"
        )
    if len(pairs) > 15:
        lines.append(f"| ... | ... | *(+{len(pairs) - 15} more pairs)* | | |")
    lines.append("")

    # 3. CSV Structure & Sampling Rates
    csv_reports = [r for r in file_reports if r.extension.lower() == ".csv"]
    lines.append("## 3. Sampling Rates & Timestamp Verification")
    lines.append("| Filename | Branch | Rows | Measured Rate | Median Δt | Duplicates | Non-Monotonic |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in csv_reports[:20]:
        if r.timestamp_stats:
            ts = r.timestamp_stats
            lines.append(
                f"| `{r.filename}` | {r.branch_name} | {r.row_count:,} | **{ts.measured_rate_hz:.2f} Hz** | "
                f"{ts.median_delta_s:.4f}s ({ts.unit}) | {ts.duplicate_count} | {ts.non_monotonic_count} |"
            )
    if len(csv_reports) > 20:
        lines.append(f"| ... | *(+{len(csv_reports) - 20} more CSVs audited)* | | | | | |")
    lines.append("")

    # 4. Data Quality Findings
    lines.append("## 4. Data Quality & Vector Extreme-Motion Findings")
    lines.append("> Extreme motion rule: Vector magnitude $|f| = \\sqrt{ax^2+ay^2+az^2} > 39.24\\text{ m/s}^2$ ($>4g$) ")
    lines.append("> and $|\\omega| = \\sqrt{gx^2+gy^2+gz^2} > 10.0\\text{ rad/s}$. Real vehicle dynamic events are preserved.")
    lines.append("")
    lines.append("| Filename | Rows | NaN Count | Inf Count | Extreme Accel (|f|>4g) | Extreme Gyro (|ω|>10 rad/s) |")
    lines.append("|---|---|---|---|---|---|")
    for r in csv_reports[:20]:
        if r.quality_stats:
            qs = r.quality_stats
            lines.append(
                f"| `{r.filename}` | {qs.total_rows:,} | {qs.nan_count} | {qs.inf_count} | "
                f"{qs.extreme_accel_count} | {qs.extreme_gyro_count} |"
            )
    if len(csv_reports) > 20:
        lines.append(f"| ... | *(+{len(csv_reports) - 20} more CSVs audited)* | | | | |")
    lines.append("")

    # 5. Candidate Stationary Segments
    stat_files = [r for r in file_reports if r.stationary_segments]
    lines.append("## 5. Candidate Stationary Segments (Dual Accel + Gyro Low Variance)")
    lines.append(f"- **Files with candidate stationary periods**: {len(stat_files)} / {len(csv_reports)}")
    lines.append(f"- **Heuristic criteria**: $\\text{{var}}(|f|) < {STATIONARY_ACCEL_VAR_THRESHOLD}\\text{{ m}}^2/\\text{{s}}^4$ AND $\\text{{var}}(|\\omega|) < {STATIONARY_GYRO_VAR_THRESHOLD}\\text{{ rad}}^2/\\text{{s}}^2$ over $\\ge {STATIONARY_WINDOW_SAMPLES}$ samples (~5.0s)")
    lines.append("")
    if stat_files:
        lines.append("| Filename | Segment Range | Duration | Mean Accel Var (m²/s⁴) | Mean Gyro Var (rad²/s²) |")
        lines.append("|---|---|---|---|---|")
        for sf in stat_files[:10]:
            for seg in sf.stationary_segments[:2]:
                lines.append(
                    f"| `{sf.filename}` | [{seg.start_index}:{seg.end_index}] | {seg.duration_s:.1f}s | "
                    f"{seg.accel_variance:.5f} | {seg.gyro_variance:.6f} |"
                )
        if len(stat_files) > 10:
            lines.append(f"| ... | *(+{len(stat_files) - 10} more files containing stationary rest periods)* | | | |")
        lines.append("")

    # 6. GPS Outage Information
    outage_files = [r for r in file_reports if r.category == "GPS outage/index file"]
    lines.append("## 6. GPS Outage Information")
    if outage_files:
        for of in outage_files:
            lines.append(f"- Found outage file: `{of.relative_path}` ({of.row_count} rows)")
    else:
        lines.append("No explicit GPS outage index CSV file was discovered in the current raw archive.")
        lines.append("The extracted `Synchronised V abd S datasets.zip` contains recordings and vehicle photos exclusively.")
        lines.append("Synthetic outage masking and self-collected tunnel logs will serve as the benchmark evaluation mechanism.")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def inspect_dataset(data_dir: Path) -> Tuple[List[FileReport], List[SVPairReport], List[str], List[str]]:
    """Walks the dataset directory and inspects all files."""
    file_reports: List[FileReport] = []

    if not data_dir.exists():
        return file_reports, [], [], []

    all_files: List[Path] = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            all_files.append(Path(root) / f)

    for f_path in all_files:
        if f_path.suffix.lower() == ".csv":
            report = inspect_csv_file(f_path)
            file_reports.append(report)
        else:
            category, subdir, branch = classify_dataset_file(f_path)
            file_reports.append(
                FileReport(
                    relative_path=str(f_path.as_posix()),
                    filename=f_path.name,
                    extension=f_path.suffix,
                    size_bytes=f_path.stat().st_size,
                    category=category,
                    subdirectory_type=subdir,
                    branch_name=branch,
                )
            )

    pairs, unmatched_s, unmatched_v = analyze_sv_pairing(file_reports)
    return file_reports, pairs, unmatched_s, unmatched_v


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect IO-VNBD dataset (Phase 0 read-only tool).")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/io_vnbd"),
        help="Path to raw IO-VNBD dataset directory (default: data/raw/io_vnbd)",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path("docs/iovnbd_inspection_report.md"),
        help="Path to output Markdown report (default: docs/iovnbd_inspection_report.md)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("docs/iovnbd_inspection_summary.json"),
        help="Optional path to output JSON summary",
    )
    args = parser.parse_args()

    print(f"[Phase 0] Auditing IO-VNBD dataset at: {args.data_dir.resolve()}")
    file_reports, pairs, unmatched_s, unmatched_v = inspect_dataset(args.data_dir)

    # Generate Markdown Report
    report_md = generate_markdown_report(args.data_dir, file_reports, pairs, unmatched_s, unmatched_v)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_report, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"[Phase 0] Inspection Markdown report written to: {args.output_report.resolve()}")

    # Generate JSON Summary
    summary_data = {
        "data_dir": str(args.data_dir.as_posix()),
        "total_files": len(file_reports),
        "csv_files_count": len([r for r in file_reports if r.extension.lower() == ".csv"]),
        "matched_pairs_count": len(pairs),
        "unmatched_s_count": len(unmatched_s),
        "unmatched_v_count": len(unmatched_v),
        "files": [asdict(r) for r in file_reports],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"[Phase 0] Inspection JSON summary written to: {args.output_json.resolve()}")

    if not file_reports:
        print(f"[Phase 0 Warning] No files discovered in '{args.data_dir}'.")
        return 0

    print(f"[Phase 0 Success] Audited {len(file_reports)} files ({len(pairs)} folder-aware matched S/V pairs).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
