"""S/V sensor and vehicle reference stream synchronization for COMPASS.

Aligns smartphone (S-) IMU/GPS records with vehicle CAN / Racelogic VBOX (V-)
ground-truth telemetry onto a unified timestamp base using genuine
timestamp-based interpolation.

Timing Architecture:
- Smartphone (S) stream:
  Raw device timestamps recorded in milliseconds since logger startup ('TIME SINCE START (ms)').
  Preserved unmodified in SynchronizedTrip.raw_timestamps_ns.
  In RELATIVE_ELAPSED mode, a monotonic working target grid is constructed by unwrapping
  session counter resets (diff < 0). Duplicate timestamps (diff == 0) are NOT advanced;
  they remain at their recorded time and are flagged by QualityTagger for omission.
- Vehicle (V) ground truth stream:
  VBOX timestamps recorded in seconds since UTC start of day ('Time Since Start of Day (seconds)').
  Validated, strictly deduplicated (keeping first occurrence) BEFORE timeline construction.
  Clock regressions in ground truth (diff < 0) are strictly rejected as invalid data.
- Relative Elapsed Alignment:
  Aligns relative elapsed time from stream origin (t - t[0]).
  Removes clock-origin discrepancy across independently initialized sensor clocks.
  Does NOT eliminate clock rate drift; drift is explicitly measured and evaluated against policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

from data.pipeline.parse import ParsedSTrip, ParsedVTrip, parse_s_file, parse_v_file
from data.pipeline.quality_tagger import QualityReport, QualityTagger

MAX_INTERPOLATION_GAP_NS: int = 1_000_000_000  # 1.0 second in nanoseconds


class SyncMode(str, Enum):
    """Synchronization mode defining stream time alignment coordinates."""
    RELATIVE_ELAPSED = "relative_elapsed"  # Default for IO-VNBD: aligns elapsed time since stream start
    ABSOLUTE = "absolute"                  # Common-clock alignment using absolute timestamps directly


class SyncValidationError(ValueError):
    """Raised when stream synchronization violates hard validity or configured policy constraints."""
    pass


@dataclass(frozen=True)
class SyncPolicy:
    """Configurable policy parameters for stream synchronization validation.

    Parameters:
        mode: Synchronization mode (RELATIVE_ELAPSED for IO-VNBD, ABSOLUTE for common-clock datasets).
              IO-VNBD requires RELATIVE_ELAPSED because smartphone S-files record elapsed time
              since logger start (~0-5s), whereas vehicle V-files record elapsed time since UTC
              start of day (~30000-50000s).
        max_gap_ns: Maximum allowable gap in source telemetry before interpolation is rejected (ns).
        min_overlap_duration_s: Minimum required overlapping duration between streams (seconds).
        min_coverage_ratio: Minimum ratio of successfully interpolated samples to target samples.
        max_clock_drift_s: Maximum allowable duration discrepancy |duration_s - duration_v| (seconds).
        enforce_strict_validation: If True, violations of min_overlap, min_coverage, or max_drift
                                   raise SyncValidationError. If False, they are recorded in
                                   diagnostics.status.
    """
    mode: SyncMode = SyncMode.RELATIVE_ELAPSED
    max_gap_ns: int = MAX_INTERPOLATION_GAP_NS
    min_overlap_duration_s: float = 1.0
    min_coverage_ratio: float = 0.50
    max_clock_drift_s: float = 120.0
    enforce_strict_validation: bool = False


@dataclass
class SyncDiagnostics:
    """Comprehensive diagnostics detailing stream alignment quality."""
    clock_origin_offset_s: float          # Raw origin offset: (t_v[0] - t_s[0]) in seconds
    relative_elapsed_offset_s: float      # Relative start offset: (t_source_clean[0] - t_target[0]) in seconds
    clock_drift_s: float                  # Elapsed duration discrepancy over full file: (dur_s - dur_v) in seconds
    overlap_duration_s: float             # Duration of valid shared overlap interval in seconds
    source_sample_count: int              # Total V samples available
    target_sample_count: int              # Total S samples targeted
    valid_interpolated_count: int         # Target samples successfully interpolated within bounds
    out_of_bounds_count: int              # Target samples falling outside source interval
    gap_violation_count: int              # Target samples falling into source gaps > max_gap_ns
    valid_coverage_ratio: float = 0.0     # Ratio of valid_interpolated_count to target_sample_count [0.0, 1.0]
    sync_mode: str = "relative_elapsed"   # Mode used: 'relative_elapsed' or 'absolute'
    status: str = "PASSED"                # 'PASSED' or description of validation warnings/failures
    sync_passed: bool = True              # Whether alignment satisfies configured sync policy


def _get_valid_gap_mask(
    t_target: np.ndarray,
    t_source: np.ndarray,
    max_gap_ns: int = MAX_INTERPOLATION_GAP_NS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute masks for valid range, out-of-bounds, and excessive gap violations.

    Returns:
        (is_valid, is_out_of_bounds, is_gap_violation)
    """
    n_target = len(t_target)
    if len(t_source) < 2 or n_target == 0:
        return np.zeros(n_target, dtype=bool), np.ones(n_target, dtype=bool), np.zeros(n_target, dtype=bool)

    t_min = t_source[0]
    t_max = t_source[-1]

    # 1. Out of bounds (before start or after end of source stream)
    is_out_of_bounds = (t_target < t_min) | (t_target > t_max)

    # 2. For points inside bounds, locate surrounding source intervals
    # searchsorted returns insertion index such that t_source[i-1] <= t < t_source[i]
    idx = np.searchsorted(t_source, t_target, side="right")
    # Clamp indices to valid interval [1, len(t_source) - 1]
    idx_clamped = np.clip(idx, 1, len(t_source) - 1)

    t_left = t_source[idx_clamped - 1]
    t_right = t_source[idx_clamped]
    gaps = t_right - t_left

    # A gap violation occurs if the surrounding source gap exceeds threshold
    # and the target point strictly lies between them
    is_gap_violation = (~is_out_of_bounds) & (gaps > max_gap_ns) & (t_target > t_left) & (t_target < t_right)
    is_valid = (~is_out_of_bounds) & (~is_gap_violation)

    return is_valid, is_out_of_bounds, is_gap_violation


def interpolate_continuous_1d(
    t_target: np.ndarray,
    t_source: np.ndarray,
    y_source: np.ndarray,
    max_gap_ns: int = MAX_INTERPOLATION_GAP_NS,
) -> np.ndarray:
    """Linearly interpolate 1D continuous variable without extrapolation or large gap fabrication."""
    if len(t_target) == 0:
        return np.zeros(0, dtype=float)
    if len(t_source) < 2 or not np.isfinite(y_source).any():
        return np.full(len(t_target), np.nan, dtype=float)

    is_valid, _, _ = _get_valid_gap_mask(t_target, t_source, max_gap_ns)

    # Standard linear interpolation
    y_interp = np.interp(t_target, t_source, y_source)
    # Mask out points that are invalid (out-of-bounds or across large gaps)
    y_interp[~is_valid] = np.nan
    return y_interp


def interpolate_continuous_2d(
    t_target: np.ndarray,
    t_source: np.ndarray,
    y_source_2d: np.ndarray,
    max_gap_ns: int = MAX_INTERPOLATION_GAP_NS,
) -> np.ndarray:
    """Linearly interpolate NxM continuous array (e.g. wheel speeds, CAN acceleration)."""
    n_target = len(t_target)
    if n_target == 0:
        return np.zeros((0, y_source_2d.shape[1]), dtype=float)
    m = y_source_2d.shape[1]
    if len(t_source) < 2:
        return np.full((n_target, m), np.nan, dtype=float)

    is_valid, _, _ = _get_valid_gap_mask(t_target, t_source, max_gap_ns)
    out = np.zeros((n_target, m), dtype=float)

    for col in range(m):
        col_vals = y_source_2d[:, col]
        if not np.isfinite(col_vals).any():
            out[:, col] = np.nan
        else:
            interp_col = np.interp(t_target, t_source, col_vals)
            interp_col[~is_valid] = np.nan
            out[:, col] = interp_col

    return out


def interpolate_circular_deg(
    t_target: np.ndarray,
    t_source: np.ndarray,
    angles_deg: np.ndarray,
    max_gap_ns: int = MAX_INTERPOLATION_GAP_NS,
) -> np.ndarray:
    """Interpolate angle in degrees using sin/cos decomposition to handle 0-360 wrap-around."""
    if len(t_target) == 0:
        return np.zeros(0, dtype=float)
    if len(t_source) < 2:
        return np.full(len(t_target), np.nan, dtype=float)

    is_valid, _, _ = _get_valid_gap_mask(t_target, t_source, max_gap_ns)

    rad = np.radians(angles_deg)
    sin_interp = np.interp(t_target, t_source, np.sin(rad))
    cos_interp = np.interp(t_target, t_source, np.cos(rad))

    out_rad = np.arctan2(sin_interp, cos_interp)
    out_deg = np.degrees(out_rad) % 360.0
    out_deg[~is_valid] = np.nan
    return out_deg


def interpolate_categorical(
    t_target: np.ndarray,
    t_source: np.ndarray,
    cat_source: np.ndarray,
    max_gap_ns: int = MAX_INTERPOLATION_GAP_NS,
) -> np.ndarray:
    """Nearest-neighbor interpolation for discrete categorical state (e.g. gear, handbrake)."""
    n_target = len(t_target)
    if n_target == 0:
        return np.zeros(0, dtype=np.int32)
    if len(t_source) < 2:
        return np.full(n_target, -1, dtype=np.int32)

    is_valid, _, _ = _get_valid_gap_mask(t_target, t_source, max_gap_ns)

    # Nearest neighbor selection
    idx = np.searchsorted(t_source, t_target)
    idx_clamped = np.clip(idx, 1, len(t_source) - 1)

    d_left = np.abs(t_target - t_source[idx_clamped - 1])
    d_right = np.abs(t_target - t_source[idx_clamped])

    nearest_idx = np.where(d_left <= d_right, idx_clamped - 1, idx_clamped)
    out = cat_source[nearest_idx].astype(np.int32)
    out[~is_valid] = -1
    return out


@dataclass
class SynchronizedTrip:
    """Unified synchronized representation of an IO-VNBD trip.

    Time Coordinates Semantics:
    - raw_timestamps_ns: Original, unmodified raw smartphone timestamps from the S-file.
      Preserved for full raw data traceability and non-destructive auditing.
    - timestamps_ns: Canonical working target timestamp grid (non-decreasing raw working axis
      with session counter wraps unwrapped). After duplicate and invalid samples are excluded,
      validated downstream samples (where is_validated is True) are strictly increasing,
      providing an unequivocal time axis for numerical integration and state estimation.
    """
    trip_id: str
    branch: str
    row_count: int
    timestamps_ns: np.ndarray               # 1D int64 non-decreasing target working grid (validated downstream is strictly increasing)
    raw_timestamps_ns: np.ndarray           # 1D int64 original device timestamps (immutable raw)

    # S-file motion (RAW DEVICE BODY FRAME)
    accel_raw: np.ndarray                   # Nx3 float64 [ax, ay, az] (m/s^2)
    gyro_raw: np.ndarray                    # Nx3 float64 [gx, gy, gz] (rad/s)
    quality_flags: np.ndarray               # 1D uint32 bitmask
    is_validated: np.ndarray                # 1D bool mask

    # S-file GNSS (Phone GPS)
    s_gnss_lat: np.ndarray                  # 1D float64
    s_gnss_lon: np.ndarray                  # 1D float64
    s_gnss_alt: np.ndarray                  # 1D float64
    s_gnss_speed_mps: np.ndarray            # 1D float64
    s_gnss_bearing_deg: np.ndarray          # 1D float64
    s_gnss_accuracy_m: np.ndarray           # 1D float64
    s_gnss_sat_count: np.ndarray            # 1D int32

    # V-file Ground Truth Reference (Racelogic VBOX & CAN, interpolated onto working timestamps)
    v_ref_lat: np.ndarray                   # 1D float64
    v_ref_lon: np.ndarray                   # 1D float64
    v_ref_alt_m: np.ndarray                 # 1D float64
    v_ref_speed_mps: np.ndarray             # 1D float64
    v_ref_heading_deg: np.ndarray           # 1D float64
    v_ref_yaw_rate_rads: np.ndarray         # 1D float64
    v_ref_wheel_speeds: np.ndarray          # Nx4 float64 [FL, FR, RL, RR]
    v_ref_can_accel_g: np.ndarray           # Nx2 float64 [longitudinal, lateral]
    v_ref_gear: np.ndarray                  # 1D int32
    v_ref_handbrake: np.ndarray             # 1D int32

    sync_offset_s: float
    overlap_duration_s: float
    diagnostics: Optional[SyncDiagnostics] = None

    def save_npz(self, cache_path: Path | str) -> Path:
        """Save synchronized trip arrays to a compressed NumPy .npz file."""
        target = Path(cache_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        diag_kwargs = {}
        if self.diagnostics is not None:
            diag_kwargs["diag_clock_origin_offset_s"] = self.diagnostics.clock_origin_offset_s
            diag_kwargs["diag_relative_elapsed_offset_s"] = self.diagnostics.relative_elapsed_offset_s
            diag_kwargs["diag_clock_drift_s"] = self.diagnostics.clock_drift_s
            diag_kwargs["diag_overlap_duration_s"] = self.diagnostics.overlap_duration_s
            diag_kwargs["diag_source_sample_count"] = self.diagnostics.source_sample_count
            diag_kwargs["diag_target_sample_count"] = self.diagnostics.target_sample_count
            diag_kwargs["diag_valid_interpolated_count"] = self.diagnostics.valid_interpolated_count
            diag_kwargs["diag_out_of_bounds_count"] = self.diagnostics.out_of_bounds_count
            diag_kwargs["diag_gap_violation_count"] = self.diagnostics.gap_violation_count
            diag_kwargs["diag_valid_coverage_ratio"] = self.diagnostics.valid_coverage_ratio
            diag_kwargs["diag_sync_mode"] = self.diagnostics.sync_mode
            diag_kwargs["diag_status"] = self.diagnostics.status
            diag_kwargs["diag_sync_passed"] = self.diagnostics.sync_passed

            # Legacy compatibility aliases
            diag_kwargs["sync_mode"] = self.diagnostics.sync_mode
            diag_kwargs["sync_status"] = self.diagnostics.status
            diag_kwargs["sync_passed"] = self.diagnostics.sync_passed
            diag_kwargs["valid_coverage_ratio"] = self.diagnostics.valid_coverage_ratio
            diag_kwargs["clock_drift_s"] = self.diagnostics.clock_drift_s
            diag_kwargs["valid_interpolated_count"] = self.diagnostics.valid_interpolated_count
            diag_kwargs["out_of_bounds_count"] = self.diagnostics.out_of_bounds_count
            diag_kwargs["gap_violation_count"] = self.diagnostics.gap_violation_count

        np.savez_compressed(
            target,
            trip_id=self.trip_id,
            branch=self.branch,
            timestamps_ns=self.timestamps_ns,
            raw_timestamps_ns=self.raw_timestamps_ns,
            accel_raw=self.accel_raw,
            gyro_raw=self.gyro_raw,
            quality_flags=self.quality_flags,
            is_validated=self.is_validated,
            s_gnss_lat=self.s_gnss_lat,
            s_gnss_lon=self.s_gnss_lon,
            s_gnss_alt=self.s_gnss_alt,
            s_gnss_speed_mps=self.s_gnss_speed_mps,
            s_gnss_bearing_deg=self.s_gnss_bearing_deg,
            s_gnss_accuracy_m=self.s_gnss_accuracy_m,
            s_gnss_sat_count=self.s_gnss_sat_count,
            v_ref_lat=self.v_ref_lat,
            v_ref_lon=self.v_ref_lon,
            v_ref_alt_m=self.v_ref_alt_m,
            v_ref_speed_mps=self.v_ref_speed_mps,
            v_ref_heading_deg=self.v_ref_heading_deg,
            v_ref_yaw_rate_rads=self.v_ref_yaw_rate_rads,
            v_ref_wheel_speeds=self.v_ref_wheel_speeds,
            v_ref_can_accel_g=self.v_ref_can_accel_g,
            v_ref_gear=self.v_ref_gear,
            v_ref_handbrake=self.v_ref_handbrake,
            sync_offset_s=self.sync_offset_s,
            overlap_duration_s=self.overlap_duration_s,
            **diag_kwargs,
        )
        return target

    @classmethod
    def load_npz(cls, cache_path: Path | str) -> SynchronizedTrip:
        """Load synchronized trip directly from .npz file."""
        target = Path(cache_path)
        with np.load(target) as data:
            row_count = len(data["timestamps_ns"])
            raw_ts = data["raw_timestamps_ns"] if "raw_timestamps_ns" in data else data["timestamps_ns"]
            diag = None
            if "diag_sync_mode" in data or "sync_mode" in data:
                diag = SyncDiagnostics(
                    clock_origin_offset_s=float(data["diag_clock_origin_offset_s"]) if "diag_clock_origin_offset_s" in data else float(data["sync_offset_s"]),
                    relative_elapsed_offset_s=float(data["diag_relative_elapsed_offset_s"]) if "diag_relative_elapsed_offset_s" in data else 0.0,
                    clock_drift_s=float(data["diag_clock_drift_s"]) if "diag_clock_drift_s" in data else float(data["clock_drift_s"]),
                    overlap_duration_s=float(data["diag_overlap_duration_s"]) if "diag_overlap_duration_s" in data else float(data["overlap_duration_s"]),
                    source_sample_count=int(data["diag_source_sample_count"]) if "diag_source_sample_count" in data else int(data.get("valid_interpolated_count", row_count)),
                    target_sample_count=int(data["diag_target_sample_count"]) if "diag_target_sample_count" in data else row_count,
                    valid_interpolated_count=int(data["diag_valid_interpolated_count"]) if "diag_valid_interpolated_count" in data else int(data.get("valid_interpolated_count", row_count)),
                    out_of_bounds_count=int(data["diag_out_of_bounds_count"]) if "diag_out_of_bounds_count" in data else int(data.get("out_of_bounds_count", 0)),
                    gap_violation_count=int(data["diag_gap_violation_count"]) if "diag_gap_violation_count" in data else int(data.get("gap_violation_count", 0)),
                    valid_coverage_ratio=float(data["diag_valid_coverage_ratio"]) if "diag_valid_coverage_ratio" in data else float(data.get("valid_coverage_ratio", 1.0)),
                    sync_mode=str(data["diag_sync_mode"]) if "diag_sync_mode" in data else str(data["sync_mode"]),
                    status=str(data["diag_status"]) if "diag_status" in data else str(data.get("sync_status", "PASSED")),
                    sync_passed=bool(data["diag_sync_passed"]) if "diag_sync_passed" in data else bool(data.get("sync_passed", True)),
                )

            return cls(
                trip_id=str(data["trip_id"]),
                branch=str(data["branch"]),
                row_count=row_count,
                timestamps_ns=data["timestamps_ns"],
                raw_timestamps_ns=raw_ts,
                accel_raw=data["accel_raw"],
                gyro_raw=data["gyro_raw"],
                quality_flags=data["quality_flags"],
                is_validated=data["is_validated"],
                s_gnss_lat=data["s_gnss_lat"],
                s_gnss_lon=data["s_gnss_lon"],
                s_gnss_alt=data["s_gnss_alt"],
                s_gnss_speed_mps=data["s_gnss_speed_mps"],
                s_gnss_bearing_deg=data["s_gnss_bearing_deg"],
                s_gnss_accuracy_m=data["s_gnss_accuracy_m"],
                s_gnss_sat_count=data["s_gnss_sat_count"],
                v_ref_lat=data["v_ref_lat"],
                v_ref_lon=data["v_ref_lon"],
                v_ref_alt_m=data["v_ref_alt_m"],
                v_ref_speed_mps=data["v_ref_speed_mps"],
                v_ref_heading_deg=data["v_ref_heading_deg"],
                v_ref_yaw_rate_rads=data["v_ref_yaw_rate_rads"],
                v_ref_wheel_speeds=data["v_ref_wheel_speeds"],
                v_ref_can_accel_g=data["v_ref_can_accel_g"],
                v_ref_gear=data["v_ref_gear"],
                v_ref_handbrake=data["v_ref_handbrake"],
                sync_offset_s=float(data["sync_offset_s"]),
                overlap_duration_s=float(data["overlap_duration_s"]),
                diagnostics=diag,
            )


def _unwrap_monotonic_elapsed_ns(timestamps_ns: np.ndarray, nominal_step_ns: int = 100_000_000) -> np.ndarray:
    """Compute relative elapsed time in nanoseconds, unwrapping session timer resets (diff < 0).

    Timing Semantics:
    - Only unwrap genuine backward counter resets where raw_curr < prev_raw (diff < 0).
    - Does NOT advance or fabricate offsets for duplicate timestamps (diff == 0).
      Duplicate records remain at their recorded elapsed time, and are flagged by
      QualityTagger (FLAG_DUPLICATE_TIMESTAMP) for exclusion from the validated stream.
    """
    n = len(timestamps_ns)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    elapsed = np.zeros(n, dtype=np.int64)
    offset = 0
    t0 = timestamps_ns[0]
    for i in range(1, n):
        raw_curr = timestamps_ns[i]
        prev_raw = timestamps_ns[i - 1]
        if raw_curr < prev_raw:
            # Genuine backward counter reset (e.g. logger restarted from 0)
            prev_unwrapped = prev_raw + offset
            curr_unwrapped = raw_curr + offset
            offset += (prev_unwrapped - curr_unwrapped) + nominal_step_ns
        elapsed[i] = (timestamps_ns[i] + offset) - t0
    elapsed[0] = 0
    return elapsed


def synchronize_s_v(
    s_trip: ParsedSTrip,
    v_trip: ParsedVTrip,
    trip_id: str,
    branch: str,
    tagger: Optional[QualityTagger] = None,
    relative_time: Optional[bool] = None,
    mode: Optional[SyncMode] = None,
    policy: Optional[SyncPolicy] = None,
    max_gap_ns: int = MAX_INTERPOLATION_GAP_NS,
) -> Tuple[SynchronizedTrip, QualityReport]:
    """Synchronize a paired S-trip and V-trip onto the S target timestamp grid.

    Timing Architecture:
    - S-trip raw timestamps are preserved unmodified in SynchronizedTrip.raw_timestamps_ns.
    - Synchronized working time grid is stored in SynchronizedTrip.timestamps_ns as a non-decreasing
      target working axis; validated downstream samples are strictly increasing after duplicate/invalid
      samples are excluded.
    - Source V timestamps are strictly validated and deduplicated BEFORE any timeline construction.
    - Genuine backward timestamps in source V raise SyncValidationError.
    - Relative elapsed alignment aligns relative elapsed time since stream origin (t - t[0]),
      removing clock-origin differences without eliminating clock rate drift.
    """
    if tagger is None:
        tagger = QualityTagger()

    # Determine synchronization policy
    if policy is None:
        if mode is not None:
            sync_mode = mode
        elif relative_time is not None:
            sync_mode = SyncMode.RELATIVE_ELAPSED if relative_time else SyncMode.ABSOLUTE
        else:
            sync_mode = SyncMode.RELATIVE_ELAPSED
        policy = SyncPolicy(mode=sync_mode, max_gap_ns=max_gap_ns)

    # 1. Quality tagging on S-file IMU
    quality_report = tagger.evaluate_arrays(
        s_trip.timestamps_ns,
        s_trip.accel_raw,
        s_trip.gyro_raw,
    )

    t_s_ns = s_trip.timestamps_ns.astype(np.int64)
    t_v_ns = v_trip.timestamps_ns.astype(np.int64)

    n_s = len(t_s_ns)
    n_v = len(t_v_ns)

    if n_s == 0:
        raise SyncValidationError(f"Cannot synchronize trip {trip_id}: S-file has 0 rows.")
    if n_v < 2:
        raise SyncValidationError(f"Cannot synchronize trip {trip_id}: V-file has {n_v} rows (minimum 2 required).")

    # 2. Target (S) stream timestamp validity checks
    if not np.isfinite(t_s_ns).all():
        raise SyncValidationError(f"Target stream (S) contains non-finite timestamps in trip {trip_id}")
    if (t_s_ns < 0).any():
        bad_s_count = int(np.sum(t_s_ns < 0))
        raise SyncValidationError(
            f"Target stream (S) contains {bad_s_count} negative/sentinel timestamp(s) (< 0) in trip {trip_id}."
        )

    # 3. Source (V) stream timestamp validity checks BEFORE any manipulation
    if not np.isfinite(t_v_ns).all():
        raise SyncValidationError(f"Source stream (V) contains non-finite timestamps in trip {trip_id}")
    if (t_v_ns < 0).any():
        bad_count = int(np.sum(t_v_ns < 0))
        raise SyncValidationError(
            f"Source stream (V) contains {bad_count} negative/sentinel timestamp(s) (< 0) in trip {trip_id}. "
            "Invalid source timestamps cannot participate in synchronization or interpolation."
        )

    # 4. Source (V) deduplication BEFORE any timeline construction or unwrapping
    # Deterministic duplicate handling: keep first occurrence of any duplicate timestamp.
    v_diffs = np.diff(t_v_ns)
    if len(v_diffs) > 0 and (v_diffs == 0).any():
        unique_mask = np.concatenate([[True], v_diffs != 0])
        t_v_clean = t_v_ns[unique_mask]
        v_lat_clean = v_trip.lat[unique_mask]
        v_lon_clean = v_trip.lon[unique_mask]
        v_alt_clean = v_trip.alt_m[unique_mask]
        v_spd_clean = v_trip.speed_mps[unique_mask]
        v_hdg_clean = v_trip.heading_deg[unique_mask]
        v_yaw_clean = v_trip.yaw_rate_rads[unique_mask]
        v_whl_clean = v_trip.wheel_speeds_rads[unique_mask]
        v_acc_clean = v_trip.can_accel_g[unique_mask]
        v_gear_clean = v_trip.gear[unique_mask]
        v_hbrk_clean = v_trip.handbrake[unique_mask]
    else:
        t_v_clean = t_v_ns
        v_lat_clean = v_trip.lat
        v_lon_clean = v_trip.lon
        v_alt_clean = v_trip.alt_m
        v_spd_clean = v_trip.speed_mps
        v_hdg_clean = v_trip.heading_deg
        v_yaw_clean = v_trip.yaw_rate_rads
        v_whl_clean = v_trip.wheel_speeds_rads
        v_acc_clean = v_trip.can_accel_g
        v_gear_clean = v_trip.gear
        v_hbrk_clean = v_trip.handbrake

    # 5. Source (V) non-monotonic / backward timestamp detection
    clean_v_diffs = np.diff(t_v_clean)
    if len(clean_v_diffs) > 0 and (clean_v_diffs < 0).any():
        raise SyncValidationError(
            f"Source stream (V) contains non-monotonic/backward timestamps (diff < 0) after deduplication in trip {trip_id}. "
            "Ground truth VBOX clock regressions violate the source timeline contract."
        )

    if len(t_v_clean) < 2:
        raise SyncValidationError(f"Source stream has fewer than 2 valid samples after cleaning in trip {trip_id}")

    clock_origin_offset_s = float(t_v_clean[0] - t_s_ns[0]) / 1e9

    # 6. Build target and source grids according to explicit policy
    if policy.mode == SyncMode.RELATIVE_ELAPSED:
        t_target = _unwrap_monotonic_elapsed_ns(t_s_ns)
        t_source_clean = t_v_clean - t_v_clean[0]
    else:
        t_target = t_s_ns
        t_source_clean = t_v_clean

    # 4. Interpolate continuous telemetry
    effective_gap = policy.max_gap_ns
    v_lat = interpolate_continuous_1d(t_target, t_source_clean, v_lat_clean, max_gap_ns=effective_gap)
    v_lon = interpolate_continuous_1d(t_target, t_source_clean, v_lon_clean, max_gap_ns=effective_gap)
    v_alt_m = interpolate_continuous_1d(t_target, t_source_clean, v_alt_clean, max_gap_ns=effective_gap)
    v_speed_mps = interpolate_continuous_1d(t_target, t_source_clean, v_spd_clean, max_gap_ns=effective_gap)
    v_yaw_rate = interpolate_continuous_1d(t_target, t_source_clean, v_yaw_clean, max_gap_ns=effective_gap)
    v_wheel_speeds = interpolate_continuous_2d(t_target, t_source_clean, v_whl_clean, max_gap_ns=effective_gap)
    v_can_accel_g = interpolate_continuous_2d(t_target, t_source_clean, v_acc_clean, max_gap_ns=effective_gap)

    # 5. Circular interpolation for heading
    v_heading_deg = interpolate_circular_deg(t_target, t_source_clean, v_hdg_clean, max_gap_ns=effective_gap)

    # 6. Nearest-neighbor for discrete categorical CAN states
    v_gear = interpolate_categorical(t_target, t_source_clean, v_gear_clean, max_gap_ns=effective_gap)
    v_handbrake = interpolate_categorical(t_target, t_source_clean, v_hbrk_clean, max_gap_ns=effective_gap)

    # 7. Compute alignment diagnostics & validate policy
    is_valid_interp, is_oob, is_gap_viol = _get_valid_gap_mask(t_target, t_source_clean, max_gap_ns=effective_gap)
    overlap_start = max(t_target[0], t_source_clean[0])
    overlap_end = min(t_target[-1], t_source_clean[-1])
    overlap_duration_s = max(0.0, float(overlap_end - overlap_start) / 1e9)
    dur_target_s = float(t_target[-1] - t_target[0]) / 1e9
    dur_source_s = float(t_source_clean[-1] - t_source_clean[0]) / 1e9
    clock_drift_s = float(dur_target_s - dur_source_s)
    rel_elapsed_offset_s = float(t_source_clean[0] - t_target[0]) / 1e9
    valid_count = int(np.sum(is_valid_interp))
    coverage_ratio = (valid_count / n_s) if n_s > 0 else 0.0

    # Synchronization validation policy evaluation
    validation_issues = []
    if overlap_duration_s < policy.min_overlap_duration_s:
        validation_issues.append(
            f"Overlap duration ({overlap_duration_s:.2f}s) below minimum ({policy.min_overlap_duration_s:.2f}s)"
        )
    if coverage_ratio < policy.min_coverage_ratio:
        validation_issues.append(
            f"Coverage ratio ({coverage_ratio * 100:.1f}%) below minimum ({policy.min_coverage_ratio * 100:.1f}%)"
        )
    if abs(clock_drift_s) > policy.max_clock_drift_s:
        validation_issues.append(
            f"Clock drift ({clock_drift_s:.2f}s) exceeds threshold ({policy.max_clock_drift_s:.2f}s)"
        )

    if validation_issues:
        status = "FAILED: " + "; ".join(validation_issues)
        sync_passed = False
        if policy.enforce_strict_validation:
            raise SyncValidationError(f"Synchronization validation failed for trip {trip_id}: {status}")
    else:
        status = "PASSED"
        sync_passed = True

    diagnostics = SyncDiagnostics(
        clock_origin_offset_s=clock_origin_offset_s,
        relative_elapsed_offset_s=rel_elapsed_offset_s,
        clock_drift_s=clock_drift_s,
        overlap_duration_s=overlap_duration_s,
        source_sample_count=n_v,
        target_sample_count=n_s,
        valid_interpolated_count=valid_count,
        out_of_bounds_count=int(np.sum(is_oob)),
        gap_violation_count=int(np.sum(is_gap_viol)),
        valid_coverage_ratio=coverage_ratio,
        sync_mode=policy.mode.value,
        status=status,
        sync_passed=sync_passed,
    )

    synced_trip = SynchronizedTrip(
        trip_id=trip_id,
        branch=branch,
        row_count=n_s,
        timestamps_ns=t_target,
        raw_timestamps_ns=s_trip.timestamps_ns,
        accel_raw=s_trip.accel_raw,
        gyro_raw=s_trip.gyro_raw,
        quality_flags=quality_report.quality_flags,
        is_validated=quality_report.is_validated,
        s_gnss_lat=s_trip.gnss_lat,
        s_gnss_lon=s_trip.gnss_lon,
        s_gnss_alt=s_trip.gnss_alt,
        s_gnss_speed_mps=s_trip.gnss_speed_mps,
        s_gnss_bearing_deg=s_trip.gnss_bearing_deg,
        s_gnss_accuracy_m=s_trip.gnss_accuracy_m,
        s_gnss_sat_count=s_trip.gnss_sat_count,
        v_ref_lat=v_lat,
        v_ref_lon=v_lon,
        v_ref_alt_m=v_alt_m,
        v_ref_speed_mps=v_speed_mps,
        v_ref_heading_deg=v_heading_deg,
        v_ref_yaw_rate_rads=v_yaw_rate,
        v_ref_wheel_speeds=v_wheel_speeds,
        v_ref_can_accel_g=v_can_accel_g,
        v_ref_gear=v_gear,
        v_ref_handbrake=v_handbrake,
        sync_offset_s=diagnostics.clock_origin_offset_s,
        overlap_duration_s=overlap_duration_s,
        diagnostics=diagnostics,
    )

    return synced_trip, quality_report
