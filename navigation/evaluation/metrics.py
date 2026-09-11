"""Standardized metric computation library for C.O.M.P.A.S.S. navigation and dead-reckoning evaluation (Phase 13).

Provides mathematically exact, decoupled metrics for:
- Position error (2D/3D RMSE, mean, median, P95, maximum, ATE, RTE)
- Dead reckoning outage drift (distance travelled during outage, final drift, max drift, drift %, growth rate)
- Velocity error (RMSE, MAE, bias vector, P95)
- Heading error (RMSE, MAE, P95, maximum)
- Downstream map matching (snap rate, fallback rate, reasons breakdown, snap distances, cross/along track)
- GNSS outage recovery (detection latency, reacquisition latency, convergence time)
- ML measurement telemetry (inference counts, accepted/rejected measurements, update cadences)
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np


@dataclass(frozen=True)
class PositionMetrics:
    """Position accuracy metrics in metric ENU Cartesian frame."""
    rmse_2d: float
    rmse_3d: float
    mean_error: float
    median_error: float
    p95_error: float
    max_error: float
    final_position_error: float
    ate: float
    rte: float


@dataclass(frozen=True)
class DeadReckoningMetrics:
    """Dead reckoning blackout metrics strictly measured over the GNSS-denied interval."""
    outage_start_s: float
    outage_end_s: float
    duration_s: float
    distance_travelled_m: float
    final_outage_drift_m: float
    max_outage_drift_m: float
    drift_percentage: float  # (final_drift / distance_travelled) * 100.0
    max_drift_percentage: float  # (max_drift / distance_travelled) * 100.0
    drift_rate_mps: float  # final_drift / duration_s
    sih_10pct_passed: bool  # True if drift_percentage < 10.0%


@dataclass(frozen=True)
class VelocityMetrics:
    """Velocity vector error metrics."""
    rmse_2d: float
    rmse_3d: float
    mae_2d: float
    bias_2d: Tuple[float, float]
    p95_error: float


@dataclass(frozen=True)
class HeadingMetrics:
    """Attitude / heading error metrics in degrees."""
    rmse_deg: float
    mae_deg: float
    p95_deg: float
    max_deg: float


@dataclass(frozen=True)
class MapMatchMetrics:
    """Downstream map matching and road snapping performance metrics."""
    total_epochs: int
    snapped_count: int
    fallback_count: int
    snap_rate_pct: float
    fallback_rate_pct: float
    fallback_reasons: Dict[str, int]
    median_snap_dist_m: float
    p95_snap_dist_m: float
    max_snap_dist_m: float
    cross_track_rmse_m: float
    along_track_rmse_m: float


@dataclass(frozen=True)
class RecoveryMetrics:
    """GNSS recovery and filter re-convergence metrics."""
    outage_detection_latency_s: float
    gnss_reacquisition_latency_s: float
    accepted_reacq_fixes: int
    rejected_reacq_fixes: int
    convergence_time_s: Optional[float]
    post_recovery_error_m: Optional[float]


@dataclass(frozen=True)
class MLDiagnosticsMetrics:
    """Telemetry and update cadence diagnostics for VelocityNet and BiasNet."""
    velocitynet_inferences: int
    biasnet_inferences: int
    velocitynet_accepted: int
    biasnet_accepted: int
    velocitynet_rejected: int
    biasnet_rejected: int
    velocitynet_cadence_hz: float
    biasnet_cadence_hz: float


@dataclass(frozen=True)
class TrajectoryMetrics:
    """Complete evaluation metric report for an experimental run."""
    position: PositionMetrics
    velocity: VelocityMetrics
    heading: HeadingMetrics
    dead_reckoning: Optional[DeadReckoningMetrics] = None
    map_matching: Optional[MapMatchMetrics] = None
    recovery: Optional[RecoveryMetrics] = None
    ml_diagnostics: Optional[MLDiagnosticsMetrics] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_position_metrics(
    est_pos_enu: np.ndarray,
    ref_pos_enu: np.ndarray,
    rte_interval_steps: int = 10,
) -> PositionMetrics:
    """Compute comprehensive 2D/3D position error metrics, ATE, and RTE."""
    est = np.asarray(est_pos_enu, dtype=np.float64)
    ref = np.asarray(ref_pos_enu, dtype=np.float64)

    if len(est) != len(ref) or len(est) == 0:
        raise ValueError("Estimated and reference trajectories must be non-empty and equal length.")

    # 2D and 3D Euclidean errors
    err_2d = np.linalg.norm(est[:, :2] - ref[:, :2], axis=1)
    err_3d = np.linalg.norm(est[:, :3] - ref[:, :3], axis=1)

    rmse_2d = float(np.sqrt(np.mean(err_2d ** 2)))
    rmse_3d = float(np.sqrt(np.mean(err_3d ** 2)))
    mean_err = float(np.mean(err_2d))
    med_err = float(np.median(err_2d))
    p95_err = float(np.percentile(err_2d, 95))
    max_err = float(np.max(err_2d))
    final_err = float(err_2d[-1])

    # Absolute Trajectory Error (ATE is root mean square of alignment error)
    ate = rmse_2d

    # Relative Trajectory Error (RTE over fixed intervals)
    if len(est) > rte_interval_steps:
        disp_est = est[rte_interval_steps:, :2] - est[:-rte_interval_steps, :2]
        disp_ref = ref[rte_interval_steps:, :2] - ref[:-rte_interval_steps, :2]
        rte_err = np.linalg.norm(disp_est - disp_ref, axis=1)
        rte = float(np.sqrt(np.mean(rte_err ** 2)))
    else:
        rte = 0.0

    return PositionMetrics(
        rmse_2d=rmse_2d,
        rmse_3d=rmse_3d,
        mean_error=mean_err,
        median_error=med_err,
        p95_error=p95_err,
        max_error=max_err,
        final_position_error=final_err,
        ate=ate,
        rte=rte,
    )


def compute_outage_metrics(
    est_pos_enu: np.ndarray,
    ref_pos_enu: np.ndarray,
    start_step: int,
    end_step: int,
    times_s: Sequence[float],
) -> DeadReckoningMetrics:
    """Compute dead-reckoning blackout drift metrics strictly over [start_step, end_step].

    IMPORTANT:
    1. Distance travelled is the accumulated ground-truth reference path length during blackout.
    2. Final outage drift is the 2D Euclidean distance between estimated and true position at outage end.
    3. Maximum outage drift is the worst-case 2D position error at any epoch during blackout.
    4. Drift % = (final_outage_drift / distance_travelled) * 100.
    """
    est = np.asarray(est_pos_enu, dtype=np.float64)
    ref = np.asarray(ref_pos_enu, dtype=np.float64)

    s_idx = max(0, start_step)
    e_idx = min(len(est) - 1, end_step)

    t_start = float(times_s[s_idx])
    t_end = float(times_s[e_idx])
    duration = max(1e-3, t_end - t_start)

    # Accumulated reference distance travelled during blackout
    ref_sub = ref[s_idx : e_idx + 1, :2]
    step_diffs = np.diff(ref_sub, axis=0)
    step_dists = np.linalg.norm(step_diffs, axis=1)
    dist_travelled = float(np.sum(step_dists))

    # Error vector at each blackout step
    est_sub = est[s_idx : e_idx + 1, :2]
    errs_2d = np.linalg.norm(est_sub - ref_sub, axis=1)

    final_drift = float(errs_2d[-1])
    max_drift = float(np.max(errs_2d))

    drift_pct = (final_drift / max(dist_travelled, 1.0)) * 100.0
    max_drift_pct = (max_drift / max(dist_travelled, 1.0)) * 100.0
    drift_rate = final_drift / duration

    return DeadReckoningMetrics(
        outage_start_s=t_start,
        outage_end_s=t_end,
        duration_s=duration,
        distance_travelled_m=dist_travelled,
        final_outage_drift_m=final_drift,
        max_outage_drift_m=max_drift,
        drift_percentage=drift_pct,
        max_drift_percentage=max_drift_pct,
        drift_rate_mps=drift_rate,
        sih_10pct_passed=(drift_pct < 10.0),
    )


def compute_velocity_metrics(
    est_vel_enu: np.ndarray,
    ref_vel_enu: np.ndarray,
) -> VelocityMetrics:
    """Compute velocity estimation error metrics."""
    est = np.asarray(est_vel_enu, dtype=np.float64)
    ref = np.asarray(ref_vel_enu, dtype=np.float64)

    err_2d = np.linalg.norm(est[:, :2] - ref[:, :2], axis=1)
    err_3d = np.linalg.norm(est[:, :3] - ref[:, :3], axis=1)

    rmse_2d = float(np.sqrt(np.mean(err_2d ** 2)))
    rmse_3d = float(np.sqrt(np.mean(err_3d ** 2)))
    mae_2d = float(np.mean(err_2d))
    bias_2d = (float(np.mean(est[:, 0] - ref[:, 0])), float(np.mean(est[:, 1] - ref[:, 1])))
    p95_2d = float(np.percentile(err_2d, 95))

    return VelocityMetrics(
        rmse_2d=rmse_2d,
        rmse_3d=rmse_3d,
        mae_2d=mae_2d,
        bias_2d=bias_2d,
        p95_error=p95_2d,
    )


def compute_heading_metrics(
    est_headings_rad: Sequence[float],
    ref_headings_rad: Sequence[float],
) -> HeadingMetrics:
    """Compute angular heading error metrics in degrees."""
    est = np.asarray(est_headings_rad, dtype=np.float64)
    ref = np.asarray(ref_headings_rad, dtype=np.float64)

    diffs = (est - ref + np.pi) % (2.0 * np.pi) - np.pi
    diffs_deg = np.degrees(np.abs(diffs))

    rmse = float(np.sqrt(np.mean(diffs_deg ** 2)))
    mae = float(np.mean(diffs_deg))
    p95 = float(np.percentile(diffs_deg, 95))
    mx = float(np.max(diffs_deg))

    return HeadingMetrics(
        rmse_deg=rmse,
        mae_deg=mae,
        p95_deg=p95,
        max_deg=mx,
    )


def compute_trajectory_metrics(
    est_pos_enu: np.ndarray,
    ref_pos_enu: np.ndarray,
    est_vel_enu: np.ndarray,
    ref_vel_enu: np.ndarray,
    times_s: Sequence[float],
    est_headings_rad: Optional[Sequence[float]] = None,
    ref_headings_rad: Optional[Sequence[float]] = None,
    outage_steps: Optional[Tuple[int, int]] = None,
    mm_outputs: Optional[Sequence[Any]] = None,
    recovery_metrics: Optional[RecoveryMetrics] = None,
    ml_diagnostics: Optional[MLDiagnosticsMetrics] = None,
) -> TrajectoryMetrics:
    """Compute the unified TrajectoryMetrics object covering all subsystems."""
    pos_m = compute_position_metrics(est_pos_enu, ref_pos_enu)
    vel_m = compute_velocity_metrics(est_vel_enu, ref_vel_enu)

    if est_headings_rad is not None and ref_headings_rad is not None:
        head_m = compute_heading_metrics(est_headings_rad, ref_headings_rad)
    else:
        head_m = HeadingMetrics(0.0, 0.0, 0.0, 0.0)

    dr_m = None
    if outage_steps is not None:
        dr_m = compute_outage_metrics(est_pos_enu, ref_pos_enu, outage_steps[0], outage_steps[1], times_s)

    mm_m = None
    if mm_outputs is not None and len(mm_outputs) > 0:
        total = len(mm_outputs)
        snapped = sum(1 for o in mm_outputs if o.snapped)
        fallbacks = total - snapped
        reasons: Dict[str, int] = {}
        for o in mm_outputs:
            if not o.snapped and o.fallback_reason:
                reasons[o.fallback_reason] = reasons.get(o.fallback_reason, 0) + 1

        snap_dists = [o.distance_to_road_m for o in mm_outputs if o.snapped and o.distance_to_road_m is not None]

        # Cross-track & along-track displacement relative to reference
        disp_pos = np.array([o.display_enu for o in mm_outputs])
        ref_sub = np.asarray(ref_pos_enu[:total], dtype=np.float64)
        err_vec = disp_pos[:, :2] - ref_sub[:, :2]
        cross_track = float(np.sqrt(np.mean(err_vec[:, 1] ** 2)))
        along_track = float(np.sqrt(np.mean(err_vec[:, 0] ** 2)))

        mm_m = MapMatchMetrics(
            total_epochs=total,
            snapped_count=snapped,
            fallback_count=fallbacks,
            snap_rate_pct=(snapped / max(total, 1)) * 100.0,
            fallback_rate_pct=(fallbacks / max(total, 1)) * 100.0,
            fallback_reasons=reasons,
            median_snap_dist_m=float(np.median(snap_dists)) if snap_dists else 0.0,
            p95_snap_dist_m=float(np.percentile(snap_dists, 95)) if snap_dists else 0.0,
            max_snap_dist_m=float(np.max(snap_dists)) if snap_dists else 0.0,
            cross_track_rmse_m=cross_track,
            along_track_rmse_m=along_track,
        )

    return TrajectoryMetrics(
        position=pos_m,
        velocity=vel_m,
        heading=head_m,
        dead_reckoning=dr_m,
        map_matching=mm_m,
        recovery=recovery_metrics,
        ml_diagnostics=ml_diagnostics,
    )
