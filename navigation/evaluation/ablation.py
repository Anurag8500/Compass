"""Ablation ladder definitions and execution suite for C.O.M.P.A.S.S. (Phase 13).

Defines two formal ablation ladders:
1. Nominal Axis A Fusion Ladder (A1 to A7):
   - A1_PURE_INS: Pure strapdown inertial mechanization without any aiding.
   - A2_ESKF_GNSS: Classical 15-state ESKF with continuous 1 Hz GNSS fixes.
   - A3_VELOCITYNET: A2 + learned forward speed pseudo-measurements (VelocityNet v1.1).
   - A4_BIASNET: A3 + learned IMU bias residual compensation (BiasNet v1.0).
   - A5_NHC: A4 + classical non-holonomic velocity constraints (lateral/vertical zero).
   - A6_ZUPT: A5 + classical standstill zero-velocity updates during detected stops.
   - A7_MAPMATCH: A6 + strictly downstream OSM HMM road snapping for presentation.

2. Dedicated GNSS-Denied Dead-Reckoning Ladder (DR-A2 to DR-A7) under identical blackout:
   - DR_A2_PURE_INS_COASTING: Pure inertial propagation without GNSS aiding.
   - DR_A3_VELOCITYNET: DR-A2 + VelocityNet speed aiding during blackout.
   - DR_A4_BIASNET: DR-A3 + BiasNet bias residual compensation during blackout.
   - DR_A5_NHC: DR-A4 + lateral/vertical non-holonomic velocity constraints.
   - DR_A6_ZUPT: DR-A5 + gated zero-velocity updates during standstill.
   - DR_A7_MAPMATCH: DR-A6 + downstream road matching (display output).

Computes exact incremental contributions to evaluate what improved because of AI/ML vs classical constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from navigation.core import NavigationCoreConfig
from navigation.evaluation.metrics import TrajectoryMetrics
from navigation.replay import ReplayConfig, ReplayResult, run_offline_replay


@dataclass(frozen=True)
class AblationLevel:
    """Specification of an ablation level configuration."""
    level_id: str
    name: str
    description: str
    core_config: NavigationCoreConfig
    enable_map_matching: bool


# Nominal Axis A Ladder Configurations
AXIS_A_LEVELS: Dict[str, AblationLevel] = {
    "A1_PURE_INS": AblationLevel(
        level_id="A1_PURE_INS",
        name="A1: Pure Strapdown INS",
        description="Strapdown inertial integration with zero external aiding or updates.",
        core_config=NavigationCoreConfig(
            gnss_enabled=False,
            velocitynet_enabled=False,
            biasnet_enabled=False,
            nhc_enabled=False,
            zupt_enabled=False,
        ),
        enable_map_matching=False,
    ),
    "A2_ESKF_GNSS": AblationLevel(
        level_id="A2_ESKF_GNSS",
        name="A2: ESKF + GNSS Baseline",
        description="Classical 15-state ESKF with continuous 1 Hz GNSS fixes.",
        core_config=NavigationCoreConfig(
            gnss_enabled=True,
            velocitynet_enabled=False,
            biasnet_enabled=False,
            nhc_enabled=False,
            zupt_enabled=False,
        ),
        enable_map_matching=False,
    ),
    "A3_VELOCITYNET": AblationLevel(
        level_id="A3_VELOCITYNET",
        name="A3: ESKF + GNSS + VelocityNet",
        description="Adds learned forward speed pseudo-measurements via VelocityNet.",
        core_config=NavigationCoreConfig(
            gnss_enabled=True,
            velocitynet_enabled=True,
            biasnet_enabled=False,
            nhc_enabled=False,
            zupt_enabled=False,
        ),
        enable_map_matching=False,
    ),
    "A4_BIASNET": AblationLevel(
        level_id="A4_BIASNET",
        name="A4: ESKF + GNSS + VelocityNet + BiasNet",
        description="Adds learned IMU bias residual compensation via BiasNet.",
        core_config=NavigationCoreConfig(
            gnss_enabled=True,
            velocitynet_enabled=True,
            biasnet_enabled=True,
            nhc_enabled=False,
            zupt_enabled=False,
        ),
        enable_map_matching=False,
    ),
    "A5_NHC": AblationLevel(
        level_id="A5_NHC",
        name="A5: A4 + Classical NHC",
        description="Adds lateral and vertical non-holonomic velocity constraints.",
        core_config=NavigationCoreConfig(
            gnss_enabled=True,
            velocitynet_enabled=True,
            biasnet_enabled=True,
            nhc_enabled=True,
            zupt_enabled=False,
        ),
        enable_map_matching=False,
    ),
    "A6_ZUPT": AblationLevel(
        level_id="A6_ZUPT",
        name="A6: A5 + Gated ZUPT",
        description="Adds classical stationary zero-velocity updates during detected stops.",
        core_config=NavigationCoreConfig(
            gnss_enabled=True,
            velocitynet_enabled=True,
            biasnet_enabled=True,
            nhc_enabled=True,
            zupt_enabled=True,
        ),
        enable_map_matching=False,
    ),
    "A7_MAPMATCH": AblationLevel(
        level_id="A7_MAPMATCH",
        name="A7: Full System (+ Downstream Map Matching)",
        description="Full integrated pipeline with strictly downstream OSM HMM road snapping.",
        core_config=NavigationCoreConfig(
            gnss_enabled=True,
            velocitynet_enabled=True,
            biasnet_enabled=True,
            nhc_enabled=True,
            zupt_enabled=True,
        ),
        enable_map_matching=True,
    ),
}


# Dedicated GNSS-Denied Dead-Reckoning Ladder (Pre-implementation Correction 3)
DR_ABLATION_LEVELS: Dict[str, AblationLevel] = {
    "DR_A2_COASTING": AblationLevel(
        level_id="DR_A2_COASTING",
        name="DR-A2: Pure Inertial Coasting",
        description="Pure inertial propagation without any aiding during blackout.",
        core_config=NavigationCoreConfig(
            gnss_enabled=True,  # Active pre-outage, blocked during outage
            velocitynet_enabled=False,
            biasnet_enabled=False,
            nhc_enabled=False,
            zupt_enabled=False,
        ),
        enable_map_matching=False,
    ),
    "DR_A3_VELOCITYNET": AblationLevel(
        level_id="DR_A3_VELOCITYNET",
        name="DR-A3: DR-A2 + VelocityNet",
        description="Learned speed aiding active during blackout.",
        core_config=NavigationCoreConfig(
            gnss_enabled=True,
            velocitynet_enabled=True,
            biasnet_enabled=False,
            nhc_enabled=False,
            zupt_enabled=False,
        ),
        enable_map_matching=False,
    ),
    "DR_A4_BIASNET": AblationLevel(
        level_id="DR_A4_BIASNET",
        name="DR-A4: DR-A3 + BiasNet",
        description="Learned speed and IMU bias compensation active during blackout.",
        core_config=NavigationCoreConfig(
            gnss_enabled=True,
            velocitynet_enabled=True,
            biasnet_enabled=True,
            nhc_enabled=False,
            zupt_enabled=False,
        ),
        enable_map_matching=False,
    ),
    "DR_A5_NHC": AblationLevel(
        level_id="DR_A5_NHC",
        name="DR-A5: DR-A4 + NHC",
        description="Full ML aiding + non-holonomic velocity constraints active during blackout.",
        core_config=NavigationCoreConfig(
            gnss_enabled=True,
            velocitynet_enabled=True,
            biasnet_enabled=True,
            nhc_enabled=True,
            zupt_enabled=False,
        ),
        enable_map_matching=False,
    ),
    "DR_A6_ZUPT": AblationLevel(
        level_id="DR_A6_ZUPT",
        name="DR-A6: DR-A5 + Gated ZUPT",
        description="Full ML aiding + NHC + ZUPT stationary pinning during blackout.",
        core_config=NavigationCoreConfig(
            gnss_enabled=True,
            velocitynet_enabled=True,
            biasnet_enabled=True,
            nhc_enabled=True,
            zupt_enabled=True,
        ),
        enable_map_matching=False,
    ),
    "DR_A7_MAPMATCH": AblationLevel(
        level_id="DR_A7_MAPMATCH",
        name="DR-A7: DR-A6 + Downstream Map Matching",
        description="Full stack with downstream road snapping display output.",
        core_config=NavigationCoreConfig(
            gnss_enabled=True,
            velocitynet_enabled=True,
            biasnet_enabled=True,
            nhc_enabled=True,
            zupt_enabled=True,
        ),
        enable_map_matching=True,
    ),
}


def run_ablation_ladder(
    ladder_levels: Dict[str, AblationLevel],
    data: Any,
    calib_gyro_bias: np.ndarray,
    base_replay_config: ReplayConfig,
    road_graph: Optional[Any] = None,
) -> Dict[str, ReplayResult]:
    """Execute all levels of an ablation ladder under identical conditions."""
    results: Dict[str, ReplayResult] = {}
    for level_key, level_spec in ladder_levels.items():
        cfg = ReplayConfig(
            start_idx=base_replay_config.start_idx,
            duration_steps=base_replay_config.duration_steps,
            rate_hz=base_replay_config.rate_hz,
            outage_start_rel_steps=base_replay_config.outage_start_rel_steps,
            outage_duration_steps=base_replay_config.outage_duration_steps,
            mounting_yaw_err_deg=base_replay_config.mounting_yaw_err_deg,
            enable_map_matching=level_spec.enable_map_matching,
        )
        res = run_offline_replay(
            data=data,
            calib_gyro_bias=calib_gyro_bias,
            core_config=level_spec.core_config,
            replay_config=cfg,
            road_graph=road_graph,
        )
        results[level_key] = res
    return results


def compute_incremental_contributions(ladder_results: Dict[str, ReplayResult]) -> Dict[str, Dict[str, float]]:
    """Compute step-by-step delta contributions (improvements/degradations) between adjacent ladder steps."""
    keys = list(ladder_results.keys())
    deltas: Dict[str, Dict[str, float]] = {}

    for i in range(1, len(keys)):
        prev_k = keys[i - 1]
        curr_k = keys[i]
        prev_m = ladder_results[prev_k].metrics
        curr_m = ladder_results[curr_k].metrics

        # Positive delta means reduction in error (improvement)
        rmse_gain = prev_m.position.rmse_2d - curr_m.position.rmse_2d

        drift_gain = 0.0
        drift_pct_gain = 0.0
        if prev_m.dead_reckoning is not None and curr_m.dead_reckoning is not None:
            drift_gain = prev_m.dead_reckoning.final_outage_drift_m - curr_m.dead_reckoning.final_outage_drift_m
            drift_pct_gain = prev_m.dead_reckoning.drift_percentage - curr_m.dead_reckoning.drift_percentage

        deltas[f"{curr_k}_vs_{prev_k}"] = {
            "rmse_improvement_m": float(rmse_gain),
            "drift_reduction_m": float(drift_gain),
            "drift_pct_reduction": float(drift_pct_gain),
        }

    return deltas
