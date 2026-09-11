"""Operating condition scenario definitions and synthetic benchmark generators (Phase 13).

Defines:
1. Axis B Operating-Condition Matrix:
   - B1: Continuous GNSS (Open-sky baseline tracking)
   - B2: 10s GNSS Outage (Urban canyon short blackout)
   - B3: 30s GNSS Outage (Extended overpass blackout)
   - B4: 60s GNSS Outage (Standard SIH benchmark blackout)
   - B5: 120s GNSS Outage (Extended tunnel blackout stress test)
   - B6: 300s GNSS Outage (Extreme 5-minute divergence stress test)
   - B7: Sharp Turn / High Dynamics (Skid relaxation & high angular velocity)
   - B8: Stop-and-Go (Standstill zero velocity pinning)
   - B9: Parallel-Road Ambiguity (Corridor disambiguation & margin safeguards)
   - B10: Zero Map Coverage (Open field / unmapped fallback)
   - B11: GNSS Reacquisition & Recovery (Smooth covariance collapse & latency)
   - B12: Real Environmental Outage (Real GNSS dropouts in session)

2. Controlled Physical Synthetic Benchmarks:
   - Benchmark 1: 50m travel in <1 min, target drift < 5m.
   - Benchmark 2: Controlled 1 km travel at 60 km/h (60s blackout), target drift < 100m.
     (Clearly designated and labeled as a controlled synthetic benchmark).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from navigation.frames.local_geo import GeoReference


@dataclass(frozen=True)
class ScenarioDefinition:
    """Definition of an evaluation scenario slice."""
    scenario_id: str
    name: str
    description: str
    session_file: str
    start_idx: int
    duration_steps: int  # at 10 Hz canonical rate
    outage_start_rel_steps: Optional[int] = None
    outage_duration_steps: Optional[int] = None
    is_synthetic: bool = False
    target_drift_pct: float = 10.0  # Official SIH requirement


# Axis B Operating Condition Matrix
AXIS_B_SCENARIOS: Dict[str, ScenarioDefinition] = {
    "B1_CONTINUOUS_GNSS": ScenarioDefinition(
        scenario_id="B1_CONTINUOUS_GNSS",
        name="B1: Continuous GNSS",
        description="Nominal open-sky driving with continuous 1 Hz GNSS fixes.",
        session_file="Categorised_S1.npz",
        start_idx=4900,
        duration_steps=600,  # 60s
        outage_start_rel_steps=None,
        outage_duration_steps=None,
    ),
    "B2_OUTAGE_10S": ScenarioDefinition(
        scenario_id="B2_OUTAGE_10S",
        name="B2: 10s GNSS Outage",
        description="Short 10-second urban canyon GNSS dropout.",
        session_file="Categorised_S1.npz",
        start_idx=4900,
        duration_steps=300,  # 10s pre, 10s outage, 10s post
        outage_start_rel_steps=100,
        outage_duration_steps=100,  # 10.0s
    ),
    "B3_OUTAGE_30S": ScenarioDefinition(
        scenario_id="B3_OUTAGE_30S",
        name="B3: 30s GNSS Outage",
        description="Intermediate 30-second overpass/tunnel GNSS blackout.",
        session_file="Categorised_S1.npz",
        start_idx=4900,
        duration_steps=500,  # 10s pre, 30s outage, 10s post
        outage_start_rel_steps=100,
        outage_duration_steps=300,  # 30.0s
    ),
    "B4_OUTAGE_60S": ScenarioDefinition(
        scenario_id="B4_OUTAGE_60S",
        name="B4: 60s GNSS Outage (Standard Benchmark)",
        description="Standard 60-second GNSS blackout benchmark.",
        session_file="Categorised_S1.npz",
        start_idx=4900,
        duration_steps=800,  # 10s pre, 60s outage, 10s post
        outage_start_rel_steps=100,
        outage_duration_steps=600,  # 60.0s
    ),
    "B5_OUTAGE_120S": ScenarioDefinition(
        scenario_id="B5_OUTAGE_120S",
        name="B5: 120s Extended Outage Stress Test",
        description="Extended 2-minute blackout stress test for filter divergence bounds.",
        session_file="Categorised_S1.npz",
        start_idx=4900,
        duration_steps=1400,  # 10s pre, 120s outage, 10s post
        outage_start_rel_steps=100,
        outage_duration_steps=1200,  # 120.0s
    ),
    "B6_OUTAGE_300S": ScenarioDefinition(
        scenario_id="B6_OUTAGE_300S",
        name="B6: 300s Extreme Outage Stress Test",
        description="Extreme 5-minute blackout stress test for covariance bounds.",
        session_file="Categorised_S1.npz",
        start_idx=4900,
        duration_steps=3200,  # 10s pre, 300s outage, 10s post
        outage_start_rel_steps=100,
        outage_duration_steps=3000,  # 300.0s
    ),
    "B7_SHARP_TURN": ScenarioDefinition(
        scenario_id="B7_SHARP_TURN",
        name="B7: Sharp Turn / High Dynamics",
        description="Cornering dynamics with significant yaw rate.",
        session_file="Categorised_S1.npz",
        start_idx=10500,
        duration_steps=400,  # 40s
        outage_start_rel_steps=None,
        outage_duration_steps=None,
    ),
    "B8_STOP_AND_GO": ScenarioDefinition(
        scenario_id="B8_STOP_AND_GO",
        name="B8: Stop-and-Go Driving",
        description="Stationary standstill periods evaluating ZUPT zero pinning.",
        session_file="Categorised_S1.npz",
        start_idx=4500,
        duration_steps=300,  # 30s
        outage_start_rel_steps=None,
        outage_duration_steps=None,
    ),
    "B9_PARALLEL_ROADS": ScenarioDefinition(
        scenario_id="B9_PARALLEL_ROADS",
        name="B9: Parallel Road Ambiguity",
        description="Dual-carriageway segment evaluating ambiguity margin safeguard.",
        session_file="Categorised_S1.npz",
        start_idx=4900,
        duration_steps=600,
        outage_start_rel_steps=None,
        outage_duration_steps=None,
    ),
    "B10_ZERO_COVERAGE": ScenarioDefinition(
        scenario_id="B10_ZERO_COVERAGE",
        name="B10: Zero Map Coverage",
        description="Off-map trajectory testing 100% graceful fallback to estimator coordinates.",
        session_file="Categorised_S1.npz",
        start_idx=4900,
        duration_steps=50,
        is_synthetic=True,
    ),
    "B11_RECOVERY": ScenarioDefinition(
        scenario_id="B11_RECOVERY",
        name="B11: GNSS Recovery & Reacquisition",
        description="Recovery phase following 60s outage testing bounded convergence.",
        session_file="Categorised_S1.npz",
        start_idx=4900,
        duration_steps=800,
        outage_start_rel_steps=100,
        outage_duration_steps=600,
    ),
    "B12_REAL_OUTAGE": ScenarioDefinition(
        scenario_id="B12_REAL_OUTAGE",
        name="B12: Real Environmental GNSS Dropout",
        description="Natural GNSS signal degradation in IO-VNBD session S3c.",
        session_file="Categorised_S3c.npz",
        start_idx=2000,
        duration_steps=500,
        outage_start_rel_steps=50,
        outage_duration_steps=200,
    ),
}


class SyntheticTrajectoryGenerator:
    """Generates controlled, physically meaningful synthetic trajectories with known ground truth."""

    @staticmethod
    def generate_50m_benchmark(
        rate_hz: float = 10.0,
        speed_mps: float = 1.0,  # 3.6 km/h -> 50m in 50s (< 1 min)
        accel_bias: Tuple[float, float, float] = (0.01, -0.01, 0.02),
        gyro_bias: Tuple[float, float, float] = (0.001, 0.001, -0.002),
        noise_std_acc: float = 0.05,
        noise_std_gyro: float = 0.005,
    ) -> Dict[str, Any]:
        """Benchmark 1: 50m GNSS-denied travel in under 1 min (Target: drift < 5m)."""
        duration_s = 50.0
        n_steps = int(duration_s * rate_hz)
        dt = 1.0 / rate_hz

        t = np.linspace(0.0, duration_s, n_steps)
        # Straight motion along East: x(t) = speed * t, y(t) = 0
        ref_x = speed_mps * t
        ref_y = np.zeros_like(t)
        ref_z = np.zeros_like(t)

        ref_pos_enu = np.column_stack([ref_x, ref_y, ref_z])
        ref_vel_enu = np.column_stack([np.full(n_steps, speed_mps), np.zeros(n_steps), np.zeros(n_steps)])
        ref_headings_rad = np.full(n_steps, 0.5 * math.pi)  # East

        # Specific force in body frame (x forward, z up):
        # When moving at constant velocity: f_b = [0, 0, +9.81]
        rng = np.random.RandomState(42)
        f_m_v = np.zeros((n_steps, 3))
        f_m_v[:, 0] = accel_bias[0] + rng.normal(0.0, noise_std_acc, n_steps)
        f_m_v[:, 1] = accel_bias[1] + rng.normal(0.0, noise_std_acc, n_steps)
        f_m_v[:, 2] = 9.81 + accel_bias[2] + rng.normal(0.0, noise_std_acc, n_steps)

        omega_m_v = np.zeros((n_steps, 3))
        omega_m_v[:, 0] = gyro_bias[0] + rng.normal(0.0, noise_std_gyro, n_steps)
        omega_m_v[:, 1] = gyro_bias[1] + rng.normal(0.0, noise_std_gyro, n_steps)
        omega_m_v[:, 2] = gyro_bias[2] + rng.normal(0.0, noise_std_gyro, n_steps)

        timestamps_ns = (t * 1e9).astype(np.int64)

        geo_ref = GeoReference(52.4, -1.5, 0.0)
        ref_lat, ref_lon, ref_alt = [], [], []
        for e, n, u in ref_pos_enu:
            la, lo, al = geo_ref.enu_to_geodetic(e, n, u)
            ref_lat.append(la)
            ref_lon.append(lo)
            ref_alt.append(al)

        return {
            "name": "Benchmark 1 (Synthetic 50m / <1 min)",
            "rate_hz": rate_hz,
            "duration_s": duration_s,
            "distance_travelled_m": float(ref_x[-1]),
            "timestamps_ns": timestamps_ns,
            "times_s": t,
            "ref_pos_enu": ref_pos_enu,
            "ref_vel_enu": ref_vel_enu,
            "ref_headings_rad": ref_headings_rad,
            "ref_lat": np.array(ref_lat),
            "ref_lon": np.array(ref_lon),
            "ref_alt_m": np.array(ref_alt),
            "f_m_v": f_m_v,
            "omega_m_v": omega_m_v,
            "calib_gyro_bias": np.array(gyro_bias),
            "outage_start_step": 0,
            "outage_duration_steps": n_steps,
            "target_max_drift_m": 5.0,
            "is_synthetic": True,
        }

    @staticmethod
    def generate_1km_60kmh_benchmark(
        rate_hz: float = 10.0,
        speed_mps: float = 16.667,  # 60.0 km/h -> 1000m in 60s
        accel_bias: Tuple[float, float, float] = (0.015, -0.01, 0.02),
        gyro_bias: Tuple[float, float, float] = (0.0005, 0.0005, -0.001),
        noise_std_acc: float = 0.05,
        noise_std_gyro: float = 0.005,
    ) -> Dict[str, Any]:
        """Benchmark 2 (Controlled Synthetic): 1 km travel at 60 km/h in 60s blackout (Target: drift < 100m)."""
        duration_s = 60.0
        n_steps = int(duration_s * rate_hz)
        dt = 1.0 / rate_hz

        t = np.linspace(0.0, duration_s, n_steps)
        ref_x = speed_mps * t
        ref_y = np.zeros_like(t)
        ref_z = np.zeros_like(t)

        ref_pos_enu = np.column_stack([ref_x, ref_y, ref_z])
        ref_vel_enu = np.column_stack([np.full(n_steps, speed_mps), np.zeros(n_steps), np.zeros(n_steps)])
        ref_headings_rad = np.full(n_steps, 0.5 * math.pi)

        rng = np.random.RandomState(123)
        f_m_v = np.zeros((n_steps, 3))
        f_m_v[:, 0] = accel_bias[0] + rng.normal(0.0, noise_std_acc, n_steps)
        f_m_v[:, 1] = accel_bias[1] + rng.normal(0.0, noise_std_acc, n_steps)
        f_m_v[:, 2] = 9.81 + accel_bias[2] + rng.normal(0.0, noise_std_acc, n_steps)

        omega_m_v = np.zeros((n_steps, 3))
        omega_m_v[:, 0] = gyro_bias[0] + rng.normal(0.0, noise_std_gyro, n_steps)
        omega_m_v[:, 1] = gyro_bias[1] + rng.normal(0.0, noise_std_gyro, n_steps)
        omega_m_v[:, 2] = gyro_bias[2] + rng.normal(0.0, noise_std_gyro, n_steps)

        timestamps_ns = (t * 1e9).astype(np.int64)

        geo_ref = GeoReference(52.4, -1.5, 0.0)
        ref_lat, ref_lon, ref_alt = [], [], []
        for e, n, u in ref_pos_enu:
            la, lo, al = geo_ref.enu_to_geodetic(e, n, u)
            ref_lat.append(la)
            ref_lon.append(lo)
            ref_alt.append(al)

        return {
            "name": "Benchmark 2 (Controlled Synthetic 1km / 60km/h / 60s Outage)",
            "rate_hz": rate_hz,
            "duration_s": duration_s,
            "distance_travelled_m": float(ref_x[-1]),
            "timestamps_ns": timestamps_ns,
            "times_s": t,
            "ref_pos_enu": ref_pos_enu,
            "ref_vel_enu": ref_vel_enu,
            "ref_headings_rad": ref_headings_rad,
            "ref_lat": np.array(ref_lat),
            "ref_lon": np.array(ref_lon),
            "ref_alt_m": np.array(ref_alt),
            "f_m_v": f_m_v,
            "omega_m_v": omega_m_v,
            "calib_gyro_bias": np.array(gyro_bias),
            "outage_start_step": 0,
            "outage_duration_steps": n_steps,
            "target_max_drift_m": 100.0,
            "is_synthetic": True,
        }
