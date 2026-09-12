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
from navigation.preprocessing.gravity import STANDARD_GRAVITY_MPS2


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
        description="Nominal open-sky driving on IO-VNBD S1 with continuous 1 Hz GNSS fixes.",
        session_file="Categorised_S1.npz",
        start_idx=4900,
        duration_steps=600,  # 60s
        outage_start_rel_steps=None,
        outage_duration_steps=None,
    ),
    "B2_OUTAGE_10S": ScenarioDefinition(
        scenario_id="B2_OUTAGE_10S",
        name="B2: 10s GNSS Outage",
        description="10-second synthetic GNSS blackout on real IO-VNBD S1 trajectory.",
        session_file="Categorised_S1.npz",
        start_idx=4900,
        duration_steps=300,  # 10s pre, 10s outage, 10s post
        outage_start_rel_steps=100,
        outage_duration_steps=100,  # 10.0s
    ),
    "B3_OUTAGE_30S": ScenarioDefinition(
        scenario_id="B3_OUTAGE_30S",
        name="B3: 30s GNSS Outage",
        description="30-second synthetic GNSS blackout on real IO-VNBD S1 trajectory.",
        session_file="Categorised_S1.npz",
        start_idx=4900,
        duration_steps=500,  # 10s pre, 30s outage, 10s post
        outage_start_rel_steps=100,
        outage_duration_steps=300,  # 30.0s
    ),
    "B4_OUTAGE_60S": ScenarioDefinition(
        scenario_id="B4_OUTAGE_60S",
        name="B4: 60s GNSS Outage (Standard Benchmark)",
        description="60-second synthetic GNSS blackout benchmark on real IO-VNBD S1 trajectory.",
        session_file="Categorised_S1.npz",
        start_idx=4900,
        duration_steps=800,  # 10s pre, 60s outage, 10s post
        outage_start_rel_steps=100,
        outage_duration_steps=600,  # 60.0s
    ),
    "B5_OUTAGE_120S": ScenarioDefinition(
        scenario_id="B5_OUTAGE_120S",
        name="B5: 120s Extended Outage Stress Test",
        description="Extended 2-minute synthetic GNSS blackout stress test on real IO-VNBD S1 trajectory.",
        session_file="Categorised_S1.npz",
        start_idx=4900,
        duration_steps=1400,  # 10s pre, 120s outage, 10s post
        outage_start_rel_steps=100,
        outage_duration_steps=1200,  # 120.0s
    ),
    "B6_OUTAGE_300S": ScenarioDefinition(
        scenario_id="B6_OUTAGE_300S",
        name="B6: 300s Extreme Outage Stress Test",
        description="Extreme 5-minute synthetic GNSS blackout stress test on real IO-VNBD S1 trajectory.",
        session_file="Categorised_S1.npz",
        start_idx=4900,
        duration_steps=3200,  # 10s pre, 300s outage, 10s post
        outage_start_rel_steps=100,
        outage_duration_steps=3000,  # 300.0s
    ),
    "B7_SHARP_TURN": ScenarioDefinition(
        scenario_id="B7_SHARP_TURN",
        name="B7: Sharp Turn / High Dynamics",
        description="Cornering dynamics with significant yaw rate on IO-VNBD S1.",
        session_file="Categorised_S1.npz",
        start_idx=10500,
        duration_steps=400,  # 40s
        outage_start_rel_steps=None,
        outage_duration_steps=None,
    ),
    "B8_STOP_AND_GO": ScenarioDefinition(
        scenario_id="B8_STOP_AND_GO",
        name="B8: Stop-and-Go Driving",
        description="Stationary standstill periods evaluating ZUPT zero pinning on IO-VNBD S1.",
        session_file="Categorised_S1.npz",
        start_idx=4500,
        duration_steps=300,  # 30s
        outage_start_rel_steps=None,
        outage_duration_steps=None,
    ),
    "B9_PARALLEL_ROADS": ScenarioDefinition(
        scenario_id="B9_PARALLEL_ROADS",
        name="B9: Parallel Road Ambiguity",
        description="Dual-carriageway segment evaluating ambiguity margin safeguard on IO-VNBD S1.",
        session_file="Categorised_S1.npz",
        start_idx=4900,
        duration_steps=600,
        outage_start_rel_steps=None,
        outage_duration_steps=None,
    ),
    "B10_ZERO_COVERAGE": ScenarioDefinition(
        scenario_id="B10_ZERO_COVERAGE",
        name="B10: Zero Map Coverage",
        description="Off-map trajectory with road graph disabled, testing 100% graceful fallback to raw estimator coordinates.",
        session_file="Categorised_S1.npz",
        start_idx=4900,
        duration_steps=600,
        is_synthetic=False,
    ),
    "B11_RECOVERY": ScenarioDefinition(
        scenario_id="B11_RECOVERY",
        name="B11: GNSS Recovery & Reacquisition",
        description="Recovery phase following 60s outage testing bounded convergence on IO-VNBD S1.",
        session_file="Categorised_S1.npz",
        start_idx=4900,
        duration_steps=800,
        outage_start_rel_steps=100,
        outage_duration_steps=600,
    ),
    "B12_REAL_OUTAGE": ScenarioDefinition(
        scenario_id="B12_REAL_OUTAGE",
        name="B12: Real IO-VNBD S3c IMU with Synthetic GNSS Blackout",
        description="Synthetic GNSS blackout injected into IO-VNBD session S3c real IMU trajectory.",
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
        noise_std_gyro: float = 0.002,
        warmup_steps: int = 100,  # 10 s of GNSS-aided warmup before blackout
    ) -> Dict[str, Any]:
        """Benchmark 1: 50m GNSS-denied travel in under 1 min (Target: drift < 5m).

        100 sample (10 s) GNSS warmup precedes the blackout so the ESKF is
        fully initialized. The blackout portion is 500 steps (50 s) over which
        exactly 50 m are travelled, satisfying the <1 min budget.
        """
        blackout_steps = 500  # 50 s @ 1 m/s  =>  50 m
        n_steps = warmup_steps + blackout_steps
        dt = 1.0 / rate_hz
        blackout_duration_s = blackout_steps * dt  # 50.0 s
        total_duration_s = n_steps * dt            # 60.0 s

        t = np.arange(n_steps, dtype=np.float64) * dt
        # Straight motion along East: x(t) = speed * t, y(t) = 0
        ref_x = speed_mps * t
        ref_y = np.zeros_like(t)
        ref_z = np.zeros_like(t)

        ref_pos_enu = np.column_stack([ref_x, ref_y, ref_z])
        ref_vel_enu = np.column_stack([np.full(n_steps, speed_mps), np.zeros(n_steps), np.zeros(n_steps)])
        ref_headings_rad = np.full(n_steps, 0.5 * math.pi)  # East

        # Specific force in body frame (x forward, z up):
        rng = np.random.RandomState(42)
        f_m_v = np.zeros((n_steps, 3))
        f_m_v[:, 0] = accel_bias[0] + rng.normal(0.0, noise_std_acc, n_steps)
        f_m_v[:, 1] = accel_bias[1] + rng.normal(0.0, noise_std_acc, n_steps)
        f_m_v[:, 2] = STANDARD_GRAVITY_MPS2 + accel_bias[2] + rng.normal(0.0, noise_std_acc, n_steps)

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

        blackout_distance_m = float(speed_mps * blackout_duration_s)  # = 50.0

        return {
            "name": "Benchmark 1 (Synthetic 50m / <1 min)",
            "rate_hz": rate_hz,
            "duration_s": blackout_duration_s,
            "blackout_duration_s": blackout_duration_s,
            "total_duration_s": total_duration_s,
            "warmup_steps": warmup_steps,
            "distance_travelled_m": blackout_distance_m,
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
            "calib_accel_bias": np.array(accel_bias),
            "outage_start_step": warmup_steps,
            "outage_duration_steps": blackout_steps,
            "target_max_drift_m": 5.0,
            "is_synthetic": True,
        }

    @staticmethod
    def generate_1km_60kmh_benchmark(
        rate_hz: float = 10.0,
        speed_mps: float = 1000.0 / 60.0,  # Exact 60.0 km/h -> 1000m in 60s
        accel_bias: Tuple[float, float, float] = (0.015, -0.01, 0.02),
        gyro_bias: Tuple[float, float, float] = (0.0005, 0.0005, -0.001),
        noise_std_acc: float = 0.05,
        noise_std_gyro: float = 0.005,
        warmup_steps: int = 100,  # 10 s GNSS-aided warmup before blackout
    ) -> Dict[str, Any]:
        """Benchmark 2 (Controlled Synthetic): 1 km travel at 60 km/h in 60s blackout (Target: drift < 100m).

        100 sample (10 s) GNSS warmup precedes the 60 s / 1000 m blackout so the
        estimator is correctly initialised. The blackout portion is exactly 600 steps (60 s),
        giving exactly 1000 m during blackout travel at (1000/60) m/s (= 60 km/h).
        """
        blackout_steps = 600  # 60 s @ (1000/60) m/s => 1000 m
        n_steps = warmup_steps + blackout_steps
        dt = 1.0 / rate_hz
        blackout_duration_s = blackout_steps * dt  # 60.0 s
        total_duration_s = n_steps * dt            # 70.0 s

        t = np.arange(n_steps, dtype=np.float64) * dt
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
        f_m_v[:, 2] = STANDARD_GRAVITY_MPS2 + accel_bias[2] + rng.normal(0.0, noise_std_acc, n_steps)

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

        blackout_distance_m = float(speed_mps * blackout_duration_s)  # = 1000.0

        return {
            "name": "Benchmark 2 (Controlled Synthetic 1km / 60km/h / 60s Outage)",
            "rate_hz": rate_hz,
            "duration_s": blackout_duration_s,
            "blackout_duration_s": blackout_duration_s,
            "total_duration_s": total_duration_s,
            "warmup_steps": warmup_steps,
            "distance_travelled_m": blackout_distance_m,
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
            "calib_accel_bias": np.array(accel_bias),
            "outage_start_step": warmup_steps,
            "outage_duration_steps": blackout_steps,
            "target_max_drift_m": 100.0,
            "is_synthetic": True,
        }
