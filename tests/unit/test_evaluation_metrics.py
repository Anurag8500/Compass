"""Unit tests for Phase 13 evaluation metrics, scenarios, and ablation modules."""

from __future__ import annotations

import math
import numpy as np
import pytest

from navigation.evaluation.metrics import (
    compute_position_metrics,
    compute_outage_metrics,
    compute_velocity_metrics,
    compute_heading_metrics,
    compute_trajectory_metrics,
)
from navigation.evaluation.scenarios import (
    AXIS_B_SCENARIOS,
    SyntheticTrajectoryGenerator,
)
from navigation.evaluation.aggregation import aggregate_scenario_results


def test_position_metrics_exact_values() -> None:
    # 3 points with constant 3m East error and 4m North error -> 2D error = 5m
    ref = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    est = np.array([[3.0, 4.0, 0.0], [13.0, 4.0, 0.0], [23.0, 4.0, 0.0]])

    m = compute_position_metrics(est, ref)
    assert m.rmse_2d == pytest.approx(5.0, abs=1e-6)
    assert m.mean_error == pytest.approx(5.0, abs=1e-6)
    assert m.max_error == pytest.approx(5.0, abs=1e-6)
    assert m.final_position_error == pytest.approx(5.0, abs=1e-6)
    assert m.ate == pytest.approx(5.0, abs=1e-6)


def test_outage_metrics_exact_values() -> None:
    # Vehicle travels 100m straight East: (0,0) -> (100,0) in 10s
    times = np.linspace(0.0, 10.0, 11)
    ref = np.column_stack([np.linspace(0.0, 100.0, 11), np.zeros(11), np.zeros(11)])
    # Estimator drifts by 8m North at the end
    est = ref.copy()
    est[-1, 1] = 8.0

    dr = compute_outage_metrics(est, ref, start_step=0, end_step=10, times_s=times)
    assert dr.distance_travelled_m == pytest.approx(100.0, abs=1e-3)
    assert dr.final_outage_drift_m == pytest.approx(8.0, abs=1e-3)
    assert dr.drift_percentage == pytest.approx(8.0, abs=1e-3)
    assert dr.sih_10pct_passed is True  # 8% < 10%

    # If drift was 12m -> 12% -> Failed
    est[-1, 1] = 12.0
    dr_fail = compute_outage_metrics(est, ref, start_step=0, end_step=10, times_s=times)
    assert dr_fail.drift_percentage == pytest.approx(12.0, abs=1e-3)
    assert dr_fail.sih_10pct_passed is False


def test_synthetic_50m_and_1km_generators() -> None:
    gen = SyntheticTrajectoryGenerator()
    b1 = gen.generate_50m_benchmark()
    assert b1["distance_travelled_m"] == pytest.approx(50.0, abs=0.5)
    assert b1["blackout_duration_s"] == pytest.approx(50.0, abs=0.1)
    assert b1["total_duration_s"] == pytest.approx(60.0, abs=0.1)

    b2 = gen.generate_1km_60kmh_benchmark()
    assert b2["distance_travelled_m"] == pytest.approx(1000.0, abs=1.0)
    assert b2["blackout_duration_s"] == pytest.approx(60.0, abs=0.1)
    assert b2["total_duration_s"] == pytest.approx(70.0, abs=0.1)
    assert b2["is_synthetic"] is True


def test_aggregation_stats() -> None:
    sample_reports = [
        {
            "position": {"rmse_2d": 1.5},
            "dead_reckoning": {"final_outage_drift_m": 7.0, "drift_percentage": 5.0},
            "map_matching": {"snap_rate_pct": 98.0},
        },
        {
            "position": {"rmse_2d": 2.5},
            "dead_reckoning": {"final_outage_drift_m": 80.0, "drift_percentage": 20.0},
            "map_matching": {"snap_rate_pct": 50.0},
        },
    ]
    agg = aggregate_scenario_results(sample_reports)
    assert agg.total_scenarios_evaluated == 2
    assert agg.continuous_rmse_stats.mean == pytest.approx(2.0, abs=1e-6)
    assert agg.sih_pass_rate_pct == pytest.approx(50.0, abs=1e-6)  # 1 pass (5%), 1 fail (20%)
