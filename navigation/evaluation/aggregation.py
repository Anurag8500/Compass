"""Cross-trip, multi-driver, and scenario aggregation library (Phase 13).

Aggregates:
- Multi-trip performance (IO-VNBD Sessions S1, S2, S3a, S3c, S4)
- Driver-stratified metrics
- Multi-scenario summary statistics (mean, median, P95, min, max, std)
- SIH Problem Statement compliance rate (% of scenarios achieving < 10% drift)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence
import numpy as np


@dataclass(frozen=True)
class SummaryStats:
    """Summary statistics across multiple evaluation samples."""
    count: int
    mean: float
    std: float
    median: float
    p95: float
    min_val: float
    max_val: float

    @classmethod
    def from_values(cls, values: Sequence[float]) -> SummaryStats:
        arr = np.asarray(values, dtype=np.float64)
        if len(arr) == 0:
            return cls(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return cls(
            count=len(arr),
            mean=float(np.mean(arr)),
            std=float(np.std(arr)),
            median=float(np.median(arr)),
            p95=float(np.percentile(arr, 95)),
            min_val=float(np.min(arr)),
            max_val=float(np.max(arr)),
        )


@dataclass(frozen=True)
class AggregatedEvaluationReport:
    """Consolidated multi-trip, multi-scenario evaluation results."""
    continuous_rmse_stats: SummaryStats
    outage_drift_stats: SummaryStats
    outage_drift_pct_stats: SummaryStats
    snap_rate_stats: SummaryStats
    sih_pass_rate_pct: float
    total_scenarios_evaluated: int
    total_passed_scenarios: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def aggregate_scenario_results(
    scenario_metrics: Sequence[Dict[str, Any]],
) -> AggregatedEvaluationReport:
    """Aggregate metric results across multiple scenario runs."""
    rmse_list: List[float] = []
    drift_list: List[float] = []
    drift_pct_list: List[float] = []
    snap_rate_list: List[float] = []
    passed_count = 0
    total_outages = 0

    for m in scenario_metrics:
        # Check continuous RMSE
        pos = m.get("position", {})
        if "rmse_2d" in pos:
            rmse_list.append(pos["rmse_2d"])

        # Check dead reckoning metrics
        dr = m.get("dead_reckoning")
        if dr:
            total_outages += 1
            drift_list.append(dr["final_outage_drift_m"])
            drift_pct_list.append(dr["drift_percentage"])
            if dr["drift_percentage"] < 10.0:
                passed_count += 1

        # Check map matching
        mm = m.get("map_matching")
        if mm:
            snap_rate_list.append(mm["snap_rate_pct"])

    pass_rate = (passed_count / max(total_outages, 1)) * 100.0 if total_outages > 0 else 100.0

    return AggregatedEvaluationReport(
        continuous_rmse_stats=SummaryStats.from_values(rmse_list),
        outage_drift_stats=SummaryStats.from_values(drift_list),
        outage_drift_pct_stats=SummaryStats.from_values(drift_pct_list),
        snap_rate_stats=SummaryStats.from_values(snap_rate_list),
        sih_pass_rate_pct=float(pass_rate),
        total_scenarios_evaluated=len(scenario_metrics),
        total_passed_scenarios=passed_count,
    )
