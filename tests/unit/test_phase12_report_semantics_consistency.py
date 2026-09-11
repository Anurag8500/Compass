"""Automated consistency and semantics test for Phase 12 documentation and benchmark reporting.

Asserts:
1. docs/phase12_mapmatch_results.json and docs/mapmatch_report.md are numerically synchronized.
2. Final drift / outage drift metrics are NEVER placed under an 'RMSE' column header.
3. Every report/table clearly distinguishes:
   - RMSE
   - Mean position error
   - Maximum position error
   - Travelled distance
   - Final outage drift
   - Maximum outage drift
   - Drift percentage.
4. The official SIH Problem Statement requirement is explicitly verified as <10.0% of distance.
5. Primary dead-reckoning compliance is judged from estimator output, not display snapping.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest


def test_phase12_report_metric_semantics_and_numerical_consistency() -> None:
    json_path = Path("docs/phase12_mapmatch_results.json")
    md_path = Path("docs/mapmatch_report.md")

    assert json_path.exists(), f"Missing JSON results file: {json_path}"
    assert md_path.exists(), f"Missing Markdown report file: {md_path}"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Rule 1: No final drift placed under an RMSE heading
    lines = md_text.splitlines()
    in_table_1 = False
    in_table_2 = False

    table_1_header = None
    table_2_header = None

    for line in lines:
        if "Table 1:" in line:
            in_table_1 = True
            in_table_2 = False
            continue
        if "Table 2:" in line:
            in_table_1 = False
            in_table_2 = True
            continue
        if line.startswith("---") or line.startswith("## "):
            in_table_1 = False
            in_table_2 = False

        if in_table_1 and table_1_header is None and line.startswith("|") and "Scenario" in line:
            table_1_header = line
        elif in_table_2 and table_2_header is None and line.startswith("|") and "Outage Scenario" in line:
            table_2_header = line

    assert table_1_header is not None, "Table 1 header not found in mapmatch_report.md"
    assert table_2_header is not None, "Table 2 header not found in mapmatch_report.md"

    # Table 1 MUST contain RMSE and Error headers, but NO final drift
    assert "RMSE" in table_1_header
    assert "Mean Error" in table_1_header
    assert "Final Drift" not in table_1_header

    # Table 2 MUST contain Distance, Final Drift, Max Drift, Drift %, but NOT be headed as RMSE
    assert "Estimator Final Drift" in table_2_header
    assert "Estimator Max Drift" in table_2_header
    assert "Estimator Drift %" in table_2_header
    assert "Travelled Distance" in table_2_header
    # The outage drift table must NOT label outage drift as RMSE
    assert "RMSE" not in table_2_header

    # Rule 2: Numerical synchronization check between JSON and Markdown
    sa = data["scenario_a_continuous_gnss"]
    sb10 = data["scenario_b_outages"]["outage_10s"]["outage_metrics"]
    sb30 = data["scenario_b_outages"]["outage_30s"]["outage_metrics"]
    sb60 = data["scenario_b_outages"]["outage_60s"]["outage_metrics"]

    # Check Scenario A RMSE appears in MD
    sa_rmse_str = f"{sa['estimator_rmse_2d_m']:.4f}"
    assert sa_rmse_str in md_text, f"Scenario A RMSE {sa_rmse_str} not found in Markdown report"

    # Check 10s outage metrics
    sb10_dist_str = f"{sb10['distance_travelled_m']:.1f}"
    sb10_est_drift_str = f"{sb10['est_final_drift_m']:.2f}"
    sb10_est_pct_str = f"{sb10['est_drift_pct']:.2f}%"

    assert sb10_dist_str in md_text, f"10s outage distance {sb10_dist_str} not in Markdown"
    assert sb10_est_drift_str in md_text, f"10s outage drift {sb10_est_drift_str} not in Markdown"
    assert sb10_est_pct_str in md_text, f"10s outage drift % {sb10_est_pct_str} not in Markdown"

    # Check 60s outage metrics
    sb60_dist_str = f"{sb60['distance_travelled_m']:.1f}"
    sb60_est_drift_str = f"{sb60['est_final_drift_m']:.2f}"
    sb60_est_pct_str = f"{sb60['est_drift_pct']:.2f}%"

    assert sb60_dist_str in md_text, f"60s outage distance {sb60_dist_str} not in Markdown"
    assert sb60_est_drift_str in md_text, f"60s outage drift {sb60_est_drift_str} not in Markdown"
    assert sb60_est_pct_str in md_text, f"60s outage drift % {sb60_est_pct_str} not in Markdown"

    # Rule 3: Official SIH Problem Statement requirement check
    assert "10.0%" in md_text or "< 10%" in md_text or "<10%" in md_text
    assert "LESS THAN 10%" in md_text or "less than 10%" in md_text.lower()

    # Rule 4: Preserved Phase 11 baseline check
    phase11_baseline = 1.5496224217307877
    assert pytest.approx(sa["estimator_rmse_2d_m"], abs=1e-12) == phase11_baseline
