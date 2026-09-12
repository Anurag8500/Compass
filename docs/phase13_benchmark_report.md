# C.O.M.P.A.S.S. Phase 13 SIH Benchmark Report
## Smart India Hackathon (SIH) Problem Statement 26168 — ISRO

---

## 1. Official Benchmark Requirement

### Official SIH Problem Statement 26168 Requirement:
> **"Dead Reckoning: positional drift must be LESS THAN 10% of the total distance travelled during GNSS blackout."**
>
> Official Examples:
> - $< 5\text{ m}$ drift over $50\text{ m}$ in $< 1\text{ min}$.
> - $< 100\text{ m}$ drift over $1\text{ km}$ at $60\text{ km/h}$ in a GNSS-denied environment.

**Note: The old internal <1.5% target is intentionally NOT referenced in this formal SIH benchmark report. It was a superseded internal aspiration; the ONLY official compliance criterion is the SIH-mandated <10%.**

---

## 2. Evidence Taxonomy

Three distinct, non-interchangeable categories are evaluated and reported SEPARATELY:

| Evidence Category | Description | Cases in this Report |
|---|---|---|
| **FULLY_CONTROLLED_SYNTHETIC** | `SyntheticTrajectoryGenerator` outputs — verifiable 50.00m / 1.0 m/s (Benchmark 1) and 1000.02m / 16.667 m/s = 60 km/h (Benchmark 2). | Synthetic Benchmark 1, Synthetic Benchmark 2 |
| **REAL_IMU_SYNTHETIC_BLACKOUT** | Real IO-VNBD IMU with synthetic GNSS mask injection. Actual distances/speeds are measured from the real trajectory segment; **never** mislabeled as the 1km@60 synthetic. | S1 10s, 30s, 60s, 120s, 300s outages; B11 recovery; B12 extended |
| **REAL_ENVIRONMENTAL_OUTAGE** | Natural GNSS blackout observed in-situ. | *None in current dataset.* |

Averaging across categories is explicitly avoided.

---

## 3. Official SIH Compliance Scorecard (ESTIMATOR only — display snaps excluded)

| Scenario / Benchmark Case | Distance Travelled | Estimator Final Drift | Drift % | Official Threshold | Status |
|---|---|---|---|---|---|
| Synth 50m (<5m target) | 49.9 m | 2.99 m | **5.99%** | < 10.0% | **PASS** |
| Synth 1km / 60km/h (<100m target) | 998.3 m | 93.32 m | **9.35%** | < 10.0% | **PASS** |
| Real S1 10s Blackout | 142.2 m | 1.60 m | **1.13%** | < 10.0% | **PASS** |
| Real S1 30s Blackout | 426.6 m | 38.66 m | **9.06%** | < 10.0% | **PASS** |
| Real S1 60s Blackout | 839.5 m | 116.15 m | **13.83%** | < 10.0% | **FAIL** |
| Real S1 120s Blackout | 1496.2 m | 3192.75 m | **213.39%** | < 10.0% | **FAIL** |

**Official scorecard pass rate (cases with explicit boolean): 4/6 = 66.7%.**

---

## 4. Detailed Benchmark Case Evidence

### 4.1 Synthetic Benchmark 1 (50m in <1 min)
| Property | Value |
|---|---|
| Evidence Category | **FULLY_CONTROLLED_SYNTHETIC** |
| Generator | `SyntheticTrajectoryGenerator.generate_50m_benchmark()` |
| Actual Distance Travelled | **49.90 m** |
| Average Speed | **1.00 m/s (3.6 km/h)** |
| Estimator Final Drift | 2.99 m |
| Estimator Max Drift | 4.27 m |
| Drift % of Distance | **5.99%** |
| Official <10% Status | **PASS** |

### 4.2 Synthetic Benchmark 2 (1 km @ 60 km/h / 60s Blackout)
| Property | Value |
|---|---|
| Evidence Category | **FULLY_CONTROLLED_SYNTHETIC** |
| Generator | `SyntheticTrajectoryGenerator.generate_1km_60kmh_benchmark()` |
| Actual Distance Travelled | **998.33 m** |
| Average Speed | **16.67 m/s (60.0 km/h)** |
| Estimator Final Drift | 93.32 m |
| Estimator Max Drift | 93.32 m |
| Drift % of Distance | **9.35%** |
| Official <10% Status | **PASS** |

### 4.3 Real-Data Synthetic Blackout (10s on IO-VNBD S1)
| Property | Value |
|---|---|
| Evidence Category | **REAL_IMU_SYNTHETIC_BLACKOUT** |
| Dataset | IO-VNBD Session S1 |
| Actual Distance Travelled | **142.21 m** |
| Estimator Final Drift | 1.60 m |
| Estimator Max Drift | 7.11 m |
| Drift % of Distance | **1.13%** |
| Official <10% Status | **PASS** |

### 4.4 Real-Data Synthetic Blackout (60s on IO-VNBD S1 — **NOT** the 1km@60 synthetic)
| Property | Value |
|---|---|
| Evidence Category | **REAL_IMU_SYNTHETIC_BLACKOUT** |
| Dataset | IO-VNBD Session S1 |
| Actual Distance Travelled | **839.54 m** |
| Average Segment Speed | **13.99 m/s = 50.4 km/h** |
| ⚠️ Notice | **NOT** a 1000 m / 60 km/h controlled synthetic benchmark — never conflate with §4.2. |
| Estimator Final Drift | 116.15 m |
| Estimator Max Drift | 134.63 m |
| Drift % of Distance | **13.83%** |
| Official <10% Status | **FAIL** |

---

## 5. Multi-Session Benchmark Generalization

| Statistic | Position 2D RMSE (m) | Drift % |
|---|---|---|
| Mean | 0.000 m | 0.00% |
| Median | 0.000 m | 0.00% |
| P95 | 0.000 m | 0.00% |
| SIH Pass Rate | **0/5 = 0.0%** | |

---

## 6. Strict Downstream Isolation & No Artificial Masking

Map matching operates strictly on the display output. When dead reckoning drifts past the configured catastrophic-snap threshold, map matching safely activates fallback to prevent snapping onto wrong streets. C.O.M.P.A.S.S. transparently evaluates and reports dead-reckoning compliance on the **ESTIMATOR trajectory**, refusing to artificially mask drift through downstream map snapping. Numerical equality of DR-A6 vs DR-A7 drift in §2 of the ablation report numerically confirms zero feedback at double precision.

---

## 7. Reproducibility Audit Status

| Item | Value |
|---|---|
| Git Commit | `d9d2b32de74846d2d51024227ff64dc90ab01f68` |
| Working Tree Dirty | `True` |
| Seed | 42 |
| Determinism Audit (Double Run) | Not yet executed; pending in test suite. |
