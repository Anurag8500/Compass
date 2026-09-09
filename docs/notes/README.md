# COMPASS Documentation: Notes & Learning Guide

**Project**: C.O.M.P.A.S.S. (Cognitive Off-grid Machine-learning Positioning And Sensor System)  
**Problem Statement**: SIH 2026 Problem Statement 26168 — ISRO  

---

## Purpose of This Directory

This `docs/notes/` directory contains **fourteen phase documentation Markdown documents** (plus this master README) organized into two complementary sets:

- **Set A: Conceptual Explanations** — What we built and why it works. Explains the mathematics, the terminology, and the design decisions in plain language. Written for someone reading the code for the first time.
- **Set B: Implementation Walkthroughs** — How we actually built each phase. Step-by-step development diaries that trace the code from first design decisions through debugging to final tests.

---

## Quick Navigation

### Set A: Conceptual Explanations
> *Start here if you want to understand WHAT something does and WHY.*

| Document | Phase | Description |
|:---|:---:|:---|
| [Phase 0 — Conceptual](./conceptual/phase_0.md) | 0 | Dataset audit, IO-VNBD forensics, what we discovered |
| [Phase 1 — Conceptual](./conceptual/phase_1.md) | 1 | Schemas, coordinate frames (ENU), quality flags, GNSS FSM |
| [Phase 2 — Conceptual](./conceptual/phase_2.md) | 2 | Sensor synchronization, quality tagging, caching |
| [Phase 3 — Conceptual](./conceptual/phase_3.md) | 3 | Calibration, mounting alignment, dual-stage filter |
| [Phase 4 — Conceptual](./conceptual/phase_4.md) | 4 | Strapdown INS, quaternion propagation, ablation baseline |
| [Phase 5 — Conceptual](./conceptual/phase_5.md) | 5 | ESKF, GNSS/ZUPT fusion, gating, covariance reset |
| [Phase 6 — Conceptual](./conceptual/phase_6.md) | 6 | ML dataset, causal windowing, driver split, normalization |

### Set B: Implementation Walkthroughs
> *Start here if you want to understand HOW we built it, step by step.*

| Document | Phase | Key Topics |
|:---|:---:|:---|
| [Phase 0 — Build Diary](./implementation/phase_0_build.md) | 0 | Inspection script architecture, hash-based branch deduplication, clock drift detection |
| [Phase 1 — Build Diary](./implementation/phase_1_build.md) | 1 | Frozen dataclass design, bitmask system, quaternion norm validation, omit-vs-keep decision |
| [Phase 2 — Build Diary](./implementation/phase_2_build.md) | 2 | CSV parser, relative-elapsed synchronization, heading wraparound fix, VBOX altitude bug |
| [Phase 3 — Build Diary](./implementation/phase_3_build.md) | 3 | Rodrigues degenerate case, zero-phase vs causal, false stationary alarms, gyro bias |
| [Phase 4 — Build Diary](./implementation/phase_4_build.md) | 4 | Quaternion renormalization, oracle initialization, 530 m vertical error diagnosis |
| [Phase 5 — Build Diary](./implementation/phase_5_build.md) | 5 | Right-multiplicative error convention, Joseph form, covariance reset audit, 99.84% improvement |
| [Phase 6 — Build Diary](./implementation/phase_6_build.md) | 6 | Jerk boundary bug, centered-window leakage fix, ddof=1 harmonization, label causality |

---

## The COMPASS Architecture at a Glance

```
IO-VNBD Raw CSV Files (288 files, 72 unique trips)
        |
        v  Phase 0: Forensic Audit
        |  - Encoding: Latin-1; timestamps: ms vs sec since midnight
        |  - Driver D (Y1) excluded: clock drift = -638.5 s
        |
        v  Phase 1: Schemas & Contracts
        |  - Coordinate frame: ENU (East-North-Up), gravity = [0,0,-9.81] m/s^2
        |  - Quaternion: Hamilton [w,x,y,z], vehicle→ENU (R_v^n)
        |  - State: 16-dim nominal, 15-dim error, 15×15 covariance P
        |
        v  Phase 2: Data Pipeline (sync.py, quality_tagger.py)
        |  - 144 S/V pairs synchronized; 142 passed (Y1 flagged twice)
        |  - 10,936 rows omitted (0.51%); 0 raw bytes changed
        |  - Cached to 144 compressed .npz files
        |
        v  Phase 3: IMU Preprocessing (calibration, alignment, filtering)
        |  - Gyro bias estimated from stationary window (≥50 samples)
        |  - R_b^v: Rodrigues rotation, gravity vector → vehicle vertical
        |  - Dual-stage: 3-sample median + 4th-order Butterworth (fc=3 Hz)
        |
        v  Phase 4: Strapdown INS (ins/attitude.py, ins/propagation.py)
        |  - Quaternion attitude propagation (renormalized every step)
        |  - Velocity and position via constant-acceleration discrete kinematic propagation
        |  - Ablation result: 60 s → 3,249.32 m horizontal drift
        |
        v  Phase 5: ESKF + GNSS + ZUPT (eskf/)
        |  - 15-state ESKF, right-multiplicative body-frame attitude error
        |  - Joseph-form covariance update + reset Jacobian after injection
        |  - GNSS: 3D position + 2D horizontal velocity (chi-sq gating)
        |  - ZUPT: zero-velocity lock during standstill
        |  - Result: 60 s → 5.27 m horizontal error (99.84% improvement)
        |
        v  Phase 6: ML Dataset Construction (ml/data/)
           - 9-channel feature tensors (20×9) at canonical 10 Hz
           - Strict causality: windows end at decision epoch T
           - Driver-level split: E→Train, B→Val, A→Test, D→Excluded
           - Training-only normalization (ddof=1 Bessel's correction)
           - 399,714 total windows; 371,472 valid
```

---

## Key Numbers to Remember

| Quantity | Value |
|:---|:---:|
| Total raw CSV files | 288 |
| Unique physical trips | 72 |
| Drivers | 4 (A, B, D, E) |
| Trips passing sync policy | 71 pairs (Y1 excluded) |
| Total driving duration | 59.48 hours |
| Synchronized rows | 2,141,490 |
| Validated rows | 2,130,554 |
| Omitted rows | 10,936 (0.51%) |
| Phase 4 drift (60 s) | 3,249.32 m horizontal |
| Phase 5 error (60 s) | 5.27 m horizontal |
| Phase 5 improvement | 99.84% |
| Phase 6 total windows | 399,714 |
| Phase 6 valid windows | 371,472 |
| Tests passing at completion | 249 |
| Feature tensor shape | (20, 9) |
| Feature time span | 2.0 seconds at 10 Hz |
| Window stride | 5 samples = 0.5 seconds |

---

## Common Misconceptions (Corrected)

> [!WARNING]
> These are things that are easy to get wrong when reading the code or documentation.

1. **The navigation frame is ENU, not NED.** Gravity is $[0, 0, -9.80665]^T\,\text{m/s}^2$, not $[0, 0, +9.80665]^T$. Vertical "Up" is positive-Z.

2. **Vehicle frame is FLU (Forward-Lateral-Up), not FRD.** $+Z_v$ points toward the vehicle roof. At rest on level ground, $f_z^v \approx +9.81\,\text{m/s}^2$ (upward reaction force from the road).

3. **Gravity is NOT removed in the ML features.** Only the strapdown INS (Phase 4/5) removes gravity. The ML input features intentionally preserve gravity-containing vehicle-frame specific force as an input.

4. **The covariance $P$ is 15×15, not 16×16.** The quaternion contributes 4 parameters but only 3 degrees of freedom. The error state uses a 3-element rotation vector $\delta\boldsymbol{\theta}$.

5. **GNSS degradation is NOT a 4th FSM state.** It is a continuous `trust_score ∈ [0,1]` on `GNSSSample`. The ESKF's chi-square gate handles implausible fixes regardless of trust_score.

6. **VBOX altitude interpretation and frozen cache scaling.** In the raw IO-VNBD CSVs, the altitude header label says `(km)`, but the numerical values empirically behave as meters. The Phase 2 pipeline multiplied these numbers by 1000 following the literal column label, resulting in cache values scaled 1000× larger than physical meters. To preserve immutability of the frozen SHA-256 cache, Phase 4 and all downstream consumers compensate for this scaling at load time by dividing by 1000.0 rather than mutating the cache.

7. **Trip Y1 (Driver D) is excluded — but it is still counted in the dataset.** It is in the manifest and cache, flagged with `sync_passed=False`. It contributes 0 windows to any ML split and 0 trajectories to any navigation evaluation.

---

## Current Project Status: Implemented vs Planned

The COMPASS project architecture spans **twenty defined phases (Phase 0 – Phase 19)**, as specified in `FINAL_IMPLEMENTATION_PLAN_SIH26168.md`. To maintain absolute technical accuracy when reviewing the codebase, tests, or documentation, the exact boundary between completed implementation and future roadmap phases is summarized below:

### Fully Implemented, Verified, and Tested (Phases 0–6)
All 249 unit and integration tests across these phases pass deterministically:
- **Phase 0 (Discovery & Forensics)**: Complete audit of 288 CSV files (72 trips) across 4 drivers; identification of Latin-1 encoding, time-since-midnight schemas, and exclusion of Driver D (Trip Y1) due to -638.5 s clock drift.
- **Phase 1 (Foundations & Contracts)**: Immutable schemas (`IMUSample`, `GNSSSample`, `NominalState`, `ErrorState`), 4 coordinate frame definitions ($b, v, n, e$) in local ENU navigation ($\mathbf{g}^n = [0, 0, -9.80665]^T\,\text{m/s}^2$), bitmask quality flags, and 3-state GNSS FSM.
- **Phase 2 (Ingestion & Synchronization)**: Stream parsing of 288 CSVs, relative-elapsed synchronization to 10 Hz master smartphone nanosecond grid, circular unwrapping interpolation, quality tagging (2,130,554 valid rows, 10,936 omitted), and SHA-256 frozen caching in 144 `.npz` files.
- **Phase 3 (Preprocessing & Alignment)**: Stationary gyroscope bias calibration (13.1× drift reduction), Rodrigues mounting tilt rotation matrix $R_b^v$ (phone body $\to$ vehicle FLU), and dual-stage vibration filtering (3-sample median + 4th-order zero-phase Butterworth, $f_c = 3.0\,\text{Hz}$).
- **Phase 4 (Strapdown INS Engine)**: First-principles Newtonian strapdown kinematic propagation; Hamilton quaternion attitude update, specific force rotation to ENU, gravity compensation, and constant-acceleration discrete kinematic double integration. Baseline open-loop drift benchmark: $3,249.32\,\text{m}$ in 60 s due to cubic gravity leakage ($p(t) \approx \frac{1}{6} g b_g t^3$).
- **Phase 5 (ESKF & Classical Fusion)**: 15-state Error-State Kalman Filter with right-multiplicative body-frame attitude error, analytical Jacobian state transition $F_d$, continuous-to-discrete $Q_d$, Joseph-form covariance update, Chi-square innovation gating, standstill ZUPT lock, and authoritative error-state covariance reset $P^+ = J_{\text{reset}} P_{\text{updated}} J_{\text{reset}}^T$. Highway benchmark error: reduced to $5.27\,\text{m}$ ($99.84\%$ reduction).
- **Phase 6 (ML Dataset Engineering)**: Canonical $(20, 9)$ feature tensor extraction at 10 Hz ($2.0\,\text{s}$ window, $0.5\,\text{s}$ stride), intact gravity preservation, causal backward jerk $\|\dot{\mathbf{f}}^v\|$, driver-level partitioning (Driver E $\to$ Train, B $\to$ Val, A $\to$ Test), train-only feature normalization (`ddof=1`), and hard-boundary causal median-filtered speed labels $v(T)$. Extracted 399,714 total windows (371,472 valid).

### Planned Roadmap Phases (Phases 7–19)
These phases build directly on top of the Phase 0–6 foundation, executing the 20-phase roadmap of `FINAL_IMPLEMENTATION_PLAN_SIH26168.md`:
- **Phase 7 (VelocityNet: Training, Validation, Export, Acceptance Gate)**: 2-layer GRU neural network (`GRU(9→64, 2L) → Dense(64→32, ReLU) → Dense(32→2)`) predicting forward speed ($v_{\text{forward}}$) and heteroscedastic log-variance ($\log \sigma_v^2$), trained on `train.npz` using causal median labels, validated on `validation.npz`, and exported to ONNX/LiteRT.
- **Phase 8 (BiasNet: Label Generation, Training, Validation, Export, Acceptance Gate)**: 2-layer GRU (`GRU(9→48, 2L) → Dense(48→24, ReLU) → Dense(24→12)`) predicting 6-axis residual bias corrections ($\Delta \mathbf{b}_a, \Delta \mathbf{b}_g$) and log-variance from computed window-level propagation-vs-GPS optimization labels; modular fallback allows clean disabling if unneeded.
- **Phase 9 (ML → ESKF Integration + Full ML-Augmented Offline Replay)**: Asynchronous measurement adapters converting VelocityNet speed and BiasNet corrections into $(z, H, R)$ innovations for ESKF ingestion; measurement-only injection with no learned weights inside the filter.
- **Phase 10 (GNSS Quality/Trust, Outage Detection, Mode FSM, Recovery)**: Multi-feature GNSS quality scoring (`trust_score` $\in [0, 1]$), automated outage detection, 3-state operational FSM (`GNSS_AIDED`, `DR_ONLY`, `REACQUIRING`), and smooth covariance re-convergence upon satellite reacquisition.
- **Phase 11 (Kinematic Constraints: NHC & Gated ZUPT Integration)**: Vehicle non-holonomic virtual velocity constraints ($v_{\text{lateral}} \approx 0, v_{\text{vertical}} \approx 0$) and Chi-square gated zero-velocity updates to constrain dead-reckoning drift during GNSS outages.
- **Phase 12 (Downstream Map Matching & Trajectory Snapping)**: Offline OpenStreetMap (OSM) extraction and Hidden Markov Model (HMM) road-network snapping; strictly downstream-only (no feedback into the ESKF state or covariance).
- **Phase 13 (Full Offline Replay Integration Test + 3-Axis Evaluation & Ablation Suite)**: Multi-trip benchmark suite running across all test trajectories, measuring 3-axis performance across full ablation ladder (INS-only vs INS+NHC vs INS+VelocityNet vs Full System).
- **Phase 14 (Navigation Core Packaging)**: Hardened Python reference navigation engine packaging (`NavigationCore`) providing a single entry-point API and behavioral oracle.
- **Phase 15 (Edge Adapter)**: Lightweight C++/Python runtime CLI adapter interfacing with physical serial/USB IMU hardware, sensor ring-buffers, and ONNX Runtime.
- **Phase 16 (Android Adapter/UI)**: Kotlin Android application with background sensor service, LiteRT (TFLite) neural inference, real-time map UI, and behaviorally equivalent Kotlin navigation core.
- **Phase 17 (Parity & Performance Validation)**: Cross-platform parity test suite ensuring the Android/Kotlin and Edge runtimes match the authoritative Python reference within strict numerical tolerances and latency budgets.
- **Phase 18 (Field Data Collection & Self-Collected Validation)**: Live vehicle road test data collection protocol using consumer smartphones and OBD-II/RTK verification rigs to validate real-world transfer.
- **Phase 19 (Final SIH Demo Readiness)**: Turnkey demonstration scripts, real-time replay dashboards, simulated GNSS denial injection toggles, and live presentation tooling for the SIH 2026 ISRO evaluation.

---

## The Journey of One Measurement Through COMPASS

To see how the entire system functions as a unified pipeline, follow the step-by-step physical journey of a single sensor reading: an accelerometer and gyroscope measurement taken by a smartphone resting on a car dashboard while driving along the M1 motorway.

```
+---------------------------------------------------------------------------------------------------+
|                            THE JOURNEY OF ONE MEASUREMENT                                         |
+---------------------------------------------------------------------------------------------------+
| [1] Physical Sensor Ingestion (Phase 0 & 1)                                                       |
|     Smartphone MEMS registers a specific force f_raw = [0.12, 4.85, 8.42] m/s^2                   |
|     and angular turn rate omega_raw = [0.002, -0.015, 0.041] rad/s.                               |
|     Parsed from Latin-1 CSV with elapsed timestamp t = 142.503 s into an IMUSample dataclass.     |
+---------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+---------------------------------------------------------------------------------------------------+
| [2] Master Nanosecond Synchronization & Quality Tagging (Phase 2)                                 |
|     Sample timestamp is aligned to the 10.0 Hz master grid: t_k = 142,500,000,000 ns.             |
|     Quality tagger checks validity: finite numbers, chronological order, within physical range.   |
|     Tagged with non-destructive bitmask FLAG_OK (0x0000). Cached to compressed .npz archive.      |
+---------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+---------------------------------------------------------------------------------------------------+
| [3] Calibration, Mounting Tilt Alignment & Denoising (Phase 3)                                    |
|     - Gyroscope Bias Subtraction: omega_cal = omega_raw - b_g_static.                             |
|     - Rodrigues Mounting Tilt Rotation (R_b^v): The phone is tilted on the dashboard!             |
|       Rotates vector from phone body frame (b) into vehicle FLU frame (v):                        |
|       f^v = R_b^v * f_raw  --> now f_z^v points straight up through vehicle roof (~9.81 m/s^2).   |
|     - Dual-Stage Filtering: 3-sample median filter strips spike outliers; 4th-order zero-phase   |
|       Butterworth filter (fc = 3.0 Hz) strips high-frequency engine acoustic vibration.           |
+---------------------------------------------------------------------------------------------------+
                                                  │
                         ┌────────────────────────┴────────────────────────┐
                         ▼                                                 ▼
+---------------------------------------------------+     +-----------------------------------------+
| [4A] Classical Navigation Backbone (Phases 4 & 5)  |     | [4B] Machine Learning Pipeline (Phase 6)|
|                                                   |     |                                         |
| 1. Strapdown INS Propagation (Phase 4):           |     | 1. Feature Engineering:                 |
|    - Quaternion kinematic update:                 |     |    - Keeps gravity intact in f_z^v.     |
|      q[k+1] = q[k] ⊗ Δq(omega^v * Δt).            |     |    - Computes causal backward jerk:     |
|    - Specific force rotated to local ENU frame:   |     |      ||f_dot^v|| = ||(f_k - f_{k-1})/Δt|||
|      f^n = R_v^n * f^v.                           |     |    - Forms 9-channel vector.            |
|    - Gravity compensation:                        |     |                                         |
|      a^n = f^n + [0, 0, -9.80665]^T.              |     | 2. Causal Windowing:                    |
|    - Discrete kinematic propagation:              |     |    - Collects 20 consecutive samples    |
|      v[k+1] = v[k] + a^n * Δt                     |     |      ending at epoch T (2.0 s history). |
|      p[k+1] = p[k] + v[k]*Δt + 0.5*a^n*Δt^2.      |     |    - Tensor X ∈ R^(20 × 9).             |
|                                                   |     |                                         |
| 2. ESKF Error Correction (Phase 5):               |     | 3. Split Isolation & Normalization:     |
|    - Error state predicted: δx[k+1] = F_d * δx.   |     |    - Assigned by Driver ID.             |
|    - Covariance propagated: P = F_d * P * F_d^T+Q.|     |    - Normalized using training-only     |
|    - GNSS fix arrives:                            |     |      parameters (ddof=1).               |
|      Innovation computed: r = z_GNSS - h(x_nom).  |     |                                         |
|    - Chi-square gate: r^T * S^(-1) * r < γ (pass).|     | 4. Labeling:                            |
|    - Kalman gain K computed; error δx estimated.  |     |    - Ground-truth speed at T: v(T).     |
|    - Error injected: nominal state corrected!     |     |    - Causal 3-sample median filtered    |
|    - Covariance reset: P^+ = J_reset * P * J_r^T. |     |      with hard invalid boundaries.      |
|    - Result: Position error bounded to ~5 meters. |     |    - Ready on disk for Phase 7/8.       |
+---------------------------------------------------+     +-----------------------------------------+
```

---

## Authoritative Project Documents

These three documents are the overall architecture specification:

| Document | Purpose |
|:---|:---|
| [`FINAL_MASTER_PLAN_SIH26168.md`](file:///d:/Hackathon/Compass/FINAL_MASTER_PLAN_SIH26168.md) | Overall project architecture and phase roadmap |
| [`EndToEnd_Trace_SIH26168.md`](file:///d:/Hackathon/Compass/EndToEnd_Trace_SIH26168.md) | End-to-end data flow from sensor to output |
| [`FINAL_IMPLEMENTATION_PLAN_SIH26168.md`](file:///d:/Hackathon/Compass/FINAL_IMPLEMENTATION_PLAN_SIH26168.md) | Detailed implementation specification per phase |

The authoritative execution reports (containing actual measured numbers) live in `docs/`:

| Report | Contents |
|:---|:---|
| [`docs/iovnbd_inspection_report.md`](../iovnbd_inspection_report.md) | Phase 0 forensic audit results |
| [`docs/data_quality_report.md`](../data_quality_report.md) | Phase 2 pipeline execution metrics |
| [`docs/preprocessing_phase3_report.md`](../preprocessing_phase3_report.md) | Phase 3 preprocessing results |
| [`docs/ablation_stage1_report.md`](../ablation_stage1_report.md) | Phase 4 open-loop drift measurement |
| [`docs/eskf_gnss_baseline_report.md`](../eskf_gnss_baseline_report.md) | Phase 5 ESKF+GNSS validation results |
| [`docs/ml_dataset_v1_report.md`](../ml_dataset_v1_report.md) | Phase 6 ML dataset statistics |
