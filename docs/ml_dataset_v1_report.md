# Phase 6: ML Dataset Construction & Leakage Audit Report
**Project**: C.O.M.P.A.S.S. (Cognitive Off-grid Machine-learning Positioning And Sensor System)  
**Problem Statement**: SIH 2026 Problem Statement 26168 — ISRO  
**Phase**: Phase 6 — ML Dataset Construction: Windowing, Feature Tensors, Normalization, Driver/File Split, Labels, and BiasNet Exclusion Plumbing  
**Status**: COMPLETE, AUDITED, REPRODUCIBLE, AND FROZEN  
**Dataset Version**: `v1.0` | **Split Version**: `split_v1`  
**Manifest Path**: `data/ml_dataset_v1/dataset_manifest.json`  

---

## Executive Summary

Phase 6 implements the shared, deterministic, leakage-audited ML dataset engineering pipeline that produces canonical $(20, 9)$ input tensors and target labels for future models:
1. **VelocityNet** (Phase 7): Predicts forward speed + uncertainty from vehicle-frame inertial motion.
2. **BiasNet** (Phase 8): Predicts 6-vector IMU bias corrections + uncertainty from vehicle-frame inertial motion.

The feature representation and window extraction logic are **100% shared**: there is zero duplicated preprocessing logic between VelocityNet and BiasNet. The pipeline was executed against all 142 downstream-ready synchronized trips from the IO-VNBD dataset, extracting **399,714 feature windows** (371,472 valid windows) across three strictly isolated splits:

| Split | Assigned Driver | Trips | Synchronized Files | Total Windows | Valid Windows | Duration | Role in Project |
|---|---|---|---|---|---|---|---|
| **TRAIN** | **Driver E** | 64 | 128 | **233,830** | **226,928** | ~32.5 h | Model training |
| **VALIDATION** | **Driver B** | 1 | 2 | **42,388** | **21,080** | ~5.9 h | Hyperparameter tuning / checkpoint selection |
| **TEST** | **Driver A** | 6 | 12 | **123,496** | **123,464** | ~17.2 h | Final held-out evaluation (includes S1 benchmark) |
| **EXCLUDED** | **Driver D** | 1 | 2 | 0 | 0 | ~3.9 h | Flagged in Phase 2 for clock-drift threshold (Y1) |
| **TOTAL** | **4 Drivers** | **72** | **144** | **399,714** | **371,472** | **59.48 h** | Full IO-VNBD synchronized archive |

---

## 1. Canonical Feature Tensor Definition

The canonical input format for all downstream neural networks is fixed at:
- **Frequency**: $10.0\,\text{Hz}$ ($\Delta t = 100\,\text{ms}$)
- **Window Duration**: $2.0\,\text{seconds}$
- **Window Size**: Exactly $20\,\text{samples}$
- **Feature Channels**: Exactly $9\,\text{channels}$
- **Tensor Shape**: $\mathbf{X} \in \mathbb{R}^{20 \times 9}$

### 1.1 The 9 Canonical Feature Channels (Strict Ordering)

All motion channels represent calibrated and aligned motion in the **VEHICLE FRAME** ($v$):

| Index | Channel Name | Description | Physical Units | Derivative Formulation |
|---|---|---|---|---|
| **0** | `f_x_v` | Vehicle forward specific force | $\text{m/s}^2$ | Direct sensor observation ($f_x^v$) |
| **1** | `f_y_v` | Vehicle lateral/right specific force | $\text{m/s}^2$ | Direct sensor observation ($f_y^v$) |
| **2** | `f_z_v` | Vehicle vertical/down specific force | $\text{m/s}^2$ | Direct sensor observation ($f_z^v$) |
| **3** | `omega_x_v` | Vehicle roll angular rate | $\text{rad/s}$ | Direct sensor observation ($\omega_x^v$) |
| **4** | `omega_y_v` | Vehicle pitch angular rate | $\text{rad/s}$ | Direct sensor observation ($\omega_y^v$) |
| **5** | `omega_z_v` | Vehicle yaw angular rate | $\text{rad/s}$ | Direct sensor observation ($\omega_z^v$) |
| **6** | `norm_f_v` | Specific force Euclidean magnitude | $\text{m/s}^2$ | $\|\mathbf{f}^v\| = \sqrt{(f_x^v)^2 + (f_y^v)^2 + (f_z^v)^2}$ |
| **7** | `norm_f_dot_v` | Specific force rate of change (jerk norm) | $\text{m/s}^3$ | $\|\dot{\mathbf{f}}^v\| = \left\|\frac{\mathbf{f}_k - \mathbf{f}_{k-1}}{\Delta t_k}\right\|$ |
| **8** | `norm_omega_v` | Angular velocity Euclidean magnitude | $\text{rad/s}$ | $\|\boldsymbol{\omega}^v\| = \sqrt{(\omega_x^v)^2 + (\omega_y^v)^2 + (\omega_z^v)^2}$ |

### 1.2 Preservation of Specific Force (No Gravity Removal)
Specific force $\mathbf{f}^v$ includes physical reaction to Earth gravity ($\sim 9.81\,\text{m/s}^2$). **Physical gravity is strictly NOT removed in the ML dataset pipeline**. Gravity compensation is the sole responsibility of the downstream ESKF strapdown mechanization ($R_v^n \mathbf{f}^v + \mathbf{g}^n$). Keeping raw specific force in the neural inputs allows networks to observe vehicle pitch/roll inclination directly through gravity projection.

---

## 2. Temporal Windowing & Strict Causality

### 2.1 Windowing Parameters
- **Cadence**: Stride of $0.5\,\text{seconds}$ ($5\,\text{samples}$ at $10\,\text{Hz}$), corresponding to a $2\,\text{Hz}$ inference update rate.
- **Window Extent**: Exactly $20\,\text{samples}$ covering $1.9\,\text{seconds}$ of history ending at the current epoch $T$.

### 2.2 Mathematical Causality Invariant
For an input window ending at timestamp $T$:
$$\text{Window}(T) = [T - 1.9\,\text{s}, \dots, T]$$
- **Zero Lookahead**: No sample with timestamp $t > T$ is ever accessed or pooled.
- **Zero Centered Windows**: The window end coincides with the current decision epoch.
- **Zero Future Padding**: Windows requiring history prior to $t_0$ are not generated ($N < 20$ yields 0 windows).
- **Verified by Unit Test**: In `test_windowing_causality.py`, corrupting future samples at $t > T$ produces bit-identical $(20, 9)$ tensors for the window ending at $T$.

---

## 3. Resampling & Decimation Engine

- **IO-VNBD Baseline**: Operates at nominal $10\,\text{Hz}$ ($\Delta t \approx 100\,\text{ms} \pm 5\,\text{ms}$). The resampling engine verifies this condition and performs an identity pass-through, preserving exact source timestamps and numerical fidelity.
- **Higher-Rate Decimation**: For synthetic or future external sensor streams (e.g. $50\,\text{Hz}$, $100\,\text{Hz}$, $200\,\text{Hz}$ FOG data), the engine establishes a deterministic $10\,\text{Hz}$ grid and pools samples causally within $(t_{k-1}, t_k]$, preventing future sample contamination.

---

## 4. Driver/File Partitioning & Leakage Audit

### 4.1 Split Assignment Policy
Adjacent windows from the same recording are correlated. To prevent data leakage, the split is constructed strictly at the **DRIVER and FILE levels**:
- **Train Split**: All 128 files from **Driver E** (trips `Vfa...`, `Vta...`, `Vtb...`, `Vw...`).
- **Validation Split**: All 2 files from **Driver B** (trip `M`).
- **Test Split**: All 12 files from **Driver A** (trips `S1`, `S2`, `S3a`, `S3b`, `S3c`, `S4`).
- **Excluded**: All 2 files from **Driver D** (trip `Y1`, sync clock drift failure).

### 4.2 Leakage Audit Verification
In `test_leakage_audit.py` and `test_ml_dataset_artifacts.py`, rigorous mathematical assertions verify:
1. $\text{Files}(\text{Train}) \cap \text{Files}(\text{Val}) = \emptyset$
2. $\text{Files}(\text{Train}) \cap \text{Files}(\text{Test}) = \emptyset$
3. $\text{Files}(\text{Val}) \cap \text{Files}(\text{Test}) = \emptyset$
4. $\text{Drivers}(\text{Train}) \cap \text{Drivers}(\text{Val}) = \emptyset$
5. $\text{Drivers}(\text{Train}) \cap \text{Drivers}(\text{Test}) = \emptyset$
6. $\text{Drivers}(\text{Val}) \cap \text{Drivers}(\text{Test}) = \emptyset$
7. Every generated window traces to exactly one source file and driver.
8. An entire driver (Driver A, representing 12 files and highway Trip S1) is held out exclusively for final testing.

---

## 5. Training-Only Normalization

Standardization parameters are computed **STRICTLY AND EXCLUSIVELY ACROSS VALID TRAINING WINDOWS** ($226,928 \times 20 = 4,538,560$ sample observations):

$$\mu_c = \frac{1}{N_{\text{obs}}} \sum_{m=1}^{M_{\text{train}}} \sum_{t=1}^{20} X_{m, t, c}, \quad \sigma_c = \sqrt{\frac{1}{N_{\text{obs}} - 1} \sum_{m=1}^{M_{\text{train}}} \sum_{t=1}^{20} (X_{m, t, c} - \mu_c)^2}$$

where $N_{\text{obs}} = M_{\text{train}} \times 20 = 4,538,560$.

> [!IMPORTANT]
> **Authoritative Standard Deviation Convention (`ddof=1`)**:
> In strict alignment with mathematical best practice and the Python implementation (`np.std(..., ddof=1)` in `ml/data/normalization.py`), the sample standard deviation with Bessel's correction (`ddof=1`) is adopted as the single authoritative convention. Dividing by $N_{\text{obs}} - 1$ guarantees an unbiased estimator of variance across training observations.

### 5.1 Exact Empirical Statistics (from `data/ml_dataset_v1/normalization.json`)

| Channel | Name | Training Mean ($\mu$) | Training Std ($\sigma$, ddof=1) | Physical Interpretation |
|---|---|---|---|---|
| 0 | `f_x_v` | $-0.069204\,\text{m/s}^2$ | $1.378226\,\text{m/s}^2$ | Longitudinal vehicle acceleration/braking |
| 1 | `f_y_v` | $+0.046116\,\text{m/s}^2$ | $1.295886\,\text{m/s}^2$ | Lateral cornering acceleration |
| 2 | `f_z_v` | $+9.838189\,\text{m/s}^2$ | $0.565357\,\text{m/s}^2$ | Vertical specific force (centered on Earth gravity) |
| 3 | `omega_x_v` | $+0.000067\,\text{rad/s}$ | $0.119624\,\text{rad/s}$ | Vehicle roll angular rate |
| 4 | `omega_y_v` | $+0.003219\,\text{rad/s}$ | $0.247268\,\text{rad/s}$ | Vehicle pitch angular rate |
| 5 | `omega_z_v` | $+0.000009\,\text{rad/s}$ | $0.121069\,\text{rad/s}$ | Vehicle yaw angular rate |
| 6 | `norm_f_v` | $+10.015576\,\text{m/s}^2$ | $0.619333\,\text{m/s}^2$ | Specific force vector Euclidean norm |
| 7 | `norm_f_dot_v` | $+9.446759\,\text{m/s}^3$ | $8.807672\,\text{m/s}^3$ | Causal backward jerk magnitude |
| 8 | `norm_omega_v` | $+0.221860\,\text{rad/s}$ | $0.202232\,\text{rad/s}$ | Angular velocity vector Euclidean norm |

### 5.2 Zero-Leakage Invariant Test
In `test_normalization_train_only.py`, validation and test tensors were intentionally corrupted with massive scalar offsets. The normalization means and standard deviations remained bit-identical to double-precision machine epsilon.

---

## 6. VelocityNet Ground-Truth Labels & Hard-Boundary Causal Median Filter

### 6.1 Target Definition
Vehicle forward speed in $\text{m/s}$ is extracted strictly at the exact **window-end timestamp** $T$:
$$\text{target} = v(T)$$
Reference source: Racelogic VBOX RTK ground truth (`v_ref_speed_mps`).

### 6.2 Hard-Boundary Causal Median Smoothing
To attenuate discretization jitter without bridging across sensor blackouts or fabricating continuity:
1. **Contiguous Valid Run Isolation**: The filter processes contiguous valid sample runs independently.
2. **Hard Boundary Reset**: Any invalid sample $k$ (unvalidated by upstream quality taggers, non-finite, or negative) immediately terminates the current contiguous segment, outputs `np.nan`, and resets the valid run length to $0$.
3. **Zero Cross-Gap Bridging**: When a new valid segment begins at $k+1$, smoothing utilizes only samples from index $k+1$ onward. Samples preceding the invalid gap **NEVER** influence any sample after the gap:
   $$\text{eff\_len} = \min(W, \text{run\_length}_k), \quad v_{\text{smoothed}}[k] = \text{median}(v[k - \text{eff\_len} + 1 : k + 1])$$
4. **Truthful Invalid Handling**: Invalid window-end samples store `np.nan` in `y_speed` and `y_speed_raw`. They are **never** silently replaced with fabricated values (such as $0.0\,\text{m/s}$).
5. **Dual Provenance**: Both smoothed targets (`y_speed`) and raw reference speeds (`y_speed_raw`) are preserved side-by-side in each `.npz` archive.

### 6.3 Empirical Label Distribution across Splits (Valid Windows)

| Split | Valid Count | Mean (m/s) | Std (m/s) | Median (m/s) | Min (m/s) | Max (m/s) | 95th Pct (m/s) |
|---|---|---|---|---|---|---|---|
| **Train** | 226,928 | 15.5711 | 9.2867 | 15.9624 | 0.0000 | 36.5831 | 29.2311 |
| **Validation** | 21,080 | 10.4388 | 5.7739 | 10.9463 | 0.0000 | 27.9537 | 20.4146 |
| **Test** | 123,464 | 8.9149 | 6.5016 | 8.4609 | 0.0000 | 32.5019 | 21.6081 |

---

## 7. BiasNet Outage-Exclusion Plumbing

- **Architecture**: `evaluate_biasnet_eligibility()` evaluates whether windows overlap real GNSS blackout periods.
- **Dataset Reality Disclosure**: As established in Phase 0/2 audits, the IO-VNBD dataset contains **NO pre-tagged real outage CSV index**. In strict compliance with scientific honesty, **no fake outage intervals were fabricated**.
- **Result**: The plumbing safely flags all 371,472 valid windows as `ELIGIBLE` with reason `ELIGIBLE`. When real outage metadata is supplied in future benchmarks, the interface automatically excludes overlapping windows without requiring architectural changes.
- **Verified by Regression Test**: `test_outage_metadata_none_invents_no_fake_exclusions` and `test_real_outage_overlap_excluded` prove that overlapping windows become ineligible while absence of metadata invents no fake exclusions.

---

## 8. Serialized Artifacts Inventory

The full dataset was serialized to `data/ml_dataset_v1/`:

| File | Size on Disk | Contents |
|---|---|---|
| `train.npz` | ~158 MB | 233,830 windows: `X`, `X_raw`, `y_speed`, `y_speed_raw`, timestamps, file IDs, driver IDs |
| `validation.npz` | ~29 MB | 42,388 windows: `X`, `X_raw`, `y_speed`, `y_speed_raw`, timestamps, file IDs, driver IDs |
| `test.npz` | ~84 MB | 123,496 windows: `X`, `X_raw`, `y_speed`, `y_speed_raw`, timestamps, file IDs, driver IDs |
| `normalization.json` | 1.1 KB | Serialized means, stds (ddof=1), sample count, channel order |
| `model_config.json` | 1.2 KB | Phase 1 `ModelConfig` schema for direct ONNX/LiteRT loading |
| `dataset_manifest.json` | 1.8 KB | Machine-readable dataset audit record and summary |

---

## 9. Test Suite Verification & Reproducibility

### 9.1 Pytest Execution Summary
```
.venv\Scripts\python.exe -m pytest -v
============================ 249 passed in 12.32s =============================
```
- **Phase 1 Schemas**: 50 passed
- **Phase 2 Pipeline & Quality**: 53 passed
- **Phase 3 Preprocessing**: 47 passed
- **Phase 4 Frame Conversion & Strapdown INS**: 23 passed
- **Phase 5 ESKF & ZUPT Fusion**: 29 passed
- **Phase 6 ML Dataset Unit Tests**: 41 passed
  - `test_resample.py`: 5 passed
  - `test_features.py`: 4 passed
  - `test_windowing_causality.py`: 4 passed
  - `test_split.py`: 6 passed
  - `test_leakage_audit.py`: 5 passed
  - `test_normalization_train_only.py`: 4 passed
  - `test_velocitynet_labels.py`: 8 passed
  - `test_exclusion_rules.py`: 5 passed
- **Phase 6 ML Dataset Integration Tests (`test_ml_dataset_artifacts.py`)**: 6 passed
- **Total Tests**: **249 passed, 0 failures, 0 errors**.

### 9.2 Reproducibility Result
The complete dataset build was executed twice independently from cold processes.
- Manifest match: `True`
- Normalization parameters match: `True` (bit-for-bit identical)
- Array data match: `True` across all 10 arrays per split (bit-for-bit deterministic)
- All 399,714 window arrays, labels, and timestamps matched identically.

### 9.3 Dataset Immutability Verification
- **Raw CSV Files**: 288 files in `data/raw/io_vnbd` (**UNTOUCHED & FROZEN**).
- **Phase 2 Cache**: 144 `.npz` files in `data/cache/iovnbd` | Collective SHA-256: `7a267e112af4193dcadb8800bd9cf4ed8abf7249a1ad53883bbd2cfd06200fa7` (**VERIFIED IDENTICAL**).
- **Phase 0–5 Code**: Byte-for-byte freeze maintained; zero regressions.

---

## 10. Conclusion & Handoff to Phase 7

Phase 6 is **100% complete, leakage-audited, verified, and frozen**.
The canonical $(20, 9)$ normalized dataset is fully prepared for Phase 7 (VelocityNet Model Architecture, Training, and Validation).
