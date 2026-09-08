# COMPASS Phase 3 Classical Preprocessing Report
**Execution Date**: 2026-09-08  
**Component**: Classical Sensor Preprocessing (`navigation/preprocessing`)  
**Phase**: Phase 3 (Hardened, Mathematically Audited, Truthful Validation)  
**Status**: COMPLETE AND VERIFIED  

---

## 1. Phase 3 Scope & Executive Summary

Phase 3 implements and verifies the complete classical sensor preprocessing transform chain for C.O.M.P.A.S.S. (SIH 2026 Problem Statement 26168 - ISRO), transforming raw smartphone device-frame IMU data from Phase 2 into calibrated, tilt-aligned, and filtered vehicle-frame specific force $\mathbf{f}_m^v$ and angular velocity $\boldsymbol{\omega}_m^v$.

```
Phase 2 Synchronized Trip (Raw Device Frame b)
               |
    1. Stationary Calibration (b_g, tilt angles)
               |
    2. Device -> Vehicle Mounting Alignment (R_b^v)
               |
    3. Dual-Stage Denoising Filter (Median Spike + 4th-Order Butterworth)
               |
    4. Recalibration Discontinuity Monitoring
               |
    Calibrated Vehicle-Frame Streams: f_m^v & omega_m^v
```

> **Core Observability Statement**:  
> Phase 3 provides a gravity/tilt-aligned inertial frame. When mounting yaw is successfully observed, the horizontal axes are additionally resolved to vehicle forward/lateral. For Categorised_S1, yaw remained unresolved, so the horizontal azimuth remains provisional and is reserved for downstream navigation fusion.

### Critical Architectural Invariants
1. **No Permanent Live Gravity Subtraction in Preprocessing**:
   - Preprocessing outputs vehicle-frame specific force $\mathbf{f}_m^v$ directly.
   - Physical gravity $\mathbf{g}^n = [0, 0, -9.80665]^T\,\text{m/s}^2$ in local East-North-Up (ENU) navigation frame is resolved downstream inside the central Error-State Kalman Filter (ESKF) strapdown mechanization using its continuously tracked attitude quaternion $R_v^n$:
     $$\mathbf{a}^n = R_v^n (\mathbf{f}_m^v - \mathbf{b}_a^v) + \mathbf{g}^n$$
2. **Bootstrap Attitude Estimator is Strictly Offline-Only**:
   - `BootstrapAttitudeEstimator` (`navigation/preprocessing/bootstrap_attitude.py`) is an **offline test and validation utility only**. It is never used as a live navigation attitude estimator, is not a production runtime dependency, and is not a Phase 3 permanent gravity-compensation mechanism.
3. **Single-Pose Accelerometer Bias Observability Limitation Preserved**:
   - Stationary gyro bias is estimated from rest data.
   - Accelerometer bias remains a nominal zero prior ($\mathbf{b}_a = [0, 0, 0]^T$) because full 3D bias is not observable from one static pose (a single static pose confounds physical tilt angles with accelerometer bias components). Dynamic bias refinement is strictly reserved for downstream navigation fusion during vehicle maneuvers.
4. **Strict Immutability of Raw and Phase 2 Datasets**:
   - All 288 raw CSV files in `data/raw/io_vnbd` and 144 Phase 2 cached trip files in `data/cache/iovnbd/*.npz` remain byte-for-byte immutable.

---

## 2. Coordinate Frame Conventions & Alignment Semantics

### Coordinate Frames
- **Device Body Frame ($b$)**: Physical casing of the sensor/smartphone.
- **Vehicle Frame ($v$)**: Right-handed Forward-Lateral-Up (FLU) convention:
  - $X_v$: Forward longitudinal direction along vehicle centerline.
  - $Y_v$: Lateral direction (perpendicular to forward, leftwards).
  - $Z_v$: Upward vertical direction (level at rest yields $f_z^v \approx +9.80665\,\text{m/s}^2$).
- **Navigation Frame ($n$)**: Local East-North-Up (ENU) tangent plane:
  - $X_n$: East, $Y_n$: North, $Z_n$: Up.

### Authoritative Transformation Equations
For any body vector $\mathbf{v}_b$, the vehicle-frame representation is:
$$\mathbf{v}_v = R_b^v \mathbf{v}_b$$
with specific force and angular velocity transformed as:
$$\mathbf{f}_m^v = R_b^v (\mathbf{f}_m^b - \mathbf{b}_{a,\text{prior}}), \quad \mathbf{b}_{a,\text{prior}} = [0, 0, 0]^T$$
$$\boldsymbol{\omega}_m^v = R_b^v (\boldsymbol{\omega}_m^b - \mathbf{b}_g)$$

The mounting rotation matrix $R_b^v$ is authoritative:
- Strictly orthogonal ($R (R)^T = \mathbf{I}$).
- Right-handed proper rotation ($\det(R) = +1.0$).
- Euler angles ($\phi, \theta, \psi$) are extracted directly from $R_b^v$ as diagnostics only.

### Alignment Semantics (Resolved vs. Unresolved)
- **When `is_yaw_aligned == True`**:
  - $R_b^v$ resolves both vertical tilt and horizontal mounting azimuth.
  - The output frame is interpreted as:
    - $X_v = \text{vehicle forward}$
    - $Y_v = \text{vehicle left/lateral}$
    - $Z_v = \text{vehicle up}$
- **When `is_yaw_aligned == False`**:
  - $R_b^v$ provides only gravity/tilt alignment.
  - $+Z_v$ is gravity/vertical aligned (support reaction $f_z^v \approx +9.81\,\text{m/s}^2$).
  - Horizontal azimuth remains provisional/unresolved.
  - $X_v$ and $Y_v$ **must NOT be described as definitively vehicle-forward/lateral**.
  - Downstream navigation fusion is responsible for resolving the remaining horizontal heading degree of freedom.
  - The reported numerical value `yaw_deg = 0.0` is strictly a sentinel/reporting value and **NOT** a measured mounting azimuth.

---

## 3. Preprocessing Component Specifications

### 3.1 Stationary Calibration (`calibration.py`)
- **Stationary Gyro Bias**: Evaluated over verified rest segments:
  $$\mathbf{b}_g = \frac{1}{N} \sum_{i=1}^N \boldsymbol{\omega}_i^b$$
- **Support Reaction Vector**:
  $$\mathbf{u}_g = \frac{\bar{\mathbf{f}}^b}{\|\bar{\mathbf{f}}^b\|}$$
- **Tilt Angles**: Extracted directly from $\mathbf{u}_g$ via $\phi = \text{atan2}(u_y, u_z)$ and $\theta = \text{atan2}(-u_x, \sqrt{u_y^2 + u_z^2})$.
- **Accelerometer Bias**: Maintained as nominal prior $\mathbf{b}_a = [0, 0, 0]^T$. Full 3D accelerometer bias is not observable from a single resting pose.
- **Input Validation**: Strict rejection of non-finite numbers (NaN/Inf), minimum sample threshold ($N \ge 20$), and timestamp monotonicity checks.

### 3.2 Mounting Alignment (`alignment.py`)
- **Tilt Alignment ($R_{\text{tilt}}$)**:
  - Rotates measured stationary support reaction vector $\bar{\mathbf{f}}^b$ onto $+Z_v = [0, 0, 1]^T$ using Rodrigues minimum-angle rotation formulation.
- **Yaw Observability & Alignment ($R_{\text{yaw}}$)**:
  - Gyro integration provides orientation *change* only; it does not provide absolute mounting yaw by itself.
  - Mounting yaw is resolved only when there is a valid observable azimuth constraint:
    1. An explicitly supplied/reference mounting yaw (`reference_yaw_deg`), OR
    2. Qualifying straight-line longitudinal acceleration correlation:
       Epochs satisfying $v_{\text{gnss}} \ge 3.0\,\text{m/s}$, $\dot{v}_{\text{gnss}} \ge 0.4\,\text{m/s}^2$, $|\omega_{z,\text{level}}| \le 0.05\,\text{rad/s}$, and $\|f_{h,\text{level}}\| \ge 0.3\,\text{m/s}^2$ are correlated with vehicle forward acceleration.
  - **Honest Fallback**: If qualifying epochs are insufficient ($< 10$) or candidate yaw angles exhibit high circular dispersion ($> 15^\circ$), mounting yaw remains unresolved (`is_yaw_aligned = False`, `yaw_deg = 0.0`). The system never fabricates an unobserved yaw.

### 3.3 Dual-Stage Denoising Filter (`filtering.py`)
- **Stage 1 (Median Filter)**: 3-sample sliding window cleanly eliminates single-sample non-Gaussian impulse spikes while preserving true step transitions.
- **Stage 2 (Butterworth Low-Pass)**: 4th-order lowpass filter with cutoff frequency $f_c = 3.0\,\text{Hz}$ at nominal sampling rate $f_s = 10.0\,\text{Hz}$.
- **Configured vs. Actual Runtime Mode**:
  - `filter.filtering_mode`: Configured nominal mode (`"zero_phase"` or `"causal"`).
  - `filter.last_filtering_mode`: Actual runtime mode applied to the most recent input sequence (`"zero_phase"`, `"causal"`, or `"passthrough"`).
  - For long real-data trips (e.g. `Categorised_S1`, $N=51,746$), the actual runtime mode is `zero_phase` using forward-backward `scipy.signal.filtfilt` with zero phase distortion.
  - For short sequences where sample count is below the filter pad length threshold ($N < 15$), the filter gracefully falls back to causal filtering or passthrough with an explicit warning, avoiding mathematical singularity or edge artifacts.
- **Nyquist Interpretation**:
  - The IO-VNBD dataset operates at nominal $f_s = 10.0\,\text{Hz}$, with a Nyquist frequency of $5.0\,\text{Hz}$.
  - Therefore, the real 10 Hz dataset cannot directly resolve or attenuate physical vibration above $5.0\,\text{Hz}$ (frequencies $> 5.0\,\text{Hz}$ alias). Preprocessing denoises signal content between $3.0\,\text{Hz}$ and $5.0\,\text{Hz}$.
  - Attenuation of higher frequency vibration (e.g. 25 Hz) is verified on synthetic 100 Hz streams where it is properly sampled above the Nyquist limit.

### 3.4 Recalibration Discontinuity Monitor (`recalibration_trigger.py`)
- **Non-Destructive Event Reporting**: Emits structured `RecalibrationEvent` rather than silently mutating calibration profiles or navigation state.
- **Stationary-Aware Confirmation**: When the vehicle is confirmed stationary, a detected gravity vector shift ($> 35^\circ$) triggers after 5 samples (`stationary_persistence_samples = 5`), exploiting the absence of dynamic motion.
- **High-Dynamics Rejection**: Severe dynamic acceleration ($|a - g| \ge 1.5\,\text{m/s}^2$) or rapid rotation suppresses false triggers during aggressive driving maneuvers.
- **Cooldown Window**: 50-sample lockout prevents event flooding.
- **Engineering Parameterization**: Thresholds are validated engineering parameters for Phase 3 evaluation and are configurable for target vehicle platforms.

---

## 4. Gravity Resolution Verification: Mathematical Identity vs. Real-Data Validation

Phase 3 rigorously separates canonical mathematical verification from empirical sanity checks:

### 4.1 Canonical Mathematical Identity Verification
In the canonical navigation mechanization with known identity attitude ($R_v^n = \mathbf{I}$), zero accelerometer bias ($\mathbf{b}_a^v = \mathbf{0}$), and static vehicle supporting gravity reaction ($\mathbf{f}_m^v = [0, 0, +9.80665]^T\,\text{m/s}^2$):
$$\mathbf{a}^n = R_v^n (\mathbf{f}_m^v - \mathbf{b}_a^v) + \mathbf{g}^n = \mathbf{I} \begin{bmatrix} 0 \\ 0 \\ +9.80665 \end{bmatrix} + \begin{bmatrix} 0 \\ 0 \\ -9.80665 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}\,\text{m/s}^2$$
This identity holds to double-precision machine epsilon ($< 10^{-14}\,\text{m/s}^2$) and is verified in `test_gravity_compensation.py`.

### 4.2 Real-Data Stationary Gravity Sanity Validation
For empirical real-world driving data (`Categorised_S1`), the offline test utility `BootstrapAttitudeEstimator` constructs an offline leveling attitude from stationary support reaction forces. Over 112 stationary segments (11,284 samples):
- Vertical coordinate acceleration: $\mathbf{a}_z^n = +0.0765\,\text{m/s}^2$ mean ($\sigma = 0.0763\,\text{m/s}^2$).
- The residual ($< 0.08\,\text{m/s}^2$) reflects unmodeled smartphone accelerometer bias and local gravity anomaly.
- This serves as a **stationary gravity-resolution sanity check**, confirming consistent coordinate frames and signs; it is not an exact proof of dynamic navigation accuracy (which requires Phase 5 closed-loop ESKF fusion).

---

## 5. Real-Data Empirical Validation (Trip `Categorised_S1`)

The complete preprocessing pipeline was executed against the real synchronized IO-VNBD trip `Categorised_S1` (51,746 samples, 5,174.5 seconds / 1.44 hours of driving):

| Metric | Raw / Unfiltered | Calibrated / Filtered | Verification Result |
|---|---|---|---|
| **Trip ID** | `S1` | `S1` | Exact match |
| **Total Samples / Duration** | 51,746 samples | 5,174.50 s | Complete timeline processed |
| **Stationary Segments Detected** | 112 segments | 11,284 samples | Identified by `StationaryDetector` |
| **Stationary Gyro Mean Norm** | $0.003513\,\text{rad/s}$ ($0.201^\circ/\text{s}$) | **$0.000269\,\text{rad/s}$ ($0.015^\circ/\text{s}$)** | **13.1x reduction** in residual gyro bias |
| **Stationary Gyro Std ($X, Y, Z$)** | $[0.0125, 0.0153, 0.0193]\,\text{rad/s}$ | $[0.0086, 0.0105, 0.0146]\,\text{rad/s}$ | Denoised stationary variance |
| **Stationary Accel Norm** | $9.8909\,\text{m/s}^2$ | $9.8909\,\text{m/s}^2$ | Norm preserved ($< 0.09\,\text{m/s}^2$ sensor bias) |
| **Stationary Vertical Accel ($\mathbf{a}_z^n$)** | N/A | **$+0.0765\,\text{m/s}^2$ mean ($\sigma = 0.0763\,\text{m/s}^2$)** | Stationary gravity-resolution sanity check |
| **Stationary 3D Coord Accel Norm** | N/A | **$0.2691\,\text{m/s}^2$ mean ($\sigma = 0.3060\,\text{m/s}^2$)** | Bounded coordinate acceleration at rest |
| **Moving Accel Variance ($X_v$)** | $1.277\,\text{m}^2/\text{s}^4$ | **$0.895\,\text{m}^2/\text{s}^4$ (-29.9%)** | Longitudinal road vibration smoothed |
| **Moving Accel Variance ($Y_v$)** | $1.311\,\text{m}^2/\text{s}^4$ | **$0.899\,\text{m}^2/\text{s}^4$ (-31.4%)** | Lateral road shock attenuated |
| **Moving Accel Variance ($Z_v$)** | $0.328\,\text{m}^2/\text{s}^4$ | **$0.154\,\text{m}^2/\text{s}^4$ (-53.0%)** | Over 50% power reduction on vertical shock |
| **Mounting Tilt Angles** | N/A | $\text{Roll} = +0.25^\circ, \text{Pitch} = -0.26^\circ$ | Level resting pose resolved |
| **Mounting Yaw Status** | N/A | `UNRESOLVED_HIGH_CIRCULAR_DISPERSION` | **Honest: Yaw unobservable ($\sigma=126.0^\circ > 15.0^\circ$)** |
| **Reported Mounting Yaw** | N/A | **$0.00^\circ$ (`is_yaw_aligned = False`)** | **Sentinel reporting value (NOT a measurement)** |
| **Recalibration Events** | 0 events | 0 events | Zero false alarms on nominal driving |
| **Filtering Mode (Configured / Actual)** | `zero_phase` | `zero_phase` / `zero_phase` | Scipy zero-phase offline processing (`last_filtering_mode="zero_phase"`) |
| **Validated Sample Count** | 51,746 (100.0%) | 51,746 (100.0%) | Zero dropped or non-finite validated samples |

> **Mounting Alignment Result Summary**:  
> Phase 3 provides a gravity/tilt-aligned inertial frame. When mounting yaw is successfully observed, the horizontal axes are additionally resolved to vehicle forward/lateral. For Categorised_S1, yaw remained unresolved, so the horizontal azimuth remains provisional and is reserved for downstream navigation fusion. The reported value of $0.00^\circ$ is strictly a sentinel/reporting value and not a measured mounting azimuth.

---

## 6. Full Test Suite & Immutability Verification

### Test Suite Execution
```
.venv\Scripts\python.exe -m pytest -v
============================= 150 passed in 4.04s =============================
```

- **Phase 1 Schemas & Core Data Contracts**: 50 passed
- **Phase 2 Pipeline & Data Quality Hardening**: 53 passed
- **Phase 3 Stationary Calibration (`test_calibration.py`)**: 10 passed
- **Phase 3 Mounting Alignment & Rotation Math (`test_alignment.py`)**: 11 passed
- **Phase 3 Gravity Resolution & Bootstrap Attitude (`test_gravity_compensation.py`)**: 7 passed
- **Phase 3 Dual-Stage Filtering (`test_filtering.py`)**: 10 passed
- **Phase 3 Recalibration Detector (`test_recalibration.py`)**: 7 passed
- **Phase 3 Real-Data Integration (`test_preprocessing_on_real_data.py`)**: 2 passed

### Dataset & Cache Immutability Audit
All raw and cached files were verified via SHA-256 digests:
- **Raw CSV Files**: 288 files in `data/raw/io_vnbd`
  - Collective SHA-256: `04aa4da188a6b0417d202f83f959ef3238dedf2a6ccfa075ff7b0751fe072a99` (UNMODIFIED)
- **Phase 2 Cached Trips**: 144 `.npz` files in `data/cache/iovnbd`
  - Collective SHA-256: `7a267e112af4193dcadb8800bd9cf4ed8abf7249a1ad53883bbd2cfd06200fa7` (UNMODIFIED)

---

## 7. Limitations & Honest Observability Disclosures

1. **Mounting Yaw Observability on Smartphone Telemetry**:
   - In unconstrained smartphone driving datasets (e.g. phones resting in console trays or loose mounts), straight-line translational acceleration may have high dispersion due to road bumps, engine vibration, and driver handling.
   - Phase 3 explicitly refuses to fabricate a yaw number. When candidate acceleration correlation exhibits circular dispersion $> 15^\circ$, Phase 3 outputs `is_yaw_aligned = False` and `yaw_deg = 0.0`. The value $0.0^\circ$ is a sentinel reporting value. Full azimuth alignment is reserved for downstream ESKF fusion with GNSS course and Non-Holonomic Constraints (NHC).
2. **Single-Pose Accelerometer Bias Observability**:
   - Single static rest cannot distinguish 3D accelerometer bias from physical tilt angles. The nominal prior $\mathbf{b}_a = [0, 0, 0]^T$ is assigned. Dynamic bias refinement is reserved for downstream navigation filtering.
3. **Sampling Rate & Nyquist Bound**:
   - The IO-VNBD dataset operates at nominal $10\,\text{Hz}$ ($f_s = 10.0\,\text{Hz}$). Phase 3 lowpass filtering attenuates sensor noise between $3\,\text{Hz}$ and $5\,\text{Hz}$ (Nyquist). Any vibration above $5\,\text{Hz}$ cannot be resolved at $10\,\text{Hz}$ due to aliasing. High-frequency vibration filtering is verified on 100 Hz synthetic streams.

---

## 8. Phase 4 Readiness & Formal Recommendation

- **Phase 0, 1, and 2**: Frozen and 100% regression-verified.
- **Phase 3**: Hardened, mathematically audited, truthful validation complete.
- **Recommendation**: **GO FOR PHASE 4 (Strapdown Inertial Navigation System Mechanization & Error Propagation)**.
