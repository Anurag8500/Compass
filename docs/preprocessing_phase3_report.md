# COMPASS Phase 3 Classical Preprocessing Report
**Execution Date**: 2026-09-08  
**Component**: Classical Sensor Preprocessing (`navigation/preprocessing`)  
**Phase**: Phase 3 (v1.0 - Hardened, Audit-Verified)  
**Status**: PASSED  

---

## 1. Executive Summary & Objective

Phase 3 implements and verifies the complete classical sensor preprocessing transform chain for C.O.M.P.A.S.S. (SIH 2026 Problem Statement 26168 - ISRO), turning raw smartphone device-frame IMU data into calibrated, aligned, and filtered vehicle-frame specific force $\mathbf{f}_m^v$ and angular velocity $\boldsymbol{\omega}_m^v$.

```
Raw IMU Stream (Device Body Frame b)
               |
    1. Stationary Calibration (b_g, tilt angles)
               |
    2. Device -> Vehicle Mounting Alignment (R_b^v)
               |
    3. Dual-Stage Denoising Filter (Median Spike + 4th-Order Butterworth)
               |
    4. Recalibration Discontinuity Detection
               |
    Calibrated Vehicle-Frame Specific Force f_m^v & Angular Velocity omega_m^v
```

### Critical Architectural Invariants Verified
1. **No Circular / Permanent Live Gravity Compensation in Preprocessing**:
   - The preprocessing pipeline outputs vehicle-frame specific force $\mathbf{f}_m^v$ directly.
   - Physical gravity $\mathbf{g}^n = [0, 0, -g]^T$ is physically resolved downstream inside the central ESKF strapdown mechanization using its continuously tracked vehicle attitude quaternion $R_v^n$:
     $$\mathbf{a}^n = R_v^n (\mathbf{f}_m^v - \mathbf{b}_a^v) + \mathbf{g}^n$$
2. **Bootstrap Attitude Estimator is Offline-Only**:
   - The complementary-filter attitude estimator in `navigation/preprocessing/bootstrap_attitude.py` is explicitly verified as an offline test/validation utility. It is not a live navigation dependency.
3. **Single-Pose Accelerometer Bias Observability Limitation**:
   - A single static resting pose cannot independently separate 3D accelerometer bias from physical tilt (5 unknowns: 2 tilt angles + 3 bias components, with only 3 accelerometer equations).
   - Preprocessing assigns the nominal prior $\mathbf{b}_a = [0, 0, 0]^T$. Dynamic 3D bias refinement is executed downstream by the ESKF during vehicle motion when GNSS, Non-Holonomic Constraints (NHC), and Zero-Velocity Updates (ZUPT) arrive.
4. **Non-Destructive Processing**:
   - Raw Phase 2 synchronized cached arrays and underlying CSV archives remain 100% byte-for-byte immutable.

---

## 2. Real Dataset Empirical Validation Results (Trip `Categorised_S1`)

The preprocessing pipeline was executed against the real synchronized IO-VNBD trip `Categorised_S1` (51,746 samples, 5,174.5 seconds of driving):

| Metric | Raw / Unfiltered | Calibrated / Filtered | Physical Interpretation |
|---|---|---|---|
| **Stationary Gyro Mean Norm** | $0.00351\,\text{rad/s}$ ($0.201^\circ/\text{s}$) | **$0.00028\,\text{rad/s}$ ($0.016^\circ/\text{s}$)** | **12.6x reduction** in stationary residual gyro drift |
| **Stationary Gyro Std** | $[0.0125, 0.0153, 0.0193]\,\text{rad/s}$ | $[0.0098, 0.0096, 0.0146]\,\text{rad/s}$ | Reduced high-frequency sensor noise variance |
| **Stationary Vertical Accel ($\mathbf{a}_z^n$)** | N/A (uncompensated) | **$+0.0690\,\text{m/s}^2$ mean ($\sigma = 0.0696\,\text{m/s}^2$)** | Accurately hovers near $0.0\,\text{m/s}^2$ ($<0.07\,\text{m/s}^2$ residual) |
| **Stationary 3D Accel Norm** | $9.823\,\text{m/s}^2$ ($1g$ reaction) | **$0.326\,\text{m/s}^2$ mean coordinate residual** | Verifies gravity subtraction without explosive drift |
| **Moving Accel Variance ($X_v$)** | $1.317\,\text{m}^2/\text{s}^4$ | **$0.916\,\text{m}^2/\text{s}^4$ (-30.4%)** | Road vibration smoothed while preserving vehicle braking/accel |
| **Moving Accel Variance ($Y_v$)** | $1.271\,\text{m}^2/\text{s}^4$ | **$0.883\,\text{m}^2/\text{s}^4$ (-30.5%)** | Lateral road shock attenuated |
| **Moving Accel Variance ($Z_v$)** | $0.328\,\text{m}^2/\text{s}^4$ | **$0.154\,\text{m}^2/\text{s}^4$ (-53.0%)** | Over 50% power reduction on vertical pothole/suspension jitter |
| **Moving Gyro Variance ($Z_v$)** | $0.0125\,\text{rad}^2/\text{s}^2$ | **$0.0069\,\text{rad}^2/\text{s}^2$ (-44.8%)** | Yaw rate sensor noise smoothed without phase lag |
| **Mounting Orientation Recovered** | $\text{Roll} = -0.25^\circ, \text{Pitch} = +0.26^\circ$ | $\text{Yaw} = -104.46^\circ$ | Full 3D mounting alignment resolved against GNSS course |
| **Recalibration False Alarms** | 0 events | 0 events | Zero false alarms on nominal driving |

---

## 3. Preprocessing Chain Verification

### Stage 1: Stationary Window Calibration (`calibration.py`)
- Gyro bias is extracted via $\mathbf{b}_g = \frac{1}{N} \sum \boldsymbol{\omega}_i$.
- Initial roll $\phi$ and pitch $\theta$ are resolved from the direction of measured specific force support reaction vector $\mathbf{f}_m^b$ at rest.
- Nominal zero prior $\mathbf{b}_a = [0, 0, 0]^T$ is assigned.
- `CalibrationProfile` is fully serializable and deserializable via JSON dictionaries.

### Stage 2: Device-to-Vehicle Alignment (`alignment.py`)
- Pitch and roll are aligned such that the measured rest reaction vector maps onto the vehicle vertical axis $+Z_v$.
- Yaw is determined by circular correlation between gyro-integrated heading change and GNSS track heading over vehicle motion exceeding $v \ge 3.0\,\text{m/s}$.
- Fixed transformation matrix $R_b^v$ transforms body vectors:
  $$\mathbf{f}_m^v = R_b^v (\mathbf{f}_m^b - \mathbf{b}_{a,\text{prior}})$$
  $$\boldsymbol{\omega}_m^v = R_b^v (\boldsymbol{\omega}_m^b - \mathbf{b}_g)$$

### Stage 3: Dual-Stage Filtering (`filtering.py`)
- **Median Filter**: 3-sample sliding window cleanly rejects non-physical single-sample impulse spikes without blurring genuine step transitions.
- **Butterworth Filter**: 4th-order lowpass filter ($f_c = 3.0\,\text{Hz}$, $f_s = 10.0\,\text{Hz}$) suppresses engine vibration and chassis shock. Uses zero-phase forward-backward filtering (`filtfilt`) for offline replay and provides causal IIR step execution for streaming.

### Stage 4: Gravity Resolution Mathematical Validation (`gravity.py`)
- Mathematical formulation:
  $$\mathbf{a}^n = R_v^n (\mathbf{f}_m^v - \mathbf{b}_a^v) + \mathbf{g}^n \quad \text{with} \quad \mathbf{g}^n = [0, 0, -9.80665]^T$$
- Verified canonical level stationary test: $R_v^n = \mathbf{I}, \mathbf{b}_a^v = \mathbf{0}, \mathbf{f}_m^v = [0, 0, +g]^T \implies \mathbf{a}^n = [0, 0, 0]^T$ identically to within $10^{-12}\,\text{m/s}^2$.
- Verified dynamic acceleration recovery under arbitrary 3D rotation and known accelerometer bias.

### Stage 5: Recalibration Trigger Detection (`recalibration_trigger.py`)
- Detects physical orientation jumps (e.g. phone dislodged from mount or bumped by driver).
- Tested against real data: zero false triggers on nominal driving; correctly flags injected $45^\circ$ sensor displacements and angular velocity shocks.

---

## 4. Test Suite Summary

- **Total Test Count**: **129 passed**
  - Phase 1 schemas: 50 passed
  - Phase 2 pipeline & quality: 53 passed
  - Phase 3 calibration unit: 6 passed
  - Phase 3 alignment unit: 6 passed
  - Phase 3 gravity & bootstrap attitude unit: 7 passed
  - Phase 3 filtering unit: 5 passed
  - Phase 3 real-data integration: 2 passed
- **Command**: `.venv\Scripts\python.exe -m pytest -v`
- **Result**: **129 passed in 4.19s (100% PASS)**

---

## 5. Phase 3 Readiness & Handoff to Phase 4

Phase 3 is complete and verified. The preprocessing transform chain provides a mathematically sound, calibrated, and filtered vehicle-frame stream ready to feed Phase 4 strapdown INS propagation.
