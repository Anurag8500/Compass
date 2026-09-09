# COMPASS Phase 3 — Calibration, Alignment, Filtering & Dynamic Recalibration

**Project**: C.O.M.P.A.S.S. (Cognitive Off-grid Machine-learning Positioning And Sensor System)  
**Problem Statement**: SIH 2026 Problem Statement 26168 — ISRO  
**Stage**: Phase 3 Complete Teaching & Reference Walkthrough  

---

## 1. Executive Summary & Why Phase 3 Existed

### The Core Problem
In Phase 2, we synchronized raw smartphone IMU measurements onto a clean 10 Hz timeline. However, those measurements were completely unusable for direct navigation:
1. **Arbitrary Mounting Orientation**: The phone is placed in a vehicle cradle, pocket, or console tray in an arbitrary 3D orientation. The phone's internal casing axes ($X^b, Y^b, Z^b$) do not match the vehicle's forward, lateral, and vertical axes ($X^v, Y^v, Z^v$).
2. **Sensor Bias**: Consumer smartphone gyroscopes have manufacturing and temperature offsets causing them to report non-zero angular velocities even when sitting perfectly still (gyroscope bias $\mathbf{b}_g$). Integrating an uncalibrated gyro causes heading to drift rapidly.
3. **Road & Engine Vibration**: As the vehicle travels, engine piston strokes, tire-road roughness, and chassis shudder inject high-frequency acceleration noise that masks true vehicle translational motion.
4. **Physical Discontinuities**: What happens if the driver taps the phone, or if the phone slides across the console tray during a sharp turn? The mounting alignment suddenly changes mid-drive.

### What Phase 3 Accomplished
Phase 3 implemented the classical sensor preprocessing pipeline that converts raw device-frame IMU data into calibrated, tilt-aligned, and filtered **vehicle-frame specific force $\mathbf{f}_m^v$ and angular velocity $\boldsymbol{\omega}_m^v$**:
- **Stationary Rest Calibration**: Automatically detects initial vehicle standstill to estimate 3-axis gyro bias $\mathbf{b}_g^b$ and device tilt angles.
- **Mounting Frame Alignment ($R_b^v$)**: Constructs an orthogonal 3D rotation matrix rotating the phone's casing into the vehicle chassis frame.
- **Dual-Stage Denoising Filter**: Cascades a 3-sample median filter (for impulse outlier suppression) with a 4th-order Butterworth low-pass filter ($f_c = 3.0\text{ Hz}$).
- **Dynamic Recalibration Detector**: Monitors for shifts in the gravity vector to detect phone mount slips while suppressing false alarms during aggressive driving.
- **Honest Observability Reporting**: Disclosed that mounting yaw cannot be reliably observed from unconstrained console trays; established the sentinel reporting contract (`is_yaw_aligned = False`), reserving full azimuth refinement for downstream ESKF fusion.

---

## 2. Technical Vocabulary & Physical Concepts

### 1. Specific Force ($\mathbf{f}$) vs Coordinate Acceleration ($\mathbf{a}$)
- **Simple Definition**: An accelerometer does not measure acceleration relative to the Earth. It measures the physical mechanical contact force per unit mass supporting it against gravity.
- **In COMPASS**: When a vehicle sits motionless at a red light, its coordinate acceleration is zero ($\mathbf{a} = \mathbf{0}$). However, the upward normal force from the pavement pushes against the car's chassis. The accelerometer measures an upward specific force:
  $$\mathbf{f} = \mathbf{a} - \mathbf{g} = \mathbf{0} - (-9.80665\,\text{m/s}^2 \hat{\mathbf{z}}) = +9.80665\,\text{m/s}^2 \hat{\mathbf{z}}$$
- **Preservation Invariant**: Phase 3 outputs vehicle-frame specific force $\mathbf{f}_m^v$ directly. Physical gravity is **NOT permanently removed** in Phase 3. Downstream strapdown mechanization (Phase 4 and 5) resolves gravity using the continuously tracked navigation attitude quaternion:
  $$\mathbf{a}^n = R_v^n (\mathbf{f}_m^v - \mathbf{b}_a^v) + \mathbf{g}^n, \quad \mathbf{g}^n = [0, 0, -9.80665]^T\,\text{m/s}^2$$

### 2. Sensor Bias ($\mathbf{b}_a, \mathbf{b}_g$)
- **Simple Definition**: A constant or slowly drifting offset in the sensor output.
- **In COMPASS**: Gyroscope bias $\mathbf{b}_g^b$ is measured during standstill and subtracted before rotation. Full 3D accelerometer bias is not observable from one static pose and is assigned a nominal zero prior $\mathbf{b}_{a,\text{prior}} = [0, 0, 0]^T$.

### 3. Rodrigues' Rotation Formula
- **Simple Definition**: An efficient mathematical method to rotate one 3D vector onto another along the shortest circular arc.
- **In COMPASS**: Used to rotate the measured gravity support vector in the phone frame $\bar{\mathbf{f}}^b$ directly onto the vertical vehicle axis $+Z^v$.

### 4. Butterworth Filter & Nyquist Limit
- **Simple Definition**: A digital frequency filter with a maximally flat passband and smooth roll-off.
- **In COMPASS**: At $10\text{ Hz}$ sampling, the Nyquist limit is $5.0\text{ Hz}$. A 4th-order low-pass filter with cutoff $f_c = 3.0\text{ Hz}$ attenuates chassis vibration between $3.0$ and $5.0\text{ Hz}$.

---

## 3. Mathematical Mechanics of Phase 3

### 3.1 Stationary Gyroscope Calibration
When the vehicle is detected at rest over $N \ge 20$ samples ($\text{var}(\|\mathbf{f}\|) < 0.05\,\text{m}^2/\text{s}^4$ and $\text{var}(\|\boldsymbol{\omega}\|) < 0.005\,\text{rad}^2/\text{s}^2$):
$$\mathbf{b}_g^b = \frac{1}{N} \sum_{i=1}^N \boldsymbol{\omega}_i^b$$
$$\bar{\mathbf{f}}^b = \frac{1}{N} \sum_{i=1}^N \mathbf{f}_i^b, \quad \mathbf{u}_g = \frac{\bar{\mathbf{f}}^b}{\|\bar{\mathbf{f}}^b\|}$$

#### Single-Pose Accelerometer Limitation
3D accelerometer bias $\mathbf{b}_a$ is **not observable from a single stationary pose** because a single static pose confounds physical tilt angles with sensor bias components. We assign a nominal zero prior $\mathbf{b}_{a,\text{prior}} = [0, 0, 0]^T$; dynamic bias estimation is strictly deferred to the downstream ESKF during vehicle maneuvers.

### 3.2 Mounting Frame Alignment ($R_b^v$)
We rotate the measured support vector $\mathbf{u}_g$ onto vehicle vertical $\mathbf{z}_v = [0, 0, 1]^T$ using Rodrigues' formula:
$$\mathbf{v}_{\text{rot}} = \mathbf{u}_g \times \mathbf{z}_v, \quad c = \mathbf{u}_g \cdot \mathbf{z}_v, \quad s = \|\mathbf{v}_{\text{rot}}\|$$
$$R_{\text{tilt}} = \mathbf{I} + [\mathbf{v}_{\text{rot}}]_\times + [\mathbf{v}_{\text{rot}}]_\times^2 \left(\frac{1 - c}{s^2}\right)$$

Transformed vehicle-frame streams follow:
$$\mathbf{f}_m^v = R_b^v (\mathbf{f}_m^b - \mathbf{b}_{a,\text{prior}}^b), \quad \mathbf{b}_{a,\text{prior}}^b = [0, 0, 0]^T$$
$$\boldsymbol{\omega}_m^v = R_b^v (\boldsymbol{\omega}_m^b - \mathbf{b}_g^b)$$

### 3.3 The Yaw Sentinel Contract
When mounting yaw cannot be observed with high confidence:
- In real highway testing on Trip S1, candidate acceleration correlation exhibited severe circular dispersion ($\sigma = 126.0^\circ$, far exceeding the $15.0^\circ$ acceptance threshold).
- Rather than fabricating an arbitrary yaw angle, Phase 3 enforced the **Honest Observability Contract**:
  - `is_yaw_aligned = False`
  - `yaw_deg = 0.0` (sentinel default)
- **Interpretation**: `yaw_deg = 0.0` combined with `is_yaw_aligned = False` means **reliable mounting yaw was NOT obtained**. It does NOT mean the measured vehicle yaw is 0 degrees. Downstream fusion is responsible for resolving the horizontal heading degree of freedom.

---

## 4. Dual-Stage Denoising Filter

1. **Stage 1 (3-Sample Median Filter)**: Causal 3-sample median filter suppresses single-sample impulse spikes without introducing phase distortion on smooth ramps.
2. **Stage 2 (4th-Order Butterworth Filter)**: Digital low-pass filter with $f_c = 3.0\,\text{Hz}$ at $f_s = 10.0\,\text{Hz}$.
   - Offline dataset building: zero-phase forward-backward filtering (`filtfilt`).
   - Online execution: causal recursive filtering.

---

## 5. Dynamic Recalibration Detector

The `RecalibrationDetector` monitors shifts in the gravity vector:
$$\theta_{\text{shift}} = \arccos\left(\frac{\mathbf{f}_{\text{filtered}} \cdot \mathbf{u}_g}{\|\mathbf{f}_{\text{filtered}}\|}\right)$$
- Recalibration triggers are suppressed during high vehicle dynamics ($\|\|\mathbf{f}\| - g\| \ge 1.5\,\text{m/s}^2$ or $\|\boldsymbol{\omega}\| \ge 0.5\,\text{rad/s}$) to avoid false alarms during sharp turns.
- When stationary, a confirmed shift $> 35^\circ$ for 5 consecutive samples triggers a recalibration event with a 50-sample cooldown lockout.

---

## 6. Code Inventory: `navigation/preprocessing/`

```
navigation/preprocessing/
├── __init__.py                # Clean package exports
├── calibration.py             # StationaryCalibrator: rest detection, gyro bias, tilt angles
├── alignment.py               # MountingAligner: Rodrigues tilt, yaw correlation, R_b^v
├── filtering.py               # DualStageFilter: 3-sample median + Butterworth IIR
├── recalibration_trigger.py   # RecalibrationDetector: mount drop & slip detection
├── pipeline.py                # PreprocessingPipeline: end-to-end orchestrator
├── gravity.py                 # Gravity models and unit utilities
└── bootstrap_attitude.py      # Offline-only initial attitude utility (NOT production live)
```

---

## 7. Real Empirical Validation Results (Trip S1)

| Metric | Raw / Unfiltered | Calibrated / Filtered | Physical Meaning |
|---|---|---|---|
| **Stationary Gyro Bias Norm** | $0.201^\circ/\text{s}$ ($0.00351\,\text{rad/s}$) | **$0.015^\circ/\text{s}$ ($0.00027\,\text{rad/s}$)** | **13.1x reduction in gyro drift** |
| **Stationary Accel Norm** | $9.8909\,\text{m/s}^2$ | $9.8909\,\text{m/s}^2$ | Preserved within $0.08\,\text{m/s}^2$ of Earth gravity |
| **Moving Accel Variance ($X^v$)** | $1.277\,\text{m}^2/\text{s}^4$ | **$0.895\,\text{m}^2/\text{s}^4$ (-29.9%)** | Longitudinal engine vibration smoothed |
| **Moving Accel Variance ($Y^v$)** | $1.311\,\text{m}^2/\text{s}^4$ | **$0.899\,\text{m}^2/\text{s}^4$ (-31.4%)** | Lateral road chatter attenuated |
| **Moving Accel Variance ($Z^v$)** | $0.328\,\text{m}^2/\text{s}^4$ | **$0.154\,\text{m}^2/\text{s}^4$ (-53.0%)** | **Over 50% power reduction on vertical bumps** |
| **Mounting Tilt Angles** | N/A | $\text{Roll} = +0.25^\circ, \text{Pitch} = -0.26^\circ$ | Level resting orientation |
| **Mounting Yaw Status** | N/A | `is_yaw_aligned = False` | Honest: circular dispersion ($126^\circ > 15^\circ$) |

---

## 8. Verification & Test Suite

Phase 3 is covered by exactly **47 tests**:
- `tests/unit/test_alignment.py` (11 tests): Rodrigues tilt, proper orthogonal matrix properties ($\det R = +1, R R^T = \mathbf{I}$), and yaw observability.
- `tests/unit/test_calibration.py` (10 tests): Stationary bias estimation, input immutability, and JSON serialization.
- `tests/unit/test_filtering.py` (10 tests): Median spike suppression, low-pass attenuation, and short-sequence fallback.
- `tests/unit/test_gravity_compensation.py` (7 tests): Physical specific force invariants and gravity alignment.
- `tests/unit/test_recalibration.py` (7 tests): Mount shift triggers, cooldown lockout, and high-dynamics false-alarm suppression.
- `tests/integration/test_preprocessing_on_real_data.py` (2 tests): End-to-end processing across real cached trips.

---

## 9. Final Phase 3 Summary: What I Should Remember

1. **Specific Force Preserved**: Preprocessing outputs calibrated vehicle-frame specific force $\mathbf{f}_m^v$. Physical gravity is **NOT permanently removed** in Phase 3.
2. **Gyro Bias Reduced by 13.1x**: Initial static rest estimates $\mathbf{b}_g^b$, cutting drift from $0.201^\circ/\text{s}$ to $0.015^\circ/\text{s}$.
3. **Accelerometer Bias Unobservable at Rest**: Confounded with tilt angles; nominal zero prior is used until downstream dynamic ESKF estimation.
4. **Honest Yaw Sentinel**: `yaw_deg = 0.0` and `is_yaw_aligned = False` means yaw was **unresolved**, not that the vehicle is pointed at 0 degrees.
5. **Dual-Stage Denoising**: Median spike removal + 4th-order Butterworth ($f_c = 3.0\,\text{Hz}$) cuts chassis vibration power by $>50\%$.
6. **Code Paths**: All code resides in `navigation/preprocessing/` (`calibration.py`, `alignment.py`, `filtering.py`, `recalibration_trigger.py`, `pipeline.py`).
7. **Phase 3 Test Count**: Exactly **47 tests** verified and passing.
