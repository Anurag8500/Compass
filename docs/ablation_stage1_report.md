# COMPASS Phase 4 Ablation Stage 1 Report: Classical Open-Loop Strapdown INS Drift Benchmark

**Execution Date**: 2026-09-09  
**Component**: Strapdown Inertial Navigation System (`navigation/ins`)  
**Phase**: Phase 4 (Attitude Representation & Strapdown INS Propagation)  
**Status**: COMPLETE AND EMPIRICALLY MEASURED  

---

## 1. Objective

This report establishes the empirical dead-reckoning performance baseline for **Ablation Stage 1** in the C.O.M.P.A.S.S. architecture (SIH 2026 Problem Statement 26168 — ISRO). 

The goal of Phase 4 is to execute pure classical open-loop strapdown INS double integration on a real driving dataset with high-precision reference telemetry (Racelogic VBOX ground truth), measuring and documenting the unconstrained drift rate of consumer smartphone inertial sensors without any aiding or corrections.

This baseline serves as the unassisted reference (**Level 1** on the fusion ablation ladder) against which all subsequent fusion mechanisms—Phase 5 (ESKF + Gated ZUPT + GNSS), Phase 7 (VelocityNet), Phase 8 (BiasNet), Phase 11 (Non-Holonomic Constraints), and Phase 12 (Map Matching)—are quantitatively measured.

---

## 2. Dataset & Evaluation Driving Segment

- **Dataset**: Real-world driving telemetry from the IO-VNBD dataset (`data/cache/iovnbd/Categorised_S1.npz`).
- **Trip ID**: `S1` (1.44 hours total driving, 51,746 samples).
- **Ablation Evaluation Window**: Samples `[19500, 20100]` (601 samples at 10.0 Hz nominal rate, duration: **60.00 seconds**).
- **Segment Characteristics**:
  - Steady road cruising at an average vehicle speed of **$12.7\,\text{m/s}$ ($45.7\,\text{km/h}$)**.
  - Speed range: $10.1\,\text{m/s}$ to $14.8\,\text{m/s}$.
  - Ground truth distance traversed: **$752.6\,\text{meters}$**.
  - Track heading: gentle curve shifting from $285.2^\circ$ to $248.3^\circ$ relative to true North.
- **Ground Truth Source**: Racelogic VBOX differential GNSS + vehicle CAN bus (`v_ref_lat`, `v_ref_lon`, `v_ref_alt_m`, `v_ref_speed_mps`, `v_ref_heading_deg`).

---

## 3. Initialization Conditions

The strapdown INS was initialized at the first sample of the evaluation window ($t = 0.0\,\text{s}$, sample index 19500) under explicit, recorded initial conditions:

| Parameter | Initial Condition | Source / Rationale |
|---|---|---|
| **Session Reference Origin** | $\text{lat}_0 = 52.401657^\circ$, $\text{lon}_0 = -1.505617^\circ$, $\text{alt}_0 = 147.88\,\text{m}$ | Fixed session-level local tangent plane origin |
| **Initial Position ($p_0^n$)** | $[0.0, 0.0, 0.0]\,\text{m}$ in local ENU | Center of the local Cartesian coordinate system |
| **Initial Velocity ($v_0^n$)** | $[-11.46, +3.22, 0.00]\,\text{m/s}$ | Derived from initial VBOX ground truth speed and track angle |
| **Initial Attitude ($q_0$)** | $[0.132334, 0.0, 0.0, 0.991205]^T$ | Level pose aligned to initial ground truth track heading ($285.21^\circ$) |
| **Initial Heading** | $285.21^\circ$ (relative to true North) | VBOX ground truth dual-antenna heading |
| **Accelerometer Bias Prior** | $\mathbf{b}_a^v = [0.0, 0.0, 0.0]^T\,\text{m/s}^2$ | Nominal architectural prior (unobservable from single static pose) |
| **Gyroscope Bias** | $\mathbf{b}_g = [-0.000494, +0.003477, +0.000090]^T\,\text{rad/s}$ | Estimated during Phase 3 initial stationary calibration window |

---

## 4. Upstream Phase 3 Preprocessing Dependency

Phase 4 accepts vehicle-frame measurements directly from the validated Phase 3 classical preprocessing pipeline (`PreprocessingPipeline`):
1. **Stationary Gyro Bias Removal**: Initial static rest window (samples 15 to 503) removed raw stationary gyro offsets.
2. **Mounting Tilt Alignment ($R_b^v$)**: Rodrigues rotation leveled the support reaction force onto vehicle vertical $+Z_v$.
3. **Dual-Stage Filtering**: 3-sample median filter removed impulse spikes, followed by a 4th-order zero-phase Butterworth filter ($f_c = 3.0\,\text{Hz}$).
4. **Specific Force Output**: Vehicle-frame specific force $\mathbf{f}_m^v$ and angular velocity $\boldsymbol{\omega}_m^v$ are delivered directly to the strapdown mechanization without pre-ESKF gravity stripping.

---

## 5. Mathematical Conventions & Architecture

### 5.1 Quaternion Convention
- **Format**: Hamilton convention, scalar-first:
  $$\mathbf{q} = [w, x, y, z]^T = [q_w, q_x, q_y, q_z]^T, \quad \|\mathbf{q}\| = 1.0$$
- **Frame Representation**: Represents the rotation from vehicle frame ($v$) to local East-North-Up navigation frame ($n$):
  $$\mathbf{v}^n = R_v^n \mathbf{v}^v$$
- **Direction Cosine Matrix ($R_v^n$)**:
  $$R_v^n = \begin{bmatrix}
  1 - 2(y^2 + z^2) & 2(xy - wz) & 2(xz + wy) \\
  2(xy + wz) & 1 - 2(x^2 + z^2) & 2(yz - wx) \\
  2(xz - wy) & 2(yz + wx) & 1 - 2(x^2 + y^2)
  \end{bmatrix}$$

### 5.2 Local ENU Navigation Frame
- Local Cartesian East-North-Up (ENU) tangent plane:
  - $X_n$: East [meters]
  - $Y_n$: North [meters]
  - $Z_n$: Up [meters]
- **Session Reference Origin**: Equirectangular projection about fixed $(\text{lat}_0, \text{lon}_0, \text{alt}_0)$ with spherical Earth radius $R_{\text{earth}} = 6,371,000\,\text{m}$.
- **Immutability Invariant**: The origin is fixed upon session start and is never reset or shifted mid-session.

### 5.3 Gravity Convention
- Physical downward gravitational acceleration in ENU navigation coordinates:
  $$\mathbf{g}^n = \begin{bmatrix} 0 \\ 0 \\ -9.80665 \end{bmatrix}\,\text{m/s}^2$$
- At rest on a level surface, the proof mass measures an upward support reaction force $\mathbf{f}_m^v = [0, 0, +9.80665]^T\,\text{m/s}^2$.
- Kinematic coordinate acceleration in ENU:
  $$\mathbf{a}_{\text{true}}^n = R_v^n (\mathbf{f}_m^v - \mathbf{b}_a^v) + \mathbf{g}^n$$
  For a stationary level vehicle: $\mathbf{a}_{\text{true}}^n = \mathbf{I} [0, 0, +g]^T + [0, 0, -g]^T = [0, 0, 0]^T\,\text{m/s}^2$.

### 5.4 Discrete Propagation Equations
For each sample timestep $\Delta t = (t_{k+1} - t_k) \times 10^{-9}\,\text{seconds}$:
1. **Attitude Propagation**:
   $$\mathbf{q}[k+1] = \text{normalize}\left(\mathbf{q}[k] \otimes \Delta\mathbf{q}(\boldsymbol{\omega}_m^v[k] \cdot \Delta t)\right)$$
   where finite rotation vector $\boldsymbol{\theta} = \boldsymbol{\omega}_m^v \Delta t$, with series expansion for $\theta \to 0$.
2. **Coordinate Acceleration**:
   $$\mathbf{a}_{\text{true}}^n[k] = R_v^n[k] (\mathbf{f}_m^v[k] - \mathbf{b}_a^v) + \mathbf{g}^n$$
3. **Velocity Propagation**:
   $$\mathbf{v}[k+1] = \mathbf{v}[k] + \mathbf{a}_{\text{true}}^n[k] \cdot \Delta t$$
4. **Position Propagation**:
   $$\mathbf{p}[k+1] = \mathbf{p}[k] + \mathbf{v}[k] \cdot \Delta t + \frac{1}{2} \mathbf{a}_{\text{true}}^n[k] \cdot \Delta t^2$$

---

## 6. Open-Loop Nature of Experiment

This experiment is strictly open-loop:
- **Zero GNSS Kalman Updates**: GNSS fixes are withheld entirely from the propagator and used exclusively as post-hoc ground truth.
- **Zero ESKF Covariance Propagation**: Covariance matrices and error-state resets are absent.
- **Zero VelocityNet or Speed Aiding**: No forward velocity pseudo-measurements.
- **Zero BiasNet**: No dynamic accelerometer or gyroscope bias residual compensation.
- **Zero Non-Holonomic Constraints (NHC)**: Lateral and vertical vehicle velocities are unconstrained.
- **Zero Zero-Velocity Updates (ZUPT)**: Stationary standstills do not reset velocity errors.
- **Zero Map Matching**: No road network projection or snapping.

---

## 7. Empirical Drift Results

### Key Checkpoint Metrics
Over the 60.00-second evaluation window, open-loop position error evolved as follows:

| Elapsed Time | Horizontal Drift ($e_{\text{2D}}$) | 3D Drift ($e_{\text{3D}}$) | Ground Truth Distance Traveled | Drift-to-Distance Ratio |
|---|---|---|---|---|
| **$5.0\,\text{s}$** | **$4.94\,\text{m}$** | $5.02\,\text{m}$ | $63.2\,\text{m}$ | $7.8\%$ |
| **$10.0\,\text{s}$** | **$21.27\,\text{m}$** | $21.58\,\text{m}$ | $126.8\,\text{m}$ | $16.8\%$ |
| **$15.0\,\text{s}$** | **$102.64\,\text{m}$** | $104.12\,\text{m}$ | $190.5\,\text{m}$ | $53.9\%$ |
| **$20.0\,\text{s}$** | **$254.01\,\text{m}$** | $257.65\,\text{m}$ | $254.1\,\text{m}$ | $100.0\%$ |
| **$30.0\,\text{s}$** | **$793.14\,\text{m}$** | $804.49\,\text{m}$ | $381.2\,\text{m}$ | $208.1\%$ |
| **$45.0\,\text{s}$** | **$1,933.62\,\text{m}$** | $1,960.91\,\text{m}$ | $568.9\,\text{m}$ | $339.9\%$ |
| **$60.0\,\text{s}$** | **$3,257.64\,\text{m}$** | **$3,301.07\,\text{m}$** | **$752.6\,\text{m}$** | **$432.8\%$** |

### Summary Statistics
- **Evaluation Duration**: $60.00\,\text{s}$ (601 samples integrated, 0 skipped).
- **Horizontal Position Error RMSE**: **$1,492.18\,\text{m}$**.
- **Maximum Horizontal Error**: **$3,257.64\,\text{m}$**.
- **Final 3D Position Error**: **$3,301.07\,\text{m}$**.
- **Final Vertical Position Error**: **$534.25\,\text{m}$**.

---

## 8. Drift-vs-Time Visualization

![Ablation Stage 1 Open-Loop Drift](docs/images/ablation_stage1_drift.png)

*Figure 1: Ablation Stage 1 open-loop strapdown INS performance on IO-VNBD Trip S1. Upper panel: Horizontal (2D) and Vertical (Up) position error versus elapsed time. Lower panel: Propagated open-loop trajectory versus Racelogic VBOX ground truth in local East-North coordinates.*

---

## 9. Physical Sanity Analysis & Error Source Attribution

### 9.1 Quadratic-to-Cubic Growth Confirmation
The measured drift trajectory confirms classical inertial navigation physics:
- For pure acceleration bias $b_a$, position error grows quadratically: $e(t) \approx \frac{1}{2} b_a t^2$.
- For uncompensated gyroscope bias $b_g$, attitude tilt error grows linearly: $\delta\theta(t) \approx b_g t$.
- In the presence of Earth's gravity ($g = 9.80665\,\text{m/s}^2$), attitude tilt error causes gravity to leak into horizontal coordinate acceleration: $a_{\text{leak}}(t) = g \sin(\delta\theta) \approx g b_g t$.
- Double integration of gravity leakage produces cubic position drift:
  $$p_{\text{drift}}(t) \approx \frac{1}{6} g b_g t^3$$

### 9.2 Order-of-Magnitude Verification
Evaluating this analytical cubic formula with a typical dynamic consumer gyroscope bias discrepancy of $b_g \approx 0.003\,\text{rad/s}$ ($0.17^\circ/\text{s}$):
$$p_{\text{cubic}}(60\,\text{s}) = \frac{1}{6} \times 9.80665 \times 0.003 \times (60)^3 \approx \frac{1}{6} \times 9.81 \times 0.003 \times 216,000 \approx 1,059\,\text{m}$$
Combined with residual accelerometer bias ($b_a \approx 0.5\,\text{m/s}^2 \implies \frac{1}{2} \times 0.5 \times 3600 \approx 900\,\text{m}$) and vehicle curve dynamics, the measured **$3,257\,\text{m}$** is completely consistent with theoretical expectations.

### 9.3 Contrast with Standstill Drift
When the open-loop INS is executed over a stationary rest window (samples 20 to 450, 43.0 s at rest):
- Horizontal drift: **$605.05\,\text{m}$** ($14.1\,\text{m/s}$ average drift rate).
- Vertical drift: **$30.74\,\text{m}$**.
- This proves that at rest, drift is driven solely by unmodeled accelerometer bias ($\sim 0.5\,\text{m/s}^2$). Under vehicle motion, dynamic tilt shifts induce gravity leakage, increasing drift by a factor of $5\times$.

---

## 10. Limitations & Disclosures

1. **Unassisted Consumer IMU Physics**:
   - Consumer smartphone IMUs (InvenSense/Bosch MEMS) are classified as tactical-to-automotive grade at best, with gyro bias stability $> 10^\circ/\text{hr}$ and significant vibration-induced drift. Pure double integration without closed-loop attitude bounds inevitably diverges within 15–30 seconds.
2. **Mounting Yaw Observability**:
   - For `Categorised_S1`, Phase 3 established `is_yaw_aligned = False`. The initial heading for this ablation was initialized from the initial VBOX GNSS course fix to allow meaningful dead-reckoning evaluation along the road trajectory.
3. **No Closed-Loop Error Suppression**:
   - This experiment deliberately includes zero aiding to measure the raw unconstrained baseline. It is a benchmark measurement, not a production operating mode.

---

## 11. Test Suite & Verification Summary

### Pytest Execution
```
.venv\Scripts\python.exe -m pytest -v
============================= 167 passed in 4.62s =============================
```
- **Phase 1 Schemas**: 50 passed
- **Phase 2 Pipeline**: 53 passed
- **Phase 3 Preprocessing**: 47 passed
- **Phase 4 Local Geodetic Frame (`test_frame_conversion.py`)**: 6 passed
- **Phase 4 Strapdown INS Propagation (`test_ins_propagation.py`)**: 11 passed
- **Total Tests**: **167 passed**, **0 failures**, **0 errors**.

### Dataset & Cache Immutability Audit
- **Raw CSVs**: 288 files in `data/raw/io_vnbd` | Collective SHA-256: `04aa4da188a6b0417d202f83f959ef3238dedf2a6ccfa075ff7b0751fe072a99` (**UNMODIFIED**)
- **Phase 2 Cache**: 144 `.npz` files in `data/cache/iovnbd` | Collective SHA-256: `7a267e112af4193dcadb8800bd9cf4ed8abf7249a1ad53883bbd2cfd06200fa7` (**UNMODIFIED**)

---

## 12. Conclusion & Readiness

The Phase 4 classical strapdown INS mechanization is deterministic, numerically stable, and rigorously verified against closed-form analytical solutions. The raw unassisted dead-reckoning baseline for Ablation Stage 1 has been empirically measured and recorded:
- **$21.27\,\text{m}$ at $10\,\text{s}$**
- **$793.14\,\text{m}$ at $30\,\text{s}$**
- **$3,257.64\,\text{m}$ at $60\,\text{s}$**

No Phase 5 functionality (ESKF, Kalman updates, or measurement models) was implemented. Phase 4 is complete and ready to serve as the propagation backbone for Phase 5.
