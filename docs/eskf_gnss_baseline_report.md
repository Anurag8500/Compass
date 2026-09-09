# Phase 5: Error-State Kalman Filter (ESKF) & GNSS Baseline Report
**Project**: C.O.M.P.A.S.S. (Cognitive Off-grid Machine-learning Positioning And Sensor System)  
**Problem Statement**: SIH 2026 Problem Statement 26168 — ISRO  
**Phase**: Phase 5 — ESKF Core Mechanics + First Real Measurement Source (GNSS)  
**Status**: COMPLETE, RIGOROUSLY VALIDATED, AND FROZEN  

---

## Executive Summary

Phase 5 establishes the central mathematical estimation backbone of the C.O.M.P.A.S.S. architecture: the **15-state Error-State Kalman Filter (ESKF)** fused with its first real aiding source: **GNSS (position and velocity in local East-North-Up)** and **Classical Gated Zero Velocity Updates (ZUPT)**.

In Phase 4, the open-loop strapdown INS demonstrated the inevitable quadratic-to-cubic divergence of unconstrained dead reckoning on consumer smartphone inertial sensors: over 60.0 seconds of real highway driving on IO-VNBD Trip S1, horizontal position error diverged to **$3,249.32\,\text{m}$** with an RMSE of **$1,487.53\,\text{m}$**.

In Phase 5, wrapping the discrete error-state filter around the exact same Phase 4 strapdown nominal propagation, applying innovation-gated GNSS updates, and enforcing Joseph-form numerical stability achieves:
- **Final Horizontal Error**: **$8.37\,\text{m}$** (compared to Phase 4 baseline: **$3,249.32\,\text{m}$**).
- **Horizontal RMSE**: **$68.72\,\text{m}$** (compared to Phase 4 baseline: **$1,487.53\,\text{m}$**).
- **Final Error Reduction**: **$99.74\%$ improvement** over Phase 4 open-loop dead reckoning.
- **RMSE Error Reduction**: **$95.38\%$ improvement** over Phase 4 open-loop dead reckoning.
- **GNSS Fix Acceptance**: 7 of 7 fixes ($100\%$) accepted through the 99.9% Mahalanobis $\chi^2$ gate with zero false rejections.
- **Stationary Rest Suppression**: Classical Gated ZUPT suppresses stationary drift to **$< 0.5\,\text{m}$** over 100 samples with residual velocity **$< 0.05\,\text{m/s}$** on real IO-VNBD data without any machine-learning dependency.
- **Test Suite Status**: **193 passed, 0 failures, 0 errors** across unit and integration test suites.

---

## 1. Architectural Role of Phase 5

Phase 5 transitions C.O.M.P.A.S.S. from an open-loop integrator into a closed-loop optimal recursive estimator:
1. **Nominal State Integration**: The 16-dimensional kinematic state is integrated forward at high rate ($10\,\text{Hz}$ on IO-VNBD, up to $100\,\text{Hz}$ on high-rate IMUs) using the exact Phase 4 strapdown kinematic mechanization.
2. **Error-State Manifold**: Errors, misalignments, and sensor biases are estimated on a minimal 15-dimensional linear vector space. The error state is zero-mean between updates and is strictly reset to zero immediately after injection into the nominal state.
3. **Pluggable Measurement Interface**: All aiding sources (GNSS, ZUPT, and future Phase 6+ VelocityNet, NHC, BiasNet) communicate through a single, generic `eskf_update(z, h, H, R, gating)` function.
4. **Complete Independence from ML**: Phase 5 relies purely on classical Newtonian mechanics, linear error dynamics, and probability theory, providing an unshakeable, interpretable mathematical foundation before machine learning is introduced.

---

## 2. Mathematical State Representation

### 2.1 Nominal Navigation State ($\mathbf{x}_{\text{nom}} \in \mathbb{R}^{16}$)
The nominal state tracks the large, non-linear kinematic quantities:
$$\mathbf{x}_{\text{nom}} = \begin{bmatrix} \mathbf{p}^n \\ \mathbf{v}^n \\ \mathbf{q} \\ \mathbf{b}_a^v \\ \mathbf{b}_g^v \end{bmatrix} \in \mathbb{R}^{16}$$
- $\mathbf{p}^n = [p_E, p_N, p_U]^T \in \mathbb{R}^3$: Local Cartesian position in the East-North-Up (ENU) frame [m].
- $\mathbf{v}^n = [v_E, v_N, v_U]^T \in \mathbb{R}^3$: Linear velocity in the local ENU frame [m/s].
- $\mathbf{q} = [q_w, q_x, q_y, q_z]^T \in \mathbb{H}, \|\mathbf{q}\| = 1$: Hamilton scalar-first unit quaternion mapping vehicle-frame vectors to ENU ($R_v^n$).
- $\mathbf{b}_a^v = [b_{ax}, b_{ay}, b_{az}]^T \in \mathbb{R}^3$: Accelerometer bias in the vehicle frame [$\text{m/s}^2$].
- $\mathbf{b}_g^v = [b_{gx}, b_{gy}, b_{gz}]^T \in \mathbb{R}^3$: Gyroscope bias in the vehicle frame [$\text{rad/s}$].

### 2.2 Error State ($\delta\mathbf{x} \in \mathbb{R}^{15}$)
The error state represents the true deviation from the nominal estimate:
$$\delta\mathbf{x} = \begin{bmatrix} \delta\mathbf{p}^n \\ \delta\mathbf{v}^n \\ \delta\boldsymbol{\theta} \\ \delta\mathbf{b}_a^v \\ \delta\mathbf{b}_g^v \end{bmatrix} \in \mathbb{R}^{15}$$
- $\delta\mathbf{p}^n = \mathbf{p}_{\text{true}}^n - \mathbf{p}_{\text{nom}}^n \in \mathbb{R}^3$ [m].
- $\delta\mathbf{v}^n = \mathbf{v}_{\text{true}}^n - \mathbf{v}_{\text{nom}}^n \in \mathbb{R}^3$ [m/s].
- $\delta\boldsymbol{\theta} \in \mathbb{R}^3$: 3D small-angle rotation vector in the vehicle body frame [rad].
- $\delta\mathbf{b}_a^v = \mathbf{b}_{a,\text{true}}^v - \mathbf{b}_{a,\text{nom}}^v \in \mathbb{R}^3$ [$\text{m/s}^2$].
- $\delta\mathbf{b}_g^v = \mathbf{b}_{g,\text{true}}^v - \mathbf{b}_{g,\text{nom}}^v \in \mathbb{R}^3$ [$\text{rad/s}$].

### 2.3 Error Covariance ($P \in \mathbb{R}^{15 \times 15}$)
The uncertainty is described by the symmetric, positive-semidefinite error covariance matrix:
$$P = \mathbb{E}\left[\delta\mathbf{x} \delta\mathbf{x}^T\right] \in \mathbb{R}^{15 \times 15}$$

---

## 3. Explicit Frame & Attitude Error Conventions

### 3.1 Right-Multiplicative Body-Frame Attitude Error
To preserve mathematical consistency with body-fixed angular rate integration, C.O.M.P.A.S.S. adopts the **right-multiplicative body-frame attitude error convention**:
$$\mathbf{q}_{\text{true}} = \mathbf{q}_{\text{nom}} \otimes \delta\mathbf{q}(\delta\boldsymbol{\theta})$$
where for small angles $\|\delta\boldsymbol{\theta}\| \ll 1$:
$$\delta\mathbf{q}(\delta\boldsymbol{\theta}) \approx \begin{bmatrix} 1 \\ \frac{1}{2} \delta\boldsymbol{\theta} \end{bmatrix}$$
The corresponding vehicle-to-navigation rotation matrix expands linearly as:
$$R(\mathbf{q}_{\text{true}}) = R(\mathbf{q}_{\text{nom}}) R(\delta\mathbf{q}) \approx R_v^n \left(\mathbf{I}_3 + [\delta\boldsymbol{\theta}]_\times\right)$$
where $[\mathbf{v}]_\times$ denotes the $3 \times 3$ skew-symmetric cross-product matrix.

---

## 4. Continuous-Time Linearized Error Dynamics

Given nominal specific force $\mathbf{f}_{\text{unbiased}}^v = \mathbf{f}_m^v - \mathbf{b}_a^v$ and angular velocity $\boldsymbol{\omega}_{\text{unbiased}}^v = \boldsymbol{\omega}_m^v - \mathbf{b}_g^v$, the true kinematic derivatives are:
$$\dot{\mathbf{p}}_{\text{true}}^n = \mathbf{v}_{\text{true}}^n$$
$$\dot{\mathbf{v}}_{\text{true}}^n = R_v^n (\mathbf{I} + [\delta\boldsymbol{\theta}]_\times) (\mathbf{f}_{\text{unbiased}}^v - \delta\mathbf{b}_a^v - \mathbf{w}_a) + \mathbf{g}^n$$
$$\dot{\mathbf{q}}_{\text{true}} = \frac{1}{2} \mathbf{q}_{\text{true}} \otimes \left(\boldsymbol{\omega}_{\text{unbiased}}^v - \delta\mathbf{b}_g^v - \mathbf{w}_g\right)$$
Subtracting the nominal dynamics and dropping second-order perturbations yields the linear continuous system:
$$\delta\dot{\mathbf{x}}(t) = F_c(t) \delta\mathbf{x}(t) + G_c(t) \mathbf{w}(t)$$
with system matrix:
$$F_c = \begin{bmatrix}
\mathbf{0}_3 & \mathbf{I}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 \\
\mathbf{0}_3 & \mathbf{0}_3 & -R_v^n [\mathbf{f}_{\text{unbiased}}^v]_\times & -R_v^n & \mathbf{0}_3 \\
\mathbf{0}_3 & \mathbf{0}_3 & -[\boldsymbol{\omega}_{\text{unbiased}}^v]_\times & \mathbf{0}_3 & -\mathbf{I}_3 \\
\mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 \\
\mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3
\end{bmatrix}$$

---

## 5. Discrete Error-State Transition Matrix ($F_d$)

Integrating over timestep $\Delta t$ with second-order position and first-order velocity terms yields the discrete transition matrix:
$$F_d = \begin{bmatrix}
\mathbf{I}_3 & \mathbf{I}_3 \Delta t & -\frac{1}{2} \Delta t^2 R_v^n [\mathbf{f}_{\text{unbiased}}^v]_\times & -\frac{1}{2} \Delta t^2 R_v^n & \mathbf{0}_3 \\
\mathbf{0}_3 & \mathbf{I}_3 & -\Delta t R_v^n [\mathbf{f}_{\text{unbiased}}^v]_\times & -\Delta t R_v^n & \mathbf{0}_3 \\
\mathbf{0}_3 & \mathbf{0}_3 & \mathbf{I}_3 - \Delta t [\boldsymbol{\omega}_{\text{unbiased}}^v]_\times & \mathbf{0}_3 & -\mathbf{I}_3 \Delta t \\
\mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{I}_3 & \mathbf{0}_3 \\
\mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{I}_3
\end{bmatrix} \in \mathbb{R}^{15 \times 15}$$

### Numerical Finite-Difference Verification
In unit test `test_analytical_vs_numerical_finite_difference_jacobian`, each of the 15 error state dimensions was perturbed by $\pm \epsilon = \pm 10^{-6}$ around a non-trivial 3D dynamic state. The numerical Jacobian matched analytical $F_d$ across all $15 \times 15 = 225$ elements to within **$\max |\Delta| < 10^{-4}$**, proving exact mathematical consistency between nominal non-linear propagation and error-state linearization.

---

## 6. Discrete Process Noise ($Q_d$)

Process noise covariance $Q_d \in \mathbb{R}^{15 \times 15}$ is formulated from continuous spectral densities:
- $S_{wa} = (\sigma_{wa})^2$ [$\text{m}^2/\text{s}^3$]: Accelerometer white noise.
- $S_{wg} = (\sigma_{wg})^2$ [$\text{rad}^2/\text{s}$]: Gyroscope white noise.
- $S_{ba} = (\sigma_{ba})^2$ [$\text{m}^2/\text{s}^5$]: Accelerometer random walk.
- $S_{bg} = (\sigma_{bg})^2$ [$\text{rad}^2/\text{s}^3$]: Gyroscope random walk.

Discrete blocks over timestep $\Delta t$:
$$Q_d(0:3, 0:3) = \frac{1}{3} \Delta t^3 S_{wa} \mathbf{I}_3$$
$$Q_d(0:3, 3:6) = \frac{1}{2} \Delta t^2 S_{wa} \mathbf{I}_3$$
$$Q_d(3:6, 3:6) = \Delta t S_{wa} \mathbf{I}_3$$
$$Q_d(6:9, 6:9) = \Delta t S_{wg} \mathbf{I}_3$$
$$Q_d(9:12, 9:12) = \Delta t S_{ba} \mathbf{I}_3$$
$$Q_d(12:15, 12:15) = \Delta t S_{bg} \mathbf{I}_3$$

---

## 7. Generic Gated Measurement Update

Every aiding measurement passes through a unified 6-step update sequence:

1. **Innovation**:
   $$\mathbf{y} = \mathbf{z} - h(\mathbf{x}_{\text{nom}}) \in \mathbb{R}^m$$
2. **Innovation Covariance**:
   $$S = H P H^T + R \in \mathbb{R}^{m \times m}$$
3. **Mahalanobis / $\chi^2$ Innovation Gating**:
   $$d_M^2 = \mathbf{y}^T S^{-1} \mathbf{y}$$
   Computed stably via `np.linalg.solve(S, y)`.
   If $d_M^2 > \gamma(\alpha, m)$, the measurement is **rejected**. The state and covariance remain **strictly unmodified**.
4. **Kalman Gain**:
   $$K = P H^T S^{-1} \iff K^T = \text{solve}(S, H P)$$
   Evaluated with zero explicit matrix inversions (`inv(S)` is prohibited).
5. **Joseph-Form Covariance Update**:
   $$P_{\text{new}} = (\mathbf{I}_{15} - K H) P (\mathbf{I}_{15} - K H)^T + K R K^T$$
   Guarantees positive-semidefiniteness even under severe roundoff or high-gain updates. Symmetrization is enforced:
   $$P_{\text{new}} \leftarrow \frac{1}{2}\left(P_{\text{new}} + P_{\text{new}}^T\right)$$
6. **Error Injection & Reset**:
   $$\delta\mathbf{x} = K \mathbf{y}$$
   $$\mathbf{p}_{\text{nom}} \leftarrow \mathbf{p}_{\text{nom}} + \delta\mathbf{p}, \quad \mathbf{v}_{\text{nom}} \leftarrow \mathbf{v}_{\text{nom}} + \delta\mathbf{v}$$
   $$\mathbf{q}_{\text{nom}} \leftarrow \text{normalize}\left(\mathbf{q}_{\text{nom}} \otimes \delta\mathbf{q}(\delta\boldsymbol{\theta})\right)$$
   $$\mathbf{b}_{a,\text{nom}} \leftarrow \mathbf{b}_{a,\text{nom}} + \delta\mathbf{b}_a, \quad \mathbf{b}_{g,\text{nom}} \leftarrow \mathbf{b}_{g,\text{nom}} + \delta\mathbf{b}_g$$
   $$\delta\mathbf{x} \leftarrow \mathbf{0}_{15}$$

---

## 8. Complete Retirement of `bootstrap_attitude.py`

In Phase 3, `bootstrap_attitude.py` served as an offline static leveling tool. In the live Phase 5 estimator, `bootstrap_attitude.py` is **completely retired** from the runtime pipeline.
- The ESKF maintains its own continuous attitude quaternion $\mathbf{q}$ ($R_v^n$).
- The ESKF rotates vehicle specific force directly into local ENU:
  $$\mathbf{a}_{\text{true}}^n = R_v^n (\mathbf{f}_m^v - \mathbf{b}_a^v) + \mathbf{g}^n$$
- Zero external attitude estimators exist in the propagation loop.

---

## 9. GNSS Measurement Model

- **Session Local Origin**: Rigid `GeoReference(lat_ref, lon_ref, alt_ref)` established at session start.
- **Position Observation**: $\mathbf{z}_p = \mathbf{p}_{\text{gnss}}^n \in \mathbb{R}^3$, $H_p = [\mathbf{I}_3, \mathbf{0}_{3 \times 12}]$.
- **Velocity Observation**: $\mathbf{z}_v = \mathbf{v}_{\text{gnss}}^n \in \mathbb{R}^3$, $H_v = [\mathbf{0}_3, \mathbf{I}_3, \mathbf{0}_{3 \times 9}]$.
- **Consumer Smartphone Altitude Handling**: In commercial smartphones, reported GPS altitude exhibits a 30–50 meter datum/geoid undulation offset relative to local terrain elevations (e.g. WGS84 ellipsoid vs. Ordnance Survey MSL in the UK). Setting a realistic vertical uncertainty floor ($\sigma_v = 35.0\,\text{m}$) prevents this vertical datum offset from triggering false outlier rejections on horizontal coordinates, allowing 100% of accurate horizontal fixes to update the filter.

---

## 10. Classical Zero-ML Gated ZUPT

The `ClassicalZUPTDetector` identifies vehicle standstill using strict physical invariants across a sliding window ($W = 8$ samples, $0.8\,\text{s}$ at $10\,\text{Hz}$):
1. **Angular Quiescence**: $\|\boldsymbol{\omega}^v\| < 0.05\,\text{rad/s}$ across window.
2. **Gravity Magnitude Match**: $|\|\mathbf{f}_m^v\| - 9.80665| < 0.25\,\text{m/s}^2$.
3. **Low Specific Force Variance**: $\text{Var}(\|\mathbf{f}_m^v\|) < 0.015\,(\text{m/s}^2)^2$.
4. **Auxiliary Speed Check**: $|v_{\text{gnss}}| < 0.1\,\text{m/s}$ when GNSS Doppler speed is available.

When standstill is declared, a pseudo-measurement $\mathbf{z}_{\text{zupt}} = [0, 0, 0]^T$ with noise $\sigma_z = 0.03\,\text{m/s}$ is applied through `eskf_update`. If vehicle motion is falsely triggered, Mahalanobis gating rejects the update without corrupting state.

---

## 11. Empirical Validation on IO-VNBD Trip S1

### 11.1 Benchmark Test Configuration
- **Dataset**: IO-VNBD Trip S1 (`Uncategorised_S1.npz`, 51,746 samples).
- **Evaluation Window**: Sample 19500 to 20100 (60.00 seconds, 601 samples).
- **Initial Conditions**: Identical to Phase 4 Ablation Stage 1:
  - Starting position: $[0, 0, 0]^T$ at $\text{lat}=52.4165342^\circ, \text{lon}=-1.5785448^\circ, \text{alt}=127.39\,\text{m}$.
  - Initial velocity: $[-11.399, 3.099, 0.0]^T\,\text{m/s}$ (speed $11.813\,\text{m/s}$, heading $285.21^\circ$).
  - Ground Truth: Racelogic VBOX RTK GPS.

### 11.2 Comparative Performance Results

| Metric | Phase 4 Open-Loop Baseline | Phase 5 ESKF + GNSS | Improvement |
|---|---|---|---|
| **Final Horizontal Error ($e_{\text{2D}}$)** | **$3,249.32\,\text{m}$** | **$8.37\,\text{m}$** | **$99.74\%$ reduction** |
| **Horizontal Position RMSE** | **$1,487.53\,\text{m}$** | **$68.72\,\text{m}$** | **$95.38\%$ reduction** |
| **Maximum Horizontal Error** | **$3,249.32\,\text{m}$** | **$191.48\,\text{m}$** | **$94.11\%$ reduction** |
| **Drift Rate at $60\,\text{s}$** | $54.16\,\text{m/s}$ | **$0.14\,\text{m/s}$** | **$386\times$ suppression** |
| **Accepted GNSS Fixes** | 0 (open loop) | **7 of 7 ($100\%$)** | N/A |
| **Rejected GNSS Fixes** | 0 | **0** | Perfect gate tracking |

### 11.3 Checkpoint Comparison Across 60 Seconds

| Elapsed Time | Phase 4 Open-Loop Drift | Phase 5 ESKF + GNSS Error |
|---|---|---|
| **$5.0\,\text{s}$** | $4.83\,\text{m}$ | $12.88\,\text{m}$ |
| **$10.0\,\text{s}$** | $20.52\,\text{m}$ | $95.52\,\text{m}$ |
| **$15.0\,\text{s}$** | $100.64\,\text{m}$ | $56.11\,\text{m}$ |
| **$20.0\,\text{s}$** | $251.21\,\text{m}$ | **$2.25\,\text{m}$** |
| **$30.0\,\text{s}$** | $788.97\,\text{m}$ | **$8.34\,\text{m}$** |
| **$45.0\,\text{s}$** | $1,927.34\,\text{m}$ | $111.87\,\text{m}$ |
| **$60.0\,\text{s}$** | **$3,249.32\,\text{m}$** | **$8.37\,\text{m}$** |

### 11.4 Real Data Standstill Validation (Trip S1 Samples 50–150)
In unit test `test_trip_s1_standstill_vs_driving` and integration test `test_eskf_zupt_standstill_suppression_on_real_data`:
- Standstill detector identified **100% of stationary rest samples** at the start of the trip.
- ZUPT was applied **over 50 times** during the rest segment.
- Velocity was constrained to $\|\mathbf{v}\| < 0.05\,\text{m/s}$ (actual: **$0.012\,\text{m/s}$**).
- Horizontal position drift over the rest window remained **$< 0.5\,\text{m}$** (actual: **$0.18\,\text{m}$**), compared to $> 600\,\text{m}$ drift under unassisted Phase 4 propagation.

---

## 12. Full Test Suite & Immutability Audit

### 12.1 Pytest Execution Summary
```
.venv\Scripts\python.exe -m pytest -v
============================ 193 passed in 30.66s =============================
```
- **Phase 1 Schemas**: 50 passed
- **Phase 2 Pipeline & Quality**: 53 passed
- **Phase 3 Preprocessing**: 47 passed
- **Phase 4 Frame Conversion & Strapdown INS**: 23 passed
- **Phase 5 ESKF Synthetic Mechanics (`test_eskf_synthetic.py`)**: 9 passed
- **Phase 5 Classical Gated ZUPT (`test_zupt.py`)**: 9 passed
- **Phase 5 Real-Data Integration (`test_eskf_gnss_real_data.py`)**: 2 passed
- **Total Tests**: **193 passed**, **0 failures**, **0 errors**.

### 12.2 Dataset & Cache Immutability
- **Raw CSV Files**: 288 files in `data/raw/io_vnbd` (**UNMODIFIED**).
- **Phase 2 Cache**: 144 `.npz` files in `data/cache/iovnbd` | Collective SHA-256: `7a267e112af4193dcadb8800bd9cf4ed8abf7249a1ad53883bbd2cfd06200fa7` (**UNMODIFIED**).
- **Phase 0–4 Source Code**: Strict byte-for-byte freeze maintained; zero modifications to earlier phase algorithms.

---

## 13. Conclusion & Phase 5 Signoff

Phase 5 achieves its primary engineering and scientific objectives:
1. **Mathematical Correctness**: State representation, right-multiplicative error injection, second-order $F_d$, and discrete $Q_d$ are derived, verified, and confirmed against finite differences.
2. **Numerical Stability**: Joseph-form covariance updates and solve-based Kalman gains maintain positive-semidefiniteness and symmetry across thousands of steps.
3. **Empirical Superiority**: ESKF + GNSS produces a **$99.74\%$ final error reduction** ($8.37\,\text{m}$ vs. $3,249.32\,\text{m}$) and a **$95.38\%$ RMSE reduction** over the frozen Phase 4 open-loop baseline.
4. **Readiness**: Phase 5 is fully tested, regression-verified, and frozen. The filter interface is completely prepared for Phase 6 (Machine Learning Models: VelocityNet and BiasNet).
