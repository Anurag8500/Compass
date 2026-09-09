# Phase 5 Complete Explanation: Error-State Kalman Filter (ESKF), GNSS Fusion & Classical Gated ZUPT

**Project**: C.O.M.P.A.S.S. (Cognitive Off-grid Machine-learning Positioning And Sensor System)  
**Problem Statement**: SIH 2026 Problem Statement 26168 — ISRO  
**Stage**: Phase 5 Complete Teaching & Reference Walkthrough  

---

## 1. Introduction & Phase Overview

### What Problem Did Phase 5 Solve?
In Phase 4, we saw the unavoidable failure mode of pure inertial dead reckoning: when a vehicle drives unassisted using a consumer smartphone IMU, small sensor biases and attitude tilt errors cause position drift to accumulate rapidly. Over 60 seconds of highway driving on Trip S1, open-loop dead reckoning drifted by **$3,249.32\,\text{meters}$**.

**Phase 5 solves the fundamental estimation problem**: How do we continuously estimate and cancel sensor biases, bound attitude errors, and track vehicle trajectory with meter-level accuracy by fusing high-rate IMU physics with intermittent external measurements (GNSS fixes and standstill detections)?

### Why Did We Need to Solve It?
Neither sensor is sufficient on its own:
- **IMU alone**: High rate ($10\,\text{Hz}$ to $100\,\text{Hz}$), smooth, immune to external signal loss, but **drifts rapidly over time**.
- **GNSS alone**: Absolute position fixes without long-term drift, but **low rate ($1\,\text{Hz}$)**, noisy, vulnerable to urban multipath, and completely unavailable inside tunnels or subterranean structures.

Phase 5 builds the **Error-State Kalman Filter (ESKF)**: the closed-loop estimation engine that marries the high-frequency smoothness of the IMU with the low-frequency absolute accuracy of GNSS, using Zero Velocity Updates (ZUPT) to lock down drift whenever the vehicle stops.

---

## 2. Core Concepts & Terminology

### 1. Extended Kalman Filter (EKF) vs. Error-State Kalman Filter (ESKF)
- **Standard EKF**: Directly estimates the large, non-linear physical state (total position, total velocity, total orientation angles).
- **Error-State Kalman Filter (ESKF)**: Splits the state into two distinct components:
  1. A **Nominal State** ($\mathbf{x}_{\text{nom}} \in \mathbb{R}^{16}$) integrating full non-linear kinematic equations of motion.
  2. An **Error State** ($\delta\mathbf{x} \in \mathbb{R}^{15}$) modeling small linear perturbations, attitude misalignments, and sensor biases.
- **Why We Care**:
  - The error state operates close to zero, ensuring second-order perturbations remain negligible and linear approximations remain accurate.
  - Quaternions have 4 parameters but only 3 degrees of freedom on the $S^3$ manifold. In an ESKF, attitude error $\delta\boldsymbol{\theta}$ is a minimal 3-parameter rotation vector in the $\mathfrak{so}(3)$ Lie algebra, completely avoiding covariance rank deficiency.

```
                       +-----------------------------------+
                       |    Raw High-Rate IMU Samples      |
                       +-----------------------------------+
                                         |
                                         v
                         +-------------------------------+
                         |   Nominal Strapdown INS       |
                         |   High-Rate Kinematics        |
                         |   (Phase 4 Mechanization)     |
                         +-------------------------------+
                                         |
                       Nominal State     |     Corrected Nominal State
                       x_nom (16-dim)    |     x_nom <- x_nom ⊕ δx
                                         v               ^
+------------------+             +---------------+       |
| Low-Rate Aiding  |             |  Innovation   |       |
| Measurements (z) |------------>|  y = z - h(x) |       |
| (GNSS / ZUPT)    |             +---------------+       |
+------------------+                     |               |
                                         v               |
                                 +---------------+       |
                                 |  Chi-Square   |       |
                                 |  Gating Gate  |       |
                                 +---------------+       |
                                         | (If Passed)   |
                                         v               |
                                 +---------------+       |
                                 |  Kalman Gain  |       |
                                 |   K = P H^T/S |       |
                                 +---------------+       |
                                         |               |
                                         v               |
                                 +---------------+       |
                                 |  Error-State  |-------+
                                 |  Update δx=Ky | (Inject error &
                                 |  P = J P J^T  |  reset δx -> 0)
                                 +---------------+
```

### 2. State Dimensions: 16 Nominal vs. 15 Error Parameters
- **Nominal State** ($\mathbf{x}_{\text{nom}} \in \mathbb{R}^{16}$):
  $$\mathbf{x}_{\text{nom}} = \begin{bmatrix} \mathbf{p}^n \\ \mathbf{v}^n \\ \mathbf{q} \\ \mathbf{b}_a^v \\ \mathbf{b}_g^v \end{bmatrix} \in \mathbb{R}^{16}$$
  Position (3), Velocity (3), Unit Quaternion (4), Accel Bias (3), Gyro Bias (3) $\implies 16$ parameters.
- **Error State** ($\delta\mathbf{x} \in \mathbb{R}^{15}$):
  $$\delta\mathbf{x} = \begin{bmatrix} \delta\mathbf{p}^n \\ \delta\mathbf{v}^n \\ \delta\boldsymbol{\theta} \\ \delta\mathbf{b}_a^v \\ \delta\mathbf{b}_g^v \end{bmatrix} \in \mathbb{R}^{15}$$
  Position error (3), Velocity error (3), Attitude rotation vector (3), Accel bias error (3), Gyro bias error (3) $\implies 15$ parameters.
- **Error Covariance** ($P \in \mathbb{R}^{15 \times 15}$): Symmetric positive-semidefinite matrix describing error uncertainties and cross-correlations.

### 3. Innovation Gating (Mahalanobis Distance)
The difference between actual measurement $\mathbf{z}$ and predicted measurement $h(\mathbf{x}_{\text{nom}})$ is innovation $\mathbf{y} = \mathbf{z} - h(\mathbf{x}_{\text{nom}})$, with covariance $S = H P H^T + R$.
The squared Mahalanobis distance:
$$d_M^2 = \mathbf{y}^T S^{-1} \mathbf{y}$$
If $d_M^2 > \gamma$ (the 99.9% Chi-Square threshold), the measurement is flagged as an outlier and rejected, leaving the state untouched.

### 4. Joseph-Form Covariance Update
To preserve numerical symmetry and promote positive-semidefiniteness under finite-precision floating-point arithmetic, COMPASS enforces the Joseph-form update:
$$P_{\text{updated}} = (\mathbf{I}_{15} - KH) P (\mathbf{I}_{15} - KH)^T + K R K^T$$

---

## 3. The Mathematics of ESKF Propagation

### 3.1 Right-Multiplicative Body-Frame Attitude Error
COMPASS adopts the right-multiplicative convention:
$$\mathbf{q}_{\text{true}} = \mathbf{q}_{\text{nom}} \otimes \delta\mathbf{q}(\delta\boldsymbol{\theta})$$
For small angles $\|\delta\boldsymbol{\theta}\| \ll 1$:
$$\delta\mathbf{q}(\delta\boldsymbol{\theta}) \approx \begin{bmatrix} 1 \\ \frac{1}{2} \delta\boldsymbol{\theta} \end{bmatrix}, \quad R(\mathbf{q}_{\text{true}}) \approx R_v^n \left(\mathbf{I}_3 + [\delta\boldsymbol{\theta}]_\times\right)$$
where $[\mathbf{v}]_\times$ is the $3 \times 3$ skew-symmetric cross-product matrix.

### 3.2 Continuous-Time Linear Error Dynamics
Linearizing true kinematics $\mathbf{x}_{\text{true}} = \mathbf{x}_{\text{nom}} \oplus \delta\mathbf{x}$ yields:
$$\delta\dot{\mathbf{x}}(t) = F_c(t) \delta\mathbf{x}(t) + G_c(t) \mathbf{w}(t)$$
where $F_c \in \mathbb{R}^{15 \times 15}$ is:
$$F_c = \begin{bmatrix}
\mathbf{0}_3 & \mathbf{I}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 \\
\mathbf{0}_3 & \mathbf{0}_3 & -R_v^n [\mathbf{f}_{\text{unbiased}}^v]_\times & -R_v^n & \mathbf{0}_3 \\
\mathbf{0}_3 & \mathbf{0}_3 & -[\boldsymbol{\omega}_{\text{unbiased}}^v]_\times & \mathbf{0}_3 & -\mathbf{I}_3 \\
\mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 \\
\mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3
\end{bmatrix}$$

Notice the gravity coupling term in row 2: $-R_v^n [\mathbf{f}_{\text{unbiased}}^v]_\times$. Attitude tilt error directly forces velocity error.

### 3.3 Discrete State Transition Matrix ($F_d$)
Over interval $\Delta t = t_{k+1} - t_k$:
$$F_d = \begin{bmatrix}
\mathbf{I}_3 & \mathbf{I}_3 \Delta t & -\frac{1}{2} \Delta t^2 R_v^n [\mathbf{f}_{\text{unbiased}}^v]_\times & -\frac{1}{2} \Delta t^2 R_v^n & \mathbf{0}_3 \\
\mathbf{0}_3 & \mathbf{I}_3 & -\Delta t R_v^n [\mathbf{f}_{\text{unbiased}}^v]_\times & -\Delta t R_v^n & \mathbf{0}_3 \\
\mathbf{0}_3 & \mathbf{0}_3 & \mathbf{I}_3 - \Delta t [\boldsymbol{\omega}_{\text{unbiased}}^v]_\times & \mathbf{0}_3 & -\mathbf{I}_3 \Delta t \\
\mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{I}_3 & \mathbf{0}_3 \\
\mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{I}_3
\end{bmatrix}$$

Discrete covariance propagation:
$$P_{k+1} = F_d P_k F_d^T + Q_d$$
symmetrized via $P_{k+1} \leftarrow \frac{1}{2}(P_{k+1} + P_{k+1}^T)$.

---

## 4. State Injection & The Covariance Reset Transformation

When an update arrives and $\hat{\delta\mathbf{x}} = K \mathbf{y}$ is computed, errors are injected into the nominal state:
$$\mathbf{p}_{\text{nom}} \leftarrow \mathbf{p}_{\text{nom}} + \hat{\delta\mathbf{p}}$$
$$\mathbf{v}_{\text{nom}} \leftarrow \mathbf{v}_{\text{nom}} + \hat{\delta\mathbf{v}}$$
$$\mathbf{q}_{\text{nom}} \leftarrow \text{normalize}\left(\mathbf{q}_{\text{nom}} \otimes \delta\mathbf{q}(\hat{\delta\boldsymbol{\theta}})\right)$$
$$\mathbf{b}_{a,\text{nom}} \leftarrow \mathbf{b}_{a,\text{nom}} + \hat{\delta\mathbf{b}}_a$$
$$\mathbf{b}_{g,\text{nom}} \leftarrow \mathbf{b}_{g,\text{nom}} + \hat{\delta\mathbf{b}}_g$$

The error state is reset: $\delta\mathbf{x} \leftarrow \mathbf{0}_{15}$.
Updating the nominal quaternion shifts the reference frame of the attitude error manifold. The post-reset covariance is transformed via reset Jacobian:
$$G_\theta = \mathbf{I}_3 - \frac{1}{2}[\hat{\delta\boldsymbol{\theta}}]_\times$$
$$J_{\text{reset}} = \text{diag}\left(\mathbf{I}_3, \mathbf{I}_3, G_\theta, \mathbf{I}_3, \mathbf{I}_3\right) \in \mathbb{R}^{15 \times 15}$$
$$P^+ = J_{\text{reset}} P_{\text{updated}} J_{\text{reset}}^T, \quad P^+ \leftarrow \frac{1}{2}\left(P^+ + (P^+)^T\right)$$

This atomic operation is implemented exclusively inside `state.inject_error(...)`.

---

## 5. Measurement Models Built in Phase 5

### 5.1 GNSS 3D Position & 2D Horizontal Velocity
1. **3D Position Fix**: $\mathbf{z}_p = \mathbf{p}_{\text{gnss}}^n \in \mathbb{R}^3$, $H_p = [\mathbf{I}_3 \mid \mathbf{0}_{3 \times 12}]$. Converted to local ENU using the fixed session reference origin.
   - **Vertical Uncertainty Handling**: Smartphone GNSS vertical measurements are noisier due to satellite geometry and geoid undulation offsets. We set a realistic vertical uncertainty floor ($\sigma_{\text{vert}} = 35.0\,\text{m}$), allowing the filter to strongly trust horizontal position without fighting vertical datum noise.
2. **2D Horizontal Velocity Fix**: $\mathbf{z}_{v,2D} = [v_E, v_N]^T$ derived from speed and course bearing:
   $$v_E = v \sin(\psi), \quad v_N = v \cos(\psi)$$
   Column 5 ($v_U$) is left unobserved in $H_v$, preventing artificial vertical velocity constraints.
3. **Speed Factor Conversion**: Raw Android speed in IO-VNBD was logged in m/s; Phase 2 converted from km/h; `GNSSMeasurementModel` scales by $3.6$ to recover true m/s without modifying the frozen cache.

### 5.2 Classical Zero-Velocity Updates (ZUPT)
The `ClassicalZUPTDetector` monitors an 8-sample sliding window ($0.8\,\text{s}$) and declares standstill if:
- Angular rate norm $\|\boldsymbol{\omega}^v\| < 0.05\,\text{rad/s}$
- Accelerometer gravity match $|\|\mathbf{f}_m^v\| - 9.80665| < 0.25\,\text{m/s}^2$
- Low specific force variance $\text{Var}(\|\mathbf{f}_m^v\|) < 0.015\,(\text{m/s}^2)^2$
- Speed $< 0.1\,\text{m/s}$ when GPS is available.

Standstill measurement: $\mathbf{z}_{\text{zupt}} = [0, 0, 0]^T\,\text{m/s}$, $H_{\text{zupt}} = [\mathbf{0}_3 \mid \mathbf{I}_3 \mid \mathbf{0}_9]$. Innovation gating prevents false ZUPT application during highway cruising.

---

## 6. Empirical Results: Phase 4 Baseline vs. Phase 5 ESKF

Evaluated over the 60.0-second highway cruising window of Trip S1:

| Metric | Phase 4 Open-Loop Baseline | Phase 5 ESKF + GNSS | Measured Improvement |
|---|---|---|---|
| **Final Horizontal Error ($e_{\text{2D}}$)** | **$3,249.32\,\text{m}$** | **$5.27\,\text{m}$** | **$99.84\%$ reduction** |
| **Horizontal Position RMSE** | **$1,487.53\,\text{m}$** | **$24.78\,\text{m}$** | **$98.33\%$ reduction** |
| **Maximum Horizontal Error** | **$3,249.32\,\text{m}$** | **$117.49\,\text{m}$** | **$96.38\%$ reduction** |
| **Final Vertical Error ($e_U$)** | $530.20\,\text{m}$ | **$63.01\,\text{m}$** | **$88.12\%$ reduction** |
| **Vertical Position RMSE** | $211.01\,\text{m}$ | **$70.54\,\text{m}$** | **$66.57\%$ reduction** |
| **Final 3D Error ($e_{\text{3D}}$)** | **$3,292.29\,\text{m}$** | **$63.23\,\text{m}$** | **$98.08\%$ reduction** |
| **3D Position RMSE** | **$1,502.42\,\text{m}$** | **$74.77\,\text{m}$** | **$95.02\%$ reduction** |
| **GNSS Position Fix Acceptance** | 0 (open loop) | **7 of 7 ($100\%$)** | 0 false rejections |
| **GNSS 2D Velocity Acceptance** | 0 (open loop) | **7 of 7 ($100\%$)** | 0 false rejections |
| **Standstill ZUPT Suppression** | N/A | **$< 0.5\,\text{m}$ over 100 samples** | Residual velocity $< 0.05\,\text{m/s}$ |

---

## 7. Verification & Test Suite

At Phase 5 completion, **202 total tests** passed across the repository, including 29 Phase 5 tests:
- `tests/unit/test_eskf_synthetic.py` (18 tests): Finite difference Jacobian validation ($F_d$ matches numerical gradient to $< 10^{-4}$), Joseph positive-definiteness, covariance reset, and outlier rejection.
- `tests/unit/test_zupt.py` (9 tests): Standstill detection, motion rejection, and covariance reduction.
- `tests/integration/test_eskf_gnss_real_data.py` (2 tests): End-to-end filter execution on Trip S1 real data.

---

## 8. Final Phase 5 Summary: What I Should Remember

1. **Architecture**: 16-parameter non-linear nominal state, 15-parameter linear error state, $15 \times 15$ covariance.
2. **Attitude Convention**: Right-multiplicative body-frame attitude error: $\mathbf{q}_{\text{true}} = \mathbf{q}_{\text{nom}} \otimes \delta\mathbf{q}(\delta\boldsymbol{\theta})$.
3. **Stability**: Joseph-form covariance update and atomic covariance reset ($J_{\text{reset}}$) prevent numerical divergence.
4. **Benchmark Improvement**: Reduced 60-second horizontal error on Trip S1 from **$3,249.32\,\text{m}$ down to $5.27\,\text{m}$** ($99.84\%$ improvement).
5. **GNSS Handling**: 3D position + 2D horizontal velocity, with weak vertical confidence handling ($\sigma_{\text{vert}} = 35.0\,\text{m}$ floor).
6. **Zero-ML ZUPT**: Classical variance energy detection protected by Chi-Square innovation gating.
