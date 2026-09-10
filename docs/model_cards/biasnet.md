# BiasNet Model Card (COMPASS Phase 8 — v1.0)

## Model Overview
- **Model Name**: BiasNet
- **Version**: v1.0
- **Model Type**: Deep 2-layer Recurrent Neural Network (GRU) for short-horizon IMU bias correction estimation.
- **Role in Navigation Architecture**:
  - BiasNet is an experimental aiding source for the 15-state Error-State Kalman Filter (ESKF).
  - It does **NOT** observe true IMU bias directly; its supervision targets are optimization-derived pseudo-ground-truth corrections from a short-horizon inverse problem.
  - It does **NOT** directly mutate or overwrite the ESKF state ($\mathbf{b}_a, \mathbf{b}_g$).
  - Its outputs reach the estimator strictly as uncertainty-weighted, innovation-gated pseudo-measurements:
    $$\mathbf{z}_b = \mathbf{b}_{\text{nominal}} + \Delta \mathbf{b}_{\text{pred}}$$
  - It can be completely disabled at runtime (`biasnet_enabled = false`) without breaking the navigation stack, falling back cleanly to the decoupled classical + VelocityNet navigation path.

## Architectural Specification
- **Input Tensor**: $(B, 20, 9)$ representing a strictly causal $2.0\text{ s}$ temporal window of 9-channel vehicle FLU motion features at $10\text{ Hz}$:
  $$[f_x, f_y, f_z, \omega_x, \omega_y, \omega_z, \|\mathbf{f}\|, \|\dot{\mathbf{f}}\|, \|\boldsymbol{\omega}\|]$$
  Features are normalized using training-only z-score parameters fitted strictly on Driver E.
- **Recurrent Backbone**: 2-layer Gated Recurrent Unit (GRU) with 48 hidden dimensions and inter-layer dropout ($p = 0.1$).
- **Dense Intermediate Layer**: $48 \to 24$ with ReLU non-linearity.
- **Output Layer**: $24 \to 6$ linear projection emitting:
  $$[\Delta b_{a, x}, \Delta b_{a, y}, \Delta b_{a, z}, \Delta b_{g, x}, \Delta b_{g, y}, \Delta b_{g, z}]$$
- **Total Parameter Count**: **23,934** parameters ($95.7\text{ KB}$ in FP32).
- **Internal Graph Safety Clamps**:
  Physical bounding is implemented inside the forward computational graph (propagated through ONNX and LiteRT graphs):
  - Accelerometer corrections: $|\Delta b_a| \le 2.00\text{ m/s}^2$
  - Gyroscope corrections: $|\Delta b_g| \le 0.150\text{ rad/s}$ ($8.59^\circ$/s)

## Training & Supervision Provenance
- **Dataset**: IO-VNBD (Inertial and Odometry Benchmark Dataset for Ground Vehicles).
- **Split Invariants**:
  - **Train**: Driver E (3,914 eligible windows from 25 trips).
  - **Validation**: Driver B (504 eligible windows across 2 trips).
  - **Held-Out Test**: Driver A (956 eligible windows across 5 trips).
- **Teacher Paradigm**: Inverse-problem short-horizon ($H = 1.0\text{ s}$, $K=10$ intervals) damped Levenberg-Marquardt optimization against synchronized Racelogic VBOX RTK GNSS position, velocity, and orientation residuals.
- **Gating Filter**:
  Windows are accepted for supervised learning only if:
  1. Jacobian condition number $\kappa \le 50.0$.
  2. Effective rank $= 6$.
  3. Residual reduction ratio $\rho \ge 1.20$.
  4. Physical bounds active flag is False ($|\Delta b_a| \le 2.0\text{ m/s}^2$, $|\Delta b_g| \le 0.15\text{ rad/s}$).
- **Optimization**: Adam optimizer, learning rate $10^{-3}$, weight decay $10^{-4}$, batch size 64, Smooth L1 loss ($\beta = 0.05$) with component weights ($W_a = 1.0, W_g = 10.0$), early stopping patience 6. Best validation checkpoint restored at Epoch 29.

## Empirical Performance Summary

### 1. Direct Label-Space Metrics on Driver B (Validation)
| Metric | Zero Baseline ($\Delta \mathbf{b} = \mathbf{0}$) | Train Mean Baseline | BiasNet v1.0 | Improvement vs Zero |
| :--- | :--- | :--- | :--- | :--- |
| **Total Vector RMSE** | $1.0430\text{ m/s}^2$ | $1.0288\text{ m/s}^2$ | **$0.7370\text{ m/s}^2$** | **+29.3%** |
| **Accel Vector RMSE** | $1.0415\text{ m/s}^2$ | $1.0272\text{ m/s}^2$ | **$0.7362\text{ m/s}^2$** | **+29.3%** |
| **Gyro Vector RMSE** | $0.0561\text{ rad/s}$ | $0.0569\text{ rad/s}$ | **$0.0350\text{ rad/s}$** | **+37.6%** |
| **Pearson Correlation $r$** | $0.000$ | $0.000$ | **$0.55\text{ to }0.98$** | Strong tracking |

### 2. Held-Out Generalization on Driver A (Test)
Evaluated strictly once post-freeze on 956 eligible test windows:
| Metric | Zero Baseline | Train Mean Baseline | BiasNet v1.0 | Improvement vs Zero |
| :--- | :--- | :--- | :--- | :--- |
| **Total Vector RMSE** | $1.0656\text{ m/s}^2$ | $1.0588\text{ m/s}^2$ | **$0.7164\text{ m/s}^2$** | **+32.8%** |
| **Accel Vector RMSE** | $1.0633\text{ m/s}^2$ | $1.0564\text{ m/s}^2$ | **$0.7150\text{ m/s}^2$** | **+32.8%** |
| **Gyro Vector RMSE** | $0.0703\text{ rad/s}$ | $0.0706\text{ rad/s}$ | **$0.0438\text{ rad/s}$** | **+37.7%** |

### 3. Indirect Navigation Outage Behavior
Evaluated during synthetic GNSS outages against the authoritative non-NHC Phase 8 baseline:
- **10s Outage**: Horizontal RMSE $= 0.424\text{ m}$ (vs Pure ESKF $0.426\text{ m}$). Innovation Mean NIS $= 0.037$ (well below $\chi^2$ gate of 25.0).
- **30s Outage**: Horizontal RMSE $= 10.099\text{ m}$ (vs Pure ESKF $11.289\text{ m}$, 10.5% improvement), Velocity RMSE $= 3.325\text{ m/s}$ (vs Pure ESKF $3.775\text{ m/s}$). Mean NIS $= 1.763$.
- **60s Outage**: Numerical stability preserved; Velocity RMSE $= 131.138\text{ m/s}$ (vs Pure ESKF $133.082\text{ m/s}$). Mean NIS $= 2.470$.

## Edge Deployment & Numerical Parity
- **Export Artifacts**:
  - `models/biasnet_v1.onnx` (`2bd6bdc8...`, $104\text{ KB}$)
  - `models/biasnet_v1.tflite` (`c678d5db...`, $240\text{ KB}$)
- **Deployment Parity (500 Real Windows)**:
  - ONNX Max Abs Error: **$5.66 \times 10^{-7}$** (Tolerance: $1.0 \times 10^{-4}$) -> **PASSED**
  - LiteRT Max Abs Error: **$3.58 \times 10^{-7}$** (Tolerance: $1.0 \times 10^{-3}$) -> **PASSED**

## Operational Constraints & Safety Rules
1. **Never mutate state directly**: BiasNet predictions must only reach the estimator via Kalman updates.
2. **Empirical covariance inflation**: Until Phase 9 joint calibration, use conservative measurement covariance $R_b = \text{diag}([1.21, 1.21, 1.21, 0.0025, 0.0025, 0.0025])$.
3. **Motion gating**: Do not fuse BiasNet updates near standstill or during unvalidated window segments.
4. **Fallback guarantee**: Setting `biasnet_enabled = false` immediately restores the pure classical / VelocityNet navigation state.
