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
  - **Train**: Driver E (4,060 eligible windows across 30 audited trips).
  - **Validation**: Driver B (480 eligible windows across 2 trips).
  - **Held-Out Test**: Driver A (956 eligible windows across 5 trips).
- **Teacher Paradigm**: Inverse-problem short-horizon ($H = 1.0\text{ s}$, $K=10$ intervals) damped Levenberg-Marquardt optimization against synchronized Racelogic VBOX RTK GNSS position, velocity, and orientation residuals.
- **Gating Filter**:
  Windows are accepted for supervised learning only if:
  1. Optimizer converges within 15 iterations (`converged == True`, rejected as `SOLVER_FAILURE` otherwise).
  2. Jacobian condition number $\kappa \le 50.0$.
  3. Effective rank $= 6$.
  4. Residual reduction ratio $\rho \ge 1.20$.
  5. Physical bounds active flag is False ($|\Delta b_a| \le 2.0\text{ m/s}^2$, $|\Delta b_g| \le 0.15\text{ rad/s}$).
- **Optimization**: Adam optimizer, learning rate $10^{-3}$, weight decay $10^{-4}$, batch size 64, Smooth L1 loss ($\beta = 0.05$) with component weights ($W_a = 1.0, W_g = 10.0$), early stopping patience 6. Best validation checkpoint restored at Epoch 15.

## Empirical Performance Summary

### 1. Direct Label-Space Metrics on Driver B (Validation)
| Metric | Zero Baseline ($\Delta \mathbf{b} = \mathbf{0}$) | Train Mean Baseline | BiasNet v1.0 | Improvement vs Zero |
| :--- | :--- | :--- | :--- | :--- |
| **Total Vector RMSE** | $1.0401\text{ m/s}^2$ | $1.0169\text{ m/s}^2$ | **$0.7372\text{ m/s}^2$** | **+29.1%** |
| **Accel Vector RMSE** | $1.0387\text{ m/s}^2$ | $1.0153\text{ m/s}^2$ | **$0.7365\text{ m/s}^2$** | **+29.1%** |
| **Gyro Vector RMSE** | $0.0554\text{ rad/s}$ | $0.0560\text{ rad/s}$ | **$0.0329\text{ rad/s}$** | **+40.6%** |

### 2. Held-Out Generalization on Driver A (Test)
Evaluated strictly once post-freeze on 956 eligible test windows:
| Metric | Zero Baseline | Train Mean Baseline | BiasNet v1.0 | Improvement vs Zero |
| :--- | :--- | :--- | :--- | :--- |
| **Total Vector RMSE** | $1.0656\text{ m/s}^2$ | $1.0563\text{ m/s}^2$ | **$0.7153\text{ m/s}^2$** | **+32.9%** |
| **Accel Vector RMSE** | $1.0633\text{ m/s}^2$ | $1.0539\text{ m/s}^2$ | **$0.7140\text{ m/s}^2$** | **+32.9%** |
| **Gyro Vector RMSE** | $0.0703\text{ rad/s}$ | $0.0704\text{ rad/s}$ | **$0.0425\text{ rad/s}$** | **+39.5%** |

### 3. Indirect Navigation Outage Behavior
Evaluated during synthetic GNSS outages on real driving segment `Categorised_S1.npz`:
- **10s Outage**: Horizontal RMSE $= 0.424\text{ m}$ (vs Pure ESKF $0.426\text{ m}$). Mean NIS $= 0.049$.
- **30s Outage**: Horizontal RMSE $= 10.049\text{ m}$ (vs Pure ESKF $11.289\text{ m}$ and +VNet $10.031\text{ m}$), Velocity RMSE $= 3.292\text{ m/s}$ (vs Pure ESKF $3.775\text{ m/s}$ and +VNet $3.361\text{ m/s}$). Mean NIS $= 1.839$.
- **60s Outage**: Velocity RMSE $= 130.002\text{ m/s}$ (vs Pure ESKF $133.082\text{ m/s}$ and +VNet $131.604\text{ m/s}$). Mean NIS $= 2.543$.

## Edge Deployment & Numerical Parity
- **Export Artifacts**:
  - `models/biasnet_v1.onnx` (`d57e6448...`, $104\text{ KB}$)
  - `models/biasnet_v1.tflite` (`551193b4...`, $240\text{ KB}$)
- **Deployment Parity (500 Real Windows)**:
  - ONNX Max Abs Error: **$6.56 \times 10^{-7}$** (Tolerance: $1.0 \times 10^{-4}$) -> **PASSED**
  - LiteRT Max Abs Error: **$3.58 \times 10^{-7}$** (Tolerance: $1.0 \times 10^{-3}$) -> **PASSED**

## Operational Constraints & Safety Rules
1. **Never mutate state directly**: BiasNet predictions must only reach the estimator via Kalman updates.
2. **Empirical covariance inflation**: Until Phase 9 joint calibration, use conservative measurement covariance $R_b = \text{diag}([1.21, 1.21, 1.21, 0.0025, 0.0025, 0.0025])$.
3. **Motion gating**: Do not fuse BiasNet updates near standstill or during unvalidated window segments.
4. **Fallback guarantee**: Setting `biasnet_enabled = false` immediately restores the pure classical / VelocityNet navigation state.
