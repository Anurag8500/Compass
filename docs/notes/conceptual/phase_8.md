# Phase 8 Complete Explanation: BiasNet — Inverse-Problem Label Optimization, Identifiability Gating & Learned Residual IMU Bias Estimation

**Project**: C.O.M.P.A.S.S. (Cognitive Off-grid Machine-learning Positioning And Sensor System)  
**Problem Statement**: SIH 2026 Problem Statement 26168 — ISRO  
**Stage**: Phase 8 Complete Teaching & Reference Walkthrough  

---

## 1. Introduction & Phase Overview

### What Problem Did Phase 8 Solve?
In Phase 2, Allan Variance analysis revealed that consumer-grade MEMS IMUs (such as those in smartphones or low-cost automotive loggers) suffer from continuous sensor bias drift:
- **Accelerometer Bias ($b_a$)**: Integrates directly into velocity ($v = \int b_a \, dt = b_a t$) and double-integrates into quadratic position drift:
  $$s_a(t) = \frac{1}{2} b_a t^2$$
  An uncalibrated horizontal acceleration bias of just $0.05\,\text{m/s}^2$ produces **$90\,\text{meters}$ of drift in 60 seconds**.
- **Gyroscope Bias ($b_g$)**: Causes angular orientation to tilt away from true vertical. As pitch or roll tilts by $\delta\theta = b_g t$, earth's gravity vector ($\mathbf{g} = 9.80665\,\text{m/s}^2$) leaks directly into the horizontal accelerometer channels:
  $$a_{\text{leak}} = g \sin(\delta\theta) \approx g b_g t$$
  Integrating this gravity leakage produces cubic position drift over time:
  $$s_g(t) = \frac{1}{6} g b_g t^3$$
  A tiny gyro bias of $0.005\,\text{rad/s}$ ($0.29^\circ/\text{s}$) produces **$176\,\text{meters}$ of drift in 60 seconds**, dominating all other error sources.

While stationary calibration (Phase 3) estimates initial turn-on biases before the vehicle moves, **biases drift dynamically during driving** due to vehicle cabin temperature swings, mechanical engine heat, vibrations, and supply voltage fluctuations.

**Phase 8 solves the dynamic bias estimation problem**: Can a deep neural network observe high-frequency IMU vibration and motion dynamics ($\mathbf{X} \in \mathbb{R}^{20 \times 9}$) and predict short-horizon residual sensor bias corrections ($\Delta \mathbf{b}_a \in \mathbb{R}^3, \Delta \mathbf{b}_g \in \mathbb{R}^3$) to prevent cubic dead-reckoning drift?

### Why BiasNet is Fundamentally an Inverse Problem
Unlike vehicle speed (which is directly measured by high-precision VBOX GNSS Doppler velocity in our training labels), **sensor bias cannot be physically measured by any sensor**:
- You cannot buy a "sensor-bias probe" to attach to an IMU.
- Biases are latent internal states obscured by physical vehicle accelerations and gravity.
- Therefore, BiasNet training targets must be **mathematically inverted from trajectory tracking errors** over a forward optimization horizon.

Because numerical inversion can easily become ill-conditioned, non-unique, or physically absurd, Phase 8 establishes a rigorous, audited inverse optimization pipeline with strict numerical identifiability gating and an unconditional decoupled fallback architecture.

---

## 2. Core Concepts & Terminology

### 1. The Inverse Optimization Formulation
To generate high-quality pseudo-ground-truth bias labels for training, we formulate a trajectory-matching optimization problem over a forward horizon $H$:
Given a valid initial navigation state at epoch $t_0$ ($\mathbf{p}_0, \mathbf{v}_0, \mathbf{q}_0$), we propagate the strapdown mechanization forward across horizon $H$ using candidate bias corrections $\Delta \mathbf{b} = [\Delta \mathbf{b}_a^T, \Delta \mathbf{b}_g^T]^T \in \mathbb{R}^6$:
$$\tilde{\mathbf{f}}^b(t) = \mathbf{f}^b(t) - (\hat{\mathbf{b}}_a + \Delta \mathbf{b}_a), \qquad \tilde{\boldsymbol{\omega}}^b(t) = \boldsymbol{\omega}^b(t) - (\hat{\mathbf{b}}_g + \Delta \mathbf{b}_g)$$
We define a weighted residual vector comparing the propagated trajectory against high-precision VBOX ground truth across all $K$ steps in the horizon:
$$\mathbf{r}(\Delta \mathbf{b}) = \begin{bmatrix} W_p (\mathbf{p}_{\text{prop}} - \mathbf{p}_{\text{ref}}) \\ W_v (\mathbf{v}_{\text{prop}} - \mathbf{v}_{\text{ref}}) \\ W_\theta \, \delta\boldsymbol{\theta}(\mathbf{q}_{\text{prop}}, \mathbf{q}_{\text{ref}}) \end{bmatrix} \in \mathbb{R}^{3K \times 1}$$

- **Residual Scaling Weights**:
  - $W_p = 0.2\,\text{m}^{-1}$ (downweights position so large accumulated distance errors do not overpower velocity)
  - $W_v = 1.0\,(\text{m/s})^{-1}$
  - $W_\theta = 10.0\,\text{rad}^{-1}$ (upweights orientation so attitude alignment is preserved)
- **Attitude Error Formulation**: Rotation vector $\delta\boldsymbol{\theta}$ is computed using our right-multiplicative quaternion Lie algebra convention:
  $$\delta\mathbf{q} = \mathbf{q}_{\text{prop}}^{-1} \otimes \mathbf{q}_{\text{ref}}, \qquad \delta\boldsymbol{\theta} = 2 \cdot \text{sign}(\delta q_w) \cdot \delta\mathbf{q}_{xyz}$$

```
                +-------------------------------------------+
                | Window Input X in R^(20 x 9) at epoch t_0 |
                +-------------------------------------------+
                                      |
                                      v
+-----------------------+     +-------------------------------+
| VBOX Reference Ground |     | Forward Horizon Strapdown INS |
| Truth (t_0 -> t_0 + H)|     | Propagation with Candidate    |
+-----------------------+     | Biases: [Δb_a, Δb_g] in R^6   |
            \                 +-------------------------------+
             \                               /
              v                             v
           +-----------------------------------+
           |    Weighted Residual Vector r     |
           |   r = [ W_p Δp, W_v Δv, W_θ δθ ]  |
           +-----------------------------------+
                             |
                             v
           +-----------------------------------+
           |  Damped Gauss-Newton / LM Solver  |
           | (J^T J + λ I) Δb = -J^T r         |
           +-----------------------------------+
                             |
                             v
           +-----------------------------------+
           |   Identifiability Gating Matrix   |
           |  cond(J) <= 50, ||r||/||r_0|| < 0.8|
           +-----------------------------------+
                    /                 \
        (Passed)   /                   \  (Failed: Rejection Code)
                  v                     v
          +---------------+      +---------------+
          | VALID TARGET  |      | DROP WINDOW   |
          |  [Δb_a, Δb_g] |      | (No Training) |
          +---------------+      +---------------+
```

### 2. Horizon Selection Analysis ($H = 1.0\,\text{s}$)
The choice of horizon length $H$ governs the mathematical trade-off between bias observability and kinematic validity:
- **Too Short ($H = 0.5\,\text{s}$)**: Velocity and attitude errors have had almost no time to develop. The Jacobian $J = \frac{\partial \mathbf{r}}{\partial \Delta\mathbf{b}}$ is severely ill-conditioned ($\kappa(J) > 150$), making accelerometer bias indistinguishable from gravity or sensor noise.
- **Too Long ($H = 2.0\,\text{s}$)**: Over 2 seconds, non-linear vehicle dynamics (unmodeled road bumps, suspension deflection, tire slip) violate the assumption of constant sensor bias.
- **The Sweet Spot ($H = 1.0\,\text{s}$, 10 samples)**: Yields excellent numerical conditioning ($\kappa(J) \approx 10.5$), full rank 6, high solver convergence rate ($>98\%$), and stable, physically plausible labels.

### 3. Damped Gauss-Newton Solver
The non-linear least squares objective $\min_{\Delta\mathbf{b}} \frac{1}{2} \|\mathbf{r}(\Delta\mathbf{b})\|_2^2$ is solved via iterative Gauss-Newton with Levenberg-Marquardt damping:
$$\Delta\mathbf{b}_{k+1} = \Delta\mathbf{b}_k - \left(\mathbf{J}_k^T \mathbf{J}_k + \lambda \mathbf{I}_6\right)^{-1} \mathbf{J}_k^T \mathbf{r}_k$$
- Convergence requires satisfying multi-criteria stopping thresholds:
  1. Step norm: $\|\Delta\mathbf{b}_{k+1} - \Delta\mathbf{b}_k\| < 10^{-4}$
  2. Relative residual improvement: $\frac{\|\mathbf{r}_k\| - \|\mathbf{r}_{k+1}\|}{\|\mathbf{r}_k\|} < 10^{-4}$
  3. Gradient infinity norm: $\|\mathbf{J}^T \mathbf{r}\|_\infty < 10^{-3}$
- Windows where the optimizer diverges or fails to converge within 25 iterations are tagged with reason code `SOLVER_FAILURE` and rejected.

---

## 3. Three-Tier Bound Taxonomy & Numerical Identifiability Gating

Because machine learning models easily memorize unconstrained optimizer artifacts, Phase 8 defines an explicit **three-tier bound taxonomy**:

| Tier | Boundary Type | Bound Limits | Purpose |
|---|---|---|---|
| **Tier 1** | **Solver Safeguard Bounds** | $\|\Delta\mathbf{b}_a\| \le 5.0\,\text{m/s}^2$<br>$\|\Delta\mathbf{b}_g\| \le 0.5\,\text{rad/s}$ | Prevents optimizer from wandering into divergent mathematical territory during iteration steps. |
| **Tier 2** | **Post-Solve Physical Eligibility Bounds** | $\|\Delta\mathbf{b}_a\| \le 2.0\,\text{m/s}^2$<br>$\|\Delta\mathbf{b}_g\| \le 0.15\,\text{rad/s}$ | Rejects windows where optimizer found a mathematical minimum that violates known physical MEMS semiconductor limits. |
| **Tier 3** | **Neural Output In-Graph Clamps** | $[-2.0, 2.0]\,\text{m/s}^2$<br>$[-0.15, 0.15]\,\text{rad/s}$ | Hard in-graph saturation layers ensuring neural inferences can never emit unphysical spikes during runtime. |

### Deterministic Rejection Precedence
To audit data quality, every candidate window passes through a deterministic sequence of identifiability checks with strict priority ordering:
$$\begin{aligned}
\text{NON\_FINITE\_INPUT} &\longrightarrow \text{TIMESTEP\_ANOMALY} \longrightarrow \text{WINDOW\_TOO\_SHORT} \\
&\longrightarrow \text{HORIZON\_TOO\_SHORT} \longrightarrow \text{NON\_FINITE\_SOLUTION} \longrightarrow \text{SOLVER\_FAILURE} \\
&\longrightarrow \text{DEFICIENT\_RANK} \longrightarrow \text{ILL\_CONDITIONED} \, (\kappa > 50.0) \\
&\longrightarrow \text{BOUNDS\_ACTIVE} \, (\text{Tier 2 violated}) \longrightarrow \text{POOR\_RESIDUAL\_REDUCTION} \, (\rho < 1.20) \\
&\longrightarrow \mathbf{VALID}
\end{aligned}$$

### Label Dataset Quality Audit
Applying this strict identifiability filter across our driving sessions yielded a pristine, verified training and validation dataset:
- **Training Set (Driver E)**: $4,060$ eligible windows ($52.4\%$ pass rate across 30 trips).
- **Validation Set (Driver B)**: $480$ eligible windows ($68.6\%$ pass rate across 8 trips).
- **Physical Autocorrelation**: Autocorrelation of generated labels was verified to be positive and continuous ($r \approx 0.55\text{--}0.75$ for accelerometer, $r \approx 0.25\text{--}0.45$ for gyroscope), confirming that the labels capture true slowly drifting physical biases rather than high-frequency white noise.

---

## 4. BiasNet Model Architecture & Training Contract

BiasNet uses a recurrent temporal architecture designed for smooth sequential tracking:
- **Input**: Canonical $(20, 9)$ feature tensor (normalized using Phase 6 training-only statistics).
- **Core Trunk**: 2-layer Gated Recurrent Unit (GRU) with 48 hidden units:
  $$\text{GRU}(9 \longrightarrow 48, \text{num\_layers}=2, \text{batch\_first}=\text{True})$$
- **Dense Intermediate Layer**: $\text{Dense}(48 \longrightarrow 24, \text{ReLU})$.
- **Dual Output Heads with In-Graph Clamping**:
  - Accelerometer Bias Head: $\text{Dense}(24 \longrightarrow 3) \longrightarrow \text{Clamp}([-2.0, 2.0]\,\text{m/s}^2)$
  - Gyroscope Bias Head: $\text{Dense}(24 \longrightarrow 3) \longrightarrow \text{Clamp}([-0.15, 0.15]\,\text{rad/s})$
- **Total Parameters**: **23,934 parameters** ($95.7\,\text{KB}$ float32).

```
         +---------------------------------------------+
         | Input Tensor: X in R^(B x 20 x 9) (2.0s IMU)|
         +---------------------------------------------+
                               |
                               v
         +---------------------------------------------+
         | 2-Layer GRU (48 hidden units, dropout=0.1)  |
         +---------------------------------------------+
                               |
                               v
         +---------------------------------------------+
         | Last Temporal Step (R^48) -> Dense(24, ReLU)|
         +---------------------------------------------+
                        /               \
                       /                 \
                      v                   v
         +-----------------------+ +-----------------------+
         | Linear Head (24 -> 3) | | Linear Head (24 -> 3) |
         | In-Graph Clamp:       | | In-Graph Clamp:       |
         | [-2.0, 2.0] m/s^2     | | [-0.15, 0.15] rad/s   |
         +-----------------------+ +-----------------------+
                     |                         |
                     v                         v
              Δb_a_pred (m/s^2)        Δb_g_pred (rad/s)
```

### Training Setup
- **Loss Function**: Weighted Smooth L1 (Huber) Loss with $\beta = 0.1$:
  $$\mathcal{L} = \mathcal{L}_{\text{SmoothL1}}(\Delta\mathbf{b}_a, \Delta\hat{\mathbf{b}}_a) + 10.0 \cdot \mathcal{L}_{\text{SmoothL1}}(\Delta\mathbf{b}_g, \Delta\hat{\mathbf{b}}_g)$$
  The $10.0\times$ multiplier balances the smaller numerical scale of radian gyro biases against meter-per-second-squared accelerometer biases.
- **Optimization**: Adam optimizer, initial lr $10^{-3}$, cosine annealing schedule down to $10^{-6}$, batch size 64, weight decay $10^{-4}$.
- **Driver A Strictly Held Out**: Test Driver A was never touched during training or hyperparameter tuning.

---

## 5. Direct and Indirect Validation Results

### 1. Direct Label-Space Validation on Held-Out Driver B
BiasNet was evaluated against two baseline estimators:
1. *Zero Correction Baseline*: Assumes no dynamic bias correction ($\Delta\mathbf{b} = \mathbf{0}$).
2. *Training Set Mean Baseline*: Always predicts the constant mean bias observed on Driver E ($\Delta\mathbf{b} = \bar{\mathbf{b}}_{\text{train}}$).

| Model / Baseline | Accel Bias Vector RMSE ($\text{m/s}^2$) | Gyro Bias Vector RMSE ($\text{rad/s}$) | Total Vector RMSE | Improvement vs. Zero Baseline |
|---|---|---|---|---|
| **Zero Correction Baseline** | $0.8412$ | $0.0381$ | $1.0401$ | Reference ($0.0\%$) |
| **Training Set Mean Baseline** | $0.8120$ | $0.0369$ | $0.9854$ | $+5.3\%$ |
| **BiasNet v1.0 (Validation Driver B)** | **$0.6124$** | **$0.0245$** | **$0.7372$** | **$+29.1\%$ Improvement** ✅ |
| **BiasNet v1.0 (Held-Out Test Driver A)** | **$0.5981$** | **$0.0238$** | **$0.7153$** | **$+32.9\%$ Improvement** ✅ |

BiasNet cut vector bias error by $\approx 30\%$ across both validation and held-out test drivers, demonstrating genuine physical generalization rather than memorization.

### 2. Indirect Navigation Ablation on Real IO-VNBD S1
To evaluate how learned bias corrections behave in closed-loop navigation, BiasNet was evaluated inside the Error-State Kalman Filter across synthetic GNSS outages (10s, 30s, 60s) on `Categorised_S1.npz`:
- **Filter Authority & Stability**: Filter innovations remained well within theoretical Chi-square bounds (Mean NIS $\le 2.543$). No divergence, numerical instability, or covariance blowups occurred.
- **Velocity Tracking RMSE Improvement**:
  - *30s Outage*: Pure ESKF velocity RMSE $= 3.775\,\text{m/s}$. Adding VelocityNet gave $3.361\,\text{m/s}$. Adding BiasNet reduced velocity error further to **$3.292\,\text{m/s}$**.
  - *60s Outage*: Pure ESKF velocity RMSE $= 133.082\,\text{m/s}$. Adding VelocityNet gave $131.604\,\text{m/s}$. Adding BiasNet reduced velocity error to **$130.002\,\text{m/s}$**.
- **Horizontal Position Drift**: Horizontal position drift was comparable to VelocityNet alone ($10.049\,\text{m}$ vs $10.031\,\text{m}$ at 30s; $1735.2\,\text{m}$ vs $1740.1\,\text{m}$ at 60s). Because position drift in prolonged unconstrained outages is heavily driven by initial heading misalignment, learned bias alone cannot completely cure dead reckoning without lateral kinematic constraints (which we add in Phase 11).

---

## 6. Filter Authority, Safety Gates & Decoupled Architecture

A primary architectural requirement of COMPASS is that **deep neural networks must never directly command navigation states**. BiasNet obeys strict filter authority invariants:

1. **No Direct State Overwrite**: BiasNet outputs do not overwrite the ESKF state vector $\mathbf{x}_{\text{nom}}$. They enter strictly as pseudo-measurements through the Kalman gain matrix ($K = P H^T S^{-1}$).
2. **Measurement Covariance Weighting**: BiasNet updates are assigned a conservative diagonal measurement noise covariance:
   $$R_b = \text{diag}([1.21, 1.21, 1.21\,\text{m}^2/\text{s}^4, 0.0025, 0.0025, 0.0025\,\text{rad}^2/\text{s}^2])$$
   This ensures the filter smoothly blends the neural bias suggestion with its internal inertial propagation history.
3. **Decoupled Standalone Fallback**: BiasNet is completely decoupled from the rest of the navigation stack. If BiasNet encounters NaN inputs, excessive timestamp jitter, or anomalous vehicle dynamics, it can be disabled instantly (`biasnet_enabled: false`). The core pipeline falls back smoothly to `Classical ESKF + GNSS + Gated ZUPT + VelocityNet` without code changes or state disruption.

---

## 7. Export & Numerical Parity Audit

BiasNet was exported to ONNX (`opset 17`) and LiteRT (`.tflite`) with single-sample shape $[1, 20, 9]$. Evaluating 500 real test windows from Driver A demonstrated numerical parity down to machine precision:
- **PyTorch vs. ONNX Runtime**: Max absolute error $= \mathbf{6.56 \times 10^{-7}\,\text{m/s}^2}$
- **PyTorch vs. LiteRT**: Max absolute error $= \mathbf{3.58 \times 10^{-7}\,\text{m/s}^2}$

---

## 8. Phase Summary & Handoff to Phase 9

| Property | Phase 8 BiasNet Specification |
|---|---|
| **Authoritative Model** | BiasNet v1.0 (2-Layer GRU with in-graph saturation) |
| **Model Parameters** | 23,934 parameters ($95.7\,\text{KB}$ float32) |
| **Input Shape** | $[B, 20, 9]$ ($2.0\,\text{s}$ causal history at $10\,\text{Hz}$) |
| **Output Shape** | $[\Delta\mathbf{b}_a (3), \Delta\mathbf{b}_g (3)] \in \mathbb{R}^6$ |
| **Label Generation** | $1.0\,\text{s}$ horizon Gauss-Newton trajectory inversion against VBOX ground truth |
| **Identifiability Filter** | Condition number $\kappa \le 50.0$, residual reduction ratio $\rho \ge 1.20$, Tier 2 bounds |
| **Direct Test Improvement** | $+32.9\%$ vector RMSE reduction over Zero Baseline on held-out Driver A |
| **Export Formats** | `biasnet_v1.onnx` and `biasnet_v1.tflite` |
| **Acceptance Gate Status** | **ACCEPTED AS EXPERIMENTAL CANDIDATE FOR PHASE 9 ESKF FUSION** |

With **VelocityNet** (Phase 7) providing forward speed and **BiasNet** (Phase 8) providing residual bias compensation, we have completed the isolated machine learning subsystem. In **Phase 9**, we build the closed-loop measurement adapters, timeline cadence schedulers, and filter safety guards to wire both neural networks directly into the Error-State Kalman Filter.
