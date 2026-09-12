# Phase 7 Complete Explanation: VelocityNet — Neural Pseudo-Velocity Estimation, 1D-CNN Architecture & Causal Smoothing

**Project**: C.O.M.P.A.S.S. (Cognitive Off-grid Machine-learning Positioning And Sensor System)  
**Problem Statement**: SIH 2026 Problem Statement 26168 — ISRO  
**Stage**: Phase 7 Complete Teaching & Reference Walkthrough  

---

## 1. Introduction & Phase Overview

### What Problem Did Phase 7 Solve?
In Phase 4, we saw the fundamental catastrophe of unassisted inertial dead reckoning: when an autonomous ground vehicle enters a GNSS-denied environment (such as a tunnel, mountain pass, or urban canyon), open-loop integration of IMU measurements causes position drift to grow quadratically and cubically over time ($s(t) \sim \frac{1}{2} a_{\text{bias}} t^2 + \frac{1}{6} g \delta\theta t^3$). On Trip S1, unassisted dead reckoning drifted by **$3,249.32\,\text{meters}$ in just 60 seconds**.

While the Error-State Kalman Filter (Phase 5) accurately bounds drift when GNSS fixes are arriving, the filter has no direct velocity observation once GNSS drops out. If the vehicle is moving at $60\,\text{km/h}$ ($16.7\,\text{m/s}$), any unmodeled acceleration bias or tilt error rapidly corrupts the estimated speed.

**Phase 7 solves the autonomous speed-sensing problem without external signals**: How can a vehicle deduce its forward speed ($v_{\text{forward}}$) purely by listening to the high-frequency vibrational, centrifugal, and rotational dynamics of its own chassis through a consumer-grade IMU?

### Why Did We Need to Solve It?
Physical vehicles are mechanical resonators:
1. **Engine and Drivetrain Harmonics**: Internal combustion engines, electric motors, transmission gears, and drive shafts transmit rotational vibrations through the vehicle frame at frequencies proportional to vehicle speed.
2. **Tire-Road Interaction & Suspension Acoustics**: As tires roll over asphalt or concrete, road roughness excites the suspension at frequencies that scale directly with vehicle velocity ($f = \frac{v}{\lambda_{\text{texture}}}$).
3. **Cornering and Longitudinal Kinematics**: Centripetal accelerations ($a_y = v \cdot \omega_z$) and pitch changes during acceleration provide strong physical constraints on speed.

Phase 7 designs, trains, selects, and exports **VelocityNet**: a compact, low-latency deep neural network that maps a $2.0\,\text{second}$ causal window of vehicle-frame IMU data ($\mathbf{X} \in \mathbb{R}^{20 \times 9}$) directly to forward driving speed ($v_{\text{forward}}$) and its accompanying heteroscedastic uncertainty ($\log \sigma_v^2$).

Crucially, **Phase 7 treats VelocityNet strictly as an isolated ML subsystem**. It does not modify or integrate with the ESKF; filter integration is deliberately deferred to Phase 9 to guarantee that model generalization, causal temporal behavior, and deployment export parity are verified before closed-loop filter risks are introduced.

---

## 2. Core Concepts & Terminology

### 1. Pseudo-Velocity Aiding
- **Definition**: In classical navigation, sensors like wheel odometers, transmission speed sensors, or Doppler radar provide velocity updates. In a software-only or retrofitted navigation system using an untethered smartphone or independent IMU box, no hardware bus (CAN-bus / OBD-II) access is guaranteed.
- **Pseudo-Velocity**: An externally estimated velocity measurement derived indirectly from sensor observations. VelocityNet acts as a "virtual odometer" or learned speedometer.

### 2. Heteroscedastic Uncertainty Head
Standard regression networks predict a single scalar $\hat{y}$. But an optimal Kalman filter (Phase 5 & 9) cannot simply ingest a raw prediction; it requires a measurement covariance $R = \sigma_v^2$ to decide whether to trust or downweight the update.
- **Homoscedastic Noise**: Assumes measurement noise is constant across all operating regimes ($\sigma^2 = \text{const}$).
- **Heteroscedastic Noise**: Recognizes that prediction confidence depends heavily on the driving context. When driving on a straight, smooth highway with clear engine hum, uncertainty $\sigma_v^2$ is low ($1.0\,\text{m}^2/\text{s}^2$). When traversing severe potholes, sudden speed bumps, or erratic stop-and-go maneuvers, uncertainty $\sigma_v^2$ expands automatically ($15.0\,\text{m}^2/\text{s}^2$).

VelocityNet outputs two values at each epoch:
$$\hat{\mathbf{y}} = \begin{bmatrix} \hat{v} \\ s \end{bmatrix} = \begin{bmatrix} \hat{v} \\ \log \sigma_v^2 \end{bmatrix}$$
Predicting log-variance $s = \log \sigma_v^2$ rather than variance directly guarantees that variance $\sigma_v^2 = \exp(s)$ remains strictly positive ($\sigma_v^2 > 0$) without numerical instability or artificial bounding during gradient descent.

```
       +-------------------------------------------------------+
       |   Input IMU Window: X in R^(20 x 9) (2.0s history)    |
       +-------------------------------------------------------+
                                   |
                                   v
       +-------------------------------------------------------+
       |      1D-CNN Feature Extractor (48 -> 64 -> 64 ch)      |
       +-------------------------------------------------------+
                                   |
                                   v
       +-------------------------------------------------------+
       |       Global Adaptive Average Pooling (R^64)          |
       +-------------------------------------------------------+
                                   |
                                   v
       +-------------------------------------------------------+
       |              Dense Shared Trunk (32 units)            |
       +-------------------------------------------------------+
                      /                         \
                     /                           \
                    v                             v
        +-----------------------+     +-----------------------+
        |   Speed Head (Dense)  |     | Log-Var Head (Dense)  |
        |   Linear Activation   |     | Linear Activation     |
        +-----------------------+     +-----------------------+
                    |                             |
                    v                             v
             v_hat (m/s)                  s = log(sigma_v^2)
```

### 3. Gaussian Negative Log-Likelihood (NLL) Loss
To train both the speed prediction and the uncertainty head simultaneously, VelocityNet is trained using Gaussian Negative Log-Likelihood loss:
$$\mathcal{L}_{\text{NLL}}(v, \hat{v}, s) = \frac{1}{2} \exp(-s) (v - \hat{v})^2 + \frac{1}{2} s + \frac{1}{2} \log(2\pi)$$
- **Precision Weighting**: The term $\frac{1}{2} \exp(-s) (v - \hat{v})^2 = \frac{(v - \hat{v})^2}{2\sigma_v^2}$ penalizes errors heavily when the network claims high confidence (small $\sigma_v^2$).
- **Uncertainty Regularizer**: The term $\frac{1}{2} s = \frac{1}{2} \log \sigma_v^2$ prevents the network from trivializing the loss by blowing up uncertainty to infinity. The network must balance predicting accurate speeds with predicting honest confidence bounds.

---

## 3. Architecture Exploration & Selection Tournament

In Phase 6, we generated $399,714$ canonical $(20, 9)$ windows partitioned by physical drivers:
- **Training Set**: Driver E ($N = 226,928$ windows across 42 trips)
- **Validation Set**: Driver B ($N = 21,080$ windows across 8 trips)
- **Held-Out Test Set**: Driver A ($N = 123,464$ windows across 21 trips)

To find the optimal network, Phase 7 established an empirical tournament evaluating three distinct architectural paradigms under identical training budgets (Adam optimizer, initial lr $10^{-3}$, `CosineAnnealingLR` down to $10^{-6}$, batch size 256, weight decay $10^{-5}$, gradient clipping 5.0, seed 42):

### Candidate Architectures

1. **Candidate A: 2-Layer GRU (Historical v1.0 Baseline)**
   - Topology: `GRU(9 → 64, 2 layers, batch_first=True) → Dense(64 → 32, ReLU) → Dual Heads (32 → 1, 32 → 1)`.
   - Parameter Count: **41,506 parameters**.
   - Characteristics: High expressive power for sequential temporal dependencies, but recurrent cell evaluation exhibits higher CPU latency and sequential memory dependencies.

2. **Candidate B: Lightweight 1D-CNN (VelocityNet v1.1)**
   - Topology:
     - `Conv1d(9 → 48, kernel=3, padding=1) → BatchNorm1d → ReLU`
     - `Conv1d(48 → 64, kernel=3, padding=1) → BatchNorm1d → ReLU`
     - `Conv1d(64 → 64, kernel=3, padding=1) → BatchNorm1d → ReLU`
     - `AdaptiveAvgPool1d(1)`
     - `Dense(64 → 32, ReLU) → Dual Heads (32 → 1, 32 → 1)`
   - Parameter Count: **25,474 parameters** ($-38.6\%$ fewer than GRU).
   - Characteristics: Parallel temporal convolutions capture multi-frequency vibration patterns efficiently with zero recurrent feedback, running exceptionally fast on mobile edge hardware.

3. **Candidate C: Conv1D-GRU Hybrid**
   - Topology: `Conv1d(9 → 32, k=3) → GRU(32 → 48, 1 layer) → Dense(48 → 32, ReLU) → Dual Heads`.
   - Parameter Count: **14,402 parameters** ($-65.3\%$ fewer than GRU).
   - Characteristics: Aggressive parameter reduction combining local convolutional smoothing with recurrent state propagation.

### Two-Stage Selection Procedure on Driver B Validation

To eliminate human bias and avoid overfitting to test data, candidate selection followed an explicit, deterministic hierarchical policy:
- **Stage 1 (Checkpoint Selection)**: Within each training run, the best model epoch was chosen by minimum Driver B validation Gaussian NLL.
- **Stage 2 (Architecture Selection)**: Candidates restored to their best-NLL checkpoint were ranked by hierarchical criteria:
  1. *Primary*: Minimum Validation RMSE (`val_rmse`)
  2. *Secondary*: Minimum Validation MAE (`val_mae`)
  3. *Tertiary*: Minimum Validation NLL (`val_nll`)
  4. *Quaternary*: High-Speed RMSE ($v > 15\,\text{m/s}$)
  5. *Tie-Breakers*: CPU Latency P50, then parameter count.

| Candidate Architecture | Parameter Count | Val RMSE ($\text{m/s}$) | Val MAE ($\text{m/s}$) | Val NLL | Mean Bias ($\text{m/s}$) | Pearson $r$ | CPU Latency (P50) | Status |
|---|---|---|---|---|---|---|---|---|
| **Candidate A (2L-GRU v1.0)** | 41,506 | $5.060$ | $3.914$ | $3.082$ | $+0.412$ | $0.664$ | $0.78\,\text{ms}$ | Baseline Retained |
| **Candidate B (1D-CNN v1.1)** | **25,474** | **$4.433$** | **$3.387$** | $2.918$ | **$+0.282$** | **$0.711$** | **$0.26\,\text{ms}$** | **WINNER (Selected)** |
| **Candidate C (Conv-GRU)** | 14,402 | $4.600$ | $3.512$ | **$2.874$** | $+0.345$ | $0.698$ | $0.45\,\text{ms}$ | Runner-Up |

**Outcome**: **Candidate B (1D-CNN, 25,474 parameters) won the selection tournament**. It achieved the lowest validation RMSE ($4.433\,\text{m/s}$ vs $5.060\,\text{m/s}$ for GRU), lowest MAE ($3.387\,\text{m/s}$), lowest speed bias ($+0.282\,\text{m/s}$), highest correlation ($r = 0.711$), and the fastest inference latency ($0.26\,\text{ms}$ on CPU). Candidate C attained slightly lower NLL but lagged on RMSE.

---

## 4. Causal Exponential Moving Average (EMA) Smoothing

Vehicle acceleration is physically bounded by tire friction and engine torque ($|a| \lesssim 4.0\,\text{m/s}^2$). However, instantaneous neural network inferences on noisy IMU windows can exhibit epoch-to-epoch jitter.

To eliminate unphysical high-frequency fluctuations while preserving strict real-time causality, Phase 7 evaluated causal Exponential Moving Average (EMA) post-filtering:
$$\bar{v}[k] = \alpha \hat{v}[k] + (1 - \alpha) \bar{v}[k-1]$$
- **Strict Trip Resets**: Whenever a new physical trip begins, the EMA state is reset ($\bar{v}[0] = \hat{v}[0]$). No smoothing state leaks across different driving sessions.
- **Tuning on Driver B**: Smoothing coefficient $\alpha = 0.20$ was selected on Driver B validation:
  - Raw 1D-CNN Val RMSE: $4.433\,\text{m/s}$
  - 1D-CNN + Causal EMA ($\alpha=0.20$) Val RMSE: **$3.510\,\text{m/s}$** (**$-20.8\%$ error reduction**).

---

## 5. Single-Pass Held-Out Test Evaluation on Driver A

Once the model architecture and EMA hyperparameters were locked and frozen, VelocityNet v1.1 was evaluated on the completely unseen **Driver A test dataset ($N = 123,464$ windows)** in a single, unrepeated validation pass:

| Model / Baseline | Test RMSE ($\text{m/s}$) | Test MAE ($\text{m/s}$) | Pearson $r$ | Error Reduction vs Baseline |
|---|---|---|---|---|
| **Operational Static Mean Baseline** ($\bar{v}_{\text{train}} = 10.42\,\text{m/s}$) | $9.305$ | $7.810$ | $0.000$ | Reference ($0.0\%$) |
| **VelocityNet v1.0 (2L-GRU)** | $7.320$ | $5.680$ | $0.395$ | $-21.3\%$ |
| **VelocityNet v1.1 Raw (1D-CNN)** | $7.070$ | $5.390$ | $0.421$ | $-24.0\%$ |
| **VelocityNet v1.1 + Causal EMA ($\alpha=0.2$)** | **$6.484$** ($23.34\,\text{km/h}$) | **$4.891$** | **$0.466$** | **$-30.3\%$** |
| *Lag-1 Doppler Oracle ($v_k = v_{\text{gnss}}[k-1]$)* | *$0.401$* | *$0.210$* | *$0.998$* | *Non-causal diagnostic bound* |

### Physical Scenario Breakdown
VelocityNet's accuracy was audited across distinct physical driving regimes:
1. **Quiescent Straight Driving** ($|\omega_z| \le 0.05\,\text{rad/s}$): RMSE = $5.92\,\text{m/s}$. When the vehicle drives in a straight line without cornering slip, chassis vibrations provide a steady, predictable speed correlation.
2. **Dynamic Cornering** ($|\omega_z| > 0.05\,\text{rad/s}$): RMSE = $7.81\,\text{m/s}$. Lateral tire slip and suspension roll introduce lateral specific force transients that moderately increase error.
3. **Low Speed Standstill** ($v < 0.5\,\text{m/s}$): Raw neural networks struggle near zero speed because stationary sensor noise resembles low-amplitude rolling noise. (This proved that Phase 11's Gated ZUPT is indispensable for locking down standstills).

### Uncertainty Calibration & Out-of-Distribution Dispersion
- On Driver B (validation), predicted uncertainty matched nominal Gaussian coverage: **$97.7\%$ of true speeds fell within the predicted $\pm 2\sigma$ envelope**.
- On Driver A (unseen test driver), coverage degraded to **$87.6\%$ at $\pm 2\sigma$**.
- **Crucial Engineering Takeaway**: Neural networks trained on one driver exhibit out-of-distribution dispersion when deployed on another driver with different driving styles and chassis dynamics. Therefore, **Phase 9 must not blindly trust the raw predicted $\sigma_v^2$**. Instead, Phase 9 must implement measurement covariance inflation ($R_v = s \cdot \sigma_v^2$) and innovation gating to protect the Kalman filter from overconfident neural predictions.

---

## 6. Mobile Export & Numerical Parity Verification

To enable real-time deployment on embedded automotive hardware (Raspberry Pi, Android vehicle dashboards, NVIDIA Jetson), VelocityNet v1.1 was exported to open inference formats:
1. **ONNX Export (`opset 17`)**: Fixed single-window tensor shape $[1, 20, 9]$.
2. **LiteRT (TensorFlow Lite `.tflite`)**: Quantization-ready flatbuffer representation.

### 500-Window Numerical Parity Audit
To guarantee that inference engines produce bit-for-bit identical outputs to the PyTorch training model, 500 real driving windows from Driver A were evaluated across all three runtimes:
- **PyTorch vs. ONNX Runtime**: Max absolute error $= \mathbf{3.12 \times 10^{-6}\,\text{m/s}}$ (Target: $\le 10^{-4}$). **PASS** ✅
- **PyTorch vs. LiteRT**: Max absolute error $= \mathbf{2.45 \times 10^{-4}\,\text{m/s}}$ (Target: $\le 10^{-3}$). **PASS** ✅

---

## 7. Phase Summary & Handoff to Phase 8

| Property | Phase 7 VelocityNet Specification |
|---|---|
| **Authoritative Model** | VelocityNet v1.1 (Lightweight 1D-CNN) |
| **Model Parameters** | 25,474 parameters ($101.9\,\text{KB}$ float32) |
| **Input Shape** | $[B, 20, 9]$ ($2.0\,\text{s}$ causal history at $10\,\text{Hz}$) |
| **Output Shape** | $[B, 2]$: Forward speed $\hat{v}$ ($\text{m/s}$) and log-variance $\log \sigma_v^2$ |
| **Post-Processing** | Causal EMA ($\alpha = 0.20$) with per-trip state resets |
| **Test Performance** | Test RMSE $6.484\,\text{m/s}$ ($-30.3\%$ over static baseline) |
| **Inference Latency** | $0.26\,\text{ms}$ on CPU |
| **Export Formats** | `velocitynet_v1_1.onnx` and `velocitynet_v1_1.tflite` |
| **Acceptance Gate Status** | **ACCEPTED AS EXPERIMENTAL CANDIDATE FOR PHASE 9 ESKF FUSION** |

VelocityNet provides the primary autonomous velocity source for GNSS-denied navigation. But linear velocity is only half the equation: consumer IMUs also suffer from continuous accelerometer and gyroscope bias drift. In **Phase 8**, we tackle the complementary problem: constructing **BiasNet** to learn and subtract residual inertial sensor biases.
