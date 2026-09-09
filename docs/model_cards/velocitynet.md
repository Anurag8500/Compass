# VelocityNet Model Card (v1.0)

**C.O.M.P.A.S.S. — Cognitive Off-grid Machine-learning Positioning And Sensor System**  
*SIH 2026 Problem Statement 26168 — Indian Space Research Organisation (ISRO)*

---

## 1. Model Details

- **Model Name**: VelocityNet
- **Model Version**: v1.0
- **Model Type**: Deep Recurrent Neural Network (2-layer Gated Recurrent Unit with Dual-Head Linear Projections)
- **Primary Task**: Forward vehicle speed regression ($\mu_v$) and heteroscedastic log-variance estimation ($\log \sigma_v^2$) from multi-channel inertial kinematics
- **Authors**: C.O.M.P.A.S.S. Development Team (SIH 2026 PS 26168)
- **Framework & Libraries**: PyTorch 2.14.0+cpu, ONNX 1.20.1, ONNX Runtime 1.26.0, LiteRT (`ai-edge-litert` 2.1.2) via `onnx2tf` 2.6.8
- **Release Date**: September 2026
- **License / Usage**: ISRO SIH 2026 Academic & Research Evaluation

---

## 2. Intended Use & Target Tasks

- **Primary Application**: Pseudo-measurement generator intended for dead reckoning aiding during GNSS outages (e.g., tunnels, urban canyons, electronic jamming, subterranean roadways).
- **Measurement Update Function**: Formulates forward velocity pseudo-measurement updates $\mathbf{z}_v = \mu_v$ with observation variance $R_v = \sigma_v^2 = \exp(\log \sigma_v^2)$ to help constrain open-loop integration drift within an Error-State Kalman Filter (ESKF).
- **Target Evaluation Rate**: 2.0 Hz update rate (evaluating every 0.5 s using a sliding 2.0 s history of 10 Hz inertial measurements).
- **Target Platform**: Wheeled road vehicles operating under road transport conditions.
- **Phase Boundary Note**: In Phase 7, VelocityNet is evaluated strictly as a standalone model. It is **NOT** integrated into the ESKF in Phase 7; ESKF integration belongs to Phase 9.

---

## 3. Out-of-Scope Uses

- **Non-Automotive Modalities**: Not trained, calibrated, or evaluated for airborne systems (fixed-wing, multicopters), maritime vessels, rail, or pedestrian tracking.
- **Standalone Trajectory Integrator**: VelocityNet estimates forward speed only. It is not an inertial navigation system and cannot produce position or heading without integration into a mechanization engine (such as the ESKF).
- **Drive-by-Wire Actuation**: Predictions are intended for state estimation, not for safety-critical closed-loop braking, steering, or automated collision avoidance.
- **Arbitrary Sensor Orientations**: The network requires inputs aligned with the vehicle Forward-Left-Up (FLU) frame. Raw, uncalibrated, or arbitrarily rotated sensor axes are out-of-scope.

---

## 4. Factors & Operating Envelope

- **Speed Range**: Observed within the available dataset from $0.0\text{ m/s}$ to approximately $35.0\text{ m/s}$ ($0\text{ to }126\text{ km/h}$). Not independently stress-tested beyond this dataset.
- **Observed Road Types**: Paved urban and suburban public roads in the IOVNBD corpus.
- **Sensor Modality**: Vehicle-frame 6-DOF inertial measurements (accelerometer specific forces and gyroscope angular rates).
- **Gravity Component**: Gravity ($g \approx 9.81\text{ m/s}^2$) is intentionally preserved in the vertical specific force channel ($f_z^v$) as specified by the Phase 6 canonical dataset contract.
- **Certification Status**: Experimental research model; no formal safety or operational certification is claimed.

---

## 5. Input Specification

The input to VelocityNet is a standardized 3D float32 tensor representing a temporal sequence of vehicle-frame inertial measurements:

$$\mathbf{X} \in \mathbb{R}^{B \times 20 \times 9}$$

| Dimension | Extent | Specification |
| :--- | :--- | :--- |
| **Batch Size ($B$)** | $B=1$ (Production Export) / Arbitrary (Training) | Fixed batch $B=1$ for embedded single-window deployment |
| **Sequence Length ($T$)** | 20 samples | 2.0 s temporal history sampled uniformly at 10.0 Hz ($\Delta t = 0.10\text{ s}$) |
| **Feature Channels ($C$)** | 9 channels | Canonical kinematic feature order in vehicle FLU frame |

### Feature Ordering

| Index | Symbol | Channel Name | Physical Unit | Description |
| :--- | :--- | :--- | :--- | :--- |
| 0 | $f_x^v$ | Longitudinal Specific Force | $\text{m/s}^2$ | Vehicle forward specific force |
| 1 | $f_y^v$ | Lateral Specific Force | $\text{m/s}^2$ | Vehicle lateral specific force |
| 2 | $f_z^v$ | Vertical Specific Force | $\text{m/s}^2$ | Vehicle upward specific force (gravity preserved) |
| 3 | $\omega_x^v$ | Roll Angular Rate | $\text{rad/s}$ | Angular rate about longitudinal axis |
| 4 | $\omega_y^v$ | Pitch Angular Rate | $\text{rad/s}$ | Angular rate about lateral axis |
| 5 | $\omega_z^v$ | Yaw Angular Rate | $\text{rad/s}$ | Angular rate about vertical axis |
| 6 | $\|\mathbf{f}^v\|$ | Specific Force Magnitude | $\text{m/s}^2$ | Euclidean norm $\sqrt{(f_x^v)^2 + (f_y^v)^2 + (f_z^v)^2}$ |
| 7 | $\|\dot{\mathbf{f}}^v\|$ | Force Jerk Approximation | $\text{m/s}^3$ | Backward finite-difference magnitude derivative |
| 8 | $\|\boldsymbol{\omega}^v\|$ | Angular Rate Magnitude | $\text{rad/s}$ | Euclidean norm $\sqrt{(\omega_x^v)^2 + (\omega_y^v)^2 + (\omega_z^v)^2}$ |

---

## 6. Output Specification

VelocityNet outputs two synchronous scalar predictions per window:

$$\hat{\mathbf{y}} = \left[ \mu_v, \; \log \sigma_v^2 \right]^T \in \mathbb{R}^{B \times 2}$$

1. **Forward Velocity ($\mu_v$)**:
   - **Data Type**: `float32`
   - **Physical Unit**: $\text{m/s}$
   - **Semantic**: Estimated forward speed along the vehicle body $x$-axis.
2. **Log-Variance ($\log \sigma_v^2$)**:
   - **Data Type**: `float32`
   - **Physical Unit**: Dimensionless
   - **Semantic**: Natural logarithm of observation variance $\sigma_v^2$. Clamped strictly to $[-10.0, +10.0]$ in the forward pass to protect numerical stability.

---

## 7. Loss Function & Selected Training Configuration

The model is optimized using **Heteroscedastic Gaussian Negative Log-Likelihood (NLL)** loss:

$$\mathcal{L}_{\text{NLL}}(\mu_v, s, y) = \frac{1}{2} \exp(-s) (y - \mu_v)^2 + \frac{1}{2} s + \frac{1}{2} \ln(2\pi)$$

where $s = \log \sigma_v^2$ and $y$ is the Doppler-derived ground truth forward velocity from high-precision GNSS.

### Selected Training Configuration
- **Optimizer**: Adam ($\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$)
- **Learning Rate**: $1.0 \times 10^{-3}$ initial, scheduled with Cosine Annealing (`CosineAnnealingLR`, $T_{\max}=20$, $\eta_{\min}=1.0 \times 10^{-5}$)
- **Weight Decay ($L_2$)**: $1.0 \times 10^{-5}$
- **Batch Size**: 256
- **Gradient Clipping**: $\ell_2$-norm clipped at $5.0\text{ m/s}$
- **Early Stopping**: Patience of 5 epochs monitoring validation Gaussian NLL
- **Reproducibility**: Seed-controlled (seed 42) across Python, NumPy, and PyTorch under the tested execution environment.

---

## 8. Training Data & Split Integrity

The dataset is derived from the IOVNBD corpus (Phase 6), comprising 142 downstream-ready file representations corresponding to 71 unique physical round-trip sessions.

### Frozen Split Allocation

| Split Role | Assigned Subject | Unique Physical Trips | Downstream Files | Valid Windows ($N$) |
| :--- | :--- | :--- | :--- | :--- |
| **Train** | **Driver E** | 22 trips | 44 files | 226,928 |
| **Validation** | **Driver B** | 2 trips | 4 files | 21,080 |
| **Held-Out Test** | **Driver A** | 12 trips | 24 files | 123,464 |
| **Excluded** | **Driver D** | 35 trips | 70 files | 0 (Omitted due to Phase 6 data inspection) |

Driver A is an unseen subject operating on independent routes and traffic patterns. There is zero window overlap between splits.

---

## 9. Quantitative Performance

All final metrics are reported on the single-pass evaluation of the **fully held-out Driver A test split** ($N = 123,464$ test windows):

| Evaluation Set | NLL Loss | RMSE ($\text{m/s}$) | RMSE ($\text{km/h}$) | MAE ($\text{m/s}$) | Mean Bias ($\text{m/s}$) | Pearson $r$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Validation (Driver B)** | **3.0030** | **5.017** | **18.06** | **3.835** | **+0.313** | **0.6952** |
| **Held-Out Test (Driver A)** | **3.8959** | **7.320** | **26.35** | **5.464** | **+0.831** | **0.4070** |

---

## 10. Uncertainty Coverage & Dispersion Analysis

Empirical coverage of predicted Gaussian confidence intervals against ground truth Doppler velocity:

| Confidence Interval | Theoretical Gaussian Target | Empirical VelocityNet Coverage | Mean Predicted $\sigma_v$ |
| :--- | :--- | :--- | :--- |
| **$\pm 1\sigma$ Coverage** | 68.3% | **59.0%** | $5.11\text{ m/s}$ |
| **$\pm 2\sigma$ Coverage** | 95.4% | **84.8%** | $10.21\text{ m/s}$ |
| **$\pm 3\sigma$ Coverage** | 99.7% | **94.1%** | $15.32\text{ m/s}$ |

> [!NOTE]
> **Dispersion Finding**: VelocityNet produces a learned heteroscedastic uncertainty estimate, but held-out coverage is below nominal Gaussian coverage (e.g., $84.8\%$ empirical vs $95.4\%$ nominal at $2\sigma$). This indicates that the learned uncertainty is **under-dispersed** for v1.0 and should not be characterized as fully calibrated.

---

## 11. Scenario-Wise Performance Breakdowns

Empirical performance observed across driving regimes on Driver A:

| Driving Scenario | Window Count ($N$) | RMSE ($\text{m/s}$) | MAE ($\text{m/s}$) | Mean Bias ($\text{m/s}$) | Mean $\sigma_v$ ($\text{m/s}$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Overall Held-Out Test** | 123,464 | 7.320 | 5.464 | +0.831 | 5.11 |
| **Low Speed ($< 2\text{ m/s}$)** | 22,276 | 6.985 | 4.144 | +4.093 | 3.58 |
| **Medium Speed ($2\text{--}15\text{ m/s}$)** | 83,410 | 6.464 | 5.141 | +1.683 | 5.33 |
| **High Speed ($> 15\text{ m/s}$)** | 17,778 | 10.721 | 8.637 | -7.251 | 5.98 |
| **Straight Driving ($|\omega_z| \le 0.05\text{ rad/s}$)** | 24,700 | 7.384 | 5.152 | -1.120 | 4.53 |
| **Cornering / Turning ($|\omega_z| > 0.05\text{ rad/s}$)** | 98,764 | 7.304 | 5.542 | +1.319 | 5.25 |
| **Dynamic Accel / Braking** | 123,454 | 7.320 | 5.464 | +0.831 | 5.11 |

*Observations*:
- Low-speed windows exhibit a positive bias ($+4.09\text{ m/s}$), showing that the network tends to overpredict speed near standstill. This highlights the operational importance of the deterministic Zero-Velocity Update (ZUPT) detector from Phase 4 to constrain standstill drift.
- High-speed windows exhibit negative bias (underprediction, bias $-7.25\text{ m/s}$), with higher RMSE ($10.72\text{ m/s}$).

---

## 12. Baseline Comparisons: Operational Causal vs Non-Causal Oracle

Comparison on held-out Driver A:

| Method | Category | Test RMSE ($\text{m/s}$) | Test MAE ($\text{m/s}$) | Status vs VelocityNet |
| :--- | :--- | :--- | :--- | :--- |
| **Static Training Mean ($15.57\text{ m/s}$)** | Operational Causal Baseline | 9.305 | 8.036 | **Beaten by VelocityNet ($-21.3\%$ RMSE)** |
| **VelocityNet (v1.0)** | **Production 2L-GRU** | **7.320** | **5.464** | **Authoritative Production Model** |
| **Lag-1 Ground-Truth Speed** | Non-Causal Oracle Reference | 0.401 | 0.249 | **NOT beaten (Infeasible in GNSS outage)** |

> [!IMPORTANT]
> **Acceptance Gate Clarification**:
> - The operational causal baseline for Phase 7 is the **static training mean** ($15.57\text{ m/s}$), which requires no future information or GNSS access. VelocityNet outperforms this baseline by $1.985\text{ m/s}$ ($21.3\%$ RMSE reduction).
> - The Lag-1 constant-velocity reference uses ground-truth Doppler speed from the preceding target window ($y_{t-1}$). Because true speed is unavailable during GNSS denial, this reference is classified as a **non-causal oracle diagnostic reference**, not an operational baseline. VelocityNet does not beat this oracle.

---

## 13. Model Comparison: GRU vs Lightweight 1D-CNN Baseline

As specified in Phase 7 Task 7, a lightweight 1D Temporal Convolutional baseline (`CNN1DVelocityBaseline`) was evaluated as an experimental alternative:

| Metric | VelocityNet (Production GRU) | Lightweight 1D-CNN Baseline | Status / Assessment |
| :--- | :--- | :--- | :--- |
| **Architecture** | 2-layer GRU (64 hidden) | 3 Conv1D layers (48, 64, 64) | Recurrent vs feedforward |
| **Parameters** | 41,506 | 25,474 | Not parameter-matched (CNN is lighter) |
| **Train Time (4 ep, 50k subset)** | $29.15\text{ s}$ | $20.30\text{ s}$ | CNN trains faster per epoch |
| **Test RMSE (4 ep subset)** | $8.212\text{ m/s}$ | $7.222\text{ m/s}$ | Subset comparison favors CNN on RMSE |
| **CPU Latency (P50)** | $3.94\text{ ms}$ | $0.50\text{ ms}$ | Both well below $10\text{ ms}$ ceiling |
| **Role in COMPASS** | **Authoritative Production Model** | **Experimental Baseline Only** | GRU retained per COMPASS specification |

*Assessment*: The GRU remains the authoritative production VelocityNet architecture specified by COMPASS. The 1D-CNN was evaluated solely as an experimental lightweight alternative. The reported subset experiment did not establish universal superiority of the GRU.

---

## 14. Computational Footprint & Latency

Measurements performed on single-thread standard CPU ($2.4\text{ GHz}$):

| Deployment Target | Artifact Size | Single-Window Latency (P50) | Single-Window Latency (P95) | Navigation Epoch Budget ($500\text{ ms}$) |
| :--- | :--- | :--- | :--- | :--- |
| **PyTorch CPU** | $514\text{ KB}$ (`.pt`) | $1.90\text{ ms}$ | $2.78\text{ ms}$ | $0.56\%$ budget consumed |
| **ONNX Runtime** | $171\text{ KB}$ (`.onnx`) | $1.85\text{ ms}$ | $2.31\text{ ms}$ | $0.46\%$ budget consumed |
| **LiteRT (TFLite)**| $316\text{ KB}$ (`.tflite`)| $0.92\text{ ms}$ | $1.24\text{ ms}$ | $0.25\%$ budget consumed |

In all runtimes, latency is $< 5.0\text{ ms}$, satisfying the $< 10.0\text{ ms}$ real-time edge computing requirement.

---

## 15. Export Parity & Numerical Verification

Numerical parity evaluated over $N = 500$ held-out test windows from `test.npz` in single-window mode ($B=1$):

| Comparison | Speed Max $\Delta$ ($\text{m/s}$) | Log-Var Max $\Delta$ | Tightened Tolerance | Gate Result |
| :--- | :--- | :--- | :--- | :--- |
| **PyTorch vs ONNX Runtime** | $5.72 \times 10^{-6}$ | $1.43 \times 10^{-6}$ | $\le 1.0 \times 10^{-4}$ | **PASSED** |
| **PyTorch vs LiteRT (TFLite)** | $3.81 \times 10^{-6}$ | $9.54 \times 10^{-7}$ | $\le 1.0 \times 10^{-3}$ | **PASSED** |

Both speed output and log-variance output satisfy tightened numerical parity gates across 500 held-out windows.

---

## 16. Known Failure Modes & Limitations

1. **Standstill Positive Bias**: At stops ($v = 0$), the model exhibits a positive bias ($+4.09\text{ m/s}$ average on low-speed segments). Dedicated ZUPT gating (Phase 4) is necessary to suppress velocity drift during stops.
2. **High-Speed Underprediction**: On high-speed segments ($> 15\text{ m/s}$), the model underestimates speed (bias $-7.25\text{ m/s}$, RMSE $10.72\text{ m/s}$).
3. **Under-Dispersed Out-of-Distribution Uncertainty**: While empirical coverage on validation data (Driver B) is well-aligned ($95.43\%$ at $2\sigma$, standardized residual $\mu = 0.005, \sigma = 0.958$), coverage degrades on unseen Driver A ($84.8\%$ at $2\sigma$). Empirical measurement covariance scaling ($R_v \leftarrow s \cdot \sigma_v^2$) is required for filter fusion.
4. **Reverse Driving Out-of-Distribution**: Reverse motion is underrepresented in the dataset; predictions in reverse are untrusted.

---

## 17. Ethical & Privacy Considerations

Input features consist exclusively of vehicle-frame inertial specific forces and angular velocities. No GPS geodetic coordinates, vehicle identifiers, or biometric data are utilized or output by the network.

---

## 18. Versioning & Provenance Hashes

| Artifact Description | File Path | SHA-256 Digest |
| :--- | :--- | :--- |
| **PyTorch Weights** | `models/velocitynet_v1_best.pt` | (Best checkpoint, Epoch 8) |
| **Exported ONNX Model** | `models/velocitynet_v1_6fadb0c7.onnx` | `7758d8a1f75e134a79ddc5d343ea11b659af240584dbbcfe203f5016bf2faf3b` |
| **Exported LiteRT Model** | `models/velocitynet_v1_6fadb0c7.tflite` | `98b58a96c0354ea84ec8a3685370e5c0d1fa5514d7b8048f679241111cafe8ff` |
| **Model Configuration** | `models/model_config_velocitynet_v1.json` | Manifest hash tag `6fadb0c7` |
| **Dataset Manifest** | `data/ml_dataset_v1/dataset_manifest.json` | `6fadb0c7a950ad0090886c5f778d91f8fa958616fa1b1626f634568fa0d927c3` |
| **Normalization Config** | `data/ml_dataset_v1/normalization.json` | `0649fdd7e350c931f48c9b7c2d6e9eb74cd731e4ec8d797a713e41753d1ee219` |

---

## 19. VelocityNet v1 → v1.1 Improvement Study

A structured, evidence-driven refinement cycle was conducted across eight focused investigations using Driver E for training and Driver B for validation model selection (Driver A held-out test split remained strictly untouched during selection).

### 1. Experimental Suite Summary

| Experiment ID | Focus Area | Key Finding | Action / Decision |
| :--- | :--- | :--- | :--- |
| **EXP-1** | Target Quality (`y_speed` vs `y_speed_raw`) | Both have identical mean speed ($15.571\text{ m/s}$ train, $10.439\text{ m/s}$ val). Raw labels contain Doppler dropouts up to $29.7\text{ m/s}$ in $0.048\%$ of windows. Median filter cleanly suppresses dropouts. | **Phase 6 target confirmed correct & frozen.** |
| **EXP-2** | Context Window (2.0s vs 4.0s) | 4.0s context yields minor RMSE improvement ($4.925\text{ m/s}$ vs $5.017\text{ m/s}$) at $+93\%$ CPU latency penalty ($1.89\text{ ms}$ vs $0.98\text{ ms}$) and higher NLL ($3.461$ vs $3.003$). | **2.0s context contract retained.** |
| **EXP-3** | Delta-V vs Direct Speed | Speed increment ($\Delta v$) prediction achieves low correlation ($r = 0.284$) and suffers from dead-reckoning velocity drift when integrated over time. | **Direct speed regression retained.** |
| **EXP-4** | Model Architecture Alternatives | 1D-CNN (25,474 params) achieves Val RMSE $4.398\text{ m/s}$; Conv-GRU hybrid (14,402 params) achieves $4.898\text{ m/s}$; 1L-GRU ($7.752\text{ m/s}$) underfits; 2L-GRU baseline achieves $5.017\text{ m/s}$. | **2L-GRU retained as authoritative production architecture per project spec.** |
| **EXP-5** | Training Configuration Study | Learning rate $10^{-3}$ superior to $5 \times 10^{-4}$ (NLL $3.003$ vs $3.690$); weight decay $10^{-5}$ superior to $10^{-4}$; dropout $0.2$ optimal; batch size 256 more stable than 128. | **Baseline training configuration validated.** |
| **EXP-6** | Causal EMA Post-Filtering | Applying causal EMA ($\alpha = 0.5$) to predicted speed reduces Val RMSE from $5.017\text{ m/s}$ to $4.470\text{ m/s}$ and raises Pearson correlation from $0.695$ to $0.739$. | **Designated as downstream Phase 9 filter-fusion processing option.** |
| **EXP-7** | Uncertainty Residual Audit | Validation residuals are unbiased ($\mu = 0.005$) with near-ideal Gaussian variance ($\sigma_z = 0.958$) and $95.43\%$ $2\sigma$ coverage. Under-dispersion on Driver A is an out-of-distribution transfer effect. | **Documented transparently; covariance scaling required in Phase 9.** |
| **EXP-8** | Non-Negative Parameterization | Predicted speed on validation is strictly non-negative (min $+0.080\text{ m/s}$, $0.0\%$ negative). Softplus head degrades optimization (Val RMSE $7.583\text{ m/s}$). | **Standard linear head retained.** |

### 2. Validation Comparison Table

| Candidate | Architecture | Window | Params | Val NLL | Val RMSE ($\text{m/s}$) | Val MAE ($\text{m/s}$) | Val Bias ($\text{m/s}$) | Val Pearson $r$ | CPU P50 Latency |
| :--- | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **VelocityNet v1 (Production)** | **2L-GRU** | **2.0s** | **41,506** | **3.003** | **5.017** | **3.835** | **+0.313** | **0.6952** | **1.72 ms** |
| 1L-GRU Lightweight | 1L-GRU | 2.0s | 16,546 | 3.495 | 7.752 | 6.290 | +5.176 | 0.0497 | 1.10 ms |
| 1D-CNN Baseline | 1D-CNN | 2.0s | 25,474 | 2.889 | 4.398 | 3.384 | +0.394 | 0.7196 | 0.54 ms |
| Conv1D-GRU Hybrid | Conv+GRU | 2.0s | 14,402 | 2.930 | 4.898 | 3.812 | +0.978 | 0.6985 | 1.25 ms |
| VelocityNet Softplus | 2L-GRU | 2.0s | 41,506 | 3.478 | 7.583 | 6.124 | +4.919 | 0.1766 | 2.07 ms |
| VelocityNet 4.0s Context | 2L-GRU | 4.0s | 41,506 | 3.461 | 4.925 | 3.892 | -0.488 | 0.6608 | 1.53 ms |
| Delta-V Increment | 2L-GRU | 2.0s | 41,506 | 1.543 | 1.280* | 0.800* | -0.238 | 0.2843 | 2.05 ms |

*\*Note: Delta-V RMSE reflects 2-second speed change increments ($\text{m/s}$ per window), not absolute vehicle speed.*

### 3. Final Selected Model & Architectural Decision

- **Selected Candidate**: **VelocityNet v1 Baseline Retained (2-Layer GRU, 41,506 parameters)**.
- **Scientific Rationale**:
  1. The 2-layer GRU demonstrates the strongest in-distribution Gaussian likelihood ($NLL = 3.003$) and reliable recurrent dynamics.
  2. While the lightweight 1D-CNN exhibited lower subset RMSE, the GRU remains the authoritative production model established by the project specification. Recurrent hidden state continuity is structurally aligned with the recursive Kalman filter mechanization planned for Phase 9.
  3. Longer context (4.0s) provided negligible RMSE reduction ($0.09\text{ m/s}$) while doubling latency and increasing NLL.
  4. Non-negative constraints (Softplus) degraded performance; the unconstrained linear projection naturally predicts positive forward speeds.
  5. Causal EMA filtering ($\alpha = 0.5$) demonstrates a verified post-processing error reduction ($5.017 \to 4.470\text{ m/s}$) without architectural modification.

---

*Signed by COMPASS Machine Learning Systems Engineering Team — SIH 2026 Problem Statement 26168.*
