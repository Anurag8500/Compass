# VelocityNet Model Card: v1.0 Baseline & v1.1 Selected Candidate

**C.O.M.P.A.S.S. — Cognitive Off-grid Machine-learning Positioning And Sensor System**  
*SIH 2026 Problem Statement 26168 — Indian Space Research Organisation (ISRO)*

---

## 1. Model Details & Versioning Governance

This model card documents both the historical baseline (**VelocityNet v1.0**) and the rigorously trained and selected candidate (**VelocityNet v1.1**).

| Attribute | VelocityNet v1.0 (Historical Baseline) | VelocityNet v1.1 (Selected Experimental Candidate) |
| :--- | :--- | :--- |
| **Release Status** | Frozen Historical Baseline | Selected Experimental Candidate for Phase 9 ESKF Integration |
| **Architecture** | 2-Layer Gated Recurrent Unit (GRU) | Lightweight 1D Temporal Convolutional Network (1D-CNN) |
| **Parameter Count** | 41,506 parameters | 25,474 parameters (-38.6% parameters) |
| **Hidden Dimensions** | GRU hidden=64, dense=32 | Conv1D: 9→48, 48→64, 64→64, GlobalAvgPool, dense=32 |
| **Output Heads** | Dual-headed: $[\mu_v, \log \sigma_v^2]^T$ | Dual-headed: $[\mu_v, \log \sigma_v^2]^T$ |
| **Causal Post-Processing**| None (Raw neural inference) | Causal Exponential Moving Average ($\alpha = 0.2$, trip-reset) |
| **Training Data** | Driver E ($N = 226,928$ windows) | Driver E ($N = 226,928$ windows, full dataset) |
| **Validation Selection** | Driver B ($N = 21,080$ windows) | Driver B ($N = 21,080$ windows, full dataset) |
| **Single Held-Out Test** | Driver A ($N = 123,464$ windows) | Driver A ($N = 123,464$ windows, evaluated once post-freeze) |
| **Validation RMSE** | $5.060\text{ m/s}$ ($5.017\text{ m/s}$ in screening) | **$4.433\text{ m/s}$** (Raw) / **$3.510\text{ m/s}$** (with EMA $\alpha=0.2$) |
| **Test RMSE (Driver A)** | $7.320\text{ m/s}$ ($26.35\text{ km/h}$) | **$7.070\text{ m/s}$** (Raw) / **$6.484\text{ m/s}$** (with EMA $\alpha=0.2$) |
| **CPU Latency (P50)** | $0.83\text{ ms}$ | **$0.26\text{ ms}$** (3.2x faster inference) |
| **Primary Artifacts** | `velocitynet_v1_best.pt` | `velocitynet_v1_1_best.pt`, `.onnx`, `.tflite` |

### Governance & Acceptance Status
- **VelocityNet v1.0** (GRU, 41,506 params) is preserved unchanged as the historical baseline.
- **VelocityNet v1.1** (1D-CNN, 25,474 params + causal EMA $\alpha = 0.2$) is the selected **experimental candidate for Phase 9 integration evaluation**. It is **not** an operationally proven or standalone high-precision speedometer. Phase 7 does not integrate it into the ESKF.

---

## 2. Intended Use & Target Tasks

- **Primary Application**: Pseudo-measurement generator intended for dead reckoning aiding during GNSS outages (e.g., tunnels, deep urban canyons, electronic jamming, subterranean roadways).
- **Measurement Update Function**: Formulates forward velocity pseudo-measurement updates $\mathbf{z}_v = \mu_v$ with observation variance $R_v = \sigma_v^2 = \exp(\log \sigma_v^2)$ to help constrain open-loop INS drift within an Error-State Kalman Filter (ESKF).
- **Target Evaluation Rate**: 2.0 Hz update rate (evaluating every 0.5 s using a sliding 2.0 s history of 10 Hz inertial measurements).
- **Target Platform**: Wheeled road vehicles operating under typical road transport conditions.
- **Phase Boundary Note**: In Phase 7, VelocityNet is evaluated strictly as a standalone model. It is **NOT** integrated into the ESKF in Phase 7; ESKF integration belongs to Phase 9.

---

## 3. Out-of-Scope Uses & Operational Limitations

- **Non-Automotive Modalities**: Not trained, calibrated, or evaluated for airborne systems (fixed-wing, multicopters), maritime vessels, rail, or pedestrian tracking.
- **Standalone Trajectory Integrator**: VelocityNet estimates forward speed only. It is not an inertial navigation system and cannot produce position or heading without integration into a mechanization engine (such as the ESKF).
- **Drive-by-Wire Actuation**: Predictions are intended for state estimation, not for safety-critical closed-loop braking, steering, or automated collision avoidance.
- **Arbitrary Sensor Orientations**: The network requires inputs aligned with the vehicle Forward-Left-Up (FLU) frame. Raw, uncalibrated, or arbitrarily rotated sensor axes are out-of-scope.
- **No Safety Certification**: Experimental research model developed under SIH 2026 Problem Statement 26168; no formal safety or operational certification is claimed.

---

## 4. Input Specification (Phase 6 Frozen Contract)

The input to VelocityNet is a standardized 3D float32 tensor representing a temporal sequence of vehicle-frame inertial measurements:

$$\mathbf{X} \in \mathbb{R}^{B \times 20 \times 9}$$

| Dimension | Extent | Specification |
| :--- | :--- | :--- |
| **Batch Size ($B$)** | $B=1$ (Production Export) / Arbitrary (Training) | Fixed batch $B=1$ for embedded single-window streaming |
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

## 5. Output Specification

VelocityNet outputs two synchronous scalar predictions per window:

$$\hat{\mathbf{y}} = \left[ \mu_v, \; \log \sigma_v^2 \right]^T \in \mathbb{R}^{B \times 2}$$

1. **Forward Velocity ($\mu_v$)**:
   - **Data Type**: `float32`
   - **Physical Unit**: $\text{m/s}$
   - **Semantic**: Estimated forward speed along the vehicle body $x$-axis. Unconstrained linear head.
2. **Log-Variance ($\log \sigma_v^2$)**:
   - **Data Type**: `float32`
   - **Physical Unit**: Dimensionless ($\ln(\text{m}^2/\text{s}^2)$)
   - **Semantic**: Natural logarithm of observation variance $\sigma_v^2$. Clamped strictly to $[-10.0, +10.0]$ in the forward pass to protect numerical stability.

---

## 6. Architecture & Training Details (VelocityNet v1.1)

### Exact Architecture Specification
- **Class**: `CNN1DVelocityBaseline` (in `ml/models/baselines/cnn1d_velocity.py`)
- **Layer Structure**:
  - `Conv1d(in_channels=9, out_channels=48, kernel_size=3, padding=1)` + `BatchNorm1d(48)` + `ReLU()` + `Dropout(p=0.2)`
  - `Conv1d(in_channels=48, out_channels=64, kernel_size=3, padding=1)` + `BatchNorm1d(64)` + `ReLU()` + `Dropout(p=0.2)`
  - `Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)` + `BatchNorm1d(64)` + `ReLU()` + `Dropout(p=0.2)`
  - `AdaptiveAvgPool1d(output_size=1)`
  - `Linear(in_features=64, out_features=32)` + `ReLU()` + `Dropout(p=0.2)`
  - `Linear(in_features=32, out_features=2)`
- **Total Parameters**: 25,474

### Actual Training Configuration (Matching Source Artifacts)
- **Optimizer**: Adam ($\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$)
- **Learning Rate**: Initial $1.0 \times 10^{-3}$, scheduled with `CosineAnnealingLR` ($T_{\max}=15$, $\eta_{\min}=1.0 \times 10^{-6}$)
- **Weight Decay ($L_2$)**: $1.0 \times 10^{-5}$
- **Batch Size**: 256
- **Max Epochs**: 15
- **Early Stopping Patience**: 5 epochs monitoring Driver B validation loss (checkpointed at epoch 1)
- **Gradient Clipping**: $\ell_2$-norm clipped at $5.0$
- **Random Seed**: 42
- **Loss Function**: Heteroscedastic Gaussian Negative Log-Likelihood (NLL):
  $$\mathcal{L}_{\text{NLL}}(\mu_v, s, y) = \frac{1}{2} \exp(-s) (y - \mu_v)^2 + \frac{1}{2} s + \frac{1}{2} \ln(2\pi)$$

---

## 7. Model Selection: Full-Data Validation on Driver B

All candidates were trained on the complete 226,928 Driver E dataset. Model selection was governed by an explicit, deterministic hierarchical policy applied **strictly on Driver B validation data**:
1. Primary: lowest `val_rmse`
2. Secondary: lowest `val_mae`
3. Tertiary: lowest `val_nll`
4. Quaternary: lowest `high_speed_rmse`
5. Tie-breaker: lowest `latency_p50_ms`, then lowest `params`

| Rank | Candidate | Architecture | Params | Val RMSE ($\text{m/s}$) | Val MAE ($\text{m/s}$) | Val NLL | Val Bias ($\text{m/s}$) | Val Pearson $r$ | High-Speed RMSE | CPU P50 Latency | Val RMSE (+EMA $\alpha=0.2$) |
| :-: | :--- | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **1** | **Candidate B (Selected)** | **1D-CNN** | **25,474** | **4.433** | **3.387** | **2.918** | **+0.282** | **0.7110** | **5.613** | **0.26 ms** | **3.510 m/s** |
| 2 | Candidate C | Conv1D-GRU | 14,402 | 4.600 | 3.540 | 2.874 | -0.422 | 0.6834 | 6.131 | 1.23 ms | 3.598 m/s |
| 3 | Candidate A | 2L-GRU | 41,506 | 5.060 | 3.907 | 2.989 | +0.429 | 0.6786 | 5.697 | 0.83 ms | 3.738 m/s |

### Selection Rationale
Candidate B was selected because it achieved the lowest validation RMSE ($4.433\text{ m/s}$ vs $4.600\text{ m/s}$ for Candidate C and $5.060\text{ m/s}$ for Candidate A), lowest validation MAE ($3.387\text{ m/s}$), competitive NLL ($2.918$ vs $2.874$ for Candidate C), best validation bias ($+0.282\text{ m/s}$), strong correlation ($0.7110$), and substantially lower CPU latency ($0.26\text{ ms}$ P50). Candidate C achieved the lowest NLL ($2.874$) but did not achieve the lowest RMSE.

---

## 8. Causal Exponential Moving Average (EMA) Downstream Filtering

To evaluate temporal smoothing without lookahead leakage, causal EMA filtering was studied on Driver B validation:

$$\hat{v}_t^{\text{filtered}} = \alpha \hat{v}_t + (1 - \alpha) \hat{v}_{t-1}^{\text{filtered}}$$

### Causal Protocol Rules
- Evaluated strictly along contiguous physical trips (grouped by `source_file_id`, sorted monotonically by timestamp).
- State reset $\hat{v}_0^{\text{filtered}} = \hat{v}_0$ at trip boundaries.
- Strictly forward in time; zero future samples.
- Post-processing only; not an alteration to neural network weights.

### Validation Sweep (Driver B)
| Smoothing Parameter $\alpha$ | Validation RMSE ($\text{m/s}$) | Validation MAE ($\text{m/s}$) | Validation Pearson $r$ |
| :--- | :--- | :--- | :--- |
| $\alpha = 1.0$ (Raw Model) | 4.433 | 3.387 | 0.7110 |
| $\alpha = 0.7$ | 4.093 | 3.125 | 0.7380 |
| $\alpha = 0.5$ | 3.864 | 2.949 | 0.7589 |
| $\alpha = 0.3$ | 3.630 | 2.766 | 0.7816 |
| **$\alpha = 0.2$ (Selected)** | **3.510** | **2.673** | **0.7937** |

$\alpha = 0.2$ was selected on Driver B validation and frozen prior to test evaluation.

---

## 9. Final Held-Out Evaluation on Driver A ($N = 123,464$)

Evaluated **EXACTLY ONCE** on Driver A after model and hyperparameters were frozen:

| Model / Configuration | Test RMSE ($\text{m/s}$) | Test RMSE ($\text{km/h}$) | Test MAE ($\text{m/s}$) | Mean Bias ($\text{m/s}$) | Pearson $r$ | Test NLL | Mean $\sigma_v$ ($\text{m/s}$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Static Training Mean Baseline** | 9.305 | 33.50 | 8.036 | +6.656 | N/A | N/A | N/A |
| **VelocityNet v1.0 Baseline (GRU)** | 7.320 | 26.35 | 5.464 | +0.831 | 0.4070 | 3.8959 | 5.107 |
| **VelocityNet v1.1 (1D-CNN Raw)** | **7.070** | **25.45** | **5.297** | **+1.520** | **0.4144** | **3.6167** | **5.366** |
| **VelocityNet v1.1 (+ EMA $\alpha=0.2$)** | **6.484** | **23.34** | **4.891** | **+1.522** | **0.4659** | N/A | **5.366** |
| *Lag-1 GT Oracle (Non-Causal Diagnostic)* | *0.401* | *1.44* | *0.249* | *-0.000* | *0.9981* | N/A | N/A |

### Key Generalization Findings
1. **Raw Model Generalization**: VelocityNet v1.1 raw neural inference reduces RMSE from $7.320\text{ m/s}$ to $7.070\text{ m/s}$ ($-0.250\text{ m/s}$) and test NLL from $3.8959$ to $3.6167$ ($-0.2792$).
2. **Observed Generalization Improvement with EMA**: Downstream causal EMA ($\alpha = 0.2$) reduces test RMSE to **$6.484\text{ m/s}$** ($23.34\text{ km/h}$) — an **$11.4\%$ error reduction** vs the v1 baseline ($7.320\text{ m/s}$) and a **$30.3\%$ error reduction** vs the operational static mean baseline ($9.305\text{ m/s}$). Causal EMA substantially reduced prediction error on both validation and held-out test data, indicating that temporal smoothing is beneficial for this model’s output.
3. **Correlation Improvement**: Pearson correlation increases from $0.4070$ to $0.4659$.
4. **Oracle Reference**: The lag-1 Doppler oracle ($0.401\text{ m/s}$) requires preceding ground-truth Doppler speed from GNSS; it is strictly a non-causal diagnostic bound, not an operational baseline, and was NOT beaten.

---

## 10. Scenario Analysis in Physical Units

Scenario masks are computed strictly in physical units (e.g., $|\omega_z| \le 0.05\text{ rad/s}$ via raw angular rates; specific-force magnitude deviation $|\|\mathbf{f}\| - 9.81| > 1.5\text{ m/s}^2$).

| Driving Scenario | Window Count ($N$) | Percentage | Raw RMSE ($\text{m/s}$) | EMA RMSE ($\text{m/s}$) | EMA MAE ($\text{m/s}$) | Mean Bias ($\text{m/s}$) | Mean $\sigma_v$ ($\text{m/s}$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Overall Held-Out Test** | 123,464 | 100.0% | 7.070 | **6.484** | 4.891 | +1.522 | 5.37 |
| **Low Speed ($< 2\text{ m/s}$)** | 22,276 | 18.04% | 7.450 | **6.946** | 4.654 | +4.624 | 3.94 |
| **Medium Speed ($2\text{--}15\text{ m/s}$)** | 83,410 | 67.56% | 6.139 | **5.449** | 4.370 | +2.511 | 5.60 |
| **High Speed ($> 15\text{ m/s}$)** | 17,778 | 14.40% | 10.039 | **9.603** | 7.628 | -7.008 | 6.08 |
| **Straight Driving ($|\omega_z| \le 0.05\text{ rad/s}$)** | 96,676 | 78.30% | 6.915 | **6.428** | 4.779 | +0.953 | 5.27 |
| **Cornering ($|\omega_z| > 0.05\text{ rad/s}$)** | 26,788 | 21.70% | 7.604 | **6.684** | 5.294 | +3.574 | 5.72 |
| **Dynamic Specific-Force Deviation Proxy** | 1,116 | 0.90% | 8.820 | **6.798** | 5.471 | +3.538 | 6.27 |

*Kinematic Regime Insights*:
- **Medium Speed Dominance**: Medium speed represents $67.6\%$ of all test driving, where VelocityNet v1.1 achieves its best performance ($5.449\text{ m/s}$ RMSE, $4.370\text{ m/s}$ MAE).
- **Standstill Overprediction**: Near standstill ($< 2\text{ m/s}$), the model exhibits a positive bias ($+4.62\text{ m/s}$). Standstill drift must be locked by the deterministic ZUPT detector from Phase 4.
- **High-Speed Underprediction**: On high-speed segments ($> 15\text{ m/s}$), the model exhibits negative bias ($-7.01\text{ m/s}$). The model predicts larger uncertainty in the high-speed regime ($\sigma_v = 6.08\text{ m/s}$); however, the associated uncertainty calibration must be validated during Phase 9 before being treated as reliable.
- **Dynamic Specific-Force Deviation Proxy**: Evaluated via $|\|\mathbf{f}\| - 9.81| > 1.5\text{ m/s}^2$. This is a heuristic physical-unit proxy based on specific-force magnitude deviation from gravity, not a direct longitudinal acceleration detector.

---

## 11. Uncertainty Coverage & Dispersion Diagnostics

Heteroscedastic Gaussian coverage evaluated on Driver A:

| Confidence Interval | Theoretical Target | Empirical Coverage (Driver A) | Empirical Coverage (Driver B Val) | Mean Predicted $\sigma_v$ |
| :--- | :--- | :--- | :--- | :--- |
| **$\pm 1\sigma$ Coverage** | 68.3% | **62.2%** | 68.9% | $5.37\text{ m/s}$ |
| **$\pm 2\sigma$ Coverage** | 95.4% | **87.6%** | 97.7% | $10.73\text{ m/s}$ |
| **$\pm 3\sigma$ Coverage** | 99.7% | **95.9%** | 99.6% | $16.10\text{ m/s}$ |

> [!NOTE]
> **Dispersion Finding**: On validation Driver B, predicted uncertainty matches nominal Gaussian coverage ($97.7\%$ at $2\sigma$). Uncertainty coverage degrades on the unseen Driver A distribution ($87.6\%$ at $2\sigma$), indicating poorer calibration under the observed train/validation-to-test distribution difference. Phase 9 must empirically calibrate or conservatively inflate the neural measurement covariance before fusion ($R_v = s \cdot \sigma_v^2$). Any covariance inflation factor $s$ must be selected using Phase 9 validation procedures rather than assumed from this Phase 7 result.

---

## 12. Deployment Footprint & Export Parity

Evaluated in single-window mode ($B=1, T=20, C=9$) over $N = 500$ real held-out Driver A test windows:

| Metric | PyTorch CPU | ONNX Runtime | LiteRT (TFLite) | Parity Gate | Result |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Artifact Size** | $107\text{ KB}$ (`.pt`) | $104\text{ KB}$ (`.onnx`) | $104\text{ KB}$ (`.tflite`) | N/A | Compact |
| **Single-Window Latency (P50)** | $0.26\text{ ms}$ | $0.21\text{ ms}$ | $0.18\text{ ms}$ | $< 10.0\text{ ms}$ | **PASSED** ($< 0.1\%$ epoch) |
| **Speed Max Absolute Error** | Reference | $5.72 \times 10^{-6}\text{ m/s}$ | $3.81 \times 10^{-6}\text{ m/s}$ | $\le 1.0 \times 10^{-4}$ / $\le 1.0 \times 10^{-3}$ | **PASSED** |
| **Speed Mean Absolute Error** | Reference | $7.81 \times 10^{-7}\text{ m/s}$ | $5.57 \times 10^{-7}\text{ m/s}$ | Informational | **PASSED** |
| **Log-Var Max Absolute Error** | Reference | $1.19 \times 10^{-6}$ | $9.54 \times 10^{-7}$ | $\le 1.0 \times 10^{-4}$ / $\le 1.0 \times 10^{-3}$ | **PASSED** |
| **Log-Var Mean Absolute Error** | Reference | $2.33 \times 10^{-7}$ | $1.47 \times 10^{-7}$ | Informational | **PASSED** |

---

## 13. Known Failure Modes & Engineering Constraints

1. **Standstill Positive Bias**: Overpredicts forward speed when stopped ($+4.62\text{ m/s}$). Deterministic ZUPT gating (Phase 4) is mandatory in Phase 9.
2. **High-Speed Underprediction**: Underestimates forward speed above $15\text{ m/s}$ (bias $-7.01\text{ m/s}$).
3. **Out-of-Distribution Dispersion**: Uncertainty coverage degrades on unseen drivers; observation variance scaling is required for Kalman filtering.
4. **Reverse Motion Inoperability**: No reverse motion data exists in the corpus; negative forward speeds must be gated out.

---

## 14. Artifact Manifest & Cryptographic Hashes

| Artifact Description | Relative File Path | SHA-256 Digest |
| :--- | :--- | :--- |
| **v1.0 Baseline Checkpoint** | `models/velocitynet_v1_best.pt` | `26b3e41b9e1150041d8eb426e6cbb41aa6db5b11153a71bbf5852445cbfdca0f` |
| **v1.1 Selected Checkpoint** | `models/velocitynet_v1_1_best.pt` | `d4b9bfb0f985d4213c2468e8b9fcc66f8e690b0e358439f9cd322c4f541210af` |
| **v1.1 Exported ONNX** | `models/velocitynet_v1_1.onnx` | `86b65c7e443970c2e27d0f1bdb6db66b42d41c9cc0053b91c837b158a707d47a` |
| **v1.1 Exported LiteRT** | `models/velocitynet_v1_1.tflite` | `f21abf87ac6d8a1c3b9180eb650fb9f4f2eeab8ef715ec405f8e607a95daa42a` |
| **v1.1 Model Configuration** | `models/model_config_velocitynet_v1_1.json` | Manifest hash tag `6fadb0c7` |
| **Model Selection Record** | `models/velocitynet_model_selection.json` | Full-data multi-candidate validation record |
| **Evaluation Record** | `models/velocitynet_v1_1_evaluation.json` | Single-pass Driver A evaluation |
| **Parity Verification Record** | `models/velocitynet_v1_1_export_parity.json` | 500-window numerical verification |
| **Dataset Manifest** | `data/ml_dataset_v1/dataset_manifest.json` | `6fadb0c74a1890fd2c5ab6997a33f3c04724f4083e862642a35244ef9a265b2b` |
| **Normalization Config** | `data/ml_dataset_v1/normalization.json` | `0649fdd7e350c931f48c9b7c2d6e9eb74cd731e4ec8d797a713e41753d1ee219` |

---

*Signed by COMPASS Machine Learning Systems Engineering Team — SIH 2026 Problem Statement 26168.*
