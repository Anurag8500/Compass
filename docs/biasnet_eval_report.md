# BiasNet Comprehensive Evaluation Report (Phase 8)

## 1. Executive Summary
This report presents the complete empirical evaluation of **BiasNet v1.0** for COMPASS Phase 8 (SIH 2026 Problem Statement 26168).

In accordance with authoritative project directives:
1. **BiasNet labels are not directly observed**: They are optimization-derived teacher targets computed by solving an inverse estimation problem over a 1.0 s horizon using Phase 4/5 strapdown mechanics against independent Racelogic VBOX RTK GNSS ground truth.
2. **Label identifiability is strictly gated**: Windows with non-converged solver solutions (`SOLVER_FAILURE`), ill-conditioned Jacobians ($\kappa > 50$), deficient rank, poor residual reduction ($< 1.20$), or unphysical bound violations ($|\Delta b_a| > 2.0\text{ m/s}^2, |\Delta b_g| > 0.15\text{ rad/s}$) are rejected.
3. **Direct label-space evaluation**: On unseen Driver B (Validation) and Driver A (Held-Out Test), BiasNet outperforms both the Zero Correction Baseline and the Train Mean Baseline by **29.1% to 32.9%** in Total Vector RMSE.
4. **Indirect navigation evaluation**: Fused through the ESKF with conservative covariance and innovation gating on synthetic GNSS outages (10s, 30s, 60s), BiasNet maintains filter numerical stability and achieves improved velocity tracking RMSE ($3.292\text{ m/s}$ vs $3.361\text{ m/s}$ for VelocityNet alone), though horizontal position error is essentially tied ($10.049\text{ m}$ vs $10.031\text{ m}$).
5. **Numerical export parity**: Both ONNX and LiteRT models match PyTorch float32 inference with maximum absolute errors $< 7 \times 10^{-7}$ across 500 real driving windows.

## 2. Methodology & Optimization Formulation

### Inverse Problem
For each window ending at $t_{\text{end}}$, with horizon $H = 1.0\text{ s}$ ($K=10$ intervals):
$$\min_{\Delta \mathbf{b}} \|\mathbf{r}(\Delta \mathbf{b})\|^2$$
where
$$\mathbf{r}(\Delta \mathbf{b}) = \frac{1}{\sqrt{K}} \sum_{k=1}^K \begin{bmatrix} W_p (\mathbf{p}_k - \mathbf{p}_{\text{ref}, k}) \\ W_v (\mathbf{v}_k - \mathbf{v}_{\text{ref}, k}) \\ W_\theta \delta \boldsymbol{\theta}(q_k, q_{\text{ref}, k}) \end{bmatrix}$$
with $W_p = 1.0\text{ m}^{-1}, W_v = 1.0\text{ (m/s)}^{-1}, W_\theta = 10.0\text{ rad}^{-1}$.

Solved via damped Levenberg-Marquardt with central finite-difference Jacobians and numerical safeguards.

### Identifiability Gating Statistics
- **Driver E (Train)**: 7,753 total windows evaluated -> 4,060 eligible (52.4%). Rejection breakdown: 3,281 for physical bounds, 291 for solver failure, 121 for poor residual reduction.
- **Driver B (Validation)**: 700 total windows evaluated -> 480 eligible (68.6%). Rejection breakdown: 113 for physical bounds, 105 for solver failure, 2 for poor residual reduction.
- **Jacobian Conditioning**: Median condition number $\kappa = 10.45$ (Driver E) and $\kappa = 10.51$ (Driver B). Effective rank is consistently 6.0 across all eligible windows.

## 3. Direct Label-Space Metrics

### Driver B (Validation Set — 480 Windows)
| Metric | Zero Baseline ($\Delta \mathbf{b}=\mathbf{0}$) | Train Mean Baseline | BiasNet v1.0 | Absolute Reduction | Relative Gain |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Total Vector RMSE** | $1.0401\text{ m/s}^2$ | $1.0169\text{ m/s}^2$ | **$0.7372\text{ m/s}^2$** | $-0.3029\text{ m/s}^2$ | **+29.1%** |
| **Total Vector MAE** | $0.9234\text{ m/s}^2$ | $0.8998\text{ m/s}^2$ | **$0.6121\text{ m/s}^2$** | $-0.3113\text{ m/s}^2$ | **+33.7%** |
| **Accel Vector RMSE** | $1.0387\text{ m/s}^2$ | $1.0153\text{ m/s}^2$ | **$0.7365\text{ m/s}^2$** | $-0.3022\text{ m/s}^2$ | **+29.1%** |
| **Gyro Vector RMSE** | $0.0554\text{ rad/s}$ | $0.0560\text{ rad/s}$ | **$0.0329\text{ rad/s}$** | $-0.0225\text{ rad/s}$ | **+40.6%** |
| **Gyro Vector MAE** | $0.0448\text{ rad/s}$ | $0.0454\text{ rad/s}$ | **$0.0270\text{ rad/s}$** | $-0.0178\text{ rad/s}$ | **+39.7%** |

### Driver A (Held-Out Test Set — 956 Windows)
Evaluated strictly once after freezing all model and training choices:
| Metric | Zero Baseline | Train Mean Baseline | BiasNet v1.0 | Absolute Reduction | Relative Gain |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Total Vector RMSE** | $1.0656\text{ m/s}^2$ | $1.0563\text{ m/s}^2$ | **$0.7153\text{ m/s}^2$** | $-0.3503\text{ m/s}^2$ | **+32.9%** |
| **Accel Vector RMSE** | $1.0633\text{ m/s}^2$ | $1.0539\text{ m/s}^2$ | **$0.7140\text{ m/s}^2$** | $-0.3493\text{ m/s}^2$ | **+32.9%** |
| **Gyro Vector RMSE** | $0.0703\text{ rad/s}$ | $0.0704\text{ rad/s}$ | **$0.0425\text{ rad/s}$** | $-0.0278\text{ rad/s}$ | **+39.5%** |

## 4. Controlled Indirect Navigation Ablation

Evaluated over real driving segment (`Categorised_S1.npz`) across synthetic GNSS outage durations under 4 conditions:
- **A. Pure Phase 5 ESKF**: Classical dead-reckoning + gated ZUPT.
- **B. ESKF + VelocityNet v1.1**: Neural forward speed aiding (1D-CNN, causal EMA $\alpha=0.2$).
- **C. ESKF + VelocityNet v1.1 + BiasNet v1**: Dual neural aiding via innovation gating.
- **D. ESKF + BiasNet v1**: Diagnostic ablation (BiasNet without VelocityNet).

### Outage Navigation Comparison
| Outage | Condition | Horiz RMSE [m] | Final Horiz Error [m] | Vel RMSE [m/s] | Mean NIS | Max Excursion [m] |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **10s** | A (Pure ESKF) | 0.426 | 0.616 | 0.012 | 0.000 | 0.616 |
| | B (+VNet) | 0.424 | 0.615 | 0.012 | 0.000 | 0.615 |
| | **C (+VNet+BNet)** | **0.424** | **0.615** | **0.012** | **0.049** | **0.615** |
| | D (+BNet only) | 0.424 | 0.615 | 0.012 | 0.049 | 0.615 |
| **30s** | A (Pure ESKF) | 11.289 | 63.880 | 3.775 | 0.000 | 63.880 |
| | B (+VNet) | 10.031 | 57.455 | 3.361 | 5.007 | 57.455 |
| | **C (+VNet+BNet)** | **10.049** | **57.543** | **3.292** | **1.839** | **57.543** |
| | D (+BNet only) | 11.258 | 63.685 | 3.696 | 1.582 | 63.685 |
| **60s** | A (Pure ESKF) | 1771.216 | 5308.950 | 133.082 | 0.000 | 5308.950 |
| | B (+VNet) | 1740.133 | 5230.277 | 131.604 | 5.007 | 5230.277 |
| | **C (+VNet+BNet)** | **1735.228** | **5202.109** | **130.002** | **2.543** | **5202.109** |
| | D (+BNet only) | 1770.020 | 5295.303 | 132.004 | 2.444 | 5295.303 |

### Key Navigation Findings
1. **Filter Stability & Authority**: In all scenarios, BiasNet innovations were accepted without filter divergence or covariance collapse. Mean NIS remained between 0.049 and 2.543, well within the 6-DOF $\chi^2$ confidence gate ($\chi^2_{0.99, 6} = 16.81$).
2. **Velocity Tracking**: Condition C (+VNet+BNet) achieved the best velocity tracking RMSE across both 30s ($3.292\text{ m/s}$) and 60s ($130.002\text{ m/s}$) outages.
3. **Position Drift Limitation**: While BiasNet improved velocity RMSE, its impact on horizontal position RMSE relative to VelocityNet alone was modest ($10.049\text{ m}$ vs $10.031\text{ m}$ on 30s outage).
4. **Single-Segment Diagnostic Scope**: This evaluation represents single-segment diagnostic evidence on `Categorised_S1.npz`. Closed-loop multi-trip trajectory tuning belongs to Phase 9.

## 5. Deployment Parity & Hash Verification
- `models/biasnet_v1_best.pt`: SHA-256 `0a309d54bfc0db5c0f74f31fe8401d86e9d42358dab713ae989c46699662f00e`
- `models/biasnet_v1.onnx`: SHA-256 `d57e64485a8722bc4041d7e73b3fed606e878c5b7ba5504e54081c4225674067`
- `models/biasnet_v1.tflite`: SHA-256 `551193b4544847c721c69d2acb572792b6c7ee37b31ad0d954015d6715a8934a`
- **Parity Results (500 Real Driving Windows)**:
  - ONNX Max Error: $6.56 \times 10^{-7}$ (Pass threshold: $1.0 \times 10^{-4}$).
  - LiteRT Max Error: $3.58 \times 10^{-7}$ (Pass threshold: $1.0 \times 10^{-3}$).

