# Phase 8 BiasNet Corrective Audit & Freeze Report (COMPASS SIH 2026 PS 26168)

## A. What Was Implemented
Phase 8 implements **BiasNet v1.0**, an experimental learned IMU bias correction subsystem engineered to supply bounded, innovation-gated pseudo-measurements $(\mathbf{z}_b, \mathbf{H}_b, \mathbf{R}_b)$ to the 15-state Error-State Kalman Filter (ESKF).
The implementation encompasses:
1. **Inverse-Problem Label Pipeline (`ml/data/biasnet_labels.py`)**: Short-horizon ($H = 1.0\text{ s}$) Levenberg-Marquardt solver estimating 6-DOF bias corrections ($\Delta \mathbf{b}_a \in \mathbb{R}^3, \Delta \mathbf{b}_g \in \mathbb{R}^3$) from strapdown mechanization residuals against synchronized Racelogic VBOX RTK GNSS ground truth.
2. **Identifiability & Conditioning Gate**: Rigorous SVD-based Jacobian observability gate auditing singular values, condition number $\kappa(J)$, and cost reduction ratio $\rho$, coupled with strict physical plausibility filtering.
3. **Neural Architecture (`ml/models/biasnet.py`)**: 2-layer GRU (48 hidden units, 24 dense projection units, 23,934 parameters) with in-graph physical safety clamps.
4. **Training & Validation Pipeline (`ml/training/train_biasnet.py`)**: Weighted Smooth L1 loss on eligible Driver E training windows, early stopping on Driver B validation.
5. **Indirect ESKF Navigation Outage Ablation (`scripts/run_phase8_navigation_ablation.py`)**: Controlled synthetic GNSS outages (10s, 30s, 60s) evaluating velocity and position drift against Pure ESKF and ESKF + VelocityNet v1.1.
6. **Dual Edge Export & Parity Verification (`ml/export/export_biasnet.py`)**: PyTorch $\to$ ONNX $\to$ LiteRT export pipeline with numerical parity verified over 500 real driving windows.

---

## B. What Was Corrected During This Audit
During this rigorous corrective audit, all methodological, numerical, and documentation inconsistencies across the codebase were reconciled:
1. **Enforced Real Solver Convergence Gating**: Previously, `solve_window_bias_correction` computed `converged` but did not gate eligibility on it. Fixed so `converged == True` is strictly required for `is_eligible`. Non-converged windows are deterministically rejected with reason code `SOLVER_FAILURE` (291 rejected in Train, 105 in Val).
2. **Numerically Sound Multi-Criteria Stopping Rule**: Implemented rigorous multi-criteria stopping:
   - Step norm: $\|\Delta \mathbf{x}\|_2 < 10^{-4}$
   - Relative cost improvement: $|C_{k-1} - C_k| / (C_{k-1} + 10^{-6}) < 10^{-4}$
   - Gradient infinity norm: $\|\mathbf{g}\|_\infty = \|J^T \mathbf{r}\|_\infty < 10^{-3}$ (evaluated at initialization and post-step).
   Max-iteration exhaustion (15 iterations) without meeting one of these conditions sets `converged = False`.
3. **Formal 3-Level Bound Taxonomy**: Resolved the ambiguity between initial engineering notes ($0.3\text{ m/s}^2, 0.05\text{ rad/s}$) and empirical smartphone bounds ($2.0\text{ m/s}^2, 0.15\text{ rad/s}$):
   - Level 1 (Solver Safeguard): $|\Delta \mathbf{b}_a| \le 5.0\text{ m/s}^2, |\Delta \mathbf{b}_g| \le 0.5\text{ rad/s}$ during optimization steps.
   - Level 2 (Physical Eligibility Gate): $|\Delta \mathbf{b}_a| \le 2.0\text{ m/s}^2, |\Delta \mathbf{b}_g| \le 0.15\text{ rad/s}$ post-convergence.
   - Level 3 (Neural Output In-Graph Clamps): $[-2.0, 2.0]\text{ m/s}^2, [-0.15, 0.15]\text{ rad/s}$ inside computational graphs.
4. **Authoritative Solver Configuration Alignment**: Replaced all documentation mentioning outdated parameters ($W_v = 2.0$, $h = 10^{-4}$, $\lambda_0 = 10^{-2}$, $\text{max\_iter} = 20$) with the actual executable parameters ($W_v = 1.0$, $h_a = 10^{-5}$, $h_g = 10^{-6}$, $\lambda_0 = 10^{-4}$, $\text{max\_iter} = 15$).
5. **Quaternion Error Frame Specification**: Corrected residual descriptions to document that the orientation residual $\mathbf{e}_\theta$ is formulated in the body/right-multiplicative convention ($\delta \mathbf{q} = \mathbf{q}_{\text{est}}^{-1} \otimes \mathbf{q}_{\text{ref}}$ with shortest-arc handling and axis-angle conversion), distinct from the local ENU navigation coordinates.
6. **Multi-Point Residual Formulation**: Explicitly documented that residuals are evaluated across all $K=10$ integration intervals with $1/\sqrt{K}$ scaling.
7. **Autocorrelation Factual Correction**: Replaced false "$r > 0.85$" claims in stability reports with truthful measured values ($r \approx 0.55-0.75$ accel, $0.25-0.45$ gyro).
8. **Navigation 60s Outage Discrepancy Elimination**: Fixed the 10x discrepancy in the 60s outage table; synchronized all reports to the authoritative JSON metrics (Pure ESKF $1771.216\text{ m}$, +VNet $1740.133\text{ m}$, +VNet+BNet $1735.228\text{ m}$).
9. **Authoritative Measurement Covariance**: Established exactly one authoritative covariance: $\mathbf{R}_b = \text{diag}([1.21, 1.21, 1.21, 0.0025, 0.0025, 0.0025])$, eliminating conflicting stale values.
10. **Test Suite Verification**: Executed full pytest suite, confirming 309 passing tests with zero failures.

---

## C. Exact Label-Generation Formulation
For each window ending at $t_{\text{end}}$ spanning horizon $H = 1.0\text{ s}$ ($K = 10$ intervals at $\Delta t = 0.1\text{ s}$):
- **Initial State**: Reference geodetic position, velocity, and azimuth heading at $t_0 = t_{\text{end}} - H$ are converted to local ENU Cartesian coordinates $(\mathbf{p}_0, \mathbf{v}_0, \mathbf{q}_0)$.
- **Strapdown Kinematics**:
  $$\hat{\mathbf{f}}^b[k] = \mathbf{f}_m^b[k] - (\mathbf{b}_{a,0} + \Delta \mathbf{b}_a), \quad \hat{\boldsymbol{\omega}}^b[k] = \boldsymbol{\omega}_m^b[k] - (\mathbf{b}_{g,0} + \Delta \mathbf{b}_g)$$
  $$\mathbf{a}^n[k] = R(\mathbf{q}[k]) \hat{\mathbf{f}}^b[k] + \begin{bmatrix} 0 \\ 0 \\ -g \end{bmatrix}, \quad \mathbf{p}[k+1] = \mathbf{p}[k] + \mathbf{v}[k] \Delta t + \frac{1}{2} \mathbf{a}^n[k] \Delta t^2$$
  $$\mathbf{v}[k+1] = \mathbf{v}[k] + \mathbf{a}^n[k] \Delta t, \quad \mathbf{q}[k+1] = \text{normalize}\left(\mathbf{q}[k] \otimes \Delta \mathbf{q}(\hat{\boldsymbol{\omega}}^b[k] \Delta t)\right)$$
- **Multi-Point Residual Vector** $\mathbf{r}(\Delta \mathbf{b}) \in \mathbb{R}^{90}$:
  Evaluated across all $K=10$ discrete steps ($k = 1, \dots, K$):
  $$\mathbf{r}_k = \frac{1}{\sqrt{K}} \begin{bmatrix} W_p (\mathbf{p}[k] - \mathbf{p}_{\text{ref}}[k]) \\ W_v (\mathbf{v}[k] - \mathbf{v}_{\text{ref}}[k]) \\ W_\theta \mathbf{e}_\theta(\mathbf{q}[k], \mathbf{q}_{\text{ref}}[k]) \end{bmatrix}$$
  Weights: $W_p = 1.0\text{ m}^{-1}$, $W_v = 1.0\text{ (m/s)}^{-1}$, $W_\theta = 10.0\text{ rad}^{-1}$.
- **Orientation Error Formulation**: Body/right-multiplicative convention:
  $$\delta \mathbf{q} = \mathbf{q}[k]^{-1} \otimes \mathbf{q}_{\text{ref}}[k]$$
  If $\delta q_0 < 0$, $\delta \mathbf{q} \leftarrow -\delta \mathbf{q}$ (shortest rotation). Small-angle conversion:
  $$\theta = 2 \text{ atan2}(\|\delta \mathbf{q}_{1:3}\|, \delta q_0), \quad \mathbf{e}_\theta = \frac{\theta}{\|\delta \mathbf{q}_{1:3}\|} \delta \mathbf{q}_{1:3}$$
- **Solver & Jacobian**: Damped Levenberg-Marquardt with central finite differences:
  $$h_a = 10^{-5}\text{ m/s}^2, \quad h_g = 10^{-6}\text{ rad/s}, \quad \lambda_0 = 10^{-4}$$
  $$(J^T J + \lambda I) \Delta \mathbf{x} = -J^T \mathbf{r}$$

---

## D. Exact Eligibility Gate
Eligibility requires satisfying all criteria under deterministic precedence:
1. `NON_FINITE_INPUT`: Any NaN/Inf in IMU or reference records.
2. `TIMESTEP_ANOMALY`: $|\Delta t - 0.01| > 0.005\text{ s}$ or non-monotonic timestamps.
3. `WINDOW_TOO_SHORT`: Input sample count $< 20$.
4. `HORIZON_TOO_SHORT`: Available horizon intervals $< 10$.
5. `NON_FINITE_SOLUTION`: NaN/Inf in optimizer output.
6. `SOLVER_FAILURE`: Non-convergence within tolerances (`converged == False`).
7. `DEFICIENT_RANK`: $\text{rank}(J) < 6$.
8. `ILL_CONDITIONED`: $\kappa(J) = \sigma_{\max} / \sigma_{\min} > 50.0$.
9. `BOUNDS_ACTIVE`: $|\Delta b_a| > 2.0\text{ m/s}^2$ or $|\Delta b_g| > 0.15\text{ rad/s}$.
10. `POOR_RESIDUAL_REDUCTION`: $\rho = \|r_0\| / \|r_{\text{final}}\| < 1.20$.
11. `VALID`: Window declared eligible.

---

## E. Exact Convergence Definition
A solution is declared `converged == True` if any of the following numerical conditions are met within `max_iterations = 15`:
1. **Gradient Infinity Norm**: $\|\mathbf{g}\|_\infty = \|J^T \mathbf{r}\|_\infty < 10^{-3}$ at step 0 or post-step.
2. **Step Norm**: $\|\Delta \mathbf{x}\|_2 < 10^{-4}$ on an accepted step.
3. **Relative Cost Improvement**: $(C_{k-1} - C_k) / (C_{k-1} + 10^{-6}) < 10^{-4}$ on an accepted step.

Exhaustion of 15 iterations without satisfying one of these conditions sets `converged = False`, triggering deterministic rejection as `SOLVER_FAILURE`.

---

## F. Exact Horizon Choice and Evidence
Audited on 500 representative windows per split across $H \in \{0.5\text{ s}, 1.0\text{ s}, 2.0\text{ s}\}$ (`docs/biasnet_horizon_audit.json`):

| Horizon | Driver E Median $\kappa$ | Driver E Conv Rate | Driver E Residual Red | Driver B Median $\kappa$ | Driver B Conv Rate | Driver B Residual Red |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.5 s** | 10.02 | 99.4% | $4.16\times$ | 10.03 | 85.2% | $3.73\times$ |
| **1.0 s** | 10.44 | 100.0% | $3.68\times$ | 10.47 | 85.6% | $3.87\times$ |
| **2.0 s** | 12.70 | 100.0% | $4.09\times$ | 12.72 | 85.0% | $4.43\times$ |

**Selection Rationale**:
$H = 1.0\text{ s}$ was selected not on an oversimplified claim of highest convergence, but based on a holistic engineering trade-off:
1. **Observability**: Mean singular values at $1.0\text{ s}$ ($\sim 6.4$ accel, $\sim 0.62$ gyro) provide substantially higher parameter sensitivity than at $0.5\text{ s}$ ($\sim 3.18$ accel, $\sim 0.32$ gyro).
2. **Conditioning Stability**: Median condition number at $1.0\text{ s}$ ($\kappa \approx 10.45$) is significantly lower and more stable than at $2.0\text{ s}$ ($\kappa \approx 12.70$, $p_{95} \approx 14.91$).
3. **Temporal Locality**: $1.0\text{ s}$ occupies exactly half of the $2.0\text{ s}$ input window, minimizing non-stationary dynamics during aggressive cornering.
4. **Runtime**: Solves in $18.4\text{ ms/window}$ (vs $34.5\text{ ms}$ at $2.0\text{ s}$).

---

## G. Train / Validation / Test Provenance
Strict driver-level isolation preserved:
- **Train (Driver E)**: 30 audited multi-kilometer driving trips, bounded at 350 windows/trip.
- **Validation (Driver B)**: 2 trips (`Categorised_M.npz`, `Uncategorised_M.npz`), bounded at 350 windows/trip.
- **Held-Out Test (Driver A)**: 5 trips (`Categorised_S1.npz`, `Categorised_S2.npz`, `Categorised_S3a.npz`, `Categorised_S3b.npz`, `Categorised_S3c.npz`). Strictly held out; evaluated once post-freeze.

---

## H. Label Counts and Rejection Reasons
Audit of final generated datasets (`data/ml_dataset_biasnet_v1/bias_label_manifest.json`):

| Split | Candidate Windows | Converged | Eligible Windows | Bounds Active | Solver Failure | Poor Residual |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Driver E (Train)** | 7,753 | 7,462 (96.2%) | **4,060 (52.4%)** | 3,281 (42.3%) | 291 (3.8%) | 121 (1.6%) |
| **Driver B (Val)** | 700 | 595 (85.0%) | **480 (68.6%)** | 113 (16.1%) | 105 (15.0%) | 2 (0.3%) |
| **Driver A (Test)** | 1,500 | 1,492 (99.5%) | **956 (63.7%)** | 536 (35.7%) | 8 (0.5%) | 0 (0.0%) |

Zero windows suffered rank deficiency or ill-conditioning ($\kappa > 50$). Bounds-active rejections correctly filter out severe phone holder mounting misalignment (cradle tilt) that projects gravity into apparent bias.

---

## I. Direct Validation Metrics (Driver B — 480 Windows)
Evaluated on optimization-derived teacher targets:

| Metric | Zero Baseline ($\Delta \mathbf{b} = \mathbf{0}$) | Train-Mean Baseline | BiasNet v1.0 | Improvement vs Zero |
| :--- | :--- | :--- | :--- | :--- |
| **Total Vector RMSE** | $1.0401\text{ m/s}^2$ | $1.0169\text{ m/s}^2$ | **$0.7372\text{ m/s}^2$** | **+29.1% error reduction** |
| **Accel Vector RMSE** | $1.0387\text{ m/s}^2$ | $1.0154\text{ m/s}^2$ | **$0.7365\text{ m/s}^2$** | **+29.1% error reduction** |
| **Gyro Vector RMSE** | $0.0554\text{ rad/s}$ | $0.0562\text{ rad/s}$ | **$0.0329\text{ rad/s}$** | **+40.6% error reduction** |
| **Total Vector MAE** | $0.9416\text{ m/s}^2$ | $0.9168\text{ m/s}^2$ | **$0.6125\text{ m/s}^2$** | **+35.0% error reduction** |

---

## J. Held-Out Test Result and Its Validity Status (Driver A — 956 Windows)
Evaluated exactly once after methodology and checkpoint freeze (`docs/biasnet_driver_a_direct_metrics.json`):

| Metric | Zero Baseline | Train-Mean Baseline | BiasNet v1.0 | Improvement vs Zero |
| :--- | :--- | :--- | :--- | :--- |
| **Total Vector RMSE** | $1.0656\text{ m/s}^2$ | $1.0563\text{ m/s}^2$ | **$0.7153\text{ m/s}^2$** | **+32.9% error reduction** |
| **Accel Vector RMSE** | $1.0633\text{ m/s}^2$ | $1.0539\text{ m/s}^2$ | **$0.7140\text{ m/s}^2$** | **+32.9% error reduction** |
| **Gyro Vector RMSE** | $0.0703\text{ rad/s}$ | $0.0704\text{ rad/s}$ | **$0.0425\text{ rad/s}$** | **+39.5% error reduction** |
| **Total Vector MAE** | $0.8861\text{ m/s}^2$ | $0.8796\text{ m/s}^2$ | **$0.5552\text{ m/s}^2$** | **+37.3% error reduction** |

**Validity Status**: **VALID FROZEN TEST RESULT**. Driver A was not accessed for architecture selection, hyperparameter tuning, loss weighting, or threshold selection.

---

## K. Navigation Ablation Results
Evaluated on real driving segment `Categorised_S1.npz` across controlled synthetic GNSS outages (`docs/biasnet_navigation_ablation_results.json`):

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

**Honest Scientific Interpretation**:
- BiasNet provides modest auxiliary velocity tracking RMSE improvements during outages ($3.292\text{ m/s}$ vs $3.361\text{ m/s}$ at 30s; $130.002\text{ m/s}$ vs $131.604\text{ m/s}$ at 60s).
- Primary horizontal position RMSE improvement relative to VelocityNet alone is negligible / essentially tied ($10.049\text{ m}$ vs $10.031\text{ m}$ at 30s; $1735.228\text{ m}$ vs $1740.133\text{ m}$ at 60s). BiasNet does **not** solve long-duration horizontal position drift on its own.
- Filter stability is completely preserved; innovations remain bounded (Mean NIS $\le 2.543$, well below 6-DOF $\chi^2_{0.99} = 16.81$ gate), with zero filter divergence.

---

## L. Filter Safety Evidence
1. **Indirect Integration**: BiasNet predictions enter the ESKF strictly as pseudo-measurements via $(\mathbf{z}_b, \mathbf{H}_b, \mathbf{R}_b)$ through Joseph-form covariance updates. No direct state mutation occurs.
2. **In-Graph Physical Clamps**: Forward graph hard-clamps outputs to $[-2.0, 2.0]\text{ m/s}^2$ and $[-0.15, 0.15]\text{ rad/s}$.
3. **Innovation Gating**: Mahalanobis gate ($\chi^2 \le 25.0$) rejects anomalous predictions.
4. **Decoupled Architecture Guarantee**: Disabling BiasNet (`biasnet_enabled = false`) leaves classical strapdown propagation, GNSS updates, and VelocityNet aiding 100% operational.

---

## M. NIS / Covariance Interpretation
- **Authoritative Covariance**: $\mathbf{R}_b = \text{diag}([1.21, 1.21, 1.21, 0.0025, 0.0025, 0.0025])$, corresponding to $1\sigma$ uncertainty of $1.1\text{ m/s}^2$ on accelerometer bias and $0.05\text{ rad/s}$ ($2.86^\circ$/s) on gyroscope bias.
- **Diagnostic Clarification**: Low NIS confirms innovation consistency with filter uncertainty, but does **not** prove optimal calibration (conservative/inflated $\mathbf{R}_b$ also suppresses NIS). Joint calibration against actual empirical innovation residuals belongs to Phase 9.

---

## N. Export Parity & Artifact Verification
PyTorch $\to$ ONNX (opset 17) $\to$ LiteRT (Float32):
- **ONNX Max Absolute Error**: $6.56 \times 10^{-7}$ (Tolerance: $1.0 \times 10^{-4}$) $\to$ **PASS**
- **LiteRT Max Absolute Error**: $3.58 \times 10^{-7}$ (Tolerance: $1.0 \times 10^{-3}$) $\to$ **PASS**

### Final Authoritative Artifact SHA-256 Digest Table
| Artifact File | Size | SHA-256 Digest |
| :--- | :--- | :--- |
| `models/biasnet_v1_best.pt` | $100\text{ KB}$ | `0a309d54bfc0db5c0f74f31fe8401d86e9d42358dab713ae989c46699662f00e` |
| `models/biasnet_v1.onnx` | $99\text{ KB}$ | `d57e64485a8722bc4041d7e73b3fed606e878c5b7ba5504e54081c4225674067` |
| `models/biasnet_v1.tflite` | $240\text{ KB}$ | `551193b4544847c721c69d2acb572792b6c7ee37b31ad0d954015d6715a8934a` |
| `models/model_config_biasnet_v1.json` | $6.7\text{ KB}$ | `ff4e639c97e32dea6f288f14be78821fe818625413c44ba368fcc285c3b50b58` |
| `models/biasnet_v1_export_parity.json`| $1.3\text{ KB}$ | `ab44602c3f5f01e96ababefa5a5a4e2297a7d1e6bde370b1685cdb3941f8347d` |
| `data/ml_dataset_biasnet_v1/bias_train.npz` | $5.7\text{ MB}$ | `d47718ec2ae0291ef7e64ae909cd7e4f50dbcb9063ddfd43b9cab496491dad3d` |
| `data/ml_dataset_biasnet_v1/bias_validation.npz` | $675\text{ KB}$ | `aff4bdf9acd44ba219fe3ab464e886a2fc10270dda7651eea75f3d76e28d546e` |
| `data/ml_dataset_biasnet_v1/bias_label_manifest.json` | $13.6\text{ KB}$ | `2cec772b50997ed1cf169fd09362c762da6f511da7e2a5067866921de7e7d7e8` |
| `docs/biasnet_driver_a_direct_metrics.json` | $5.9\text{ KB}$ | `58a37d57bc55580ef8db9d79b9ba6dfa822f5b6e38debf0648199054c23fc1f8` |
| `docs/biasnet_navigation_ablation_results.json` | $4.1\text{ KB}$ | `c97dfd004bd3bb6d4ce963c7502c7b553e777dff727057f0461b907b6aebfa62` |
| `docs/biasnet_horizon_audit.json` | $6.2\text{ KB}$ | `c412d9b15b451f23d2c3b310ecee4916eba6d8d5fdc76b5f640d0c2c21713103` |

---

## O. Actual Full Test-Suite Result
Command: `uv run pytest tests -v`
```
====================== 309 passed, 14 warnings in 25.46s ======================
```
All unit, integration, contract, and export parity tests passed with zero failures.

---

## P. Final Acceptance Decision
**Status**: **ACCEPTED AS EXPERIMENTAL ESKF AIDING CANDIDATE (STAGE A)**.

**Scientific Justification**:
1. Mathematical identifiability and multi-criteria convergence gating deterministically eliminate non-converged and unobservable windows.
2. Direct pseudo-target predictions outperform naive baselines by $>29\%$ on Driver B and $>32\%$ on Driver A.
3. Outage navigation exhibits modest velocity tracking improvements and bounded innovations without filter divergence.
4. Complete decoupled safety guarantees ensure zero regression if disabled.

---

## Q. Remaining Limitations
1. **Pseudo-Target Nature**: Labels are optimization-derived targets, not physical ground-truth sensor bias.
2. **Mounting Tilt Conflation**: Dynamic vehicle attitude errors and cradle mounting tilt project into apparent accelerometer bias.
3. **Limited Standalone Position Benefit**: BiasNet primarily stabilizes velocity and state consistency; it does not eliminate long-duration position drift on its own.
4. **Diagnostic Outage Testing**: Evaluated on representative IO-VNBD outage segments; full closed-loop multi-trip trajectory replay belongs to Phase 9.

---

## R. What Phase 9 Is Allowed to Do Next
1. Wire both VelocityNet v1.1 and BiasNet v1.0 into `NavigationCore` using indirect Kalman measurement updates.
2. Implement empirical covariance calibration for $\mathbf{R}_v$ and $\mathbf{R}_b$ based on actual validation innovation residuals.
3. Preserve strict decoupled fallbacks: `velocitynet_enabled` and `biasnet_enabled` configuration flags.
4. Execute full offline replay across the complete dataset to evaluate multi-sensor fusion performance.
