# BiasNet Label Stability & Identifiability Report (Phase 8)

## 1. Dataset & Split Provenance
- **Dataset**: IO-VNBD (Inertial and Odometry Benchmark Dataset for Ground Vehicles).
- **Split Structure (Strict Phase 6 Invariant)**:
  - **Train**: Driver E (represented by 25 audited multi-kilometer driving trips).
  - **Validation**: Driver B (2 trips: `Categorised_M.npz`, `Uncategorised_M.npz`).
  - **Held-Out Test**: Driver A (strictly withheld from all methodology choices).
- **Teacher Paradigm**: Short-horizon inverse-problem optimization against synchronized Racelogic VBOX RTK GNSS ground truth.
- **Horizon Configuration**: $H = 1.0$ s ($K = 10$ integration intervals at 10 Hz canonical sampling).

## 2. Identifiability Methodology & Gating Rule
A 6-parameter bias correction $\Delta \mathbf{b} = [\Delta \mathbf{b}_a^T, \Delta \mathbf{b}_g^T]^T$ is solved per window via damped Levenberg-Marquardt against multi-point ENU position, velocity, and orientation residuals.

### Formal Eligibility Gating Policy
A window is accepted as a supervised learning target **if and only if** all of the following pass:
1. **Solver Convergence**: Gauss-Newton / LM solver converges within 2.0 iterations.
2. **Numerical Conditioning**: Jacobian condition number $\kappa = \sigma_{\max} / \sigma_{\min} \le 50.0$.
3. **Residual Reduction**: $\rho = \|\mathbf{r}_{\text{before}}\| / \|\mathbf{r}_{\text{after}}\| \ge 1.20$ (at least 20% residual reduction).
4. **Physical Plausibility**:
   - Accelerometer bias correction: $|\Delta b_a| \le 2.00\text{ m/s}^2$
   - Gyroscope bias correction: $|\Delta b_g| \le 0.150\text{ rad/s}$ (8.6$^\circ$/s)

## 3. Candidate Windows & Rejection Statistics

| Metric | Driver E (Train) | Driver B (Validation) |
| :--- | :--- | :--- |
| **Total Candidate Windows** | 7192 | 800 |
| **Eligible Windows Passed** | 3914 (54.4%) | 504 (63.0%) |
| **Rejected Windows** | 3278 (45.6%) | 296 (37.0%) |

### Rejection Reason Breakdown
**Driver E (Train)**:
{
  "BOUNDS_ACTIVE": 3165,
  "VALID": 3914,
  "POOR_RESIDUAL_REDUCTION": 113
}

**Driver B (Validation)**:
{
  "BOUNDS_ACTIVE": 294,
  "POOR_RESIDUAL_REDUCTION": 2,
  "VALID": 504
}

## 4. Robust Distribution Statistics (Eligible Windows)

### Driver E (Train) Bias Target Distributions
| Component | Mean | Median | Std | MAD | 5th Pct | 95th Pct |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| $\Delta b_{a, x}$ [m/s$^2$] | -0.0316 | 0.0006 | 0.8122 | 0.4834 | -1.4772 | 1.4128 |
| $\Delta b_{a, y}$ [m/s$^2$] | -0.0319 | -0.0120 | 0.7755 | 0.4364 | -1.4149 | 1.3642 |
| $\Delta b_{a, z}$ [m/s$^2$] | 0.0143 | 0.0359 | 0.3333 | 0.1382 | -0.5453 | 0.5658 |
| $\Delta b_{g, x}$ [rad/s] | 0.00211 | 0.00129 | 0.04465 | 0.02335 | -0.07396 | 0.08070 |
| $\Delta b_{g, y}$ [rad/s] | 0.00210 | 0.00033 | 0.06740 | 0.04330 | -0.11627 | 0.12106 |
| $\Delta b_{g, z}$ [rad/s] | 0.00224 | 0.00070 | 0.04789 | 0.02322 | -0.07980 | 0.08934 |

## 5. Temporal Smoothness & Adjacent Autocorrelation
Between consecutive windows (stride = 0.5 s, 15 samples overlap):
- $\Delta b_{a, x}$ Autocorrelation: 0.7231
- $\Delta b_{a, y}$ Autocorrelation: 0.7461
- $\Delta b_{a, z}$ Autocorrelation: 0.5943
- $\Delta b_{g, x}$ Autocorrelation: 0.2944
- $\Delta b_{g, y}$ Autocorrelation: 0.4446
- $\Delta b_{g, z}$ Autocorrelation: 0.2994

Moderate to high autocorrelation ($r \approx 0.60 - 0.75$ for accelerometer components) confirms that the inverse-problem targets exhibit spatial continuity along continuous driving trajectories, while gyroscope corrections exhibit lower autocorrelation reflecting local angular dynamics.

## 6. Physical Plausibility & Domain-Shift Analysis
1. **Driver E (Train)**:
   The majority (54.4%) of windows yield physically consistent bias targets within consumer smartphone IMU tolerances ($|\Delta b_a| \le 2.0\text{ m/s}^2$, $|\Delta b_g| \le 0.15\text{ rad/s}$).
2. **Driver B (Validation)**:
   Driver B exhibits a severe domain shift in unconstrained optimization solutions ($|\Delta b_g| > 0.5\text{ rad/s}$). Investigation demonstrates that Driver B has an unresolved physical mounting angle difference in the smartphone holder that causes the body rotation to mix heavily into body axes. Under the strict physical bound rule, these windows are appropriately rejected (294 windows rejected as `BOUNDS_ACTIVE`).
3. **Identifiability Conclusion**:
   The inverse problem is mathematically well-conditioned (median $\kappa \approx 10.4$, rank 6), but the physical meaning of the correction is strictly conditional on the mounting orientation of the specific trip.

## 7. Gate Decision: CONDITIONAL (PROCEED TO STAGE A EVALUATION)
- **Status**: **CONDITIONAL**
- **Justification**:
  - The inverse problem is well-posed, full rank, and reduces residuals by $3.5\times$ to $5\times$.
  - Within Driver E (Train), a rich population of 3914 valid, identifiable windows is established.
  - The strict physical gating successfully rejects non-identifiable and mounting-distorted windows.
- **Protocol**:
  - Proceed to train BiasNet Stage A (mean model with internal hard clamps).
  - Compare strictly against Zero Correction and Train Mean baselines on Driver B.
  - If BiasNet fails to outperform the baselines or destabilizes the ESKF during synthetic outage testing, trigger the decoupled fallback outcome (`biasnet_enabled = false`) as required.
