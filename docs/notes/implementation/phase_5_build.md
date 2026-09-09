# COMPASS Phase 5 — Implementation Build Walkthrough

**Project**: C.O.M.P.A.S.S. (Cognitive Off-grid Machine-learning Positioning And Sensor System)  
**Problem Statement**: SIH 2026 Problem Statement 26168 — ISRO  
**Document Type**: Step-by-Step Implementation Build Diary

---

## Overview: What We Were Trying to Do

Phase 5 was the heart of the classical navigation system: building the **Error-State Kalman Filter (ESKF)** that merges the high-rate IMU physics from Phase 4 with low-rate GNSS measurements and ZUPT (Zero Velocity Update) aiding.

This phase moved COMPASS from an open-loop dead-reckoning system to a closed-loop estimator, estimating sensor biases and attitude corrections in real time.

---

## Step 1: Designing the ESKF State (`navigation/eskf/state.py`)

### Two-Level State Design
The ESKF maintains two representations:
1. **16-element nominal state** (non-linear kinematics): position (3), velocity (3), quaternion (4), accel bias (3), gyro bias (3).
2. **15-element error state** (linear perturbations): position error (3), velocity error (3), rotation vector error (3), accel bias error (3), gyro bias error (3).

```python
@dataclass(frozen=True)
class ESKFNominalState:
    position_enu: np.ndarray   # (3,) [East, North, Up] meters
    velocity_enu: np.ndarray   # (3,) [vE, vN, vU] m/s
    q: np.ndarray              # (4,) [w, x, y, z] Hamilton unit quaternion
    accel_bias: np.ndarray     # (3,) m/s^2, vehicle frame
    gyro_bias: np.ndarray      # (3,) rad/s, vehicle frame
    timestamp_ns: int

@dataclass(frozen=True)
class ESKFState:
    nominal: ESKFNominalState
    covariance: np.ndarray     # (15, 15) P matrix
```

### Right-Multiplicative Attitude Error Convention
Attitude error is parameterized in the body (vehicle) frame:
$$\mathbf{q}_{\text{true}} = \mathbf{q}_{\text{nom}} \otimes \delta\mathbf{q}(\delta\boldsymbol{\theta})$$
This aligns with how gyroscopes physically measure body-fixed angular velocities.

---

## Step 2: Building the Propagation Step (`navigation/eskf/predict.py`)

At each timestep $\Delta t$:
1. **Propagate nominal kinematics** via Phase 4 strapdown equations.
2. **Compute discrete error transition matrix** $F_d \in \mathbb{R}^{15 \times 15}$.
3. **Propagate covariance**:
   $$P_{k+1} = F_d P_k F_d^T + Q_d, \quad P_{k+1} \leftarrow \frac{1}{2}(P_{k+1} + P_{k+1}^T)$$

The continuous system matrix $F_c$:
$$F_c = \begin{bmatrix}
\mathbf{0}_3 & \mathbf{I}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 \\
\mathbf{0}_3 & \mathbf{0}_3 & -R_v^n[\mathbf{f}_{\text{unbiased}}^v]_\times & -R_v^n & \mathbf{0}_3 \\
\mathbf{0}_3 & \mathbf{0}_3 & -[\boldsymbol{\omega}_{\text{unbiased}}^v]_\times & \mathbf{0}_3 & -\mathbf{I}_3 \\
\mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 \\
\mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3
\end{bmatrix}$$

---

## Step 3: Generic Measurement Update (`navigation/eskf/update.py`)

A single `eskf_update()` handles all measurement sources $(z, h, H, R)$:
```python
def eskf_update(state, z, h_val, H, R, gating=None, timestamp_ns=None):
    y = z - h_val                        # Innovation
    S = H @ state.covariance @ H.T + R  # Innovation covariance
    
    # Innovation gating
    if gating is not None and not gating.evaluate(y, S).accepted:
        return state, rejection_diagnostics
    
    # Kalman gain via linear solve (no matrix inversion)
    X = np.linalg.solve(S, H @ state.covariance)
    K = X.T  # (15, m)
    
    delta_x = K @ y
    
    # Joseph-form covariance update
    I_KH = np.eye(15) - K @ H
    P_new = I_KH @ state.covariance @ I_KH.T + K @ R @ K.T
    P_new = 0.5 * (P_new + P_new.T)
    
    # Inject error & reset covariance
    new_state = state.inject_error(delta_x, new_covariance=P_new, timestamp_ns=timestamp_ns)
    return new_state, update_diagnostics
```

---

## Step 4: The Error-State Covariance Reset (`navigation/eskf/state.py`)

When error $\hat{\delta\mathbf{x}}$ is injected into the nominal state:
$$\mathbf{q}_{\text{nom}} \leftarrow \text{normalize}(\mathbf{q}_{\text{nom}} \otimes \delta\mathbf{q}(\hat{\delta\boldsymbol{\theta}}))$$
The reference frame of the attitude error manifold shifts. To maintain mathematical consistency, the covariance matrix must be transformed using the reset Jacobian:
$$G_\theta = \mathbf{I}_3 - \frac{1}{2}[\hat{\delta\boldsymbol{\theta}}]_\times$$
$$J_{\text{reset}} = \text{diag}(\mathbf{I}_3, \mathbf{I}_3, G_\theta, \mathbf{I}_3, \mathbf{I}_3) \in \mathbb{R}^{15 \times 15}$$
$$P^+ = J_{\text{reset}} P_{\text{updated}} J_{\text{reset}}^T$$

This is enforced atomically inside `ESKFState.inject_error()`.

---

## Step 5: Measurement Models (`navigation/eskf/measurements/`)

1. **GNSS 3D Position (`gnss.py`)**: $\mathbf{z}_p = \mathbf{p}_{\text{gnss}}^n$, $H_p = [\mathbf{I}_3 \mid \mathbf{0}_{3 \times 12}]$. Includes a realistic vertical uncertainty floor ($\sigma_{\text{vert}} = 35.0\,\text{m}$) to account for noisy smartphone GPS altitude.
2. **GNSS 2D Horizontal Velocity (`gnss.py`)**: $\mathbf{z}_{v,2D} = [v_E, v_N]^T$ from speed and course. Vertical velocity $v_U$ is left unobserved.
3. **Classical ZUPT (`zupt.py`)**: Standstill detector monitors low gyro variance, gravity norm match, and low accel variance over an 8-sample window. Standstill measurement $\mathbf{z}_{\text{zupt}} = [0, 0, 0]^T\,\text{m/s}$, $H_{\text{zupt}} = [\mathbf{0}_3 \mid \mathbf{I}_3 \mid \mathbf{0}_9]$.

---

## Step 6: The Critical Jacobian Sign Audit & Bug Fix History

During the development of Phase 5, a rigorous mathematical audit cycle resolved key implementation issues:

### The $F_d$ Attitude-Velocity Coupling Sign Bug
1. **What the original implementation did**:
   In continuous error dynamics, true acceleration is:
   $$\dot{\mathbf{v}}_{\text{true}}^n = R_v^n (\mathbf{I} + [\delta\boldsymbol{\theta}]_\times) (\mathbf{f}_{\text{unbiased}}^v - \delta\mathbf{b}_a^v) + \mathbf{g}^n$$
   Expanding the attitude perturbation term gives $R_v^n [\delta\boldsymbol{\theta}]_\times \mathbf{f}_{\text{unbiased}}^v$.
   The original draft code converted this to matrix multiplication using $+ R_v^n [\mathbf{f}_{\text{unbiased}}^v]_\times \delta\boldsymbol{\theta}$, inadvertently omitting the anti-symmetry minus sign of the cross product.
2. **What test exposed the issue**:
   The finite-difference Jacobian test in `tests/unit/test_eskf_synthetic.py` (`test_analytical_vs_numerical_finite_difference_jacobian`). This test numerically perturbed each error state dimension by $\pm \epsilon = \pm 10^{-6}$ and computed $\frac{\Delta \mathbf{x}_{k+1}}{\Delta \mathbf{x}_k}$. The numerical derivatives for the attitude-velocity block had the opposite sign from the analytical $F_d$ implementation.
3. **What the correct mathematical relationship is**:
   The vector cross-product is anti-symmetric: $[\mathbf{a}]_\times \mathbf{b} = - [\mathbf{b}]_\times \mathbf{a}$.
   Therefore:
   $$R_v^n [\delta\boldsymbol{\theta}]_\times \mathbf{f}_{\text{unbiased}}^v = - R_v^n [\mathbf{f}_{\text{unbiased}}^v]_\times \delta\boldsymbol{\theta}$$
   The correct coupling term must have a negative sign: $- R_v^n [\mathbf{f}_{\text{unbiased}}^v]_\times$.
4. **What was changed in the code**:
   In `navigation/eskf/predict.py`, the velocity-attitude block was corrected to:
   ```python
   F_d[3:6, 6:9] = - dt * (R_v_n @ skew(f_unbiased))
   ```
   and the second-order position-attitude block was corrected to:
   ```python
   F_d[0:3, 6:9] = -0.5 * (dt * dt) * (R_v_n @ skew(f_unbiased))
   ```
5. **What final implementation is authoritative**:
   Lines 97 and 100 of `navigation/eskf/predict.py`.
6. **What test passed**:
   `test_analytical_vs_numerical_finite_difference_jacobian` passed with a maximum residual difference of $3.80 \times 10^{-6} < 10^{-4}$ across all 225 elements.

### Additional Audit Fixes
- **Covariance Reset Invariant**: Ensured $J_{\text{reset}}$ is applied automatically during error injection in `state.py`.
- **Joseph-Form Stability**: Enforced $P = (I - KH)P(I - KH)^T + KRK^T$ in `update.py` to prevent negative eigenvalues.
- **Predict Symmetrization**: Enforced $P \leftarrow 0.5(P + P^T)$ at every prediction step.

---

## Step 7: Real-Data Validation Results (Trip S1)

Executed via [`tests/integration/test_eskf_gnss_real_data.py`](file:///d:/Hackathon/Compass/tests/integration/test_eskf_gnss_real_data.py) over the 60.0-second highway segment:
- **Final horizontal error**: **$5.27\,\text{m}$** (vs. Phase 4 open-loop: $3,249.32\,\text{m}$) $\implies$ **$99.84\%$ reduction**.
- **Horizontal RMSE**: **$24.78\,\text{m}$** (vs. Phase 4: $1,487.53\,\text{m}$) $\implies$ **$98.33\%$ reduction**.
- **Maximum horizontal error**: **$117.49\,\text{m}$**.
- **Final vertical error**: **$63.01\,\text{m}$**; **Vertical RMSE**: **$70.54\,\text{m}$**.
- **Final 3D error**: **$63.23\,\text{m}$**; **3D RMSE**: **$74.77\,\text{m}$**.
- **GNSS Fixes**: 7 of 7 position fixes and 7 of 7 velocity fixes accepted through the Chi-Square gate with 0 false rejections.
- **Standstill ZUPT**: Drift suppressed to $< 0.5\,\text{m}$ over 100 samples.

---

## Final Outputs

| Artifact | Path | Purpose |
|:---|:---|:---|
| State representation | `navigation/eskf/state.py` | 16-nominal, 15-error state, reset Jacobian |
| Prediction | `navigation/eskf/predict.py` | Nominal propagation & covariance prediction |
| Update | `navigation/eskf/update.py` | Generic gated Joseph-form update |
| GNSS models | `navigation/eskf/measurements/gnss.py` | 3D position & 2D velocity models |
| ZUPT model | `navigation/eskf/measurements/zupt.py` | Standstill detector & measurement model |
| Gating | `navigation/eskf/gating.py` | Mahalanobis Chi-Square gate |
| Report | `docs/eskf_gnss_baseline_report.md` | Full validation report |
| Unit tests | `tests/unit/test_eskf_synthetic.py` | 18 synthetic tests |
| ZUPT tests | `tests/unit/test_zupt.py` | 9 ZUPT unit tests |
| Integration tests | `tests/integration/test_eskf_gnss_real_data.py` | 2 real-data tests |

**Test count contributed by Phase 5**: Exactly **29 tests** (total repository tests at Phase 5 completion: **202 passed**).
