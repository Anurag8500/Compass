# COMPASS Phase 4 — Implementation Build Walkthrough

**Project**: C.O.M.P.A.S.S. (Cognitive Off-grid Machine-learning Positioning And Sensor System)  
**Problem Statement**: SIH 2026 Problem Statement 26168 — ISRO  
**Document Type**: Step-by-Step Implementation Build Diary

---

## Overview: What We Were Trying to Do

Phase 4 introduced pure **inertial navigation physics**: integrating angular rates to track attitude, then rotating specific force through that attitude to compute acceleration, velocity, and position in the local ENU frame.

Two objectives:
1. Build the correct, production-quality strapdown INS mechanization used by all subsequent phases.
2. Run an empirical experiment (Ablation Stage 1) measuring how fast a consumer smartphone IMU drifts when running completely unassisted.

This drift measurement established the empirical baseline against which Phase 5 (ESKF), Phase 7 (VelocityNet), and all subsequent fusion layers are judged.

---

## Step 1: Building Quaternion Attitude Propagation (`navigation/ins/attitude.py`)

### The Quaternion Update Equation
For discrete timestep $\Delta t$ with vehicle angular velocity $\boldsymbol{\omega}_m^v$:
$$\mathbf{q}[k+1] = \text{normalize}\left(\mathbf{q}[k] \otimes \Delta\mathbf{q}(\boldsymbol{\omega}_m^v \Delta t)\right)$$

where the incremental rotation quaternion from rotation vector $\boldsymbol{\theta} = \boldsymbol{\omega}_m^v \Delta t$ is:
$$\Delta\mathbf{q}(\boldsymbol{\theta}) = \begin{bmatrix} \cos\tfrac{\|\boldsymbol{\theta}\|}{2} \\ \sin\tfrac{\|\boldsymbol{\theta}\|}{2} \cdot \hat{\boldsymbol{\theta}} \end{bmatrix}$$

For small angles ($\|\boldsymbol{\theta}\| < 10^{-8}$), a Taylor expansion avoids numerical singularity:
```python
theta_norm = np.linalg.norm(theta_vec)
if theta_norm < 1e-8:
    dq = np.array([1.0 - theta_norm**2 / 8.0,
                   *(0.5 - theta_norm**2 / 48.0) * theta_vec])
else:
    half = theta_norm / 2.0
    dq = np.array([np.cos(half), *(np.sin(half) / theta_norm * theta_vec)])
```

### Hamilton Product & Renormalization
Hamilton scalar-first convention ($[w, x, y, z]^T$) is used, and explicit normalization (`q = q / np.linalg.norm(q)`) is executed every step to avoid scaling degradation.

---

## Step 2: Building Position-Velocity Propagation (`navigation/ins/propagation.py`)

### Propagation Equations
Given attitude quaternion $\mathbf{q}[k]$, specific force $\mathbf{f}_m^v[k]$, accelerometer bias prior $\mathbf{b}_a^v$, and gravity $\mathbf{g}^n = [0, 0, -9.80665]^T\,\text{m/s}^2$:

```python
def propagate_kinematics(p, v, q, f_m_v, b_a_v, g_n, dt):
    # 1. Rotation matrix from quaternion (R_v^n)
    R = quaternion_to_rotation_matrix(q)
    
    # 2. Bias-compensated specific force
    f_unbiased = f_m_v - b_a_v
    
    # 3. True coordinate acceleration in ENU navigation frame
    a_true = R @ f_unbiased + g_n
    
    # 4. Constant-acceleration discrete kinematic propagation
    p_new = p + v * dt + 0.5 * a_true * dt**2
    v_new = v + a_true * dt
    
    return p_new, v_new, a_true
```

Downward gravity $[0, 0, -9.80665]^T\,\text{m/s}^2$ is added in the navigation frame. When stationary and level:
$$\mathbf{a}_{\text{true}}^n = \mathbf{I} [0, 0, +9.80665]^T + [0, 0, -9.80665]^T = [0, 0, 0]^T\,\text{m/s}^2$$

---

## Step 3: Building the Ablation Stage 1 Runner Script (`scripts/run_ablation_stage1.py`)

### Evaluation Window
We selected Trip S1 samples `[19500, 20100]` (601 samples = 60.00 seconds) from `Categorised_S1.npz`:
- Steady cruising at $\sim 12.7\,\text{m/s}$ ($45.7\,\text{km/h}$).
- Ground truth distance traversed: $763.2\,\text{meters}$.

### Controlled Oracle Initialization Disclosure
Initial velocity ($\mathbf{v}_0^n = [-11.399, +3.099, 0.0]^T\,\text{m/s}$) and heading ($285.21^\circ$) were initialized directly from Racelogic VBOX ground truth at $t = 0$. This controlled setup isolates sensor integration divergence over time from startup alignment errors. Following $t = 0$, **zero GNSS or external aiding was provided**.

### VBOX Altitude Unit Scaling
Raw VBOX CSV column header said `' Height (km)'`, but raw numerical values were in physical meters ($127.39\text{ m}$ to $131.68\text{ m}$). Phase 2 multiplied by 1000 based on the label, storing values in mm-range in cache. In `run_ablation_stage1.py`, cached `v_ref_alt_m` is scaled by $1/1000.0$ to recover true meters without mutating the frozen Phase 2 cache.

---

## Step 4: Running the Ablation Experiment

```bash
python scripts/run_ablation_stage1.py
```

Checkpoint horizontal error evolution over 60 seconds:
- 5 s: $4.83\,\text{m}$ (7.8% of distance traveled)
- 10 s: $20.52\,\text{m}$ (16.9%)
- 15 s: $100.64\,\text{m}$ (54.6%)
- 20 s: $251.21\,\text{m}$ (100.3%)
- 30 s: $788.97\,\text{m}$ (209.8%)
- 45 s: $1,927.34\,\text{m}$ (341.0%)
- 60 s: **$3,249.32\,\text{m}$** (425.7% of ground-truth distance)

Final 3D error: **$3,292.29\,\text{m}$**; Final vertical error: **$-530.20\,\text{m}$**; Horizontal RMSE: **$1,487.53\,\text{m}$**.

### Error Interpretation
Theoretical models demonstrate that attitude tilt errors misproject gravity into horizontal axes, causing a $t^3$-type error component ($\frac{1}{6} g b_g t^3$). The empirical trajectory demonstrated severe drift consistent with the accumulation of inertial errors across the 60-second window.

---

## Step 5: Tests (`tests/unit/test_ins_propagation.py`)

Phase 4 propagation is covered by **16 unit tests** in `tests/unit/test_ins_propagation.py`:
- **Stationary test**: Level stationary specific force yields zero coordinate acceleration and zero drift.
- **Constant acceleration**: Propagating constant specific force matches $p = \frac{1}{2}at^2$.
- **Circular motion**: Constant turn rate and centripetal force propagate a circular arc.
- **Normalization stability**: Quaternion norm remains within $10^{-14}$ of 1.0.
- **Input validation**: Non-finite values and invalid timesteps raise descriptive `ValueError`.

---

## Final Outputs

| Artifact | Path | Purpose |
|:---|:---|:---|
| Attitude propagation | `navigation/ins/attitude.py` | Quaternion integration & DCM mapping |
| Kinematic propagation | `navigation/ins/propagation.py` | Constant-acceleration kinematic propagation |
| Ablation runner | `scripts/run_ablation_stage1.py` | 60-second open-loop benchmark |
| Report | `docs/ablation_stage1_report.md` | Full empirical benchmark report |
| Unit tests | `tests/unit/test_ins_propagation.py` | 16 propagation unit tests |

**Test count contributed by Phase 4**: Exactly **16 tests** passing.
