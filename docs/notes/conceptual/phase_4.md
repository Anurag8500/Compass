# Phase 4 Complete Explanation: Strapdown Inertial Navigation System (INS) Mechanization & The Open-Loop Baseline

**Project**: C.O.M.P.A.S.S. (Cognitive Off-grid Machine-learning Positioning And Sensor System)  
**Problem Statement**: SIH 2026 Problem Statement 26168 — ISRO  
**Stage**: Phase 4 Complete Teaching & Reference Walkthrough  

---

## 1. Introduction & Phase Overview

### What Problem Did Phase 4 Solve?
In Phase 3, we successfully calibrated our raw sensor readings: we estimated and subtracted the initial stationary gyroscope bias, rotated the IMU's accelerations from the phone's physical body frame into the vehicle's Forward-Lateral-Up (FLU) frame, and filtered high-frequency structural engine vibrations.

However, an aligned stream of acceleration ($\text{m/s}^2$) and turn rates ($\text{rad/s}$) is **not** a position or trajectory. A self-driving car or navigation app cannot use "you are experiencing $1.2\,\text{m/s}^2$ forward acceleration" to know where it is on a map.

**Phase 4 solves the fundamental problem of Inertial Navigation**: How do we take discrete, high-frequency acceleration and angular velocity measurements in a moving vehicle and mathematically integrate them forward through time to calculate where the vehicle is, how fast it is moving, and which direction it is pointing?

### Why Did We Need to Solve It?
Inertial navigation is the ultimate autonomous, self-contained navigation method. Unlike GNSS (which relies on microwave signals traveling 20,000 kilometers from space that can be jammed, spoofed, blocked by skyscrapers, or lost inside tunnels), an IMU relies on pure Newtonian mechanics inside the vehicle.

Before we can build an advanced sensor fusion system (like the Phase 5 Kalman Filter) or train deep neural networks (like Phase 7 VelocityNet), we must build the **classical physics engine** of navigation: the **Strapdown Inertial Navigation System (INS)**.

Furthermore, we needed to establish an empirical **Ablation Stage 1 baseline**: we measured exactly how fast a consumer smartphone IMU drifts when left completely unassisted. This provides the quantitative benchmark against which subsequent fusion mechanisms in COMPASS are evaluated.

---

## 2. Core Concepts & Terminology

### 1. Strapdown INS vs. Gimballed INS
- **Simple Definition**: An inertial navigation system where the sensors are rigidly fixed ("strapped down") directly to the vehicle chassis, rather than mounted on motorized, gyro-stabilized gimbals that keep the sensors physically pointed north and level.
- **In COMPASS**: A smartphone sitting in a dashboard mount is strapped down to the car. As the car pitches, rolls, and yaws, the phone moves with it. Coordinate transformations are handled mathematically using quaternions.

### 2. Double Integration
- **Simple Definition**: Integrating acceleration once with respect to time gives velocity; integrating velocity once with respect to time gives position:
  $$\mathbf{v}(t) = \mathbf{v}_0 + \int \mathbf{a}(t)\,dt, \quad \mathbf{p}(t) = \mathbf{p}_0 + \int \mathbf{v}(t)\,dt$$
- **Why We Care**: Double integration allows us to calculate 3D position purely from motion. But any small error or bias in acceleration is integrated twice, causing position errors to accumulate quadratically ($t^2$) or cubically ($t^3$) over time.

### 3. Specific Force vs. Kinematic Acceleration
- **Simple Definition**: Specific force is what an accelerometer physically measures: the mechanical contact force per unit mass supporting the proof mass against gravity. Kinematic acceleration is the actual rate of change of coordinate velocity with respect to an inertial reference frame.
- **In COMPASS**: When a car is parked at a red light, its velocity is zero and its coordinate acceleration is zero. But an accelerometer on the dashboard measures $+9.80665\,\text{m/s}^2$ pointing straight UP because the ground pushes up on the vehicle to prevent it from falling through the Earth.
- **Why We Care**: We must explicitly add downward gravitational acceleration in the local navigation frame to recover true kinematic coordinate acceleration:
  $$\mathbf{a}_{\text{true}}^n = R_v^n (\mathbf{f}_m^v - \mathbf{b}_a^v) + \mathbf{g}^n, \quad \mathbf{g}^n = \begin{bmatrix} 0 \\ 0 \\ -9.80665 \end{bmatrix}\,\text{m/s}^2$$

### 4. Attitude Quaternion
- **Simple Definition**: A four-element hypercomplex number $\mathbf{q} = [w, x, y, z]^T$ that represents a 3D rotation without gimbal lock.
- **In COMPASS**: $\mathbf{q}$ represents the 3D orientation that rotates vectors from the vehicle's body frame ($v$) into the local East-North-Up navigation frame ($n$):
  $$\mathbf{v}^n = R_v^n(\mathbf{q})\,\mathbf{v}^v$$

### 5. Local East-North-Up (ENU) Tangent Plane
- **Simple Definition**: A local Cartesian $(X, Y, Z)$ coordinate system centered at a fixed point on the Earth's surface where $X$ points East, $Y$ points North, and $Z$ points Up into the sky.
- **In COMPASS**: Every trip establishes an immutable session reference point $(\text{lat}_0, \text{lon}_0, \text{alt}_0)$ at sample 0. All positions are computed in meters relative to this local origin.

---

## 3. Coordinate Frames & Transformations

```
      +Z_n (Up)
         ^
         |      +Y_n (North)
         |     /
         |    /
         |   /
         |  /
         | /
         +----------------> +X_n (East)
       Local ENU Frame (n)
            ^
            |  Rotated by Attitude Quaternion q = R_v^n
            |
            |     +X_v (Forward)
            |    /
            |   /
            |  /
            | /
            +----------------> +Y_v (Lateral/Left)
            |
            v  +Z_v (Up/Roof)
       Vehicle FLU Frame (v)
```

### Frame Transformation Pipeline
1. IMU measures angular rate $\boldsymbol{\omega}_m^v$ and specific force $\mathbf{f}_m^v$ in vehicle coordinates.
2. $\boldsymbol{\omega}_m^v$ rotates the attitude quaternion $\mathbf{q}$ from step $k$ to $k+1$.
3. The updated quaternion produces Direction Cosine Matrix $R_v^n \in \mathbb{R}^{3 \times 3}$.
4. Vehicle specific force is rotated into navigation coordinates, compensated for bias and gravity:
   $$\mathbf{a}_{\text{true}}^n = R_v^n (\mathbf{f}_m^v - \mathbf{b}_a^v) + \mathbf{g}^n$$
5. $\mathbf{a}_{\text{true}}^n$ is integrated to update velocity $\mathbf{v}^n$ and position $\mathbf{p}^n$.

---

## 4. The Mathematics of Strapdown INS

### 4.1 Quaternion Representation & Hamilton Convention
COMPASS strictly follows the **Hamilton scalar-first convention**:
$$\mathbf{q} = [w, x, y, z]^T = [q_w, q_x, q_y, q_z]^T, \quad \|\mathbf{q}\| = \sqrt{w^2 + x^2 + y^2 + z^2} = 1.0$$

The product of two quaternions $\mathbf{p} \otimes \mathbf{q}$ corresponds to successive rotations. The 3x3 rotation matrix $R_v^n$ mapped from $\mathbf{q}$ is:
$$R_v^n = \begin{bmatrix}
1 - 2(y^2 + z^2) & 2(xy - wz) & 2(xz + wy) \\
2(xy + wz) & 1 - 2(x^2 + z^2) & 2(yz - wx) \\
2(xz - wy) & 2(yz + wx) & 1 - 2(x^2 + y^2)
\end{bmatrix}$$

### 4.2 Discrete Attitude Propagation
Over timestep $\Delta t = t_{k+1} - t_k$, the incremental rotation vector is $\boldsymbol{\theta} = \boldsymbol{\omega}_m^v \Delta t \in \mathbb{R}^3$, with magnitude $\theta = \|\boldsymbol{\theta}\|$.
The incremental rotation quaternion $\Delta \mathbf{q}$ is:
$$\Delta \mathbf{q} = \begin{bmatrix} \cos(\theta / 2) \\ \frac{\boldsymbol{\theta}}{\theta} \sin(\theta / 2) \end{bmatrix}$$

For $\theta < 10^{-8}\,\text{rad}$, a Taylor series expansion avoids numerical singularity:
$$\Delta \mathbf{q} \approx \begin{bmatrix} 1 - \frac{\theta^2}{8} \\ \left(\frac{1}{2} - \frac{\theta^2}{48}\right)\boldsymbol{\theta} \end{bmatrix}$$

Attitude propagates via right-multiplication:
$$\mathbf{q}[k+1] = \text{normalize}\left(\mathbf{q}[k] \otimes \Delta \mathbf{q}\right)$$

### 4.3 Coordinate Acceleration & Bias Distinction
In Phase 4:
- $\mathbf{f}_m^v$: Raw measured vehicle-frame specific force from Phase 3 preprocessing.
- $\mathbf{b}_a^v$: Accelerometer bias in the vehicle frame. In Phase 4 open-loop benchmark, this is configured as a nominal prior $\mathbf{b}_a^v = [0, 0, 0]^T$. Dynamic accelerometer bias estimation is not solved here; that belongs to the downstream ESKF architecture.
- $\mathbf{f}_{\text{unbiased}}^v = \mathbf{f}_m^v - \mathbf{b}_a^v$: Bias-compensated specific force.
- $\mathbf{g}^n = [0, 0, -9.80665]^T\,\text{m/s}^2$: Gravitational acceleration in local ENU.
- True kinematic coordinate acceleration:
  $$\mathbf{a}_{\text{true}}^n[k] = R_v^n[k] (\mathbf{f}_m^v[k] - \mathbf{b}_a^v) + \mathbf{g}^n$$

### 4.4 Discrete Kinematic Propagation
Velocity and position are updated via **constant-acceleration discrete kinematic propagation**:
$$\mathbf{v}[k+1] = \mathbf{v}[k] + \mathbf{a}_{\text{true}}^n[k] \cdot \Delta t$$
$$\mathbf{p}[k+1] = \mathbf{p}[k] + \mathbf{v}[k] \cdot \Delta t + \frac{1}{2} \mathbf{a}_{\text{true}}^n[k] \cdot \Delta t^2$$

---

## 5. Error Mechanisms vs. Observed Empirical Drift

### Theoretical Error Mechanisms
1. **Accelerometer Bias ($t^2$)**: A constant uncompensated acceleration bias $b_a$ integrates twice into quadratic position error:
   $$\delta p_a(t) \approx \frac{1}{2} b_a t^2$$
2. **Gyroscope Bias & Gravity Leakage ($t^3$)**: An uncompensated gyroscope bias $b_g$ causes an accumulating attitude tilt error: $\delta \theta(t) \approx b_g t$. In the presence of Earth's gravity, this tilt error misprojects downward gravity into the horizontal plane:
   $$a_{\text{leak}}(t) = g \sin(\delta \theta) \approx g b_g t$$
   Double integrating this spurious horizontal acceleration produces a cubic position error component:
   $$\delta p_{\text{leak}}(t) \approx \frac{1}{6} g b_g t^3$$

### Observed Empirical Drift
In the real Stage 1 benchmark on Trip S1, the unassisted trajectory demonstrated severe drift consistent with the accumulation of inertial errors:
- Theoretical formulas show how attitude errors leak gravity to cause $t^3$-type divergence.
- In practice, the empirical error reflects a combination of residual gyro bias, unmodeled accelerometer bias, chassis vibration residuals, and non-linear vehicle dynamics. We do not claim the $3,249.32\,\text{m}$ error is mathematically proven to be purely cubic gravity leakage; rather, the real trajectory demonstrates the classic divergence expected when dead-reckoning consumer IMUs open loop.

---

## 6. Code Architecture: `navigation/ins/`

```
navigation/ins/
├── __init__.py           # Package exports (INSState, INSTrajectory, StrapdownINS)
├── attitude.py          # Pure quaternion algebra, rotation matrices, conversions
└── propagation.py       # Strapdown mechanization, state classes, discrete kinematic propagation
```

- **`attitude.py`**: `quaternion_multiply`, `quaternion_normalize`, `delta_quaternion`, `quaternion_to_rotation_matrix`, `quaternion_to_euler_deg`.
- **`propagation.py`**: `INSState` (immutable dataclass), `INSTrajectory`, and `StrapdownINS` propagation engine enforcing timestep validation ($0 < \Delta t \le 1.0\,\text{s}$) and monotonic timestamps.

---

## 7. The Ablation Stage 1 Benchmark (Trip S1)

Executed via `scripts/run_ablation_stage1.py`:
- **Window**: Samples 19500 to 20100 of `Categorised_S1.npz` (601 samples at $10\,\text{Hz}$, duration: $60.00\,\text{s}$).
- **Dynamics**: Highway cruising at average speed $12.7\,\text{m/s}$ ($45.7\,\text{km/h}$), traversing $763.2\,\text{meters}$.
- **Oracle Initialization Disclosure**: Initial velocity ($\mathbf{v}_0^n = [-11.399, +3.099, 0.0]^T\,\text{m/s}$) and heading ($285.21^\circ$) were initialized from Racelogic VBOX ground truth at $t = 0$. This controlled condition isolates inertial integration divergence over time from startup alignment errors. Following $t = 0$, **zero GNSS or external aiding information influenced the propagation**.
- **VBOX Altitude Unit Scaling**: In `V-S1.csv`, the column header said `' Height (km)'` but raw values ($92$ to $144$) were in meters. In `run_ablation_stage1.py`, cached `v_ref_alt_m` was scaled by $1/1000.0$ to recover physical meters without modifying the frozen Phase 2 cache.

---

## 8. Empirical Results: What We Actually Measured

| Elapsed Time | Horizontal Drift ($e_{\text{2D}}$) | 3D Drift ($e_{\text{3D}}$) | Vertical Drift ($e_{\text{Up}}$) | True Distance Traveled | Drift / Distance Ratio |
|---|---|---|---|---|---|
| **$5.0\,\text{s}$** | **$4.83\,\text{m}$** | $4.95\,\text{m}$ | $+1.08\,\text{m}$ | $62.0\,\text{m}$ | $7.8\%$ |
| **$10.0\,\text{s}$** | **$20.52\,\text{m}$** | $20.60\,\text{m}$ | $+1.83\,\text{m}$ | $121.6\,\text{m}$ | $16.9\%$ |
| **$15.0\,\text{s}$** | **$100.64\,\text{m}$** | $100.74\,\text{m}$ | $-4.42\,\text{m}$ | $184.5\,\text{m}$ | $54.6\%$ |
| **$20.0\,\text{s}$** | **$251.21\,\text{m}$** | $252.18\,\text{m}$ | $-22.08\,\text{m}$ | $250.5\,\text{m}$ | $100.3\%$ |
| **$30.0\,\text{s}$** | **$788.97\,\text{m}$** | $793.73\,\text{m}$ | $-86.89\,\text{m}$ | $376.1\,\text{m}$ | $209.8\%$ |
| **$45.0\,\text{s}$** | **$1,927.34\,\text{m}$** | $1,943.15\,\text{m}$ | $-247.41\,\text{m}$ | $565.3\,\text{m}$ | $341.0\%$ |
| **$60.0\,\text{s}$** | **$3,249.32\,\text{m}$** | **$3,292.29\,\text{m}$** | **$-530.20\,\text{m}$** | **$763.2\,\text{m}$** | **$425.7\%$** |

### Summary Statistics
- **Horizontal Position RMSE**: **$1,487.53\,\text{meters}$**.
- **Final Horizontal Position Error**: **$3,249.32\,\text{meters}$**.
- **Final Vertical Position Error**: **$-530.20\,\text{meters}$**.
- **Final 3D Position Error**: **$3,292.29\,\text{meters}$**.
- **Integrated Intervals**: Exactly 600 intervals (601 state records, 0 skipped).

---

## 9. Verification & Unit Testing

Phase 4 propagation physics is validated by **16 unit tests** in [`tests/unit/test_ins_propagation.py`](file:///d:/Hackathon/Compass/tests/unit/test_ins_propagation.py):
1. **Stationary Gravity Cancellation**: Asserts that level stationary specific force $[0, 0, +9.80665]^T$ yields coordinate acceleration $[0, 0, 0]^T$ and zero position drift.
2. **Constant Angular Velocity**: Verifies quaternion integration against analytical rotation angles.
3. **Quaternion Normalization Stability**: Verifies $|\|\mathbf{q}\| - 1.0| < 10^{-14}$ and $R R^T = \mathbf{I}$ over successive integration steps.
4. **Invalid-Sample Protection**: Rejects non-finite inputs and negative timesteps with clean `ValueError`.

---

## 10. Final Phase 4 Summary: What I Should Remember

1. **Propagation Equations**: Constant-acceleration discrete kinematic propagation:
   $$\mathbf{v}[k+1] = \mathbf{v}[k] + \mathbf{a}_{\text{true}}^n[k] \Delta t, \quad \mathbf{p}[k+1] = \mathbf{p}[k] + \mathbf{v}[k] \Delta t + \frac{1}{2} \mathbf{a}_{\text{true}}^n[k] \Delta t^2$$
2. **Gravity & Coordinate Acceleration**: $\mathbf{a}_{\text{true}}^n = R_v^n (\mathbf{f}_m^v - \mathbf{b}_a^v) + \mathbf{g}^n$, where $\mathbf{g}^n = [0, 0, -9.80665]^T\,\text{m/s}^2$.
3. **Empirical Baseline**: Open-loop dead reckoning over 60 seconds on Trip S1 diverged to **$3,249.32\,\text{m}$ horizontal error** ($425.7\%$ of distance traveled).
4. **Controlled Benchmark**: Initialized from oracle ground truth at $t=0$ to isolate sensor drift from startup misalignment.
5. **Phase 4 Test Count**: Exactly **16 tests** passing in `tests/unit/test_ins_propagation.py`.
