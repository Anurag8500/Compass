# COMPASS Phase 1 — Schemas, Adapters & Architectural Contracts

**Project**: C.O.M.P.A.S.S. (Cognitive Off-grid Machine-learning Positioning And Sensor System)  
**Problem Statement**: SIH 2026 Problem Statement 26168 — ISRO  
**Stage**: Phase 1 Complete Teaching & Reference Walkthrough

---

## 1. Executive Summary & Why Phase 1 Existed

### The Core Problem
In complex robotics and sensor fusion projects, the single most common cause of catastrophic bugs is **semantic ambiguity**:
- Is this acceleration vector in the phone's casing coordinate frame, or in the car's forward-lateral-up frame?
- Is this timestamp in milliseconds, microseconds, or nanoseconds?
- Is this quaternion formatted as $[w, x, y, z]$ (scalar first) or $[x, y, z, w]$ (vector first)?
- Does this altitude represent height above mean sea level or height in kilometers above the WGS-84 ellipsoid?
- What happens when a sensor disconnects or outputs `NaN`? Does the code crash, or does it track the failure non-destructively?

If developers jump directly into writing filtering algorithms without rigid schemas, every module invents its own ad-hoc dictionaries. A function expects meters per second, but receives kilometers per hour. A matrix expects vehicle-frame acceleration, but gets raw phone-frame acceleration. The math runs without syntax errors, but the vehicle drifts into a digital ditch.

### What Phase 1 Accomplished
Phase 1 established the **immutable software foundation and architectural contracts** of COMPASS. We built:
1. Rigid, type-safe data containers (`navigation/schemas/`) with runtime validation enforcing units, boundaries, non-zero norms, and coordinate frame semantics.
2. The Non-Destructive Quality Bitmask System, allowing bad samples to be flagged without dropping rows from time series.
3. The Three-State GNSS Finite State Machine (`GNSSMode`).
4. Abstract Sensor and GNSS Adapters (`navigation/adapters/`), creating clean boundaries between external hardware logs and core navigation math.

---

## 2. Technical Vocabulary & Physical Concepts

Before looking at the files, let us define the core physical and computational terms used throughout Phase 1.

### 1. Coordinate Frames: Device, Vehicle, and Navigation
- **Simple Definition**: An origin point and a set of perpendicular axes ($X, Y, Z$) used to measure positions, velocities, and forces.
- **In COMPASS**: We strictly distinguish three physical coordinate systems:
  1. **Device Frame ($b$)**: The physical casing of the smartphone. Axes are oriented however the driver placed the phone in the mount or console.
  2. **Vehicle Frame ($v$)**: Rigidly attached to the vehicle chassis using the **Forward-Lateral-Up (FLU)** convention ($X_v$ forward along centerline, $Y_v$ lateral leftwards, $Z_v$ upward toward the roof).
  3. **Navigation Frame ($n$)**: The local geographic tangent plane using the right-handed **East-North-Up (ENU)** convention ($X_n$ East, $Y_n$ North, $Z_n$ Up).
- **External Comparison Only**: Aerospace systems frequently use North-East-Down (NED). COMPASS uses **ENU** as its authoritative navigation convention.
- **Why We Care**: $10\,\text{m/s}^2$ of acceleration along the phone's screen axis means something completely different from $10\,\text{m/s}^2$ forward along the car's hood. Mixing frames creates massive false drift.

### 2. Quaternion
- **Simple Definition**: A four-dimensional hypercomplex number $\mathbf{q} = [q_w, q_x, q_y, q_z]^T$ used to represent 3D rotations without suffering from gimbal lock.
- **In COMPASS**: Used to rotate vectors from the vehicle frame to the local navigation frame ($R_v^n$). Stored in the Hamilton scalar-first convention with $\|\mathbf{q}\| = 1.0$.
- **Why We Care**: Euler angles (roll, pitch, yaw) suffer from mathematical singularities (division by zero at $90^\circ$ pitch, known as gimbal lock). Quaternions provide smooth, continuous, singularity-free 3D attitude tracking.

### 3. WGS-84 (World Geodetic System 1984)
- **Simple Definition**: The standard ellipsoidal coordinate model of the Earth used by GPS.
- **In COMPASS**: Latitude and longitude angles defining the vehicle's position on the globe.
- **Why We Care**: Satellites measure positions relative to an ellipsoidal Earth model. We convert these curvilinear coordinates into a local flat tangent plane (East-North-Up) for vehicle navigation math.

### 4. Heteroscedastic Uncertainty (Conceptual Contract)
- **Simple Definition**: Uncertainty that is not constant, but changes dynamically depending on the operating environment.
- **In COMPASS**: Our future ML network architecture predicts both a physical value (e.g. forward speed) and an uncertainty variance ($\sigma^2$).
- **Illustrative Example Only**: For instance, on smooth dry asphalt, an estimated forward speed might have low uncertainty (e.g. illustrative $\sigma \approx 0.2\,\text{m/s}$), whereas over cobblestones or during wheel slip, uncertainty increases (e.g. illustrative $\sigma \approx 3.0\,\text{m/s}$). Phase 1 establishes the schema contract (`MLPrediction`) to represent dynamic uncertainty.
- **Project Status Clarification**: The actual training of VelocityNet belongs to Phase 7, and BiasNet belongs to Phase 8. Phase 1 defines the data contracts; models are not yet trained here.

### 5. Innovation Gating (Chi-Square Validation)
- **Simple Definition**: A statistical test checking whether a new sensor measurement is plausible given our current estimate and uncertainty.
- **In COMPASS**: Comparing a GPS fix or auxiliary speed measurement against the predicted state.
- **Why We Care**: If GPS experiences a multipath jump (satellite signal reflecting off a building), the indicated position might jump by 50 meters in 100 milliseconds. Innovation gating detects that this is statistically implausible and rejects the measurement before it can corrupt the filter.

---

## 3. Coordinate Frames in COMPASS

Understanding COMPASS requires understanding the exact relationship between the three physical frames used throughout the system.

> [!IMPORTANT]
> COMPASS uses **East-North-Up (ENU)**, **not** North-East-Down (NED). The local navigation frame origin is fixed at the session reference point. $Z_n$ points **Up** toward the sky, $X_n$ points **East**, and $Y_n$ points **North**.

```
+-------------------------------------------------------------------------+
|                              EARTH                                      |
|                                                                         |
|   1. Local Navigation Frame (n): East-North-Up (ENU)                   |
|      - Cartesian tangent plane at fixed session reference point        |
|      - X^n points True East                                            |
|      - Y^n points True North                                           |
|      - Z^n points Up away from Earth                                   |
|      - Gravity vector: g^n = [0, 0, -9.80665]^T  m/s^2               |
+--------------------------------|----------------------------------------+
                                 |
                                 v  Attitude Rotation: q (R_v^n)
+--------------------------------|----------------------------------------+
|                             VEHICLE                                     |
|                                                                         |
|   2. Vehicle Body Frame (v): Forward-Lateral-Up (FLU)                  |
|      - X^v points Forward along vehicle centerline                     |
|      - Y^v points Lateral (leftwards)                                  |
|      - Z^v points Up toward vehicle roof                               |
|      - At rest on level ground: f^v ≈ [0, 0, +9.81]^T  m/s^2         |
+--------------------------------|----------------------------------------+
                                 |
                                 v  Mounting Alignment Matrix: R_b^v
+--------------------------------|----------------------------------------+
|                             DEVICE                                      |
|                                                                         |
|   3. Device Sensor Frame (b): Smartphone casing coordinate system      |
|      - Arbitrary orientation as placed by the driver                   |
|      - Axes align with physical phone casing, not vehicle              |
+-------------------------------------------------------------------------+
```

### Why Frame Alignment is Mandatory
When a driver mounts a smartphone in a cradle or leaves it on the console tray, the phone's $Y$-axis might point backwards at a $30^\circ$ tilt, while its $Z$-axis points toward the vehicle's door. If you feed raw device-frame accelerations into navigation equations without rotating them first, the computed East/North/Up acceleration components are meaningless.

### The Gravity Convention
In the **ENU** navigation frame, Earth gravity acts **downward**, hence:
$$\mathbf{g}^n = \begin{bmatrix} 0 \\ 0 \\ -9.80665 \end{bmatrix}\,\text{m/s}^2$$

When a vehicle is stationary on level ground, the specific force in the vehicle frame is:
$$\mathbf{f}^v = [0, 0, +9.80665]^T\,\text{m/s}^2$$

This positive-Z reading represents the upward normal force of the ground supporting the vehicle proof mass.

### Frame Transformation and Bias Conventions
Sensor biases originate physically inside the device hardware ($b$). Gyroscope bias $\mathbf{b}_g^b$ is estimated during stationary rest in the body frame. Transformed vehicle-frame angular rates and specific forces follow:
$$\boldsymbol{\omega}_m^v = R_b^v (\boldsymbol{\omega}_m^b - \mathbf{b}_g^b)$$
$$\mathbf{f}_m^v = R_b^v (\mathbf{f}_m^b - \mathbf{b}_{a,\text{prior}}^b), \quad \mathbf{b}_{a,\text{prior}}^b = [0, 0, 0]^T$$
In downstream navigation (Phase 5 ESKF), residual biases $\mathbf{b}_a^v$ and $\mathbf{b}_g^v$ are maintained and dynamically estimated in the vehicle frame.

---

## 4. State Vector Dimensions: The Crucial 16 vs 15 Distinction

A fundamental architectural distinction in COMPASS:

### 1. Nominal Navigation State ($\mathbf{x}_{\text{nom}} \in \mathbb{R}^{16}$)
The non-linear nominal kinematic state carries **16 parameters**:
$$\mathbf{x}_{\text{nom}} = \begin{bmatrix} \mathbf{p}^n \\ \mathbf{v}^n \\ \mathbf{q} \\ \mathbf{b}_a^v \\ \mathbf{b}_g^v \end{bmatrix} \in \mathbb{R}^{16}$$
- $\mathbf{p}^n \in \mathbb{R}^3$: Position in local ENU coordinates [m] (3).
- $\mathbf{v}^n \in \mathbb{R}^3$: Velocity in local ENU coordinates [m/s] (3).
- $\mathbf{q} \in \mathbb{H}, \|\mathbf{q}\| = 1$: Unit attitude quaternion (4).
- $\mathbf{b}_a^v \in \mathbb{R}^3$: Accelerometer bias in vehicle coordinates [$\text{m/s}^2$] (3).
- $\mathbf{b}_g^v \in \mathbb{R}^3$: Gyroscope bias in vehicle coordinates [$\text{rad/s}$] (3).
$$\text{Total parameters} = 3 + 3 + 4 + 3 + 3 = 16$$

### 2. Error State ($\delta\mathbf{x} \in \mathbb{R}^{15}$)
The linear perturbation state carries **15 parameters**:
$$\delta\mathbf{x} = \begin{bmatrix} \delta\mathbf{p}^n \\ \delta\mathbf{v}^n \\ \delta\boldsymbol{\theta} \\ \delta\mathbf{b}_a^v \\ \delta\mathbf{b}_g^v \end{bmatrix} \in \mathbb{R}^{15}$$
- $\delta\mathbf{p}^n \in \mathbb{R}^3$: Position error (3).
- $\delta\mathbf{v}^n \in \mathbb{R}^3$: Velocity error (3).
- $\delta\boldsymbol{\theta} \in \mathbb{R}^3$: Small-angle rotation vector on the $\mathfrak{so}(3)$ Lie algebra manifold (3).
- $\delta\mathbf{b}_a^v \in \mathbb{R}^3$: Accelerometer bias error (3).
- $\delta\mathbf{b}_g^v \in \mathbb{R}^3$: Gyroscope bias error (3).
$$\text{Total parameters} = 3 + 3 + 3 + 3 + 3 = 15$$

### 3. Error Covariance Matrix ($P \in \mathbb{R}^{15 \times 15}$)
Because attitude error is parameterized by the minimal 3-vector $\delta\boldsymbol{\theta}$, the error covariance matrix is strictly:
$$P = \mathbb{E}\left[\delta\mathbf{x} \delta\mathbf{x}^T\right] \in \mathbb{R}^{15 \times 15}$$

> [!IMPORTANT]
> The nominal state is never referred to as a "15-state vector". The shorthand phrase "15-state ESKF" refers specifically to the **15-dimensional error state** and its $15 \times 15$ covariance.

---

## 5. Phase 1 Architecture: What We Built

### 5.1 Schema Files (`navigation/schemas/`)

**`imu.py`** — Raw and Aligned IMU sample containers:
- `RawIMUSample`: Device-frame specific force and angular velocity with quality bitmask.
- `AlignedIMUSample`: Vehicle-frame specific force $\mathbf{f}^v$ and angular rate $\boldsymbol{\omega}^v$ after mounting alignment. Physical gravity is **NOT** removed here; strapdown mechanization handles gravity downstream.
- Quality flags (`FLAG_OK`, `FLAG_NAN_OR_NONFINITE`, `FLAG_INVALID_TIMESTAMP`, `FLAG_NON_MONOTONIC_TIMESTAMP`, `FLAG_DUPLICATE_TIMESTAMP`, `FLAG_EXTREME_MOTION`, `FLAG_SENSOR_DROPOUT`).

**`gnss.py`** — GNSS fix container:
- `GNSSSample`: WGS-84 latitude, longitude, altitude, speed, bearing, accuracy, satellite count, and continuous `trust_score ∈ [0, 1]`.

**`state.py`** — Navigation state containers:
- `GNSSMode` enum: Three discrete FSM states (`GNSS_AIDED`, `DR_ONLY`, `REACQUIRING`). Continuous signal degradation is represented by `trust_score`, **not** as a 4th mode.
- `OrientationState`: Hamilton scalar-first quaternion `(w, x, y, z)` representing rotation from vehicle frame to ENU navigation frame ($R_v^n$), plus estimated gyroscope bias.
- `NavigationState`: Complete navigation snapshot — 16-parameter nominal state representation, $15 \times 15$ error covariance matrix $P$, fixed session reference point $(lat_0, lon_0)$, current GNSS mode, and nanosecond timestamp.

**`ml.py`** — ML inference output containers:
- `MLPrediction`: Canonical output of VelocityNet (1 scalar: forward speed) or BiasNet (6 values: $[\delta b_{ax}, \delta b_{ay}, \delta b_{az}, \delta b_{gx}, \delta b_{gy}, \delta b_{gz}]$), each paired with log-variance for heteroscedastic uncertainty.

**`config.py`** — Feature engineering constants and schema:
- `CANONICAL_CHANNELS`: 9-tuple defining strict channel ordering for all feature tensors:
  `("f_x_v", "f_y_v", "f_z_v", "omega_x_v", "omega_y_v", "omega_z_v", "norm_f_v", "norm_f_dot_v", "norm_omega_v")`.
- `ExternalSensorPacket`: Schema for non-phone external sensors (FOG, CAN, MEMS).
- `ModelConfig`: JSON-serializable normalization schema loaded by VelocityNet/BiasNet at inference time.

**`mapmatch.py`** — Map-matching schemas (for later integration).

### 5.2 Adapter Interface (`navigation/adapters/`)
The abstract `SensorAdapter` and `GNSSAdapter` base classes define the canonical interface between any external hardware log format and COMPASS internal schemas.

---

## 6. The Quality Bitmask System

The bitmask system allows bad samples to be **flagged without being deleted** — preserving time-series continuity:

| Flag Constant | Hex Value | Meaning | Pipeline Policy |
|:---|:---:|:---|:---|
| `FLAG_OK` | `0x00` | Clean nominal sample | Pass to filter |
| `FLAG_NAN_OR_NONFINITE` | `0x01` | Contains NaN, +Inf, or -Inf | Omit from validated stream |
| `FLAG_INVALID_TIMESTAMP` | `0x02` | Negative timestamp | Omit from validated stream |
| `FLAG_NON_MONOTONIC_TIMESTAMP` | `0x04` | $t_k \le t_{k-1}$ | Omit from validated stream |
| `FLAG_DUPLICATE_TIMESTAMP` | `0x08` | Same timestamp repeated | Omit from validated stream |
| `FLAG_EXTREME_MOTION` | `0x10` | $\|\mathbf{f}\| > 39.24\,\text{m/s}^2$ or $\|\boldsymbol{\omega}\| > 10\,\text{rad/s}$ | **KEEP** — real vehicle dynamics |
| `FLAG_SENSOR_DROPOUT` | `0x20` | $\Delta t > 3 \times \Delta t_\text{nominal}$ | **KEEP** — propagate across gap |

Only flags `0x01`, `0x02`, `0x04`, and `0x08` represent non-computable samples. Flags `0x10` and `0x20` represent real physical dynamics that downstream estimators must handle rather than discard.

---

## 7. The Three-State GNSS FSM

```
+----------------+          +----------------+         +--------------+
|                |  Signal  |                |  Stable |              |
|  GNSS_AIDED    |--------> |  DR_ONLY       |-------> | REACQUIRING  |
|                |  Lost    |                | Return  |              |
+----------------+          +----------------+         +--------------+
        ^                                                      |
        |                    Validated                         |
        +------------------------------------------------------+
                             Convergence
```

- **`GNSS_AIDED`**: Active GNSS position and horizontal velocity measurements are fused in the ESKF.
- **`DR_ONLY`**: GNSS signal is lost (tunnels, parking garages). The ESKF propagates forward using IMU dead reckoning and ZUPT.
- **`REACQUIRING`**: GNSS signal has returned but is still being validated. Full updates resume only after convergence.

Degraded GNSS is represented continuously by `trust_score`, not as a 4th FSM mode.

---

## 8. Verification & Test Suite

Phase 1 delivered exactly **50 unit tests**:
- `tests/unit/test_schemas.py`: 43 unit tests covering schema field validation, round-trip JSON serialization, flag bitmask arithmetic, `GNSSMode` enumeration, covariance shape enforcement ($15 \times 15$), and `MLPrediction` dimension consistency.
- `tests/unit/test_frame_conversion.py`: 7 unit tests validating coordinate conversions between Geodetic (WGS84) and local Cartesian ENU tangent plane.

---

## 9. Final Phase 1 Summary: What I Should Remember

1. **Navigation Frame**: Local **East-North-Up (ENU)** right-handed Cartesian frame. Downward gravity is $[0, 0, -9.80665]^T\,\text{m/s}^2$.
2. **Vehicle Frame**: **Forward-Lateral-Up (FLU)**. At rest on level ground, $\mathbf{f}^v \approx [0, 0, +9.81]^T\,\text{m/s}^2$.
3. **Quaternion**: Hamilton scalar-first $[w, x, y, z]^T$ representing rotation from vehicle to ENU ($R_v^n$).
4. **State Dimensions**: Nominal state has **16 parameters**; Error state has **15 parameters**; Covariance matrix $P$ is **$15 \times 15$**.
5. **GNSS FSM**: Exactly 3 discrete modes. Degraded GNSS is handled via continuous `trust_score`.
6. **Quality Flags**: 6 flags. Only non-computable samples are omitted; extreme motion and dropouts are preserved.
7. **Phase 1 Test Count**: Exactly **50 tests** verified and passing.
