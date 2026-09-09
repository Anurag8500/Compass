# COMPASS Phase 3 — Implementation Build Walkthrough

**Project**: C.O.M.P.A.S.S. (Cognitive Off-grid Machine-learning Positioning And Sensor System)  
**Problem Statement**: SIH 2026 Problem Statement 26168 — ISRO  
**Document Type**: Step-by-Step Implementation Build Diary

---

## Overview: What We Were Trying to Do

Phase 3 was the **signal conditioning pipeline**: taking the raw device-frame (phone casing) accelerations and gyroscope readings from Phase 2 and turning them into calibrated, tilt-aligned, filtered **vehicle-frame** measurements.

Before Phase 3, the IMU data was physically meaningful (correctly parsed and quality-tagged) but it was in the wrong coordinate system and contaminated with:
- Engine and road vibration (high-frequency noise).
- Gyroscope bias (the gyro reads non-zero angular velocity even at rest).
- An arbitrary phone mounting angle.

Phase 3 resolved these challenges while preserving physical specific force intact for downstream navigation.

---

## Step 1: Building the Stationary Calibration Module (`calibration.py`)

### What It Does
When the vehicle is stationary, we can measure:
1. The **gyroscope bias** $\mathbf{b}_g^b$ — because true angular velocity is zero at rest, so any non-zero reading is pure bias.
2. The **tilt angles** (roll and pitch) — because the specific force vector equals the upward gravity support reaction vector $[0, 0, +g]^T$ in a level vehicle frame.

### Implementation
```python
def calibrate_stationary_window(accel: np.ndarray, gyro: np.ndarray, ...) -> CalibrationProfile:
    # Gyro bias: mean of all gyro readings at rest in body frame
    gyro_bias = np.mean(gyro, axis=0)  # shape (3,)
    
    # Gravity direction: mean specific force vector
    g_measured = np.mean(accel, axis=0)  # should be [0, 0, +9.81] for level phone
    g_norm = np.linalg.norm(g_measured)
    
    # Roll and pitch from spherical trigonometry
    gz = g_measured[2] / g_norm
    gy = g_measured[1] / g_norm
    gx = g_measured[0] / g_norm
    pitch_rad = -np.arcsin(gx)
    roll_rad  = np.arctan2(gy, gz)
```

**Why not estimate accelerometer bias $\mathbf{b}_a$?**
A single static pose gives 3 measurements but has 5 unknowns (2 tilt angles + 3 bias components). The problem is mathematically underdetermined. We set $\mathbf{b}_{a,\text{prior}} = [0, 0, 0]^T$ as a nominal prior and let the downstream ESKF (Phase 5) dynamically refine it during motion.

---

## Step 2: Building the Mounting Alignment Module (`alignment.py`)

### Part A: Tilt Alignment (Always Observable)
We used Rodrigues' rotation formula to build the minimum-angle 3D rotation matrix that rotates the measured gravity support vector $\bar{\mathbf{f}}^b$ onto vehicle vertical $+Z_v = [0, 0, 1]^T$:

```python
def rotation_matrix_from_vectors(v_from, v_to):
    """Compute R such that R @ v_from = v_to."""
    a = v_from / np.linalg.norm(v_from)
    b = v_to / np.linalg.norm(v_to)
    dot = np.dot(a, b)
    if dot > 0.99999999:
        return np.eye(3)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    K = skew(v)
    return np.eye(3) + K + (K @ K) * (1.0 - dot) / (s * s)
```

### Part B: Yaw Observability & Sentinel Contract
Yaw (horizontal mounting angle) cannot be determined from gravity alone. When candidate acceleration correlation was tested on real highway trip S1:
- Wild acceleration fluctuations caused circular dispersion $\sigma = 126.0^\circ > 15.0^\circ$.
- Rather than fabricating an unreliable yaw, Phase 3 sets:
  ```python
  alignment.is_yaw_aligned = False
  alignment.yaw_deg = 0.0  # sentinel default
  ```
- **Sentinel Rule**: `yaw_deg = 0.0` with `is_yaw_aligned = False` means reliable mounting yaw was NOT obtained. It does NOT mean the measured vehicle yaw is 0 degrees.

### Transform Equations
$$\mathbf{f}_m^v = R_b^v (\mathbf{f}_m^b - \mathbf{b}_{a,\text{prior}}^b), \quad \mathbf{b}_{a,\text{prior}}^b = [0, 0, 0]^T$$
$$\boldsymbol{\omega}_m^v = R_b^v (\boldsymbol{\omega}_m^b - \mathbf{b}_g^b)$$

---

## Step 3: Building the Dual-Stage Filter (`filtering.py`)

1. **Stage 1 (3-Sample Median Filter)**: Causal median filter eliminates single-sample impulse spikes.
2. **Stage 2 (4th-Order Butterworth Filter)**: Digital low-pass filter ($f_c = 3.0\,\text{Hz}$, $f_s = 10.0\,\text{Hz}$).
   - Offline: zero-phase (`filtfilt`).
   - Online: causal recursive (`lfilter`).
   - Short sequence fallback: passthrough if $N < 15$ samples to avoid filter padding crashes.

---

## Step 4: Building the Recalibration Detector (`recalibration_trigger.py`)

Monitors gravity vector shifts $\theta_{\text{shift}}$ in vehicle frame. Suppresses triggers during high vehicle dynamics ($\|\|\mathbf{f}\| - g\| \ge 1.5\,\text{m/s}^2$ or $\|\boldsymbol{\omega}\| \ge 0.5\,\text{rad/s}$); when stationary, triggers on confirmed $>35^\circ$ shift with a 50-sample cooldown lockout.

---

## Step 5: Assembling the Pipeline (`pipeline.py`)

The `PreprocessingPipeline` orchestrates calibration, tilt alignment, dual-stage filtering, and recalibration monitoring into clean `PreprocessedTrip` objects.

---

## Step 6: Tests

Phase 3 is validated by exactly **47 tests**:
- **`test_alignment.py`** (11 tests): Proper rotation matrices ($\det R = +1, R R^T = \mathbf{I}$), Euler angles, yaw observability logic.
- **`test_calibration.py`** (10 tests): Ground-truth gyro bias and tilt angle recovery from stationary windows.
- **`test_filtering.py`** (10 tests): Spike suppression and Butterworth frequency attenuation.
- **`test_gravity_compensation.py`** (7 tests): Level stationary specific force invariants ($f_z \approx +9.81, f_x \approx 0, f_y \approx 0$).
- **`test_recalibration.py`** (7 tests): Mounting shift detection and dynamic false-alarm suppression.
- **`test_preprocessing_on_real_data.py`** (2 tests): End-to-end execution on real cached data.

---

## Debugging Episodes

**Bug 1: Rodrigues formula degenerate case at 180°**
When the measured gravity vector was anti-parallel to target, $\mathbf{u}_g \times \mathbf{z}_v = \mathbf{0}$, causing division by zero. Fixed by selecting an orthogonal fallback axis and applying $R = \mathbf{I} + 2 K^2$.

**Bug 2: Zero-phase filter padding crash on short sequences**
`filtfilt` raised `ValueError` on trips $< 15$ samples due to default padding. Fixed by adding a length check that routes short sequences to causal filtering or passthrough.

---

## Final Outputs

| Artifact | Path | Purpose |
|:---|:---|:---|
| Calibration | `navigation/preprocessing/calibration.py` | Gyro bias and tilt angles |
| Alignment | `navigation/preprocessing/alignment.py` | Rodrigues mounting rotation $R_b^v$ |
| Filtering | `navigation/preprocessing/filtering.py` | Dual-stage median + Butterworth filter |
| Recalibration | `navigation/preprocessing/recalibration_trigger.py` | Mounting slip detector |
| Orchestrator | `navigation/preprocessing/pipeline.py` | Unified preprocessing pipeline |
| Utilities | `navigation/preprocessing/gravity.py` | Gravity model utilities |
| Report | `docs/preprocessing_phase3_report.md` | Full Phase 3 execution report |

**Test count contributed by Phase 3**: Exactly **47 tests** passing.
