# Phase 6 Complete Explanation: Machine Learning Dataset Construction, Causal Windowing & Leakage-Audited Engineering

---

## 1. Introduction & Phase Overview

### What Problem Did Phase 6 Solve?
In Phase 5, we successfully built the 15-state Error-State Kalman Filter (ESKF). When GPS fixes are available, the filter bounds horizontal position error to $5.27\,\text{meters}$. But when GPS signals are completely lost (e.g., inside long highway tunnels, subterranean parking structures, or dense urban canyons), the filter receives zero position updates. In that blind window, it is forced to rely on open-loop inertial propagation, which Phase 4 showed drifts quadratically and cubically to thousands of meters.

To navigate without GPS, an autonomous vehicle needs an alternate source of speed and bias aiding:
1. **VelocityNet** (Phase 7): A 2-layer GRU neural network (`GRU(9→64, 2L) → Dense(64→32, ReLU) → Dense(32→2)`) that observes high-frequency inertial vibration patterns from the chassis and predicts the vehicle's forward driving speed ($v_{\text{forward}}$) and uncertainty ($\log \sigma_v^2$).
2. **BiasNet** (Phase 8): A 2-layer GRU neural network (`GRU(9→48, 2L) → Dense(48→24, ReLU) → Dense(24→12)`) that estimates residual IMU sensor biases during dynamic driving.

**Phase 6 solves the foundational data-engineering problem for these models**: How do we transform 60 hours of raw, multi-sensor vehicle driving telemetry into a deterministic, leakage-audited, standardized, and strictly causal machine learning dataset ready for deep learning training?

### Why Did We Need to Solve It?
In Machine Learning for robotics and physical systems, **data pipeline errors are fatal and silent**:
- If future sensor samples accidentally leak into an input window (lookahead leakage), a model will achieve an impressive $99\%$ accuracy on your laptop during training, but will completely fail when deployed in real time on a vehicle because the future does not exist yet.
- If data from the same driving trip is randomly shuffled into both the training and test sets, the neural network simply memorizes the vehicle's route and track geometry rather than learning general physical motion principles.
- If normalization statistics (means and standard deviations) are computed across the entire dataset, validation and test information leaks into the training process.

Phase 6 implements a **shared, leakage-audited, deterministic data engineering pipeline** that extracts **399,714 feature tensors** ($\mathbf{X} \in \mathbb{R}^{20 \times 9}$) and ground-truth speed labels across 142 downstream-ready file representations corresponding to 71 unique physical trips, enforcing and testing defined leakage-prevention invariants, strict physical causality, and bit-for-bit reproducibility.

---

## 2. Core Concepts & Terminology

### 1. The Canonical $(20, 9)$ Feature Tensor
- **Simple Definition**: Every single input example fed to our future neural networks is a 2D matrix of 20 time rows and 9 feature columns:
  $$\mathbf{X} \in \mathbb{R}^{20 \times 9}$$
- **Time Extent**: At our canonical $10.0\,\text{Hz}$ rate ($\Delta t = 100\,\text{ms}$), 20 samples span exactly $2.0\,\text{seconds}$ ($1.9\,\text{s}$ of history ending at the current timestamp $T$).
- **Why We Care**: Two seconds of inertial history is the sweet spot for vehicle dynamics: it is long enough to capture several complete engine cylinder strokes, wheel revolutions, suspension oscillations, and road surface texture vibrations, but short enough that vehicle speed does not change drastically across the window.

### 2. The 9 Canonical Feature Channels
All 9 channels represent calibrated motion in the **Vehicle FLU Frame** ($v$):

| Channel Index | Channel Name | Description | Physical Units | Mathematical Formulation |
|---|---|---|---|---|
| **0** | `f_x_v` | Longitudinal specific force (forward/braking) | $\text{m/s}^2$ | Direct vehicle-frame sensor reading |
| **1** | `f_y_v` | Lateral specific force (cornering/centrifugal) | $\text{m/s}^2$ | Direct vehicle-frame sensor reading |
| **2** | `f_z_v` | Vertical specific force (road bumps & gravity) | $\text{m/s}^2$ | Direct vehicle-frame sensor reading |
| **3** | `omega_x_v` | Roll angular velocity | $\text{rad/s}$ | Direct vehicle-frame sensor reading |
| **4** | `omega_y_v` | Pitch angular velocity | $\text{rad/s}$ | Direct vehicle-frame sensor reading |
| **5** | `omega_z_v` | Yaw angular velocity (turning rate) | $\text{rad/s}$ | Direct vehicle-frame sensor reading |
| **6** | `norm_f_v` | Specific force vector magnitude | $\text{m/s}^2$ | $\|\mathbf{f}^v\| = \sqrt{(f_x^v)^2 + (f_y^v)^2 + (f_z^v)^2}$ |
| **7** | `norm_f_dot_v` | Causal backward jerk magnitude | $\text{m/s}^3$ | $\|\dot{\mathbf{f}}^v\| = \|\frac{\mathbf{f}_k - \mathbf{f}_{k-1}}{\Delta t_k}\|$ |
| **8** | `norm_omega_v` | Angular velocity vector magnitude | $\text{rad/s}$ | $\|\boldsymbol{\omega}^v\| = \sqrt{(\omega_x^v)^2 + (\omega_y^v)^2 + (\omega_z^v)^2}$ |

### 3. Why Gravity is Kept Intact in the ML Features
In classical strapdown navigation (Phase 4 and 5), gravity is subtracted inside the strapdown mechanization step. **In our ML features, gravity-containing vehicle-frame specific force is intentionally preserved as an input**.
- **Why?** In dynamic driving, gravity projects across the vehicle axes as vehicle tilt varies ($f_x^v = -g \sin(\text{pitch})$). Keeping specific force intact retains this physical signal directly in the input tensors without premature subtraction or unvalidated compensation.

### 4. Causal Backward Jerk ($\|\dot{\mathbf{f}}^v\|$)
- **Simple Definition**: Jerk is the time-derivative of acceleration ($\text{m/s}^3$).
- **In COMPASS**: Computed using a strictly backward finite difference:
  $$\dot{\mathbf{f}}[k] = \frac{\mathbf{f}[k] - \mathbf{f}[k-1]}{\Delta t_k}$$
  For the very first sample ($k=0$), jerk is set to $0.0$.
- **Why We Care**: Central differences ($\frac{\mathbf{f}[k+1] - \mathbf{f}[k-1]}{2\Delta t}$) look into the future, which is illegal in real-time robotics. Backward differencing uses only the past. High-frequency jerk is a direct physical signature of road roughness and vehicle chassis vibration, which correlates strongly with rolling speed.

### 5. Stride and Inference Cadence
- **Simple Definition**: How many samples we step forward between successive sliding windows.
- **In COMPASS**: We use a stride of **5 samples** ($0.5\,\text{seconds}$ at $10\,\text{Hz}$).
- **Why We Care**: A stride of $0.5\,\text{s}$ means adjacent windows overlap by $1.5\,\text{s}$ ($75\%$). This provides rich training augmentation for the neural network while defining a natural $2\,\text{Hz}$ inference update cadence for the real-time vehicle system.

### 6. Mathematical Causality Invariant
- **Simple Definition**: An algorithm is causal if its output at time $T$ depends only on inputs from times $t \le T$, with zero access to future time $t > T$.
- **In COMPASS**: Every window is aligned such that the window **ends** at the decision epoch $T$:
  $$\text{Window}(T) = [T - 1.9\,\text{s}, \dots, T]$$
  The target speed label is strictly the vehicle speed at the exact same instant:
  $$\text{label} = v(T)$$
- **Why We Care**: In academic literature, researchers often center their windows ($[T - 1.0\,\text{s}, \dots, T + 1.0\,\text{s}]$) to predict speed at $T$. In an autonomous car driving at $100\,\text{km/h}$, you cannot wait $1.0\,\text{second}$ into the future before knowing how fast you are currently driving! COMPASS strictly enforces zero-lookahead causality.

---

## 3. Data Leakage Prevention: Driver & File-Level Splitting

In standard computer vision (like classifying photos of dogs and cats), researchers often shuffle images randomly into train and test sets. **Doing this on driving time-series data is scientific malpractice**.

### The Mechanism of Time-Series Leakage
If you extract 2-second windows with a 0.5-second stride from a single drive, window $k$ (time $10.0\,\text{s}$ to $12.0\,\text{s}$) and window $k+1$ (time $10.5\,\text{s}$ to $12.5\,\text{s}$) share $1.5\,\text{seconds}$ of identical road bumps, engine vibrations, and vehicle trajectory.
If window $k$ goes to the Training set and window $k+1$ goes to the Test set, the neural network does not learn how to estimate speed from inertial physics; it simply memorizes the specific bumpy curve of that road segment. When tested on a new road, its performance collapses.

### The COMPASS Partitioning Policy
To enforce strict split independence and eliminate spatial-temporal correlation between subsets, we partitioned the dataset strictly at the **DRIVER and FILE boundary**:

```
IO-VNBD Dataset (72 Trips, 144 Files, 59.48 Hours)
│
├── TRAIN SPLIT: Driver E (64 Trips, 128 Files, ~32.5 Hours)
│   └── 233,830 Windows (226,928 Valid)
│   └── Used strictly to train neural network weights
│
├── VALIDATION SPLIT: Driver B (1 Trip, 2 Files, ~5.9 Hours)
│   └── 42,388 Windows (21,080 Valid)
│   └── Used for hyperparameter tuning & early stopping
│
├── TEST SPLIT: Driver A (6 Trips, 12 Files, ~17.2 Hours)
│   └── 123,496 Windows (123,464 Valid)
│   └── Held out completely; includes Highway Trip S1
│
└── EXCLUDED: Driver D (1 Trip, 2 Files, ~3.9 Hours)
    └── Trip Y1 (Excluded in Phase 2 due to -638.5s clock drift)
```

### Set-Theoretic Leakage Prevention Invariants
Our automated audit suite enforces six set-theoretic invariants:
1. $\text{Files}(\text{Train}) \cap \text{Files}(\text{Val}) = \emptyset$
2. $\text{Files}(\text{Train}) \cap \text{Files}(\text{Test}) = \emptyset$
3. $\text{Files}(\text{Val}) \cap \text{Files}(\text{Test}) = \emptyset$
4. $\text{Drivers}(\text{Train}) \cap \text{Drivers}(\text{Val}) = \emptyset$
5. $\text{Drivers}(\text{Train}) \cap \text{Drivers}(\text{Test}) = \emptyset$
6. $\text{Drivers}(\text{Val}) \cap \text{Drivers}(\text{Test}) = \emptyset$

Driver A (who drove the famous S1 highway trip we used for our Phase 4 and Phase 5 baselines) is held out exclusively for final evaluation. The neural networks will never see a single millisecond of Driver A's data during training!

---

## 4. Training-Only Feature Normalization

Neural networks train best when input features have zero mean ($\mu = 0$) and unit variance ($\sigma = 1$). However, calculating these statistics across the whole dataset causes **normalization leakage**.

### The Normalization Formulation
Normalization statistics are computed **STRICTLY AND EXCLUSIVELY ACROSS VALID TRAINING WINDOWS** ($226,928 \text{ windows} \times 20 \text{ timesteps} = 4,538,560 \text{ observations}$):
$$\mu_c = \frac{1}{N_{\text{obs}}} \sum_{m=1}^{M_{\text{train}}} \sum_{t=1}^{20} X_{m, t, c}$$
$$\sigma_c = \sqrt{\frac{1}{N_{\text{obs}} - 1} \sum_{m=1}^{M_{\text{train}}} \sum_{t=1}^{20} (X_{m, t, c} - \mu_c)^2}$$

#### Bessel's Correction (`ddof=1`)
In `ml/data/normalization.py`, we strictly use Bessel's correction (`ddof=1` in `np.std(..., ddof=1)`), dividing by $N_{\text{obs}} - 1$ rather than $N_{\text{obs}}$. This provides an unbiased sample standard deviation estimator across the 4.5 million training observations.

### Exact Empirical Training Statistics (from `normalization.json`)

| Channel | Name | Training Mean ($\mu$) | Training Std ($\sigma$, ddof=1) | Physical Meaning |
|---|---|---|---|---|
| **0** | `f_x_v` | $-0.069204\,\text{m/s}^2$ | $1.378226\,\text{m/s}^2$ | Long-term vehicle acceleration/braking |
| **1** | `f_y_v` | $+0.046116\,\text{m/s}^2$ | $1.295886\,\text{m/s}^2$ | Lateral cornering acceleration |
| **2** | `f_z_v` | $+9.838189\,\text{m/s}^2$ | $0.565357\,\text{m/s}^2$ | Vertical specific force (Earth gravity + bumps) |
| **3** | `omega_x_v` | $+0.000067\,\text{rad/s}$ | $0.119624\,\text{rad/s}$ | Roll rate |
| **4** | `omega_y_v` | $+0.003219\,\text{rad/s}$ | $0.247268\,\text{rad/s}$ | Pitch rate (road gradient changes) |
| **5** | `omega_z_v` | $+0.000009\,\text{rad/s}$ | $0.121069\,\text{rad/s}$ | Yaw rate (turning left/right) |
| **6** | `norm_f_v` | $+10.015576\,\text{m/s}^2$ | $0.619333\,\text{m/s}^2$ | Total acceleration vector magnitude |
| **7** | `norm_f_dot_v` | $+9.446759\,\text{m/s}^3$ | $8.807672\,\text{m/s}^3$ | High-frequency chassis jerk |
| **8** | `norm_omega_v`| $+0.221860\,\text{rad/s}$ | $0.202232\,\text{rad/s}$ | Total angular turn rate |

Notice Channel 2: the training mean is $+9.838\,\text{m/s}^2$, perfectly matching Earth's gravitational acceleration!

When transforming Validation and Test datasets:
$$X_{\text{val, norm}} = \frac{X_{\text{val}} - \mu_{\text{train}}}{\sigma_{\text{train}}}, \quad X_{\text{test, norm}} = \frac{X_{\text{test}} - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
Validation and test data never contribute to the calculation of $\mu$ or $\sigma$.

---

## 5. Causal Speed Labels & The Hard-Boundary Median Filter

The target label for VelocityNet is the vehicle's true forward speed in $\text{m/s}$ taken from the high-precision Racelogic VBOX RTK reference:
$$\text{target} = v(T)$$

### The Need for Smoothing
While Racelogic VBOX is highly accurate, CAN bus wheel-speed counters and GPS Doppler receivers have minor discretization steps and quantization jitter. To provide a clean target for neural network regression, we apply a short 3-sample median filter ($W = 3$ samples = $300\,\text{ms}$ at $10\,\text{Hz}$).

### The Hard-Boundary Invariant
Standard filtering libraries (like SciPy's `medfilt`) smooth across the entire array, including across gaps where data was missing, corrupted, or invalid. That causes severe data contamination.

In `ml/data/velocitynet_labels.py`, we built `causal_median_filter_1d` with **hard invalid boundaries**:
1. **Contiguous Valid Run Isolation**: The filter processes contiguous valid sample runs independently.
2. **Hard Boundary Reset**: If sample $k$ is invalid (unvalidated by upstream quality taggers, non-finite, or negative), the filter immediately sets `filtered[k] = np.nan` and resets the valid run length to $0$.
3. **Zero Cross-Gap Bridging**: When a new valid segment starts at $k+1$, smoothing uses only samples from index $k+1$ onward:
   $$\text{eff\_len} = \min(W, \text{run\_length}_k), \quad v_{\text{smoothed}}[k] = \text{median}(v[k - \text{eff\_len} + 1 : k + 1])$$
   Samples before the gap **never** influence any sample after the gap.
4. **Truthful Invalid Preservation**: Windows that end on an invalid sample store `np.nan` in `y_speed` and `y_speed_raw`. They are **never** silently replaced with fake values like $0.0\,\text{m/s}$.
5. **Dual Provenance**: Both smoothed targets (`y_speed`) and raw reference speeds (`y_speed_raw`) are saved side-by-side in the archive.

---

## 6. Shared BiasNet Plumbing & Outage Disclosure

BiasNet (Phase 8) is designed to estimate IMU biases during dynamic driving and GNSS outages.
In `ml/data/exclusion_rules.py`, we created the shared plumbing function `evaluate_biasnet_eligibility(...)` to determine which feature windows are eligible for training BiasNet. Here, "eligible" specifically denotes that a window satisfies all programmatic dataset and exclusion criteria (valid timestamps, finite inertial readings, and absence of known outage exclusions); it is an explicit pipeline eligibility filter, not an infallible claim of perfect physical ground truth.

### Scientific Honesty Regarding Real Outages
As documented in our Phase 0 audit, the IO-VNBD dataset was collected during open-sky driving in the UK and **contains no pre-tagged real GNSS outage CSV files**.

In strict adherence to scientific integrity:
- We **did not invent or fabricate fake outage metadata** to pretend real outages existed.
- All 371,472 valid windows were marked as `ELIGIBLE` with reason `ELIGIBLE` under default open-sky conditions.
- We wrote regression tests (`test_outage_metadata_none_invents_no_fake_exclusions` and `test_real_outage_overlap_excluded`) proving that when future benchmark scripts provide simulated or real tunnel outage timestamps, the plumbing automatically flags overlapping windows as excluded.

---

## 7. Code Architecture & Implementation Walkthrough

```
ml/data/
├── __init__.py                # Package exports
├── resample.py               # Deterministic 10 Hz resampling / decimation engine
├── features.py               # 9-channel feature computation (intact gravity, causal jerk)
├── windowing.py              # Strictly causal sliding window extractor (20 samples, stride 5)
├── split.py                  # Driver/file isolation & set-theoretic partitioner
├── normalization.py          # Training-only normalizer (ddof=1, zero leakage)
├── velocitynet_labels.py     # Causal window-end speed labeler with hard-boundary median filter
├── exclusion_rules.py        # BiasNet outage-overlap plumbing
└── builder.py                # End-to-end dataset builder script & serialization pipeline
```

### Complete Data Flow from Phase 2 to Phase 6 Artifacts

```
[Phase 2 Synchronized Trip (.npz)]
               │
               ▼
[Phase 3 PreprocessingPipeline]
  - Stationary gyro bias subtracted
  - Mounting tilt rotation R_b^v applied (phone -> vehicle FLU)
  - Dual-stage median + Butterworth filtering
               │
               ▼
[resample_to_canonical_10hz]
  - Verifies 10 Hz cadence (Δt ≈ 100 ms)
  - Synchronizes auxiliary reference speed
               │
               ▼
[compute_canonical_features]
  - Arranges channels: [fx, fy, fz, wx, wy, wz, ||f||, ||f_dot||, ||w||]
  - Computes causal backward jerk: (f[k] - f[k-1]) / Δt
  - Gravity preserved intact in fz
               │
               ▼
[extract_causal_windows]
  - Sliding window: length = 20 samples (2.0s), stride = 5 samples (0.5s)
  - Extracts 3D tensors: (M, 20, 9)
               │
               ▼
[create_driver_file_split]
  - Partitions by driver: Driver E (Train), Driver B (Val), Driver A (Test)
               │
               ▼
[FeatureNormalizer.fit (TRAIN ONLY)]
  - Fits μ and σ (ddof=1) strictly on 226,928 Train windows
  - Standardizes Train, Val, and Test tensors
               │
               ▼
[extract_velocitynet_labels]
  - Extracts reference speed at window end: target = v(T)
  - Applies 3-sample causal median filter with hard invalid boundaries
               │
               ▼
[Versioned Disk Serialization]
  - train.npz (~158 MB)
  - validation.npz (~29 MB)
  - test.npz (~84 MB)
  - normalization.json (1.1 KB)
  - model_config.json (1.2 KB)
  - dataset_manifest.json (1.8 KB)
```

---

## 8. Empirical Dataset Numbers & Validation Statistics

The complete dataset build was executed across the entire synchronized archive:

### 8.1 Split Partitioning & Window Counts

| Split | Assigned Driver | Trips | Synchronized Files | Total Windows | Valid Windows | Duration | Role in Project |
|---|---|---|---|---|---|---|---|
| **TRAIN** | **Driver E** | 64 | 128 | **233,830** | **226,928** | ~32.5 h | Model training |
| **VALIDATION** | **Driver B** | 1 | 2 | **42,388** | **21,080** | ~5.9 h | Hyperparameter tuning / checkpoint selection |
| **TEST** | **Driver A** | 6 | 12 | **123,496** | **123,464** | ~17.2 h | Final held-out evaluation (includes S1 benchmark) |
| **EXCLUDED** | **Driver D** | 1 | 2 | 0 | 0 | ~3.9 h | Excluded in Phase 2 for clock drift (Trip Y1) |
| **TOTAL** | **4 Drivers** | **72** | **144** | **399,714** | **371,472** | **59.48 h** | Full IO-VNBD synchronized archive |

### 8.2 Label Speed Distribution Across Splits (Valid Windows)

| Split | Valid Windows | Mean Speed | Std Speed | Median Speed | Min Speed | Max Speed | 95th Percentile |
|---|---|---|---|---|---|---|---|
| **Train** | 226,928 | $15.57\,\text{m/s}$ ($56.1\,\text{km/h}$) | $9.29\,\text{m/s}$ | $15.96\,\text{m/s}$ | $0.0\,\text{m/s}$ | $36.58\,\text{m/s}$ ($131.7\,\text{km/h}$) | $29.23\,\text{m/s}$ |
| **Val** | 21,080 | $10.44\,\text{m/s}$ ($37.6\,\text{km/h}$) | $5.77\,\text{m/s}$ | $10.95\,\text{m/s}$ | $0.0\,\text{m/s}$ | $27.95\,\text{m/s}$ ($100.6\,\text{km/h}$) | $20.41\,\text{m/s}$ |
| **Test** | 123,464 | $8.91\,\text{m/s}$ ($32.1\,\text{km/h}$) | $6.50\,\text{m/s}$ | $8.46\,\text{m/s}$ | $0.0\,\text{m/s}$ | $32.50\,\text{m/s}$ ($117.0\,\text{km/h}$) | $21.61\,\text{m/s}$ |

### 8.3 Reproducibility Verification
The entire pipeline was run twice from clean cold processes. The generated manifests, JSON normalization parameters, and 399,714 array elements matched **bit-for-bit** across all files.

---

## 9. Verification & Unit Testing

The Phase 6 test suite adds 47 rigorous tests (41 unit tests and 6 end-to-end integration tests), bringing the project total to **249 passed tests**.

### Key Tests Explained
1. **Causality Invariant (`test_windowing_causality.py`)**:
   - **Scenario**: Modifies or corrupts samples at future timestamps $t > T$.
   - **Verification**: Asserts that the generated $(20, 9)$ feature tensor ending at $T$ is 100% bit-identical.
   - **Result**: Proves zero future lookahead.
2. **Leakage Audit Invariant (`test_leakage_audit.py`)**:
   - **Scenario**: Checks driver IDs and file IDs across Train, Validation, and Test sets.
   - **Verification**: Enforces that all pairwise intersections are empty ($\emptyset$).
3. **Train-Only Normalization Invariant (`test_normalization_train_only.py`)**:
   - **Scenario**: Injects massive artificial spikes ($10^6$) into Validation and Test tensors.
   - **Verification**: Confirms that the fitted $\mu$ and $\sigma$ remain bit-identical to machine precision.
4. **Hard Invalid Boundary Filter (`test_causal_median_hard_boundaries`)**:
   - **Scenario**: Inserts an invalid $\text{NaN}$ gap between two valid segments.
   - **Verification**: Asserts that samples before the gap do not smooth into samples after the gap, and invalid epochs store `np.nan` rather than $0.0$.

---

## 10. Summary & Lessons Learned

### Final Result
Phase 6 established a fully leakage-audited machine learning dataset engineering pipeline. It processed 59.48 hours of multi-sensor vehicle telemetry into **371,472 valid $(20, 9)$ standardized feature tensors** with causal ground-truth speed labels, strictly partitioned across drivers.

### What We Learned
- **Never shuffle time-series data**: Time-series windows overlap; splitting at the driver/file boundary is the only way to prevent severe data leakage.
- **Normalize strictly on training data**: Using full-dataset statistics leaks test distribution data into model training.
- **Respect causality in robotics**: Centered windows and forward finite differences look into the future and cannot be deployed on a real vehicle. Backward differences and window-end labels preserve physical causality.

### Important Limitations
- **No Neural Network Training Yet**: Phase 6 constructs the dataset. It does not train VelocityNet or BiasNet (that belongs to Phases 7 and 8).
- **Outage Metadata**: The IO-VNBD dataset does not contain real pre-tagged GPS blackout periods. Real outage evaluations in future phases will require simulated outage masks or supplementary tunnel datasets.

### What Phase 6 Handed to Phase 7
Phase 6 produced the complete, versioned ML artifacts:
- `train.npz` (233,830 windows, 226,928 valid)
- `validation.npz` (42,388 windows, 21,080 valid)
- `test.npz` (123,496 windows, 123,464 valid)
- `normalization.json` & `model_config.json`
Phase 7 can now immediately load these tensors to train the authoritative 2-layer GRU VelocityNet architecture (`GRU(9→64, 2L) → Dense(64→32, ReLU) → Dense(32→2)`)!

### What I Should Remember
- Feature tensors are shaped $\mathbf{X} \in \mathbb{R}^{20 \times 9}$ ($2.0\,\text{seconds}$ at $10\,\text{Hz}$, stride $0.5\,\text{s}$).
- The 9 channels are: $[f_x^v, f_y^v, f_z^v, \omega_x^v, \omega_y^v, \omega_z^v, \|\mathbf{f}^v\|, \|\dot{\mathbf{f}}^v\|, \|\boldsymbol{\omega}^v\|]$.
- Gravity is **NOT** subtracted from ML features; gravity-containing vehicle-frame specific force is intentionally preserved as an input.
- Normalization parameters are fitted **exclusively on the training split** using Bessel's correction (`ddof=1`).
- Splitting is performed at the **Driver level**: Driver E (Train), Driver B (Val), Driver A (Test).
- Speed labels represent ground-truth velocity at the exact **window end** $v(T)$, smoothed causally with hard invalid boundaries.

---

## 11. FINAL PROJECT-LEVEL RECAP: THE COMPLETE STORY FROM PHASE 0 TO PHASE 6

This section provides the unified, end-to-end technical story of what we have built in C.O.M.P.A.S.S. from the very beginning up to this point.

```
+-------------------------------------------------------------------------------+
|                        THE COMPASS PROGRESSION STORY                          |
+-------------------------------------------------------------------------------+
| Phase 0: DISCOVERY     --> Audited 360 raw files; found Latin-1 headers,      |
|                            non-monotonic time, and Y1 clock drift.            |
| Phase 1: FOUNDATION    --> Defined schemas, 4 coordinate frames, bitmasks,    |
|                            and the 3-state GNSS FSM.                          |
| Phase 2: PIPELINE      --> Ingested & synchronized 2.14M rows to 10 Hz master |
|                            nanosecond grid; hashed cache (SHA-256).           |
| Phase 3: PREPROCESSING --> Calibrated gyro bias (13.1x reduction), leveled   |
|                            mounting tilt (R_b^v), and filtered vibrations.    |
| Phase 4: PROPAGATION   --> Built strapdown INS; measured open-loop drift:     |
|                            3,249.32 m in 60 seconds (cubic gravity leakage).  |
| Phase 5: ESTIMATION    --> Built 15-state ESKF + GNSS + ZUPT; reduced drift   |
|                            to 5.27 m (99.84% error reduction).                |
| Phase 6: ML DATASET    --> Built 371,472 causal (20, 9) feature tensors       |
|                            enforcing leakage-prevention invariants.           |
+-------------------------------------------------------------------------------+
```

### 1. The Original Problem
Problem Statement 26168 from ISRO asks for a **Cognitive Off-grid Machine-learning Positioning And Sensor System (C.O.M.P.A.S.S.)**. Ground vehicles operating off-grid (in remote border regions, subterranean tunnels, urban canyons, or under electronic warfare jamming) lose GNSS positioning. Dead reckoning with consumer-grade smartphone IMUs diverges catastrophically. The project's goal is to fuse classical Newtonian mechanics, optimal estimation (Kalman filtering), and deep learning to achieve robust, drift-resilient positioning when satellite navigation fails.

### 2. What We Discovered First (Phase 0)
Before writing any algorithms, we conducted an exhaustive forensic audit of the **IO-VNBD** dataset (360 files, 72 trips across Drivers A, B, D, and E). We discovered:
- The data files were encoded in Windows `Latin-1`, not UTF-8.
- Smartphone IMU clocks had non-monotonic time jumps and jitter.
- The Racelogic VBOX hardware and phone sensors logged at different sampling rates.
- Trip Y1 suffered from severe non-linear hardware clock drift ($-638.5\,\text{seconds}$), requiring exclusion.
- The dataset contained zero pre-tagged real outage CSV files, establishing our commitment to scientific honesty.

### 3. The Software Foundation (Phase 1)
To ensure long-term architectural stability, we established strict software contracts:
- Defined immutable dataclass schemas for IMU, GNSS, navigation states, and ML configs.
- Rigorously defined 4 coordinate frames: Sensor body ($b$), Vehicle FLU ($v$), Local ENU navigation ($n$), and Earth ECEF ($e$).
- Implemented non-destructive bitmask quality flags (`FLAG_OK`, `FLAG_NAN`, `FLAG_TIMESTAMP_OUT_OF_ORDER`, etc.).
- Designed the 3-state GNSS Finite State Machine (`GNSS_AIDED`, `DR_ONLY`, `REACQUIRING`) with a continuous $[0, 1]$ trust score.

### 4. Raw Data to Usable Data (Phase 2)
In Phase 2, we built the automated ingestion and synchronization pipeline:
- Established a master nanosecond time grid using the smartphone's steady clock.
- Implemented type-aware interpolation: linear kinematics for accelerations, circular unwrapped $\text{atan2}$ interpolation for headings, and nearest-neighbor for CAN gear states.
- Audited 2,141,490 synchronized rows (99.49% validated).
- Serialized the synchronized data into compressed `.npz` archives and locked them with an immutable SHA-256 digest (`7a267e112af4193dcadb8800bd9cf4ed8abf7249a1ad53883bbd2cfd06200fa7`).

### 5. Calibrated & Aligned Sensors (Phase 3)
In Phase 3, we turned raw sensor readings into physically aligned vehicle forces:
- Estimated stationary gyroscope bias from initial resting intervals, cutting angular drift by **13.1x**.
- Honestly disclosed that accelerometer bias is unobservable from a single static pose, setting a nominal prior $\mathbf{b}_a = [0, 0, 0]^T$.
- Implemented the Rodrigues mounting tilt rotation matrix $R_b^v$, aligning the phone's tilted physical frame with the car's Forward-Lateral-Up axes.
- Disclosed that mounting yaw is unobservable without external heading fixes.
- Built a dual-stage denoising filter (3-sample median + 4th-order zero-phase Butterworth, $f_c = 3.0\,\text{Hz}$), attenuating structural engine vibrations by $53\%$.

### 6. The Physics Engine & Open-Loop Divergence (Phase 4)
In Phase 4, we built the classical strapdown Inertial Navigation System (INS):
- Implemented Hamilton quaternion kinematics: $\mathbf{q}[k+1] = \mathbf{q}[k] \otimes \Delta \mathbf{q}(\boldsymbol{\omega}^v \Delta t)$.
- Added downward gravity in local ENU: $\mathbf{a}_{\text{true}}^n = R_v^n \mathbf{f}_m^v + \mathbf{g}^n$.
- Executed constant-acceleration discrete kinematic double integration to calculate coordinate velocity and 3D position.
- Fixed the VBOX altitude unit scale factor ($1/1000.0$) without mutating the frozen Phase 2 cache.
- Benchmarked open-loop dead reckoning over 60.0 seconds of highway driving on Trip S1:
  - Error at 5s: $4.83\,\text{m}$
  - Error at 15s: $100.64\,\text{m}$
  - Error at 60s: **$3,249.32\,\text{meters}$** ($425\%$ of distance traveled).
- Identified **cubic gravity leakage** ($p(t) \approx \frac{1}{6} g b_g t^3$) as a primary theoretical error mechanism contributing to this rapid divergence.

### 7. Closed-Loop Estimation with ESKF (Phase 5)
In Phase 5, we solved the divergence problem using the Error-State Kalman Filter:
- Maintained a 16-state non-linear nominal kinematic state and a 15-state linear error state ($P \in \mathbb{R}^{15 \times 15}$).
- Implemented right-multiplicative body-frame attitude error: $\mathbf{q}_{\text{true}} = \mathbf{q}_{\text{nom}} \otimes \delta \mathbf{q}(\delta \boldsymbol{\theta})$.
- Derived analytical discrete state transition $F_d$ (verified against numerical finite differences to $3.8 \times 10^{-6}$) and continuous-to-discrete process noise $Q_d$.
- Enforced Joseph-form covariance updates: $P = (I - KH)P(I - KH)^T + KRK^T$.
- Derived and implemented the authoritative error-state covariance reset transformation: $P^+ = J_{\text{reset}} P_{\text{updated}} J_{\text{reset}}^T$.
- Fused 3D GNSS position, 2D horizontal course velocity ($v_E, v_N$), and zero-ML standstill ZUPT protected by Chi-Square innovation gating.
- **Empirical Result on Trip S1**: Reduced 60-second horizontal position error from **$3,249.32\,\text{m}$ down to $5.27\,\text{m}$** ($99.84\%$ reduction).

### 8. The ML Data Pipeline (Phase 6)
In Phase 6, we paved the highway for machine learning aiding during GNSS blackouts:
- Defined the canonical $(20, 9)$ feature tensor spanning $2.0\,\text{seconds}$ of vehicle-frame motion at $10\,\text{Hz}$ (stride $0.5\,\text{s}$).
- Preserved physical gravity intact in specific force inputs.
- Implemented strictly causal backward jerk: $\|\dot{\mathbf{f}}^v\| = \|\frac{\mathbf{f}_k - \mathbf{f}_{k-1}}{\Delta t_k}\|$.
- Prevented data leakage through strict driver-level partitioning: Driver E (Train), Driver B (Val), Driver A (Test).
- Computed normalization parameters strictly on the training set using Bessel's correction (`ddof=1`).
- Extracted ground-truth speed labels at the exact window end $v(T)$ using a causal 3-sample median filter with hard invalid boundaries.
- Serialized 371,472 valid feature tensors and labels across `train.npz`, `validation.npz`, and `test.npz`.

### 9. Where COMPASS Stands Today & What Lies Ahead
With Phases 0 through 6 complete, tested, and frozen:
- **We have a tested classical navigation and estimation backbone**: Strapdown INS + 15-state ESKF + GNSS + ZUPT.
- **We have a leakage-audited, standardized ML dataset ready on disk**.
- **What Remains for Future Phases (Executing the 20-Phase Roadmap of `FINAL_IMPLEMENTATION_PLAN_SIH26168.md`)**:
  - **Phase 7**: VelocityNet model architecture (2-layer GRU), training, validation, and ONNX export.
  - **Phase 8**: BiasNet model architecture (2-layer GRU) and residual IMU bias training.
  - **Phase 9**: ML $\to$ ESKF integration and full ML-augmented offline replay.
  - **Phase 10**: GNSS trust scoring, outage detection FSM, and reacquisition recovery.
  - **Phase 11**: Non-Holonomic Constraints (NHC) and gated ZUPT integration.
  - **Phase 12**: Downstream OpenStreetMap (OSM) + HMM map-matching.
  - **Phase 13–19**: Full 3-axis ablation ladder, core packaging, Edge/Android adapters, cross-platform parity, field collection, and SIH demo readiness.

COMPASS has evolved from raw, uncalibrated CSV files into a mathematically validated, empirically benchmarked, and machine-learning-ready navigation system.
