# COMPASS Phase 6 — Implementation Build Walkthrough

**Project**: C.O.M.P.A.S.S. (Cognitive Off-grid Machine-learning Positioning And Sensor System)  
**Problem Statement**: SIH 2026 Problem Statement 26168 — ISRO  
**Document Type**: Step-by-Step Implementation Build Diary

---

## Overview: What We Were Trying to Do

Phase 6 was the **ML data engineering** phase: transforming 60 hours of synchronized vehicle sensor recordings into a clean, leakage-audited, reproducible machine learning dataset ready for training VelocityNet and BiasNet.

The core engineering challenge was correctness under many subtle failure modes:
1. **Causal data leakage**: accidentally including future sensor samples in the input window.
2. **Normalization leakage**: computing normalization statistics using test or validation data.
3. **Driver leakage**: splitting training and test windows from the same driver's trips.
4. **Label jitter**: raw VBOX speed has quantization noise at low speeds.
5. **Invalid epoch leakage**: smoothing a speed label across a sensor dropout boundary.

Each of these issues is silent — the code runs without errors, but the resulting model will fail in deployment.

---

## Step 1: Designing the Module Structure

### ML Package Layout
```
ml/
├── data/
│   ├── builder.py          ← main pipeline orchestrator
│   ├── exclusion_rules.py  ← BiasNet eligibility logic
│   ├── features.py         ← 9-channel feature computation
│   ├── normalization.py    ← training-only normalizer
│   ├── resample.py         ← canonical 10 Hz decimator
│   ├── split.py            ← driver/file split assignment
│   ├── velocitynet_labels.py ← causal speed labels
│   └── windowing.py        ← causal 20-sample window extractor
```

Each module was designed to be independently unit-testable without requiring the full dataset.

---

## Step 2: The 9-Channel Feature Engine (`features.py`)

### Feature Computation
Given preprocessed vehicle-frame measurements $\mathbf{f}^v \in \mathbb{R}^{N \times 3}$ and $\boldsymbol{\omega}^v \in \mathbb{R}^{N \times 3}$:

```python
def compute_canonical_features(timestamps_ns, f_m_v, omega_m_v) -> np.ndarray:
    """Compute 9-channel feature array from vehicle-frame IMU."""
    N = len(timestamps_ns)
    features = np.zeros((N, 9), dtype=np.float64)
    
    # Channels 0-2: direct specific force components
    features[:, 0:3] = f_m_v  # [fx, fy, fz] in m/s^2
    
    # Channels 3-5: direct angular velocity components
    features[:, 3:6] = omega_m_v  # [omega_x, omega_y, omega_z] in rad/s
    
    # Channel 6: specific force norm
    features[:, 6] = np.linalg.norm(f_m_v, axis=1)
    
    # Channel 7: specific force jerk norm (causal backward difference)
    # f_dot[k] = (f[k] - f[k-1]) / dt_k
    # f_dot[0] = 0.0  (hard boundary, no previous sample)
    dt_s = np.diff(timestamps_ns) / 1e9  # convert ns to seconds
    df = np.diff(f_m_v, axis=0)
    f_dot = df / dt_s[:, np.newaxis]  # (N-1, 3)
    f_dot_full = np.vstack([[0.0, 0.0, 0.0], f_dot])  # prepend zero for k=0
    features[:, 7] = np.linalg.norm(f_dot_full, axis=1)
    
    # Channel 8: angular velocity norm
    features[:, 8] = np.linalg.norm(omega_m_v, axis=1)
    
    return features
```

### Gravity Retention
Notice: the raw specific force `f_m_v` includes gravity's reaction force (~9.81 m/s² in the vertical). This was deliberately **not** subtracted; gravity-containing vehicle-frame specific force is intentionally preserved as an input.

### Jerk Hard Boundary
The jerk computation at sample $k=0$ sets `f_dot = [0, 0, 0]`. This prevents backward-looking across a trip boundary or a sensor dropout. For a sensor dropout (flagged `FLAG_SENSOR_DROPOUT`), the invalid sample is treated as a hard boundary: the jerk immediately after the gap is zeroed, not computed across the gap.

---

## Step 3: The Causal Window Extractor (`windowing.py`)

### Window Parameters
- **Window size**: 20 samples (2.0 seconds at 10 Hz)
- **Stride**: 5 samples (0.5 seconds at 10 Hz)
- **Window end alignment**: each window ends at its decision epoch $T$

```python
def extract_causal_windows(features, timestamps_ns, is_validated, stride=5):
    """Extract causal 20-sample windows with 5-sample stride."""
    N, C = features.shape
    windows = []
    
    for end_idx in range(19, N, stride):  # need at least 20 samples of history
        start_idx = end_idx - 19
        window_features = features[start_idx:end_idx + 1]  # shape (20, 9)
        window_valid = is_validated[start_idx:end_idx + 1].all()
        windows.append(ExtractedWindow(
            features=window_features,
            end_idx=end_idx,
            start_idx=start_idx,
            timestamp_end_ns=timestamps_ns[end_idx],
            timestamp_start_ns=timestamps_ns[start_idx],
            is_valid=window_valid,
        ))
    return windows
```

**Causality invariant**: `features[start_idx:end_idx+1]` — the last sample in the window is at `end_idx`, which is the current epoch $T$. All samples have indices $\le$ end_idx. Zero samples from $>T$ are ever accessed.

### Valid Window Mask
A window is marked `is_valid=True` only if **all 20 samples** in the window are validated. If any sample is flagged with a non-computable quality flag (NaN, duplicate timestamp), the entire window is marked invalid.

---

## Step 4: The Resampler (`resample.py`)

IO-VNBD data is already at ~10 Hz, so the resampler performs an **identity pass-through** with verification:

```python
def resample_to_canonical_10hz(timestamps_ns, f_m_v, omega_m_v, ...):
    median_dt_ms = np.median(np.diff(timestamps_ns)) / 1e6
    if 90.0 <= median_dt_ms <= 110.0:
        # Already approximately 10 Hz: identity pass-through
        return ResampleResult(timestamps_ns=timestamps_ns, f_m_v=f_m_v, ...)
    else:
        # Higher rate: decimate causally using causal pooling
        # (For future 50/100 Hz external sensors)
        ...
```

For future high-rate sensors ($50–200$ Hz), the resampler pools samples causally within each $100$ ms grid cell: only samples with timestamp $\le t_k$ (the grid epoch) and $> t_{k-1}$ are pooled. This prevents future samples from entering the current epoch's pool.

---

## Step 5: The Driver/File Split (`split.py`)

### The Split Assignment Policy

```python
class DriverFileSplit:
    DRIVER_TO_SPLIT = {
        "A": "test",        # Driver A: 6 trips, ~17.2 h
        "B": "validation",  # Driver B: 1 trip, ~5.9 h
        "D": "excluded",    # Driver D: excluded (clock drift)
        "E": "train",       # Driver E: 64 trips, ~32.5 h
    }
    
    def get_split(self, driver_id: str) -> str:
        return self.DRIVER_TO_SPLIT.get(driver_id, "unknown")
```

The trip metadata in `data/manifests/iovnbd_manifest_v1.csv` contains the driver ID for each file. This is used to assign every file to exactly one split with **zero overlap**.

### Why Driver-Level Split (Not File-Level or Window-Level)?
If you split windows randomly, window $k$ from Trip S1 (in training) and window $k+1$ from Trip S1 (in test) share 15 samples of identical road data. The neural network memorizes the sequence instead of learning physics.

Driver-level split is even stronger: the neural network must generalize across:
- Different vehicle (Driver A uses different mounting positions).
- Different road geometry (Driver A on motorways, Driver E on urban roads).
- Different driving style (Driver A high-speed, Driver E urban stop-and-go).

---

## Step 6: VelocityNet Label Generation (`velocitynet_labels.py`)

### What We Need
For each window ending at epoch $T$, the target speed label is $v(T)$ — the VBOX ground-truth forward speed at that exact moment.

### Why Not Just Use `v_ref_speed_mps[end_idx]` Directly?
The Racelogic VBOX speed is Doppler-derived and very accurate (~0.05 km/h error). But at very low speeds (< 1 km/h), there is visible quantization noise. A 3-sample median filter smooths this.

### Hard-Boundary Causal Median Filter

The critical design decision: the median filter must be **strictly causal** and must **not bridge across invalid sample boundaries**.

```python
def causal_median_filter_1d(signal, valid_mask=None, window_size=3):
    """Causal median filter with hard boundaries at invalid samples."""
    filtered = np.full(n, np.nan)
    run_length = 0
    
    for k in range(n):
        if not is_valid_sample[k]:
            run_length = 0          # hard boundary: reset run
            filtered[k] = np.nan
        else:
            run_length += 1
            eff_len = min(window_size, run_length)
            # Only use samples within the current contiguous valid run
            filtered[k] = np.median(signal[k - eff_len + 1 : k + 1])
    
    return filtered
```

If a gap (sensor dropout) occurs between samples $k-2$ and $k-1$:
- The run resets at $k-2$.
- Sample $k$ only sees `run_length=1`, so `eff_len=1`, and `filtered[k] = signal[k]` (unsmoothed).

**No speed values from before the gap ever influence labels after the gap.**

---

## Step 7: Training-Only Normalization (`normalization.py`)

### The Leakage Invariant
```
NORMALIZATION PARAMETERS MUST BE COMPUTED STRICTLY AND EXCLUSIVELY
FROM THE TRAINING SPLIT.
```

The normalizer is fitted only on training windows:
```python
train_tensors = np.concatenate([trip_windows for trip in train_trips])  # shape (N_train, 20, 9)
normalizer = FeatureNormalizer.fit(train_tensors)

# Apply to all splits
train_normalized = normalizer.transform(train_tensors)
val_normalized = normalizer.transform(val_tensors)   # same parameters!
test_normalized = normalizer.transform(test_tensors)  # same parameters!
```

### Bessel's Correction (ddof=1)
The standard deviation uses `np.std(..., ddof=1)` — Bessel's correction for unbiased sample variance. This was the subject of a bug fix:

**Original bug**: Used `ddof=0` (population std). For very large $N$ (226,928 training windows × 20 samples = 4.5 million observations per channel), the difference between `ddof=0` and `ddof=1` is negligible (~1/4.5M). But the standard was harmonized to `ddof=1` for mathematical correctness and consistency with tests.

### Clamping Near-Zero Stds
Some channels (e.g., `norm_omega_v` during trips with minimal turning) can have very small standard deviation. Division by near-zero std causes numerical explosions. We clamp:
```python
stds = np.where(stds < 1e-6, 1.0, stds)  # use identity scaling for near-constant channels
```

---

## Step 8: BiasNet Exclusion Plumbing (`exclusion_rules.py`)

BiasNet requires dynamic driving conditions with full GNSS to generate training labels (the labels come from ESKF bias estimates). Windows during GNSS outages or stationary periods must be excluded from BiasNet training.

```python
def evaluate_biasnet_eligibility(
    is_stationary: np.ndarray,
    is_gnss_valid: np.ndarray,
    speed_mps: np.ndarray,
    min_speed_mps: float = 2.0,
) -> Tuple[np.ndarray, List[str]]:
    eligible = np.ones(len(is_stationary), dtype=bool)
    reasons = ["OK"] * len(is_stationary)
    
    stat_mask = is_stationary
    eligible[stat_mask] = False
    for i in np.where(stat_mask)[0]:
        reasons[i] = "STATIONARY"
    
    low_speed_mask = speed_mps < min_speed_mps
    eligible[low_speed_mask] = False
    for i in np.where(low_speed_mask)[0]:
        if reasons[i] == "OK":
            reasons[i] = "LOW_SPEED"
    
    return eligible, reasons
```

**Note**: BiasNet labels (the actual ESKF-estimated biases) are **not generated in Phase 6**. Phase 6 only establishes the exclusion plumbing — marking which windows are eligible. The actual label generation optimization belongs to Phase 8.

---

## Step 9: The Build Script (`scripts/build_ml_dataset.py`)

```python
# High-level execution flow
representations = load_all_142_representations_from_cache()  # 142 downstream-ready file representations (71 physical trips)
split_assignments = DriverFileSplit().assign_all(representations)

# Process each representation through the feature + windowing pipeline
all_results = [process_trip_to_windows(r) for r in representations]

# Compute normalization from training trips only
train_windows = concat([r.windows for r in all_results if r.split == "train"])
normalizer = FeatureNormalizer.fit(train_windows)

# Normalize all splits using the same parameters
for result in all_results:
    result.windows_normalized = normalizer.transform(result.windows)

# Save artifacts
save_dataset_manifest(all_results)
save_model_config(normalizer)
save_per_file_npz(all_results)
```

---

## Step 10: Debugging Episodes

**Bug 1: Jerk computed across trip boundary**
The first implementation computed jerk by `np.diff(f_m_v, axis=0)` across the entire trip before dividing by `dt`. If sample $k=0$ was the start of the dataset (no previous sample), `f_dot[0]` was set to the difference between sample 0 and sample −1 of the previous trip in memory. Fixed by always prepending `[0, 0, 0]` for the first jerk sample, making it strictly zero.

**Bug 2: Speed labels not causal — centered windows**
Early code tried `median(speed[end_idx-1], speed[end_idx], speed[end_idx+1])`. The `+1` was future lookahead! Changed to strictly backward: `median(speed[end_idx-2], speed[end_idx-1], speed[end_idx])`.

**Bug 3: Standard deviation normalization used `ddof=0`**
Discovered during audit that `np.std` defaults to `ddof=0` (population standard deviation). Changed to `ddof=1` (sample standard deviation, Bessel's correction) throughout. The numerical impact on the very large training set is negligible, but the test specification and documentation required `ddof=1`.

**Bug 4: Normalization applied before split assignment**
An early version fit the normalizer on all 142 file representations then split. Fixed: split first, fit normalizer on train split only, then transform each split separately.

---

## Final Outputs

| Artifact | Purpose |
|:---|:---|
| `ml/data/features.py` | 9-channel feature computation |
| `ml/data/windowing.py` | Causal 20-sample window extractor |
| `ml/data/velocitynet_labels.py` | Hard-boundary causal median label filter |
| `ml/data/normalization.py` | Training-only standardizer (ddof=1) |
| `ml/data/split.py` | Driver/file level split assignment |
| `ml/data/resample.py` | Canonical 10 Hz decimator |
| `ml/data/exclusion_rules.py` | BiasNet eligibility plumbing |
| `ml/data/builder.py` | Unified pipeline orchestrator |
| `scripts/build_ml_dataset.py` | Build runner |
| `data/ml_dataset_v1/` | Generated dataset artifacts (gitignored) |
| `docs/ml_dataset_v1_report.md` | Full dataset statistics report |
| `tests/unit/test_features.py` | Feature computation tests (4 tests) |
| `tests/unit/test_velocitynet_labels.py` | Causal label and boundary tests (8 tests) |
| `tests/unit/test_windowing_causality.py` | Causality invariant enforcement tests (4 tests) |
| `tests/unit/test_normalization_train_only.py` | Normalization leakage tests (4 tests) |
| `tests/unit/test_split.py` | Split exclusivity tests (6 tests) |
| `tests/unit/test_leakage_audit.py` | Cross-split driver/file disjointness tests (5 tests) |
| `tests/unit/test_resample.py` | Decimation and pass-through tests (5 tests) |
| `tests/unit/test_exclusion_rules.py` | BiasNet eligibility & outage plumbing tests (5 tests) |
| `tests/integration/test_ml_dataset_artifacts.py` | Integration test on full built dataset (6 tests) |

**Final dataset statistics**:
- Total windows: 399,714
- Valid windows: 371,472
- Train: 226,928 valid | Validation: 21,080 valid | Test: 123,464 valid
- 47 Phase 6 tests; all 249 repository tests passed at Phase 6 completion.
