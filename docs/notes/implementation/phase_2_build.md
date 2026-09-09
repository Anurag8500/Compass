# COMPASS Phase 2 — Implementation Build Walkthrough

**Project**: C.O.M.P.A.S.S. (Cognitive Off-grid Machine-learning Positioning And Sensor System)  
**Problem Statement**: SIH 2026 Problem Statement 26168 — ISRO  
**Document Type**: Step-by-Step Implementation Build Diary

---

## Overview: What We Were Trying to Do

Phase 2 was the **offline data factory** — turning raw CSV text into clean NumPy binary arrays. The challenge was aligning two independent sensor streams that used completely different time references, handling sensor data flaws, and doing it reproducibly for all 144 file representations.

The deliverable was a set of cached `.npz` files — one per representation — that downstream phases (Phases 3 through 6) load in milliseconds instead of re-parsing CSVs every time.

---

## Step 1: Designing the CSV Parser (`data/pipeline/parse.py`)

### S-File (Smartphone) Parser
The smartphone CSV columns were discovered in Phase 0. The parser handles:
1. **Latin-1 encoding**: `encoding='iso-8859-1'`
2. **Leading spaces in headers**: `df.columns = df.columns.str.strip()`
3. **Unit conversions**: Speed from km/h → m/s (`* 1/3.6`), altitude in meters.
4. **Timestamp conversion**: `int(ms * 1_000_000)` converting milliseconds to int64 nanoseconds.

```python
# parse.py (conceptual snippet)
df = pd.read_csv(path, encoding='iso-8859-1')
df.columns = df.columns.str.strip()
timestamps_ns = (df['TIME SINCE START (ms)'].values * 1_000_000).astype(np.int64)
accel_raw = df[['ACCELEROMETER X (m/s²)', 'ACCELEROMETER Y (m/s²)', 'ACCELEROMETER Z (m/s²)']].values
```

### V-File (Vehicle VBOX) Parser
Vehicle files also used Latin-1 with distinct column semantics:
1. **Timestamp**: `'Time Since Start of Day (seconds)'` — floating-point seconds since midnight UTC. Converted via `int(round(t_sec * 1_000_000_000))`.
2. **Speed**: `'Velocity (km/hr)'` → m/s.
3. **Altitude**: `'Height (km)'` — the raw column header said km, but empirical terrain analysis showed values were in meters ($92\text{ to }144\text{ m}$). The parser applied `* 1000.0` based on the literal header string, storing values in the range 92,000–144,000 in the cache. Phase 2 cache files are frozen with SHA-256; Phase 4 corrects this with `/ 1000.0` on load.

---

## Step 2: Designing the Quality Tagger (`data/pipeline/quality_tagger.py`)

### The Non-Destructive Contract
The tagger evaluates the parsed arrays and produces a `quality_flags` bitmask array of the same length without deleting rows:

```python
class QualityTagger:
    def tag(self, timestamps_ns, accel, gyro) -> np.ndarray:
        flags = np.zeros(len(timestamps_ns), dtype=np.uint32)
        
        # NaN/Inf check
        nan_mask = ~np.isfinite(accel).all(axis=1) | ~np.isfinite(gyro).all(axis=1)
        flags[nan_mask] |= FLAG_NAN_OR_NONFINITE
        
        # Non-monotonic check
        dt = np.diff(timestamps_ns, prepend=timestamps_ns[0] - 1)
        flags[dt <= 0] |= FLAG_NON_MONOTONIC_TIMESTAMP
        
        # Duplicate check
        flags[dt == 0] |= FLAG_DUPLICATE_TIMESTAMP
        
        # Extreme motion
        accel_norm = np.linalg.norm(accel, axis=1)
        gyro_norm = np.linalg.norm(gyro, axis=1)
        flags[(accel_norm > 39.24) | (gyro_norm > 10.0)] |= FLAG_EXTREME_MOTION
        
        # Sensor dropout (> 300 ms)
        nominal_dt_ns = 100_000_000
        flags[dt > 3 * nominal_dt_ns] |= FLAG_SENSOR_DROPOUT
        
        return flags
```

The validated integration mask:
```python
is_validated = (flags & (FLAG_NAN_OR_NONFINITE | FLAG_INVALID_TIMESTAMP |
                          FLAG_NON_MONOTONIC_TIMESTAMP | FLAG_DUPLICATE_TIMESTAMP)) == 0
```
Extreme motion and sensor dropouts are **kept** in the validated stream.

---

## Step 3: Designing the Synchronizer (`data/pipeline/sync.py`)

### Relative Elapsed Alignment
The phone clock starts at 0 and the VBOX clock starts at seconds past midnight. We align by subtracting each stream's own start time:

```python
t_s_relative = timestamps_s_ns - timestamps_s_ns[0]    # master working grid
t_v_relative = timestamps_v_ns - timestamps_v_ns[0]
```

### Type-Aware Interpolation
- Continuous kinematics (position, speed): linear interpolation.
- Angular headings: circular trigonometric interpolation ($\sin, \cos \to \text{atan2}$).
- Discrete CAN states (gear, handbrake): nearest-neighbor.

### Gap Policy
If the vehicle reference data has a gap $> 1.0\text{ second}$, interpolation for smartphone timestamps falling inside that gap is assigned `NaN`.

---

## Step 4: The Stationary Detector (`data/pipeline/stationary_detect.py`)

```python
def detect_stationary_windows(accel, gyro, min_samples=50):
    accel_norm = np.linalg.norm(accel, axis=1)
    gyro_norm = np.linalg.norm(gyro, axis=1)
    var_accel = pd.Series(accel_norm).rolling(min_samples).var()
    var_gyro = pd.Series(gyro_norm).rolling(min_samples).var()
    stationary = (var_accel < 0.05) & (var_gyro < 0.005)
    return stationary
```
Criteria: 50 consecutive samples, $\text{var}(\|\mathbf{f}\|) < 0.05\,\text{m}^2/\text{s}^4$, $\text{var}(\|\boldsymbol{\omega}\|) < 0.005\,\text{rad}^2/\text{s}^2$.

---

## Step 5: Synthetic Outage Index (`data/pipeline/outage_index.py`)

Because Phase 0 confirmed that no real GPS outage files exist in the archive, we built a synthetic outage generator marked explicitly with `is_synthetic=True`. No fake outage labels were fabricated.

---

## Step 6: Pipeline Execution (`scripts/run_pipeline.py`)

The pipeline runner processed all 144 representations, generating `data/manifests/iovnbd_manifest_v1.csv` and saving compressed `.npz` files in `data/cache/iovnbd/`.

---

## Step 7: Tests (`test_pipeline_quality.py`, `test_pipeline_full_file.py`)

The Phase 2 test suite consists of **46 tests**:
- **`tests/unit/test_pipeline_quality.py`** (45 unit tests): Validates bitmask flag logic, saturation thresholds, non-monotonic row detection, and interpolation algorithms.
- **`tests/integration/test_pipeline_full_file.py`** (1 integration test): Tests full pipeline ingestion, >99.5% reference coverage, and roundtrip cache serialization on real data.

---

## Final Outputs & Verified Numbers

| Artifact | Purpose |
|:---|:---|
| `data/manifests/iovnbd_manifest_v1.csv` | Master catalog of 144 representations |
| `data/cache/iovnbd/*.npz` | 144 compressed binary trip files |
| `docs/data_quality_report.md` | Phase 2 execution report |
| `tests/unit/test_pipeline_quality.py` | 45 quality & interpolation unit tests |
| `tests/integration/test_pipeline_full_file.py` | 1 full-trip integration test |

**Key results**:
- 142 of 144 representations passed sync policy.
- 10,936 rows omitted (0.5107% of 2,141,490 total).
- 34 extreme motion sample occurrences and 6 dropouts 100% preserved.
- Trip Y1 excluded ($-638.51\text{ s}$ clock drift).
- Raw archive size: ~194.2 MB (288 CSV files).
