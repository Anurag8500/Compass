# COMPASS Phase 1 — Implementation Build Walkthrough

**Project**: C.O.M.P.A.S.S. (Cognitive Off-grid Machine-learning Positioning And Sensor System)  
**Problem Statement**: SIH 2026 Problem Statement 26168 — ISRO  
**Document Type**: Step-by-Step Implementation Build Diary

---

## Overview: What We Were Trying to Do

Phase 1 was about building the **contractual foundation** before writing any physics. At this stage, no navigation code existed. The goal was to answer: what are the exact data structures that every other module in the project will pass between each other?

Without this foundation, every module would invent its own dictionary or tuple format. A function expecting `(fx, fy, fz)` might receive `(fx, fz, fy)` from another module that swapped Y and Z. A quaternion could be `[w, x, y, z]` in one place and `[x, y, z, w]` in another.

Phase 1 created **frozen, type-checked Python dataclasses** that enforced correct semantics at construction time.

---

## Step 1: Choosing Python Dataclasses (Frozen)

### Why `@dataclass(frozen=True)`?
We chose `frozen=True` to make all schemas immutable after construction. This prevents a very common class of bug where code modifies a sensor reading in-place, corrupting shared references.

```python
@dataclass(frozen=True)
class RawIMUSample:
    timestamp_ns: int
    accel: Tuple[float, float, float]
    gyro: Tuple[float, float, float]
    quality_flags: int = FLAG_OK
    source: SensorSource = SensorSource.PHONE
    sensor_id: str = "phone_internal"
```

Every field uses basic Python types (`int`, `float`, `Tuple`) to ensure JSON serializability. We deliberately avoided NumPy arrays inside schemas to keep them lightweight and JSON-round-trippable.

---

## Step 2: Designing `imu.py`

### Deciding on Timestamps in Nanoseconds
Phase 0 showed that smartphones record milliseconds, but the Android sensor API internally uses nanoseconds. NumPy's `int64` can represent timestamps up to ~292 years in nanoseconds. We chose nanoseconds as the universal timestamp unit throughout COMPASS.

```python
timestamp_ns: int  # always nanoseconds, always int (never float)
```

A `float` nanosecond timestamp would lose precision due to IEEE 754 floating-point representation limits. `int64` nanoseconds avoids this.

### Designing the Quality Flag Bitmask
We defined 6 mutually compatible power-of-2 bit flags:

```python
FLAG_OK:                  0x00  # no flags set
FLAG_NAN_OR_NONFINITE:    0x01  # payload contains NaN/Inf
FLAG_INVALID_TIMESTAMP:   0x02  # negative or zero timestamp
FLAG_NON_MONOTONIC_TIMESTAMP: 0x04  # t_k <= t_{k-1}
FLAG_DUPLICATE_TIMESTAMP: 0x08  # same timestamp as previous
FLAG_EXTREME_MOTION:      0x10  # ||f|| > 4g or ||omega|| > 10 rad/s
FLAG_SENSOR_DROPOUT:      0x20  # gap > 3x nominal dt
```

The bitmask design allows multiple simultaneous flags:
```python
flags = FLAG_EXTREME_MOTION | FLAG_SENSOR_DROPOUT  # 0x30 — both true
```

### Key Design Decision: Omit vs. Keep
A critical discussion happened here: should extreme-motion samples (`FLAG_EXTREME_MOTION`) be **omitted** (dropped) or **kept**?

We decided to **keep** them. A pothole shock at $5g$ is physically real. If we drop it, the inertial integrator sees a continuous trajectory with a missing bump, which creates a velocity discontinuity. The downstream ESKF's Mahalanobis gating can handle real physical extremes without rejecting them globally. Only non-computable samples (NaN, duplicate timestamps) are omitted.

### The `AlignedIMUSample` Gravity Decision
Early design considered removing gravity from `AlignedIMUSample` in Phase 3. We explicitly decided **not to**. The docstring was written to be completely unambiguous:

```python
"""
Physical gravity is NOT removed here; strapdown INS propagation in ESKF
resolves gravity internally via vehicle attitude R_v^n.
"""
```

This prevents Phase 3 from ever "helpfully" subtracting gravity during alignment.

---

## Step 3: Designing `gnss.py`

### The Trust Score Design
Rather than a binary "GPS valid/invalid" flag, we modeled GNSS reliability as a **continuous** score:

```python
trust_score: float = 1.0  # range [0.0, 1.0]
```

This was a deliberate rejection of an arbitrary 4th GNSS FSM state called `DEGRADED`. By keeping degradation continuous, the Mahalanobis chi-square gate in the ESKF naturally handles it: a measurement from a low-confidence fix has an expanded covariance, allowing the filter to down-weight it statistically.

---

## Step 4: Designing `state.py`

### The Reference Point Invariant
The `NavigationState.reference_point` field was the subject of a specific design note written directly into the class docstring:

```
CRITICAL ARCHITECTURAL INVARIANT:
The reference_point (lat0, lon0) defines the origin of the Cartesian ENU
tangent plane for the ENTIRE navigation session. It is fixed upon session
start and MUST NOT be silently reset or shifted upon every GNSS update.
Resetting the origin during a session would invalidate integrated trajectory
history and corrupt the ESKF error-state covariance P.
```

### State Dimensions: 16 Nominal vs 15 Error Parameters
The nominal kinematic state carries 16 parameters:
- Position in ENU (3)
- Velocity in ENU (3)
- Hamilton quaternion (4)
- Accelerometer bias in vehicle frame (3)
- Gyroscope bias in vehicle frame (3)
$$\text{Total nominal parameters} = 16$$

The error state carries 15 parameters:
- Position error (3)
- Velocity error (3)
- Rotation vector $\delta\boldsymbol{\theta}$ in body frame (3)
- Accelerometer bias error (3)
- Gyroscope bias error (3)
$$\text{Total error parameters} = 15$$

The error covariance $P$ is strictly $15 \times 15$. The `__post_init__` validator enforces this:
```python
if len(self.covariance) != 15:
    raise ValueError(...)
for i, row in enumerate(self.covariance):
    if len(row) != 15:
        raise ValueError(...)
```

---

## Step 5: Designing `ml.py` and `config.py`

### The `MLPrediction` Schema Contract
We enforced that VelocityNet predictions have dimension 1 and BiasNet predictions have dimension 6:
```python
if self.model == MLModelType.VELOCITY_NET and len(self.value) != 1:
    raise ValueError("VELOCITY_NET prediction must have 1 element")
if self.model == MLModelType.BIAS_NET and len(self.value) != 6:
    raise ValueError("BIAS_NET prediction must have 6 elements")
```
This defines the ML output contract; model training occurs in Phase 7 and Phase 8.

### `CANONICAL_CHANNELS` Constant
The channel ordering for all feature windows was locked down as an immutable tuple:
```python
CANONICAL_CHANNELS: Tuple[str, ...] = (
    "f_x_v", "f_y_v", "f_z_v",
    "omega_x_v", "omega_y_v", "omega_z_v",
    "norm_f_v", "norm_f_dot_v", "norm_omega_v",
)
```

---

## Step 6: Designing the Adapter Interface

```python
# navigation/adapters/base.py
class SensorAdapter(ABC):
    @abstractmethod
    def get_raw_samples(self, trip_id: str) -> List[RawIMUSample]: ...

class GNSSAdapter(ABC):
    @abstractmethod
    def get_gnss_fixes(self, trip_id: str) -> List[GNSSSample]: ...
```

These abstract classes define the contract that external data sources implement.

---

## Step 7: Writing the Schema & Frame Tests

Phase 1 delivered exactly **50 tests**:
1. `tests/unit/test_schemas.py` (43 unit tests):
   - Valid construction and default behavior for all schemas.
   - Rejection of invalid dimensions, negative timestamps, and non-15×15 covariances.
   - Quality flag bitmask operations.
   - Real JSON round-trip serialization.
2. `tests/unit/test_frame_conversion.py` (7 unit tests):
   - WGS84 Geodetic to Cartesian ENU tangent plane coordinate conversion.
   - Preserves session origin immutability and right-handed orientation.

---

## Debugging Episodes

**Bug 1: Quaternion norm check was too strict**
Initial validation checked `norm == 1.0` strictly, failing on slight floating-point imprecision. Changed to checking squared norm $> 10^{-12}$, preventing only truly degenerate zero-norm quaternions.

**Bug 2: Deserialization integer-to-float coercion**
When reading from JSON, `0` was parsed as `int`. Added explicit `float()` casts across all `from_dict()` methods.

---

## What Phase 1 Delivered

| Artifact | Path | Purpose |
|:---|:---|:---|
| IMU schemas | `navigation/schemas/imu.py` | Raw and aligned IMU containers |
| GNSS schema | `navigation/schemas/gnss.py` | GNSS fix with trust score |
| Navigation state | `navigation/schemas/state.py` | 16-nominal, 15-error state snapshots |
| ML schemas | `navigation/schemas/ml.py` | Prediction output containers |
| Config schema | `navigation/schemas/config.py` | Feature channel constants |
| Adapter interfaces | `navigation/adapters/base.py` | Hardware abstraction |
| Frame conversions | `navigation/frames/local_geo.py` | WGS84 to local ENU |
| Unit tests | `tests/unit/test_schemas.py` | 43 schema tests |
| Frame tests | `tests/unit/test_frame_conversion.py` | 7 frame tests |

**Total Phase 1 Tests**: Exactly **50 tests** passing.
