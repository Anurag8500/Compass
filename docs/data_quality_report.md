# COMPASS IO-VNBD Data Quality Report (Phase 2)
**Execution Date**: 2026-09-08 15:11:33 UTC  
**Pipeline Version**: Phase 2 (v1.0 - Hardened, Audit-Verified)  
**Manifest Path**: `data/manifests/iovnbd_manifest_v1.csv`  

---

## 1. Executive Summary & Inventory

The COMPASS Phase 2 offline data pipeline ingested, audited, and processed the IO-VNBD synchronized dataset archive. All 144 matched S/V pairs (representing 72 unique physical driving trips across Driver A, Driver B, Driver D, Driver E) were parsed and quality-tagged. Genuine timestamp-based interpolation was performed for all pairs. Under the configured synchronization acceptance policy, 142 pairs passed all criteria and 2 pairs were flagged as policy failures.

| Metric | Value | Architectural Interpretation |
|---|---|---|
| **Total Matched Pairs** | **144 pairs** (72 unique trips) | Programmatic discovery audit: 144 matched pairs, 0 unmatched S-files, 0 unmatched V-files across 288 discovered CSV files. |
| **Physical Unique Trips** | **72 unique drives** | Evaluated across folder branches |
| **Sync Policy Passing Pairs** | **142 pairs** (98.6%) | Met all overlap, coverage, and drift criteria |
| **Sync Policy Flagged Pairs** | **2 pairs** (1.4%) | Flagged for clock drift policy threshold (Y1) |
| **Downstream-Ready Pairs** | **142 pairs** (98.6%) | Validated and fully accepted for Phase 3 filter integration |
| **Total Raw S Rows** | **2,141,490 rows** | Raw smartphone IMU + GPS telemetry |
| **Total Raw V Rows** | **2,142,070 rows** | Raw Racelogic VBOX ground-truth telemetry |
| **Total Synchronized Rows** | **2,141,490 rows** | Unified time-aligned records on target S working grid |
| **Total Validated Rows** | **2,130,554 rows** | Structurally computable rows kept for filter integration |
| **Total Omitted Rows** | **10,936 rows** (0.5107%) | Corrupt/non-computable rows omitted from validated stream |
| **Total Driving Duration** | **59.48 hours** | Real-world Indian road driving telemetry |
| **Measured Sampling Rate** | **10.00 Hz median** (range [10.00, 10.00] Hz, IQR 0.00 Hz) | Empirical sensor rate distribution across files |

---

## 2. Dataset Branch Breakdown

| Branch | Matched Pairs | Synced Rows | Validated Rows | Omitted Rows | Files with Stationary Periods |
|---|---|---|---|---|---|
| Categorised IOVNB Dataset | 72 | 1,070,745 | 1,065,277 | 5,468 | 44 |
| Uncategorised IOVNB Dataset | 72 | 1,070,745 | 1,065,277 | 5,468 | 44 |

---

## 3. Non-Destructive Quality Tagging Statistics

Every raw sensor record is strictly preserved without modification. Samples evaluated by the `QualityTagger` received non-destructive bitmask flags. Flag counts represent row-level occurrences across all processed pairs:

| Quality Flag | Flag Bitmask | Rows Affected | Pipeline & Filter Handling Policy |
|---|---|---|---|
| `FLAG_OK` | `0x00` | 2,130,514 | Clean nominal sample; passed directly to filter integration. |
| `FLAG_NAN_OR_NONFINITE` | `0x01` | 0 | Structurally non-computable; **omitted from validated stream**. |
| `FLAG_INVALID_TIMESTAMP` | `0x02` | 0 | Negative timestamps; omitted from validated stream. |
| `FLAG_NON_MONOTONIC_TIMESTAMP` | `0x04` | 16 | Session counter restarts across 5 unique trip(s); **omitted from validated stream**. |
| `FLAG_DUPLICATE_TIMESTAMP` | `0x08` | 10,920 | Duplicate timestamps across 2 unique trip(s); omitted from validated stream. |
| `FLAG_EXTREME_MOTION` | `0x10` | 34 (14 accel events, 28 gyro events) | **KEPT IN VALIDATED STREAM**. Physical dynamics (potholes, bumps, sharp turns). Innovation gate inflates measurement variance without discarding real motion. |
| `FLAG_SENSOR_DROPOUT` | `0x20` | 6 | **KEPT IN VALIDATED STREAM**. Timing gap event (> 300 ms); strapdown INS propagates over the larger $\Delta t$. |

### Omission Policy Verification
- **Total Rows Omitted from Validated Stream**: 10,936 out of 2,141,490 (0.5107% omission rate).
- **Data-Driven Cause Analysis**: Omissions are strictly limited to non-computable records: 16 non-monotonic session restart samples across 5 unique trip(s), 10,920 duplicate timestamp samples across 2 unique trip(s). Zero physical extreme-motion events and zero timing dropouts were discarded.

---

## 4. Extreme Motion Analysis

Extreme motion is detected using vector Euclidean magnitudes:
$$\|\mathbf{f}\| = \sqrt{a_x^2 + a_y^2 + a_z^2} > 39.24\text{ m/s}^2 \quad (>4g)$$
$$\|\boldsymbol{\omega}\| = \sqrt{\omega_x^2 + \omega_y^2 + \omega_z^2} > 10.0\text{ rad/s} \quad (\approx 573^\circ/\text{s})$$

- **Total Extreme Accel Events**: 14 events.
- **Total Extreme Gyro Events**: 28 events.
- **Trips Exhibiting Extreme Dynamics**:
  * `M` (Driver B): 3 accel, 3 gyro events per branch
  * `S2` (Driver A): 1 accel, 2 gyro events per branch
  * `S3b` (Driver A): 2 accel, 3 gyro events per branch
  * `S4` (Driver A): 1 accel, 2 gyro events per branch
  * `Y1` (Driver D): 0 accel, 4 gyro events per branch
- **Significance**: These trips capture real roadway shock vibrations and aggressive vehicle turns. Because COMPASS does not clip or drop these samples, the downstream ESKF's innovation gating mechanism can dynamically adjust measurement covariance without losing tracking.

---

## 5. Candidate Stationary Segments & Calibration Feasibility

Stationary periods are detected using the dual-signal requirement:
$$\text{var}(\|\mathbf{f}\|) < 0.05\text{ m}^2/\text{s}^4 \quad \text{AND} \quad \text{var}(\|\boldsymbol{\omega}\|) < 0.005\text{ rad}^2/\text{s}^2 \quad \text{over } \ge 50\text{ samples}$$

- **Timing Semantics**: The detection criterion requires $\ge 50$ consecutive stationary samples. For 50 samples at nominal 10 Hz, the timestamp span from sample 0 to sample 49 covers 4.90 s, spanning 50 discrete measurement epochs.
- **Files with Stationary Periods**: 88 / 144 (44 unique trips).
- **Total Stationary Segments**: 2146 segments (1073 unique intervals per branch).
- **Total Rest Duration**: 10.21 hours across all trips.
- **Application**: Validates the Phase 3 startup calibration requirement. Gyroscope bias $\mathbf{b}_g$ and initial roll/pitch alignment from the gravity reaction vector can be estimated at rest across 44 trips.

---

## 6. S/V Timestamp Synchronization & Audit Summary

- **Clock Alignment**:
  * S-file timestamps (`TIME SINCE START (ms)`) converted safely to signed `int64` nanoseconds (`int(t_ms * 1_000_000)`).
  * V-file timestamps (`Time Since Start of Day (seconds)`) converted safely to signed `int64` nanoseconds (`int(round(t_sec * 1_000_000_000))`).
- **Alignment Coordination**:
  * Mode: `relative_elapsed` (explicit `SyncMode.RELATIVE_ELAPSED` policy).
  * Aligns relative elapsed time from stream origin (t - t[0]), removing clock origin offsets.
  * Clock rate drift is explicitly measured and evaluated against acceptance thresholds.
  * S timestamps form the master working target grid; V ground truth reference telemetry is interpolated onto those target timestamps.
- **Interpolation Rules**:
  * Continuous position/velocity (lat, lon, alt, speed, wheel speeds, CAN accel, yaw rate): linear interpolation.
  * Heading azimuth: circular angle interpolation (via $\text{atan2}(\sin, \cos)$) eliminating $0^\circ/360^\circ$ wrap-around spikes.
  * Discrete CAN signals (gear, handbrake): nearest-neighbor integer preservation (never fractional).
  * Maximum Gap Policy: source gaps $> 1.0\text{ s}$ are rejected without fabrication (marked NaN).
  * Extrapolation Policy: zero extrapolation outside valid source timestamp intervals.
- **Policy Audit Status**:
  * Total Pairs Processed: 144 pairs.
  * Policy Passing Rate: 142 / 144 pairs (98.6%).
  * Policy Flagged Rate: 2 / 144 pairs (1.4%).
  * Mean Overlap Duration: 1487.03 seconds.
  * Mean Interpolation Coverage: 99.91%.
  * Mean Full-Trip Clock Drift: -9.25 seconds.
- **Policy Failure Audit**:
  * `Y1` (Categorised IOVNB Dataset): FAILED: Clock drift (-638.51s) exceeds threshold (120.00s) (drift = -638.50s)
  * `Y1` (Uncategorised IOVNB Dataset): FAILED: Clock drift (-638.51s) exceeds threshold (120.00s) (drift = -638.50s)

---

## 7. GPS Outage Availability

- **Dataset Real Outages**: False (0 real outage index files exist in this archive).
- **Synthetic Outage Evaluation**:
  * Benchmark outage windows (e.g. 50 m / 1 min blackout; 1 km @ 60 km/h blackout) are generated via `OutageIndex.generate_synthetic_outage()` and tagged strictly with `is_synthetic=True` for Phase 4+ evaluation.

---

## 8. Cache Integrity & Format

- **Format**: Compressed NumPy binary archives (`.npz`).
- **Location**: `data/cache/iovnbd/` (gitignored per project policy).
- **Arrays & Metadata Cached per Trip**:
  * `timestamps_ns`: 1D int64 strictly monotonic unwrapped working time axis on target S grid.
  * `raw_timestamps_ns`: 1D int64 unmodified original device timestamps from raw S-file.
  * `accel_raw`: Nx3 float64 specific force (device body frame, m/s²).
  * `gyro_raw`: Nx3 float64 angular velocity (device body frame, rad/s).
  * `quality_flags`: 1D uint32 bitflags.
  * `is_validated`: 1D bool integration mask.
  * `s_gnss_*`: Lat, Lon, Alt, Speed, Bearing, Accuracy, Sat Count.
  * `v_ref_*`: Ground-truth VBOX Lat, Lon, Alt, Speed, Heading, Yaw Rate, Wheel Speeds, CAN Accel, Gear, Handbrake.
  * `diag_*`: Exact synchronization diagnostics preserved identically across save and load.
- **Access Performance**: NumPy compressed binary arrays (`.npz`) eliminate repeated CSV parsing overhead for downstream training and ESKF replay.

---

## 9. Immutability Verification

- **Raw Data Directory**: `data/raw/io_vnbd`
- **Verification Method**: SHA-256 digests, file sizes, and modification times evaluated for all raw CSV files before and after full pipeline execution.
- **Result**: **100% UNCHANGED**. All discovered raw CSV files verified byte-for-byte identical.
