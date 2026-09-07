# IO-VNBD Dataset Notes (Phase 0 Empirical Discovery)

This document records the empirical facts, structure, and findings discovered from auditing the physical **IO-VNBD** (Input-Output Vehicle Navigation Benchmark Dataset) located in `data/raw/io_vnbd/`.

All findings strictly separate **`[CONFIRMED FROM ACTUAL DATASET]`**, **`[SPECIFICATION EXPECTATION]`**, and **`[OPEN QUESTION]`**.

---

## Dataset Location
- **Local Path**: `data/raw/io_vnbd/` `[CONFIRMED FROM ACTUAL DATASET]`
- **Extracted Root**: `data/raw/io_vnbd/Synchronised V abd S datasets/` `[CONFIRMED FROM ACTUAL DATASET]`
- **Archive Status**: The original downloaded ZIP archive (`Synchronised V abd S datasets.zip`, ~194.2 MB) was extracted and the redundant archive was removed to prevent duplication. Exactly 360 extracted dataset files remain. `[CONFIRMED FROM ACTUAL DATASET]`
- **Upstream Source**: Onyekpe et al., *Applied Sciences* 2021 (`https://github.com/onyekpeu/IO-VNBD`) `[SPECIFICATION EXPECTATION]`

---

## Dataset Structure
`[CONFIRMED FROM ACTUAL DATASET]`
The extracted dataset contains a total of **360 files** (288 CSV sensor files + 72 JPG trip photos).
The directory is organized into two primary branches:

1. **`Categorised IOVNB Dataset/`** (216 files: 144 CSVs + 72 JPGs):
   Recordings are grouped into driver-specific folders:
   - `M (Driver B)/`: 1 trip (`M`) -> `S-M.csv`, `V-M.csv`, `V-M.JPG`
   - `S (Driver A)/`: 6 trip subfolders (`S1`, `S2`, `S3a`, `S3b`, `S3c`, `S4`)
   - `Vf (Driver E)/`: 2 trip subfolders (`Vfa01`, `Vfa02`)
   - `Vta (Driver E)/`: 30 trip subfolders (`Vta1a`, `Vta1b`, `Vta2` .. `Vta30`)
   - `Vtb (Driver E)/`: 12 trip subfolders (`Vtb1` .. `Vtb12`)
   - `Vw (Driver E)/`: 20 trip subfolders (`Vw1` .. `Vw20`)
   - `Y (Driver D)/`: 1 trip subfolder (`Y`) -> `S-Y.csv`, `V-Y.csv`, `V-Y.JPG`
   *Total unique trips*: **72 trips** across 4 drivers (Driver A, Driver B, Driver D, Driver E).

2. **`Uncategorised IOVNB Dataset/`** (144 files: 144 CSVs):
   Files are separated into two flat folders:
   - `S-Dataset/`: 72 `S-*.csv` files
   - `V-Dataset/`: 72 `V-*.csv` files

*Total CSVs*: 288 (144 in Categorised, 144 in Uncategorised). Exactly 72 trips are represented identically in both branches.

---

## S- Files
- **Role**: Smartphone IMU and GPS telemetry. `[CONFIRMED FROM ACTUAL DATASET]`
- **Total Count**: 144 CSV files (72 in Categorised, 72 in Uncategorised). `[CONFIRMED FROM ACTUAL DATASET]`
- **File Sizes**: Range from 12.8 KB (`S-Vtb1b.csv`, 128 rows) to 25.8 MB (`S-Vw4.csv`, 126,526 rows). `[CONFIRMED FROM ACTUAL DATASET]`
- **Encoding**: ISO-8859-1 / Latin-1 (contains degree symbol `°` as `0xB0`, squared `²` as `0xB2`, micro `μ` as `0xCE 0xBC`). `[CONFIRMED FROM ACTUAL DATASET]`
- **Exact Column Names (24 columns)**: `[CONFIRMED FROM ACTUAL DATASET]`
  1. `GPS LATITUDE (degrees)`
  2. `GPS LONGITUDE (degrees)`
  3. `GPS ALTITUDE (m)`
  4. `GPS SPEED (Kmh)`
  5. `GPS ACCURACY (m)`
  6. `GPS ORIENTATION (°)`
  7. `GPS SATELLITES IN RANGE`
  8. `TIME SINCE START (ms)` *(Timestamp)*
  9. `DATE (YYYY-MO-DD HH-MI-SS_SSS)`
  10. `ACCELEROMETER X (m/s²)`
  11. `ACCELEROMETER Y (m/s²)`
  12. `ACCELEROMETER Z (m/s²)`
  13. `GRAVITY X (m/s²)`
  14. `GRAVITY Y (m/s²)`
  15. `GRAVITY Z (m/s²)`
  16. `GYROSCOPE Yaw (rad/s)`
  17. `GYROSCOPE Pitch (rad/s)`
  18. `GYROSCOPE Roll (rad/s)`
  19. `MAGNETIC FIELD X (μT)`
  20. `MAGNETIC FIELD Y (μT)`
  21. `MAGNETIC FIELD Z (μT)`
  22. `ORIENTATION (Yaw) (°)`
  23. `ORIENTATION (Pitch) (°)`
  24. `ORIENTATION (Roll ) (°)`
  *(Note: CSV headers contain leading spaces after each comma, e.g. `' TIME SINCE START (ms)'`, which require whitespace stripping).*

---

## V- Files
- **Role**: Vehicle CAN bus and high-precision reference telemetry (Racelogic VBOX). `[CONFIRMED FROM ACTUAL DATASET]`
- **Total Count**: 144 CSV files (72 in Categorised, 72 in Uncategorised). `[CONFIRMED FROM ACTUAL DATASET]`
- **File Sizes**: Range from 25.3 KB (`V-vta1b.csv`, 128 rows) to 28.0 MB (`V-Vw4.csv`, 126,527 rows). `[CONFIRMED FROM ACTUAL DATASET]`
- **Exact Column Names (29 columns)**: `[CONFIRMED FROM ACTUAL DATASET]`
  1. `No of GPS Satellites Available`
  2. `Time Since Start of Day (seconds)` *(Timestamp)*
  3. `Latitude (degrees)`
  4. `Longitude (degrees)`
  5. `Velocity (km/hr)`
  6. `Heading (degrees)`
  7. `Height (km)`
  8. `Vertical velocity (km/hr)`
  9. `Sample period (seconds)`
  10. `Steering Angle (degrees)`
  11. `Wheel Speed Front Left (rad/sec)`
  12. `Wheel Speed Front Right (rad/sec)`
  13. `Wheel Speed Rear Left (rad/sec)`
  14. `Wheel Speed Rear Right (rad/sec)`
  15. `Yaw Rate (deg/sec)`
  16. `Indicated Vehicle Speed (km/hr)`
  17. `Indicated Longitudinal Acceleration (g)`
  18. `Indicated Lateral Acceleration (g)`
  19. `Handbrake (0 or 1)`
  20. `Gear Requested (Number fof gear employed 1-5)`
  21. `Gear (Number fof gear employed 1-5)`
  22. `Engine Speed (rev/min)`
  23. `Coolant Temperature (degrees)`
  24. `Clutch Position (0 or 1)`
  25. `Brake Pressure (psi)`
  26. `Brake Position (0 or 1)`
  27. `Battery Voltage (volts)`
  28. `Air Temperature (degrees)`
  29. `Accelerator Pedal Position (0 or 1)`

---

## Synchronised V and S
- Both `Categorised IOVNB Dataset` and `Uncategorised IOVNB Dataset` contain synchronized pairs. `[CONFIRMED FROM ACTUAL DATASET]`
- For every trip, the `S-` and `V-` files have virtually identical row counts (e.g. `S-S1.csv` has 51,746 rows, `V-S1.csv` has 51,746 rows; $\Delta = 0$). `[CONFIRMED FROM ACTUAL DATASET]`
- Minor row count offsets exist in a few files (e.g. `S-Vfa01` has 11,486 rows vs `V-Vfa01` has 11,535 rows, $\Delta = -49$ rows = 4.9s at 10 Hz), which will be aligned during Phase 2 linear timestamp interpolation. `[CONFIRMED FROM ACTUAL DATASET]`

---

## Unsynchronised V and S
- `[CONFIRMED FROM ACTUAL DATASET]`: The currently downloaded archive contains only the synchronized dataset.
- `[SPECIFICATION EXPECTATION]`: An upstream `Unsynchronised V and S Dataset` exists on GitHub (~58h smartphone logs). It is not required for Phase 0/1, and will only be needed if exploratory self-supervised pretraining is conducted.

---

## GPS Outage Information
- `[CONFIRMED FROM ACTUAL DATASET]`: **No GPS outage index file** (`gps_outage_index.csv`) is present in the extracted dataset.
- All 360 files in the archive are `.csv` sensor logs or `.JPG` photos.
- `[SPECIFICATION EXPECTATION]`: The master plan anticipated that real-world GPS outage windows might be referenced by a supporting index.
- `[RESOLUTION / OUR ARCHITECTURAL FACT]`: Because no outage index is bundled in the synchronized archive, synthetic outage masking (e.g. 50 m / 1 min and 1 km @ 60 km/h blackout intervals per ISRO Problem Statement) and self-collected tunnel logs will serve as the primary evaluation mechanism for Phase 2+.

---

## Measured Sampling Rates
- `[CONFIRMED FROM ACTUAL DATASET]`: **The measured median sampling rate across all 288 CSV files is 10 Hz.**
  - Formula: $\text{measured\_rate\_hz} = 1 / \text{median}(\Delta t)$.
  - In `S-` files: $\Delta t$ median is $100.0\text{ ms} = 0.1000\text{ s} \implies \text{measured rate } 10.00\text{ Hz}$.
  - In `V-` files: $\Delta t$ median is $0.1000\text{ s} \implies \text{measured rate } 10.00\text{ Hz}$.
  - Real sampling rate satisfies the specification's canonical 10 Hz ML input rate.

---

## Timestamp Findings
- `[CONFIRMED FROM ACTUAL DATASET]`:
  - `duplicate_timestamps`: Absent across `S-` files (0 duplicates in all 144 files) and absent in virtually all `V-` files (1 duplicate observed in `V-M.csv`).
  - `non_monotonic_timestamps`: Absent across most files (0 in almost all files), with isolated instances detected in large files (1 in `S-M.csv`, 1 in `S-S2.csv`, 1 in `S-S3b.csv`, 2 in `S-S4.csv` out of ~100k rows).
  - Out-of-order and non-positive timestamps will receive `FLAG_NON_MONOTONIC_TIMESTAMP` / `FLAG_INVALID_TIMESTAMP` during Phase 2 ingestion.

---

## Data Quality Findings
- `[CONFIRMED FROM ACTUAL DATASET]`:
  - `NaN / Inf values`: **0 NaN and 0 Inf values** across all audited numeric sensor columns in the 288 CSVs.
  - `Vector Extreme Motion (|f| > 39.24 m/s²)`: Rare physical dynamic shock events detected using vector magnitude $\|\mathbf{f}\| = \sqrt{a_x^2 + a_y^2 + a_z^2} > 4g$ (e.g. 3 samples in `S-M.csv`, 1 sample in `S-S2.csv`, 2 samples in `S-S3b.csv`).
  - `Vector Extreme Gyro (|ω| > 10.0 rad/s)`: Rare sharp rotation spikes detected using vector magnitude $\|\boldsymbol{\omega}\| = \sqrt{\omega_x^2 + \omega_y^2 + \omega_z^2} > 10.0\text{ rad/s}$ (e.g. 3 samples in `S-M.csv`, 2 in `S-S2.csv`, 3 in `S-S3b.csv`).
  - *Non-destructive policy*: These samples represent real physical vehicle dynamics (potholes, severe bumps) and are **preserved**, not deleted.

---

## Stationary Segments
- `[CONFIRMED FROM ACTUAL DATASET]`:
  - Audited using dual low-variance criteria: $\text{var}(\|\mathbf{f}\|) < 0.05\text{ m}^2/\text{s}^4$ AND $\text{var}(\|\boldsymbol{\omega}\|) < 0.005\text{ rad}^2/\text{s}^2$ over $\ge 50$ samples (~5.0s).
  - **88 out of 144 S- files contain candidate stationary segments**.
  - Example: `S-S1.csv` contains 6,955 stationary samples (~695.5 seconds = ~11.5 minutes) across red-light and stop periods.
  - Confirms feasibility of startup calibration (estimating gyro bias and gravity tilt) on real IO-VNBD data.

---

## S/V Pairing
- `[CONFIRMED FROM ACTUAL DATASET]`:
  - **Folder-Aware Matched Pairs**: Exactly **144 matched pairs** (72 in `Categorised IOVNB Dataset`, 72 in `Uncategorised IOVNB Dataset`).
  - **Unmatched S- Files**: 0.
  - **Unmatched V- Files**: 0.
  - *Filename convention quirk resolved*: In `Categorised IOVNB Dataset`, several trips use uppercase `Vta` / `Vtb` for S- files (e.g. `S-Vta12.csv`) and lowercase `vta` / `vtb` for V- files (e.g. `V-vta12.csv`). Case-insensitive trip ID extraction (`extract_base_trip_id(...).lower()`) achieves 100% pairing.

---

## Known Discrepancies
1. **Archive Naming Typo**:
   - `[CONFIRMED FROM ACTUAL DATASET]`: Folder name is `Synchronised V abd S datasets` (contains "abd" instead of "and"). Handled cleanly by path resolution.
2. **File Casing Discrepancy**:
   - `[CONFIRMED FROM ACTUAL DATASET]`: `S-Vta*.csv` vs `V-vta*.csv`. Handled via case-insensitive trip matching.
3. **No Outage Index File in Archive**:
   - `[CONFIRMED FROM ACTUAL DATASET]`: `gps_outage_index.csv` is not present in the synchronized archive.
   - `[SPECIFICATION ALIGNMENT]`: Synthetic masking will be the primary outage simulation method.
4. **Phone Gyro Column Naming**:
   - `[CONFIRMED FROM ACTUAL DATASET]`: Phone gyroscope axes are named `GYROSCOPE Yaw (rad/s)`, `GYROSCOPE Pitch (rad/s)`, `GYROSCOPE Roll (rad/s)`, rather than `gyro_x, gyro_y, gyro_z`.

---

## Open Questions
1. `[OPEN QUESTION]`: *Vehicle Wheel Speed Conversion*: Wheel speeds in `V-` files are recorded in `rad/sec` (`Wheel Speed Front Left (rad/sec)`), while indicated speed is in `km/hr`. In Phase 2, converting wheel angular speed to linear velocity will require determining the effective rolling radius per vehicle model.
2. `[OPEN QUESTION]`: *Unsynchronised Dataset*: The unsynchronised branch was not included in this archive; it will be retrieved only if unsupervised pretraining is prioritized.
3. `[OPEN QUESTION]`: *Licensing Terms*: Upstream licensing terms for IO-VNBD are not explicitly stated in a LICENSE file within the extracted archive. Exact terms for competition submission and redistribution remain an open question to be clarified with SIH / ISRO mentors and the dataset authors.
