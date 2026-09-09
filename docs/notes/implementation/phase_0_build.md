# COMPASS Phase 0 — Implementation Build Walkthrough

**Project**: C.O.M.P.A.S.S. (Cognitive Off-grid Machine-learning Positioning And Sensor System)  
**Problem Statement**: SIH 2026 Problem Statement 26168 — ISRO  
**Document Type**: Step-by-Step Implementation Build Diary

---

## Overview: What We Were Trying to Do

Before writing any navigation code, we needed to answer a fundamental question: **Do we actually understand our data?**

IO-VNBD is an academic benchmark dataset published by UK researchers. Reading the paper tells you the number of trips and their general structure. But to build reliable algorithms, you need to know things the paper does not tell you:
- How exactly is each column encoded?
- Does every file use the same encoding?
- Are timestamps truly monotonic?
- Are there hidden clock drift failures?
- Does a trustworthy pre-tagged "GPS outage index" actually exist in the downloaded archive?

Phase 0 was built as a **forensic audit pass**: a standalone Python script that reads every file in the archive and produces a machine-readable diagnostic report.

---

## Step 1: Setting Up the Python Environment

### What We Did
We created the project as an installable Python package using `pyproject.toml` with the `compass` package name. The project layout was:

```
d:/Hackathon/Compass/
├── pyproject.toml           ← package metadata + dependencies
├── .venv/                   ← virtual environment
├── data/raw/io_vnbd/        ← raw dataset (gitignored)
├── scripts/                 ← standalone runner scripts
├── navigation/              ← core nav library
├── tests/                   ← pytest suite
└── docs/                    ← reports and documentation
```

Dependencies used: `numpy`, `scipy`, `pandas`, `pytest`.

### Why Editable Install?
We installed the package with `pip install -e .` to allow `import navigation`, `import data.pipeline` etc. without path manipulation. This made every script in `scripts/` and every test in `tests/` automatically find the project's Python packages.

---

## Step 2: Writing the Archive Inspection Script

### What We Built
[`scripts/inspect_iovnbd.py`](file:///d:/Hackathon/Compass/scripts/inspect_iovnbd.py) — a standalone audit script.

### How We Structured It

**Phase A: File Discovery**
```python
# Walk the entire data/raw/io_vnbd/ tree recursively
# Classify every file as S-file, V-file, or photo
# Match S-TripID.csv to V-TripID.csv using TripID extraction
```
This immediately revealed the two-branch structure: `Categorised IOVNB Dataset/` and `Uncategorised IOVNB Dataset/`. We used `pathlib.Path.rglob("*.csv")` rather than `os.walk()` for cleaner code.

**Phase B: Encoding Discovery**
We tried reading a sample file with `pd.read_csv(..., encoding='utf-8')` first. It raised a `UnicodeDecodeError` on the degree symbol (`°`) in column headers. We then tried `'iso-8859-1'` (Latin-1) — success. All 288 CSV files used Latin-1 encoding.

Header columns also had leading spaces: `' ACCELEROMETER X (m/s²)'`. We added `.str.strip()` on the column name list.

**Phase C: Sampling Rate Audit**
For every file, we computed:
```python
delta_t_ms = np.diff(timestamps_ms)
median_dt = np.median(delta_t_ms)
sampling_rate_hz = 1000.0 / median_dt
```
Result across all 288 files: **median $\Delta t$ = 100.0 ms, i.e., 10.0 Hz exactly**. Smartphone files showed ±5 ms jitter from Android Linux OS thread scheduling. Vehicle VBOX files were rigidly hardware-clocked.

**Phase D: Non-Monotonic Timestamp Detection**
```python
non_monotonic_mask = np.diff(timestamps_ms) <= 0
```
Found exactly 4 files with anomalies: `S-M.csv`, `S-S2.csv`, `S-S3b.csv`, `S-S4.csv`. Each had 1–2 entries where `t_k <= t_{k-1}`.

**Phase E: Cross-Stream Clock Alignment Check**
Attempted to align smartphone `TIME SINCE START (ms)` (starting at 0) with vehicle `Time Since Start of Day (seconds)` (starting at ~30,000–50,000). Since these have different origins, we aligned by relative elapsed time from each stream's origin and evaluated clock drift over the full trip.

For Trip Y1 (Driver D), the elapsed durations differed by **638.5 seconds** (clock drift threshold: 120 seconds). Trip Y1 was flagged as a policy failure.

**Phase F: Stationary Window Detector**
Using a rolling variance approach on the 144 smartphone S-files:
```python
# Compute rolling variance of |f| and |omega| with window_size=50 samples (~5 sec)
var_f = pd.Series(accel_norm).rolling(window_size).var()
var_omega = pd.Series(gyro_norm).rolling(window_size).var()
stationary_mask = (var_f < 0.05) & (var_omega < 0.005)
```
Found candidate stationary periods in **88 out of 144 S-file representations** (representing 44 unique trips across the two branches), with 1,073 unique rest intervals per branch.

**Phase G: GPS Outage Index Scan**
Searched `data/raw/io_vnbd/` exhaustively for any CSV, JSON, or text file naming patterns like `outage`, `gps_mask`, `tunnel_index`. **Found zero such files.** The dataset has no pre-labeled GPS outage windows.

---

## Step 3: Forensic Comparison of the Two Branches

### The Question
The archive contains two directories with seemingly independent sets of CSV files — 144 in `Categorised/` and 144 in `Uncategorised/`. Are these 288 independent trips, or 144 trip files stored twice?

### How We Resolved It
We computed SHA-256 cryptographic hashes of every CSV file:
```python
import hashlib
sha = hashlib.sha256(file_bytes).hexdigest()
```
Then compared hashes between branches for each matching TripID.

**Result**: Every `S-TripID.csv` in `Categorised/` was byte-for-byte identical to the corresponding `S-TripID.csv` in `Uncategorised/`. Same for V-files. The dataset contains exactly **144 matched S/V file representations corresponding to 72 unique physical trips**, duplicated into two branches.

This finding prevented us from accidentally training on 144 files thinking they were independent, which would have inflated our dataset size by 2× and duplicated training samples.

---

## Step 4: Output Report Generation

The script generated two output files:

**[`docs/iovnbd_inspection_report.md`](file:///d:/Hackathon/Compass/docs/iovnbd_inspection_report.md)** — human-readable Markdown summary with tables for every finding.

**[`docs/iovnbd_inspection_summary.json`](file:///d:/Hackathon/Compass/docs/iovnbd_inspection_summary.json)** — machine-readable JSON with per-file statistics (trip ID, rows, sampling rate, clock drift, stationary segments, non-monotonic counts, extreme motion flags).

---

## Step 5: Writing the Unit Tests

Phase 0 also delivered [`tests/test_inspect_iovnbd.py`](file:///d:/Hackathon/Compass/tests/test_inspect_iovnbd.py) — a test file with exactly **9 unit tests** that verify the inspection script's core detection logic in isolation, using synthetic constructed CSV data rather than requiring the full raw data archive to be present.

---

## Step 6: Key Decisions Made Based on Phase 0 Findings

| Finding | Decision Made |
|:---|:---|
| Latin-1 encoding, leading spaces in headers | Mandated `encoding='iso-8859-1'` and `.strip()` on all column names for all future CSV readers |
| 72 unique physical trips (144 matched representations) | Track 72 unique TripIDs; never count duplicated branches separately |
| Smartphone timestamp in ms (0-based) vs Vehicle timestamp in sec (midnight-based) | Phase 2 must use relative-elapsed alignment onto smartphone target working grid |
| Trip Y1 clock drift -638.5 s | Formally exclude Driver D from all active splits |
| No pre-tagged real GPS outage index exists | Design synthetic masking architecture; no fabricated real outage data |
| 88 S-file representations have stationary windows | Phase 3 stationary calibration is feasible; exact detection threshold: 50 samples, var(f) < 0.05, var(omega) < 0.005 |
| VBOX height column header says `(km)` but values are in meters | Flag for Phase 2 parser investigation; resolve in Phase 4 audit |

---

## Debugging Episodes

**Bug 1: File pairing by TripID failed for Trip Y1**
Trip Y1 was stored as two separate files (`2021-03-24-15-58-29` and `2021-03-24-16-08-27`) in the Categorised branch. The simple regex `r'S-(.+)\.csv'` matched the wrong pattern. Fixed by adding directory-name awareness (grouping files by `Y/` subdirectory).

**Bug 2: Stationary detector false-fired on aggressive acceleration onset**
The rolling variance window of 50 samples was sometimes triggered as "stationary" at the very start of sudden braking. Fixed by requiring the **first** sample of the stationary window itself to be in the low-variance regime.

---

## What Phase 0 Delivered

| Artifact | Path | Purpose |
|:---|:---|:---|
| Inspection script | `scripts/inspect_iovnbd.py` | One-shot full archive audit |
| Human report | `docs/iovnbd_inspection_report.md` | Readable summary of all findings |
| Machine report | `docs/iovnbd_inspection_summary.json` | Per-file statistics for all 288 files |
| Unit tests | `tests/test_inspect_iovnbd.py` | Exactly 9 regression tests for detection logic |

**Test count contributed by Phase 0**: Exactly **9 unit tests** inside `tests/test_inspect_iovnbd.py`.
