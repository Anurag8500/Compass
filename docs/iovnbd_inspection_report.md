# IO-VNBD Dataset Inspection Report (Phase 0)
**Target Directory**: `data/raw/io_vnbd`  
**Total Files Discovered**: 360  

## 1. Dataset Overview & Inventory
### Breakdown by Branch
| Dataset Branch | File Count |
|---|---|
| Categorised IOVNB Dataset | 216 |
| Uncategorised IOVNB Dataset | 144 |

### Breakdown by File Category
| Category | File Count |
|---|---|
| S- file (Smartphone) | 144 |
| Trip Photo | 72 |
| V- file (Vehicle CAN/VBOX) | 144 |

## 2. Folder-Aware S / V Pairing Analysis
- **Total Matched Pairs**: 144
- **Unmatched S- Files**: 0
- **Unmatched V- Files**: 0

### Pairs per Branch
| Branch | Matched S/V Pairs |
|---|---|
| Categorised IOVNB Dataset | 72 pairs |
| Uncategorised IOVNB Dataset | 72 pairs |

### Matched Pairs Sample (First 15)
| Branch | Trip ID | S- File (Rows, Rate) | V- File (Rows, Rate) | Row Δ |
|---|---|---|---|---|
| Categorised IOVNB Dataset | `M` | `S-M.csv` (105,974, 10.00Hz) | `V-M.csv` (105,974, 10.00Hz) | +0 |
| Categorised IOVNB Dataset | `S1` | `S-S1.csv` (51,746, 10.00Hz) | `V-S1.csv` (51,746, 10.00Hz) | +0 |
| Categorised IOVNB Dataset | `S2` | `S-S2.csv` (93,876, 10.00Hz) | `V-S2.csv` (93,876, 10.00Hz) | +0 |
| Categorised IOVNB Dataset | `S3a` | `S-S3a.csv` (24,621, 10.00Hz) | `V-S3a.csv` (24,621, 10.00Hz) | +0 |
| Categorised IOVNB Dataset | `S3b` | `S-S3b.csv` (6,813, 10.00Hz) | `V-S3b.csv` (6,813, 10.00Hz) | +0 |
| Categorised IOVNB Dataset | `S3c` | `S-S3c.csv` (37,183, 10.00Hz) | `V-S3c.csv` (37,183, 10.00Hz) | +0 |
| Categorised IOVNB Dataset | `S4` | `S-S4.csv` (94,600, 10.00Hz) | `V-S4.csv` (94,600, 10.00Hz) | +0 |
| Categorised IOVNB Dataset | `Vfa01` | `S-Vfa01.csv` (11,486, 10.00Hz) | `V-Vfa01.csv` (11,535, 10.00Hz) | -49 |
| Categorised IOVNB Dataset | `Vfa02` | `S-Vfa02.csv` (67,523, 10.00Hz) | `V-Vfa02.csv` (67,755, 10.00Hz) | -232 |
| Categorised IOVNB Dataset | `Vta1a` | `S-Vta1a.csv` (25,676, 10.00Hz) | `V-Vta1a.csv` (25,676, 10.00Hz) | +0 |
| Categorised IOVNB Dataset | `Vta1b` | `S-Vta1b.csv` (954, 10.00Hz) | `V-Vta1b.csv` (953, 10.00Hz) | +1 |
| Categorised IOVNB Dataset | `Vta2` | `S-Vta2.csv` (10,991, 10.00Hz) | `V-vta2.csv` (10,991, 10.00Hz) | +0 |
| Categorised IOVNB Dataset | `Vta3` | `S-Vta3.csv` (645, 10.00Hz) | `V-vta3.csv` (645, 10.00Hz) | +0 |
| Categorised IOVNB Dataset | `Vta4` | `S-Vta4.csv` (1,789, 10.00Hz) | `V-vta4.csv` (1,789, 10.00Hz) | +0 |
| Categorised IOVNB Dataset | `Vta5` | `S-Vta5.csv` (307, 10.00Hz) | `V-vta5.csv` (307, 10.00Hz) | +0 |
| ... | ... | *(+129 more pairs)* | | |

## 3. Sampling Rates & Timestamp Verification
| Filename | Branch | Rows | Measured Rate | Median Δt | Duplicates | Non-Monotonic |
|---|---|---|---|---|---|---|
| `S-M.csv` | Categorised IOVNB Dataset | 105,974 | **10.00 Hz** | 0.1000s (milliseconds) | 0 | 1 |
| `V-M.csv` | Categorised IOVNB Dataset | 105,974 | **10.00 Hz** | 0.1000s (seconds) | 1 | 0 |
| `S-S1.csv` | Categorised IOVNB Dataset | 51,746 | **10.00 Hz** | 0.1000s (milliseconds) | 0 | 0 |
| `V-S1.csv` | Categorised IOVNB Dataset | 51,746 | **10.00 Hz** | 0.1000s (seconds) | 0 | 0 |
| `S-S2.csv` | Categorised IOVNB Dataset | 93,876 | **10.00 Hz** | 0.1000s (milliseconds) | 0 | 1 |
| `V-S2.csv` | Categorised IOVNB Dataset | 93,876 | **10.00 Hz** | 0.1000s (seconds) | 0 | 0 |
| `S-S3a.csv` | Categorised IOVNB Dataset | 24,621 | **10.00 Hz** | 0.1000s (milliseconds) | 0 | 0 |
| `V-S3a.csv` | Categorised IOVNB Dataset | 24,621 | **10.00 Hz** | 0.1000s (seconds) | 0 | 0 |
| `S-S3b.csv` | Categorised IOVNB Dataset | 6,813 | **10.00 Hz** | 0.1000s (milliseconds) | 0 | 1 |
| `V-S3b.csv` | Categorised IOVNB Dataset | 6,813 | **10.00 Hz** | 0.1000s (seconds) | 0 | 0 |
| `S-S3c.csv` | Categorised IOVNB Dataset | 37,183 | **10.00 Hz** | 0.1000s (milliseconds) | 0 | 0 |
| `V-S3c.csv` | Categorised IOVNB Dataset | 37,183 | **10.00 Hz** | 0.1000s (seconds) | 0 | 0 |
| `S-S4.csv` | Categorised IOVNB Dataset | 94,600 | **10.00 Hz** | 0.1000s (milliseconds) | 0 | 2 |
| `V-S4.csv` | Categorised IOVNB Dataset | 94,600 | **10.00 Hz** | 0.1000s (seconds) | 0 | 0 |
| `S-Vfa01.csv` | Categorised IOVNB Dataset | 11,486 | **10.00 Hz** | 0.1000s (milliseconds) | 0 | 0 |
| `V-Vfa01.csv` | Categorised IOVNB Dataset | 11,535 | **10.00 Hz** | 0.1000s (seconds) | 0 | 0 |
| `S-Vfa02.csv` | Categorised IOVNB Dataset | 67,523 | **10.00 Hz** | 0.1000s (milliseconds) | 0 | 0 |
| `V-Vfa02.csv` | Categorised IOVNB Dataset | 67,755 | **10.00 Hz** | 0.1000s (seconds) | 0 | 0 |
| `S-Vta1a.csv` | Categorised IOVNB Dataset | 25,676 | **10.00 Hz** | 0.1000s (milliseconds) | 0 | 0 |
| `V-Vta1a.csv` | Categorised IOVNB Dataset | 25,676 | **10.00 Hz** | 0.1000s (seconds) | 0 | 0 |
| ... | *(+268 more CSVs audited)* | | | | | |

## 4. Data Quality & Vector Extreme-Motion Findings
> Extreme motion rule: Vector magnitude $|f| = \sqrt{ax^2+ay^2+az^2} > 39.24\text{ m/s}^2$ ($>4g$) 
> and $|\omega| = \sqrt{gx^2+gy^2+gz^2} > 10.0\text{ rad/s}$. Real vehicle dynamic events are preserved.

| Filename | Rows | NaN Count | Inf Count | Extreme Accel (|f|>4g) | Extreme Gyro (|ω|>10 rad/s) |
|---|---|---|---|---|---|
| `S-M.csv` | 105,974 | 0 | 0 | 3 | 3 |
| `V-M.csv` | 105,974 | 0 | 0 | 0 | 0 |
| `S-S1.csv` | 51,746 | 0 | 0 | 0 | 0 |
| `V-S1.csv` | 51,746 | 0 | 0 | 0 | 0 |
| `S-S2.csv` | 93,876 | 0 | 0 | 1 | 2 |
| `V-S2.csv` | 93,876 | 0 | 0 | 0 | 0 |
| `S-S3a.csv` | 24,621 | 0 | 0 | 0 | 0 |
| `V-S3a.csv` | 24,621 | 0 | 0 | 0 | 0 |
| `S-S3b.csv` | 6,813 | 0 | 0 | 2 | 3 |
| `V-S3b.csv` | 6,813 | 0 | 0 | 0 | 0 |
| `S-S3c.csv` | 37,183 | 0 | 0 | 0 | 0 |
| `V-S3c.csv` | 37,183 | 0 | 0 | 0 | 0 |
| `S-S4.csv` | 94,600 | 0 | 0 | 1 | 2 |
| `V-S4.csv` | 94,600 | 0 | 0 | 0 | 0 |
| `S-Vfa01.csv` | 11,486 | 0 | 0 | 0 | 0 |
| `V-Vfa01.csv` | 11,535 | 0 | 0 | 0 | 0 |
| `S-Vfa02.csv` | 67,523 | 0 | 0 | 0 | 0 |
| `V-Vfa02.csv` | 67,755 | 0 | 0 | 0 | 0 |
| `S-Vta1a.csv` | 25,676 | 0 | 0 | 0 | 0 |
| `V-Vta1a.csv` | 25,676 | 0 | 0 | 0 | 0 |
| ... | *(+268 more CSVs audited)* | | | | |

## 5. Candidate Stationary Segments (Dual Accel + Gyro Low Variance)
- **Files with candidate stationary periods**: 88 / 288
- **Heuristic criteria**: $\text{var}(|f|) < 0.05\text{ m}^2/\text{s}^4$ AND $\text{var}(|\omega|) < 0.005\text{ rad}^2/\text{s}^2$ over $\ge 50$ samples (~5.0s)

| Filename | Segment Range | Duration | Mean Accel Var (m²/s⁴) | Mean Gyro Var (rad²/s²) |
|---|---|---|---|---|
| `S-M.csv` | [2145:2231] | 8.5s | 0.05175 | 0.003356 |
| `S-M.csv` | [4896:4949] | 5.2s | 0.15145 | 0.002014 |
| `S-S1.csv` | [15:503] | 48.7s | 0.00979 | 0.000238 |
| `S-S1.csv` | [810:917] | 10.6s | 0.07416 | 0.001132 |
| `S-S2.csv` | [0:121] | 12.0s | 0.01744 | 0.001567 |
| `S-S2.csv` | [88:168] | 7.9s | 0.01189 | 0.002999 |
| `S-S3a.csv` | [5:58] | 5.2s | 0.04966 | 0.003330 |
| `S-S3a.csv` | [2084:2252] | 16.7s | 0.05355 | 0.000253 |
| `S-S3b.csv` | [20:70] | 4.9s | 0.06970 | 0.000263 |
| `S-S3b.csv` | [89:149] | 5.9s | 0.10766 | 0.000273 |
| `S-S3c.csv` | [206:624] | 41.7s | 0.01906 | 0.000239 |
| `S-S3c.csv` | [815:959] | 14.3s | 0.05486 | 0.000456 |
| `S-S4.csv` | [70:169] | 9.8s | 0.04401 | 0.008984 |
| `S-S4.csv` | [202:366] | 16.3s | 0.03802 | 0.006062 |
| `S-Vfa01.csv` | [5241:5346] | 10.4s | 0.04136 | 0.004047 |
| `S-Vfa01.csv` | [5360:5456] | 9.5s | 0.01548 | 0.009761 |
| `S-Vfa02.csv` | [0:126] | 12.5s | 0.01067 | 0.001297 |
| `S-Vfa02.csv` | [1987:2148] | 16.0s | 0.09211 | 0.008436 |
| `S-Vta1a.csv` | [0:335] | 33.4s | 0.00766 | 0.001116 |
| `S-Vta1a.csv` | [1257:1386] | 12.8s | 0.04378 | 0.001200 |
| ... | *(+78 more files containing stationary rest periods)* | | | |

## 6. GPS Outage Information
No explicit GPS outage index CSV file was discovered in the current raw archive.
The extracted `Synchronised V abd S datasets.zip` contains recordings and vehicle photos exclusively.
Synthetic outage masking and self-collected tunnel logs will serve as the benchmark evaluation mechanism.
