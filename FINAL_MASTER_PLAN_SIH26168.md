# FINAL_MASTER_PLAN_SIH26168.md
## AI-ML Based Intelligent Dead Reckoning System — Single Master Technical/Context Document

This document reconciles and supersedes the loose architecture sketches spread across `PRD_Intelligent_Dead_Reckoning_SIH26168.md`, `Workflow_Intelligent_Dead_Reckoning_SIH26168.md`, `FeatureSpec_TechStack_Implementation_SIH26168.md`, and `AIML_Implementation_Blueprint_SIH26168.md`. It is not a merge of those four files — it is a reconciled, authoritative restatement, following the hierarchy: **the AI/ML Blueprint is the source of truth for the AI/ML pipeline specifically; the PRD/Workflow/FeatureSpec remain authoritative for everything else.** Where a decision changed across the four source documents during their own iterative review, the *final* version is what appears here, with the change noted, not the original draft.

**Labels used throughout**: `[OFFICIAL ISRO REQUIREMENT]` `[IO-VNBD FACT]` `[RESEARCH FINDING]` `[ENGINEERING RECOMMENDATION]` `[OUR DESIGN DECISION]` `[ASSUMPTION]` `[OPEN QUESTION]`.

---

# 1. Project Identity

| | |
|---|---|
| Project | AI-ML Based Intelligent Dead Reckoning (IDR) System |
| SIH Problem Statement | ID 26168 |
| Organization/owner | Indian Space Research Organisation (ISRO), Department of Space |
| Theme | Smart Vehicles |
| One-line problem | Smartphone-based vehicle navigation breaks down whenever GNSS is blocked or degraded (tunnels, underground parking, urban canyons, dense forest, jamming), because there is no factory INS or OBD-II feed to fall back on in most Indian vehicles. |
| One-line solution | A lightweight, AI-augmented dead-reckoning engine — implementing one shared navigation architecture via an authoritative Python reference implementation and a behaviorally equivalent Kotlin production port for Android — that keeps the vehicle's position on the road through a GNSS outage and seamlessly reacquires GNSS when it returns. |
| Objective | Meet the PS's own stated drift benchmark on IO-VNBD-derived and ISRO-provided test data, with a working mobile app and a working edge engine, by the SIH finale. |
| Primary use cases | Delivery/logistics riders and drivers losing GNSS in a tunnel/underpass; fleet vehicles through underground parking; continuity of navigation through an urban canyon with intermittent signal. |
| Target users | Drivers/riders of vehicles without factory INS (trucks, older cars, two-wheelers) — the PS's own stated majority-case Indian vehicle population; secondarily, systems integrators wanting a portable navigation core independent of a phone. |
| Scope | Android mobile app + Python/ONNX edge engine, sharing one navigation core; offline-first; IO-VNBD-trained AI models; OSM-based map matching. |
| Non-goals | iOS; full HD-map/lane-graph infrastructure; a backend/cloud service; real OBD-II/CAN-bus integration; absolute (non-relative) indoor positioning without any road-graph anchor; a spoofing-resistant GNSS receiver. |

---

# 2. Official Requirements — Definitive Table

| ID | Requirement | Official / Team | Priority | Verification |
|---|---|---|---|---|
| R1 | In-Vehicle Alignment & Calibration Engine: automatic pitch/roll/yaw estimation regardless of mount | Official | Must | Mis-mounted-phone demo; attitude error vs. GPS-heading reference |
| R2 | AI Speed & Vibration Filter: forward velocity from IMU alone, non-navigation motion filtered | Official | Must | Speed RMSE vs. GPS/wheel-speed on held-out IO-VNBD |
| R3 | Advanced Map-Matching & Kinematic Constraints (NHC) | Official | Must | % trajectory correctly snapped; overlay plot |
| R4 | GNSS+INS Fusion Engine, AI-based | Official | Must | Position RMSE on non-outage segments |
| R5 | Seamless GNSS Deficit Handler, both directions, millisecond-scale | Official | Must | Synthetic-outage transition latency test |
| R6 | Real-time Navigation Interface | Official | Must | Live demo |
| R7 | Smartphone IMU support | Official | Must | Runs on ≥2 demo phones |
| R8 | External IMU support (not restricted to smartphone) | Official | Must | Edge CLI ingesting a non-phone IMU log |
| R9 | Edge-deployable software engine | Official | Must | Parity test: mobile vs. edge output on same input |
| R10 | 10 Hz smartphone position update | Official | Must | On-device profiling |
| R11 | Higher-rate edge processing, ~200 Hz with FOG-grade IMU | Official | Must (edge) | Replay throughput test |
| R12 | Drift < 10% of distance travelled during GNSS blackout | Official | Must | % drift on synthetic + real outage windows |
| R13 | Example benchmark: <5 m drift / 50 m / <1 min | Official | Must (headline metric) | Same |
| R14 | Example benchmark: <100 m drift / 1 km @ 60 km/h | Official | Must (headline metric) | Same |
| R15 | Preliminary AI models + position plots on IO-VNBD subset, submitted for screening | Official | Must (screening) | Proposal document |
| E1 | ESKF as the fusion backbone (not UKF) | Engineering decision | — | Literature-grounded (Section 16) |
| E2 | Exactly two ML models (VelocityNet, BiasNet), both GRU | Engineering decision | — | Section 7-8 |
| E3 | Canonical 10 Hz decimation for all ML input, regardless of source sensor rate | Engineering decision | — | Section 12, 21 |
| E4 | HMM map matching over offline OSM extract | Engineering decision | — | Section 19 |
| E5 | Offline-first, no backend | Engineering decision | — | Section 4, 29 |
| D1 | Mode-switch/re-convergence target ~1-2 s | Our proposed target | — | `[ENGINEERING RECOMMENDATION]` — not stated numerically by the PS, an interpretation of "seamless" |
| D2 | Outage-detection/mode-switch latency < 200 ms | Our proposed target | — | `[ENGINEERING RECOMMENDATION]` — interpretation of "within milliseconds" |
| D3 | Inference latency ≤ 40 ms/model, ≤100 ms total pipeline per 10 Hz cycle | Our proposed target | — | `[ENGINEERING ESTIMATE, unverified]` |

**Rule enforced throughout this document**: R1-R15 are load-bearing and must never be silently altered. E1-E5 are this project's own settled engineering decisions, made with reasons documented in the sections below, and open to revision only with a new, equally-documented reason. D1-D3 are proposed internal targets, explicitly not ISRO-mandated numbers — never cite these as if the PS specified them.

---

# 3. The Problem, From First Principles

**GNSS** gives position/velocity/time by timing radio signals from ≥4 satellites and trilaterating (with the 4th satellite resolving the receiver's own clock error). **It fails** when line-of-sight to enough satellites is blocked (tunnels, underground parking — total loss) or degraded (urban canyons, dense forest — multipath, reduced satellite count) or actively disrupted (jamming).

**IMU** = accelerometer + gyroscope (+ magnetometer). The **accelerometer** measures *specific force* — true acceleration *plus* the reaction to gravity — never true acceleration alone; a stationary phone reads ~9.81 m/s² "up," not zero. The **gyroscope** measures angular velocity (rate of rotation), not absolute orientation. The **magnetometer** measures local magnetic field, useful as a coarse compass but unreliable inside a vehicle's steel/electronics environment.

**INS** = an IMU plus the mechanization math that integrates those raw readings into a continuously updated position/velocity/attitude, without needing any external reference — which is exactly what makes it the natural GNSS fallback.

**Dead reckoning** = the general technique of estimating current position from a known start plus measured motion since. INS is one specific implementation of dead reckoning.

**Why this is mathematically hard**: position requires *double*-integrating acceleration. A constant accelerometer bias of just 0.02 m/s² — smaller than typical MEMS specification tolerances — produces roughly 1.2 m/s of velocity error and **36 m of position error after only 60 seconds** of unaided integration (worked in full in the End-to-End Trace document, Part 11). Bias, noise, and gravity-frame coupling (you must know orientation accurately to subtract gravity correctly, and gyro bias corrupts orientation, which then corrupts the gravity subtraction — a compounding effect) together make naive "just integrate the accelerometer" fail within seconds on consumer MEMS hardware. `[RESEARCH FINDING]`

**Coordinate frames** — device frame (however the phone happens to be mounted), vehicle frame (forward/lateral/vertical of the car), local navigation frame (a flat, locally-Euclidean frame used for the actual integration math), and the geographic frame (WGS84 lat/lon, used only for display and GNSS fusion). A frame mistake produces plausible-looking garbage no downstream filtering or ML can fix — this is why calibration/alignment is treated as foundational, not a minor setup step (Section 14).

---

# 4. Final System Concept

```mermaid
flowchart TB
    subgraph Sensors[Sensor Layer — ALWAYS AVAILABLE]
        IMU[IMU: accel, gyro, mag]
        GNSS["GNSS (when present)"]
    end
    subgraph Classical[Classical Stack — ALWAYS ACTIVE]
        PP[Calibrate → Align → Filter (Body Specific Force & Gyro)]
        STRAP[Strapdown INS Propagation (Attitude Rotation + Gravity Addition)]
        ESKF[ESKF]
        NHC[NHC]
        MM[HMM Map Matching]
        FSM[Mode FSM]
    end
    subgraph AIML[AI/ML — ALWAYS ACTIVE, not GNSS-gated]
        VN[VelocityNet]
        BN[BiasNet]
    end
    IMU --> PP --> STRAP --> ESKF
    PP --> VN --> ESKF
    PP --> BN --> ESKF
    GNSS -. "quality-gated measurement" .-> ESKF
    ESKF --> NHC --> MM --> FSM --> OUT["Mobile UI / Edge output"]
```

- **Normal GNSS navigation**: GNSS quality-checked → ESKF update alongside the always-running VelocityNet/BiasNet updates → NHC → map matching → output. Low, roughly steady-state uncertainty.
- **GNSS outage**: GNSS branch simply stops firing (no special code path); IMU propagation, VelocityNet, BiasNet, NHC, and gated ZUPT continue; prediction increases uncertainty while measurement updates reduce uncertainty in observed directions; position uncertainty generally increases overall, though VelocityNet, NHC, and ZUPT updates locally constrain velocity and orientation drift; mode → `DR_ONLY`.
- **GNSS recovery**: new fix passes a plausibility/innovation check → mode → `REACQUIRING` → bounded-rate blend over ~1-2 s → mode → `GNSS_AIDED`. Never an instant snap.
- **Training pipeline & Deployment matrix**: PyTorch training → ONNX export → ONNX Runtime on edge and LiteRT on Android, with export/numerical parity tests. Quantization (FP32 baseline, FP16/INT8 empirical evaluation based on on-device profiling) is evaluated against accuracy/latency budgets, not assumed.
- **Mobile deployment**: Android app, Kotlin core / LiteRT inference, OSMDroid map, foreground service.
- **Edge deployment**: Python + ONNX Runtime, identical shared navigation core, CLI I/O, no UI.

**CLASSICAL** (deterministic, no learned weights): timestamp sync, calibration, alignment, filtering, strapdown mechanization (attitude rotation + gravity addition), the ESKF's own predict/update math, NHC, HMM map matching, mode FSM, GNSS quality scoring. **AI/ML**: VelocityNet, BiasNet — nothing else. **HYBRID**: the ESKF itself, in the sense that its math is 100% classical but two of its measurement *inputs* are learned. **GNSS-dependent**: only the GNSS quality-check → ESKF-update branch. **GNSS-independent**: everything else, including both ML models.

---

# 5. Definitive Architecture — Every Layer/Module

| # | Module | Purpose | Inputs | Outputs | Algorithm | Frequency | Dependencies | Failure handling / fallback | AI involved? |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Sensor Ingestion | Unify async sensor callbacks into one ordered stream; attach quality flags | Raw IMU/GNSS/external packets | Time-ordered records with quality flags | Timestamp-tagged buffering + non-destructive quality tagging | Native rate | — | Flag/interpolate short gaps, flag long dropouts; preserve raw data | No |
| 2 | Calibration | Estimate gyro bias and initialize attitude tilt from rest; set accelerometer bias prior | Ingested stream | Gyro bias and calibration prior | Rest-period mean/variance | Per session | 1 | Use last-session profile if no rest period observed | No |
| 3 | Alignment | Device→vehicle rotation | Calibrated stream, GPS heading | `R_b^v` | Gravity-vector pitch/roll + GPS-heading yaw | Per session, re-triggered on event | 2 | Hold previous alignment; coarse magnetometer seed until yaw resolves | No |
| 4 | Filtering | Remove vibration/pothole nuisance signal | Aligned vehicle-frame specific force & gyro | Denoised vehicle stream (`f^v, ω^v`) | Median spike filter + 4th-order Butterworth | Native rate | 3 | Retain flagged extreme-motion samples without deleting | No |
| 5 | Gravity Resolution & Strapdown | Rotate specific force to nav frame, add physical gravity | Filtered `f_m^v, ω_m^v`, ESKF attitude `R_v^n` | Navigation-frame coordinate acceleration `a^n` | `a^n = R_v^n · (f_m^v − b_a^v) + g^n` (`g^n = [0,0,-g]ᵀ`) | Native rate | 4, ESKF state | Bounded by NHC + fusion | No |
| 6 | INS Propagation | Kinematic state integration | `a^n`, `ω_m^v`, prior state | Propagated position/velocity/attitude | Quaternion attitude integration + double integration | Native rate (up to ~200 Hz edge) | 5 | Bounded by NHC + fusion | No |
| 7 | Feature Builder | Windowed ML input tensor | Filtered vehicle stream (`f^v, ω^v`) | (20,9) tensor | Decimate to canonical 10 Hz, derive `\|f^v\|,\|f_dot^v\|,\|ω^v\|`, normalize | Every 0.5 s stride | 4 | Skip cycle if timestamp gap too large | No |
| 8 | VelocityNet | Forward-speed measurement source | (20,9) tensor | speed + log-variance | GRU(9→64, 2L) | ~2 Hz | 7 | Skip cycle if input invalid/OOD | **Yes** |
| 9 | BiasNet | Bias-correction measurement source | (20,9) tensor | 6-vector Δbias + log-variance | GRU(9→48, 2L) | ~1 Hz | 7 | Same | **Yes** |
| 10 | GNSS Quality Scoring | Continuous trust score | GNSS fix, filter innovation | Trust score / R | Rule-based (accuracy, sat count, plausibility, innovation test) | Per fix | — | Conservative thresholds | No |
| 11 | ESKF | Central state estimator | Propagated state, GNSS (if trusted), VelocityNet, BiasNet, NHC | Fused state + covariance | Error-state Kalman filter | Native predict rate; update whenever a measurement arrives | 6, 8, 9, 10 | Innovation gate rejects implausible updates; covariance reset on divergence | Hybrid (classical math, 2 learned inputs) |
| 12 | NHC | Vehicle-frame kinematic constraint | Fused state | Constrained state | Pseudo-measurement, lateral/vertical velocity ≈0 | Every fusion cycle | 11 | Relaxed/skipped during detected skid | No |
| 13 | Map Matching | Snap to road network | Constrained trajectory, OSM graph | Snapped (or unsnapped) position | HMM (Newson & Krumm) | ≤2 Hz | 12 | Falls back to unsnapped estimate | No |
| 14 | Mode FSM | Track GNSS_AIDED/DR_ONLY/REACQUIRING | Trust score, fusion state | Mode + smoothed output | Hysteresis state machine | Per cycle | 10, 11 | Minimum dwell time prevents flapping | No |
| 15 | Mobile App | User-facing product | Final state stream | Rendered map, mode indicator | Kotlin production port (behaviorally equivalent to Python reference), OSMDroid | 10 Hz nav output, 30-60 fps render | 1-14 | Graceful degradation, no crash on any module failure | Hosts AI inference (LiteRT) |
| 16 | Edge Engine | Sensor-agnostic navigation core | External IMU/GNSS stream | Trajectory log/API output | Authoritative Python reference implementation + ONNX Runtime | Native rate (classical), ~1-2 Hz (ML) | 1-14 | Same | Hosts AI inference (ONNX Runtime) |
| 17 | Logging/Telemetry | Debuggability, offline re-evaluation | Raw + intermediate signals, mode transitions | Timestamped logs | Local files/DB | Continuous | — | — | No |
| 18 | Evaluation Pipeline | Reproduce the PS's own drift metric | Held-out IO-VNBD, self-collected data | Metrics, plots | Three-axis evaluation suite (Section 26) | Offline | 1-16 outputs | — | No |

---

# 6. Classical vs. AI/ML Boundary — Definitive Table

| Component | Classical / ML / Hybrid | Exact responsibility |
|---|---|---|
| Timestamp sync | Classical | Merge async streams onto one ordered timeline |
| Calibration | Classical | Estimate gyro bias and attitude tilt from rest; initialize accelerometer bias prior for dynamic ESKF refinement |
| Alignment | Classical | Device→vehicle rotation (`R_b^v`) |
| Filtering | Classical | Fixed Butterworth + median filter on body specific force & gyro |
| Gravity resolution & strapdown | Classical | Rotates body specific force to nav frame using ESKF attitude and adds physical gravity (`g^n = [0,0,-g]ᵀ`) |
| Strapdown INS | Classical | Attitude/velocity/position propagation every sample |
| VelocityNet | **AI/ML** | Predicts forward speed (the one quantity nothing classical can supply without OBD-II) |
| BiasNet | **AI/ML** | Predicts context-dependent bias correction beyond static calibration |
| ESKF | **Hybrid** | 100% classical predict/update math; two of its measurement *sources* are learned (VelocityNet, BiasNet); GNSS and NHC remain classical measurement sources feeding the same mechanism |
| NHC | Classical | Zero lateral/vertical velocity pseudo-measurement |
| GNSS quality scoring | Classical | Rule-based trust score, feeding a continuous, time-varying `R` — not a trained classifier |
| Map matching | Classical | HMM over OSM road graph |
| Mode FSM | Classical | Hysteresis state machine over the trust score and fusion state |

**What AI does**: predicts forward speed (VelocityNet) and IMU bias correction (BiasNet), each with its own uncertainty, both feeding the ESKF as ordinary weighted measurements. **What AI does NOT do**: run the filter itself, decide GNSS trust, detect outages, perform map matching, enforce NHC, or ever directly overwrite the navigation state. **What ESKF does**: the single authoritative state estimator, combining every measurement source (classical and learned) through one consistent, covariance-aware mechanism. **What INS does**: the physics-based propagation step that runs every sample, regardless of GNSS or ML availability. **What NHC does**: injects the free, sensor-independent fact that a car doesn't slide sideways or fly, as a pseudo-measurement. **What map matching does**: the one correction source external to the sensor chain entirely — snaps the trajectory onto the real road network when confident, never overriding when not. **What GNSS does**: the highest-value, but not sole, absolute-position correction source, quality-gated before it's trusted.

---

# 7. Final AI/ML Architecture

**Number of ML models: exactly two.** `[OUR DESIGN DECISION, re-confirmed across three rounds of review]`

**Model names**: `VelocityNet`, `BiasNet`.

### Model 1 — VelocityNet
- **Purpose**: virtual speedometer — the one quantity classical dead reckoning cannot supply without an OBD-II feed the PS explicitly says not to assume.
- **Input**: (20, 9) tensor — 2.0 s @ canonical 10 Hz, [f_x^v, f_y^v, f_z^v, ω_x^v, ω_y^v, ω_z^v, `|f^v|`, `|f_dot^v|`, `|ω^v|`], representing aligned vehicle-frame specific force and angular rate (aligned via `R_b^v`, filtered; not gravity-compensated), normalized.
- **Output**: scalar forward speed (m/s) + scalar log-variance.
- **Architecture**: GRU(9→64, 2 layers) → Dense(64→32, ReLU) → Dense(32→2).
- **Training labels**: median-smoothed GPS/wheel-speed at window end (causally consistent with live inference), from IO-VNBD's synchronized `S-`/`V-` files.
- **Runtime role**: continuous velocity pseudo-measurement into the ESKF, ~2 Hz, active at all times (not GNSS-gated).

### Model 2 — BiasNet
- **Purpose**: context-dependent correction to accel/gyro bias, beyond what static calibration captures — the one technique with direct published precedent on this exact dataset (Onyekpe, Palade & Kanarachos, *Applied Sciences* 2021).
- **Input**: identical (20, 9) tensor to VelocityNet (aligned vehicle-frame specific force and angular rate; deliberately excludes the ESKF's own current bias state, to avoid a feedback loop).
- **Output**: 6-vector [Δb_ax,ay,az, Δb_gx,gy,gz] + 6-vector log-variance.
- **Architecture**: GRU(9→48, 2 layers) → Dense(48→24, ReLU) → Dense(24→12).
- **Training labels**: **computed, not observed** — a short-horizon propagation-vs-GPS-reference least-squares optimization per window (full procedure in Section 11).
- **Runtime role**: continuous bias-state pseudo-measurement into the ESKF, ~1 Hz, active at all times.
- **Decoupled Standalone Fallback**: `VelocityNet + Classical ESKF + GNSS + NHC (+ gated ZUPT)` forms the fully functional, validated standalone baseline. BiasNet is an experimental add-on. If BiasNet fails its validation gate or proves unstable, it is cleanly disabled without impacting system functionality.

**Deviation note, explicit**: earlier drafts of the FeatureSpec document described the residual-correction model ambiguously, sometimes as "the same model" as the speed estimator, and left the filter choice open between ESKF and UKF. Both ambiguities were resolved during this project's own cross-document review (documented in Section 30) — this section states only the final, settled answer.

**Third/fourth candidate models considered and rejected** (motion classifier, GNSS-quality classifier, vibration-suppression model, heading model, map-matching model, a separate uncertainty model, a learned Kalman gain) — full per-candidate reasoning is not needed for these since they were **not built**; the summary: each has an adequate, simpler classical or rule-based alternative already in the architecture, and no evidence justified the added model.

---

# 8. Exact Model Contracts

| | VelocityNet | BiasNet |
|---|---|---|
| Input tensor shape | `(batch, 20, 9)` | `(batch, 20, 9)` |
| Input channels | accel_x,y,z (m/s²), gyro_x,y,z (rad/s), `\|a\|`, `\|jerk\|`, `\|ω\|` | Identical |
| Units (pre-normalization) | m/s², rad/s, m/s³ for jerk | Identical |
| Sampling rate | Canonical 10 Hz (decimated from any native rate) | Identical |
| Window duration | 2.0 s | Identical |
| Stride | 0.5 s | Identical |
| Normalization | Fixed per-channel z-score, computed once on the training split, stored in `model_config.json` | Identical |
| Output | `(speed, log_variance)`, shape `(batch, 2)` | `(Δbias×6, log_variance×6)`, shape `(batch, 12)` |
| Uncertainty | Heteroscedastic auxiliary head, Gaussian NLL loss | Same mechanism; loss starts as Huber (noisier, optimization-derived label), graduates to NLL once label pipeline validated |
| Inference cadence | ~2 Hz | ~1 Hz |
| Latency target | `[ENGINEERING ESTIMATE, unverified]` < 5 ms/inference | Same |
| Training data | IO-VNBD `S-`/`V-` synchronized files, driver/file-level split | Same, `V-` GPS preferred for reference quality; windows overlapping a real GPS outage excluded from the label set |
| Loss | Gaussian NLL | Huber → Gaussian NLL (staged) |
| Fallback | Skip pseudo-measurement this cycle if input invalid/OOD or innovation gate rejects it | Skip pseudo-measurement if gate rejects, output clamped to safe range, or disable BiasNet entirely if validation fails |
| Acceptance Gate | Held-out RMSE beats naive baseline, export parity verified, scenario-wise stability | Measurable, statistically significant RMSE reduction over no-BiasNet baseline; zero regressions > baseline uncertainty; calibrated NIS variance |
| Deployment Pipeline | PyTorch training → ONNX export → LiteRT on Android, with export & numerical parity tests | PyTorch training → ONNX export → ONNX Runtime on edge, with export & numerical parity tests |
| Runtime & Quantization | LiteRT on Android (FP32 baseline; FP16/INT8 evaluated empirically on-device, not assumed) | ONNX Runtime on edge (FP32 baseline; FP16/INT8 evaluated empirically on-device, not assumed) |

---

# 9. Complete Sensor Inventory

| Sensor/source | Required? | Used by ML? | Used by classical nav? | Rate | Units | Frame | Device-dependent? |
|---|---|---|---|---|---|---|---|
| Accelerometer | **Required, guaranteed** | Yes | Yes | 10-200+ Hz | m/s² | Device | No |
| Gyroscope | **Required, guaranteed** | Yes | Yes | Same | rad/s | Device | No |
| Magnetometer | Optional, common | **No** | Fallback-only (coarse yaw seed before GPS-heading alignment converges) | 10-50 Hz | µT | Device | Common, not guaranteed |
| Gravity (`TYPE_GRAVITY`, virtual) | Optional | No | Sanity-check only (ESKF's own attitude-derived gravity is authoritative) | OS-fused | m/s² | Device | Common, not guaranteed |
| Rotation vector (`TYPE_ROTATION_VECTOR`, virtual) | Optional | No | Sanity-check only | OS-fused | quaternion | Device→ENU | Common, not guaranteed |
| Linear acceleration (`TYPE_LINEAR_ACCELERATION`, virtual) | Optional | No | Not used (we compute our own gravity compensation) | OS-fused | m/s² | Device | Common, not guaranteed |
| GNSS location (lat/lon) | Required when available | Label source only (training); measurement only (inference) | Yes | ~1 Hz (phone), ~10 Hz (`V-` VBOX) | degrees | WGS84 | No |
| GNSS speed | Required when available | Label/cross-check source | Yes | Same as fix | m/s | — | **Device-dependent reliability** (Doppler vs. differenced) |
| GNSS bearing | Optional | No | Yaw-alignment input, above a min-speed threshold | Same | degrees | — | Unreliable at low speed |
| GNSS accuracy | Required when available | No | Feeds trust score | Same | metres | — | Chipset/OS estimate |
| Satellite count / raw GNSS measurements | Optional | No | Optional enrichment to trust score | Per fix | dB-Hz etc. | — | **Device-dependent, not all chipsets expose** |
| External IMU (edge) | Required for edge deployment | Yes (via canonical decimation) | Yes (native rate) | Up to ~200 Hz | m/s², rad/s | Configured per sensor | By definition, external/configurable |
| FOG-grade IMU (edge) | Per PS `[OFFICIAL ISRO REQUIREMENT]` | Yes (via canonical decimation) | Yes (native rate) | ~200 Hz | m/s², rad/s | Configured | Domain gap acknowledged (Section 28) |
| Barometer | Not used | No | No | — | hPa | — | Optional, many phones lack it |

---

# 10. Dataset Strategy

- **Primary training dataset**: **IO-VNBD** `[OFFICIAL ISRO REQUIREMENT + IO-VNBD FACT]` — `S-` (smartphone, 10 Hz IMU + 1 Hz GPS, ~24 columns incl. pre-separated gravity XYZ) and `V-` (vehicle CAN/VBOX, 10 Hz, ~29 columns incl. wheel speeds and 10 Hz GPS) files, ~40h vehicle recordings + ~58h smartphone recordings across 8 drivers, 4 vehicles, UK/France/Nigeria.
- **Which files, exactly**: the **Synchronised V and S** folder, for both models — VelocityNet uses the `V-` wheel-speed (preferred) or GPS speed (fallback) as label; BiasNet uses the `V-` 10 Hz GPS as the higher-quality reference for its optimization-derived label, with windows overlapping a real GPS outage excluded.
- **Stationary segments** (>20 min, IO-VNBD): used to validate the classical calibration module, not directly for ML training.
- **GPS outage index file** (IO-VNBD): real-world outage windows, used as a secondary "found in the wild" evaluation category alongside synthetic outage masking.
- **Validation data**: held-out files from drivers also seen in training. **Test data**: at least one entirely held-out driver (Section 26).
- **Known IO-VNBD limitations**: no tagged tunnel/underground-parking scenario (synthetic masking compensates, disclosed as such); 10 Hz native rate (production phones sample faster — handled by the canonical-decimation rule, Section 21); no FOG/200 Hz data at all (cannot be validated until ISRO provides it); GPS-derived reference is not RTK-grade ground truth.
- **Additional public datasets**: **not used** — OxIOD/RIDI/RoNIN (pedestrian, wrong motion regime), comma2k19/KITTI/nuScenes (vehicle, but camera/LiDAR-centric autonomous-driving sensor rigs, not phone-mounted IMU) were investigated and rejected; none fills a real gap IO-VNBD and self-collection don't already address better `[RESEARCH FINDING]`.
- **Self-collected data**: **yes**, to close the 10 Hz-vs-native-rate gap and the tunnel-scenario gap — `[ENGINEERING ESTIMATE, unverified]` ~3-5 hours for initial validation, 10+ hours to support actual fine-tuning.
- **Synthetic GNSS-outage masking**: yes, sized to the PS's own benchmark examples (≈50 m/1 min; ≈1 km at 60 km/h) — explicitly not a substitute for real tunnel/garage multipath physics, only for the pure "no signal" condition.

---

# 11. Complete Data Pipeline (Offline/Training)

```
Raw IO-VNBD CSVs
→ Ingestion & Quality Tagging (preserve immutable raw records; attach quality flags for non-finite values, timing gaps, and extreme motion without destructive dropping)
→ Validated Downstream Stream (exclude only structurally unusable records like NaNs; retain flagged physical impact/pothole events for estimator awareness)
→ Timestamp Synchronization & Pairing (align S-/V- files where paired, cross-reference GPS outage index)
→ Calibration (estimate gyro bias at rest; initialize roll/pitch from gravity vector; set nominal accelerometer bias prior)
→ Device→Vehicle Alignment (estimate R_b^v from gravity vector + GPS track heading)
→ Temporal Filtering (Butterworth low-pass + median spike filter on aligned vehicle-frame specific force f^v and angular rate ω^v)
→ Resampling & Windowing (decimate to canonical 10 Hz, 2.0 s causal window [t-2.0, t], 0.5 s stride)
→ Feature Engineering (compute 9 channels: [f_x^v, f_y^v, f_z^v, ω_x^v, ω_y^v, ω_z^v, ||f^v||, ||f_dot^v||, ||ω^v||])
→ Normalization (per-channel z-score using training-split-only statistics)
→ Label Generation (VelocityNet: smoothed GPS/wheel speed at causal window end; BiasNet: optimization-derived bias correction)
→ Train/Validation/Test Split (driver + file level, strictly zero window-level leakage)
→ Training (Gaussian NLL / staged Huber→NLL, Section 8)
→ Evaluation (held-out drivers, three-axis evaluation suite — Section 26)
→ Export (PyTorch → ONNX → LiteRT for mobile; ONNX Runtime for edge)
```
Each stage's detailed why/how is documented in the Implementation Plan (Phase 1-8) and companion trace.

---

# 12. Complete Live Sensor Pipeline

```
Physical vehicle motion
→ Sensor measurement (accel/gyro native rate; GNSS when present)
→ Timestamping & Quality Tagging (elapsedRealtimeNanos; non-destructive quality flags attached)
→ Sync (buffer into unified time-ordered stream; unusable non-finite samples excluded from estimators)
→ Calibration & Alignment (subtract gyro bias; apply device→vehicle rotation R_b^v)
→ Filtering (median + Butterworth low-pass filter on vehicle-frame motion: f^v = R_b^v · (f_raw^b − b_a_prior), ω^v = R_b^v · (ω_raw^b − b_g))
→ [in parallel:]
    → ESKF Strapdown Mechanization (every sample, native rate): takes debiased vehicle specific force f_m^v − b_a^v, uses ESKF vehicle attitude quaternion R_v^n to compute a^n = R_v^n·(f_m^v − b_a^v) + g^n (g^n = [0, 0, -g]ᵀ in ENU), propagating velocity and position
    → Feature generation (canonical 10 Hz windowing on aligned vehicle-frame motion: [f_x^v, f_y^v, f_z^v, ω_x^v, ω_y^v, ω_z^v, ||f^v||, ||f_dot^v||, ||ω^v||]) → (20, 9) tensor → ML Inference (VelocityNet + BiasNet, ~1-2 Hz) → feeds ESKF measurement updates
→ ESKF Updates (GNSS-if-trusted + VelocityNet + BiasNet + Gated ZUPT, every cycle)
→ NHC (pseudo-measurement, every fusion cycle: lateral/vertical velocity ≈ 0)
→ Map Matching (≤2 Hz, downstream HMM over OSM graph; strictly zero feedback into ESKF)
→ State (position, velocity, heading, covariance, mode)
→ Local-frame → latitude/longitude conversion (equirectangular approximation about the single session reference origin)
→ UI (mobile) / output log-API (edge)
```
**Architecture Note on Gravity Handling**: In the final live architecture, the preprocessor does **not** attempt to strip gravity before passing data to the estimator. The preprocessor aligns and filters specific force into the vehicle frame. The ESKF holds the running vehicle attitude state $R_v^n$ and estimated vehicle-frame accelerometer bias $\mathbf{b}_a^v$, rotating vehicle specific force to navigation coordinates and adding physical gravity $\mathbf{g}^n = [0, 0, -g]^T$ internally inside strapdown propagation. (The standalone bootstrap attitude estimator is strictly a Phase 3 offline test utility, not a live dependency).

---

# 13. Coordinate Frames

| Frame | Definition | Used for |
|---|---|---|
| Device/body frame | Whatever orientation the phone chip physically has | Raw sensor readings arrive here |
| Vehicle frame | Forward/lateral/vertical of the car | NHC, both ML models' input frame |
| Local navigation frame | Flat, locally-Euclidean, ENU, centered on a single session-level reference point (frozen at the first trusted 3D GNSS fix; no mid-session resets) | All strapdown/ESKF integration math |
| Geographic frame | WGS84 lat/lon/alt | Display, GNSS fusion, map matching |

**Rotations**: quaternions internally (no gimbal lock, cheap to integrate/compose), converted to roll/pitch/yaw only for human-readable display/debugging. `R_b^v` (device→vehicle) from Section 14; `R_v^n` (vehicle→nav frame) from the ESKF's current attitude state.

**Why integrate in a local Cartesian frame, not directly in lat/lon**: a degree of longitude represents a different physical distance depending on latitude (`Δlon ≈ Δx / (R_earth · cos(lat))`), while a degree of latitude is nearly constant — running the integration equations directly on angular coordinates would require correcting for this distortion inside the loop itself and would behave inconsistently as latitude changes. The system instead integrates in local ENU Cartesian meters and converts to lat/lon only at the display/GNSS-fusion boundary, using a flat-Earth (equirectangular) approximation valid over the short distances relevant to driving trajectories. Crucially, a single session-level origin is used throughout each run: mid-session resets are avoided because they introduce artificial state jumps, disrupt covariance continuity, corrupt sliding-window trajectory buffers, and unnecessarily complicate downstream map matching. For typical driving distances (<50 km), the flat-Earth curvature error is under 0.2 m vertically and distortion is negligible horizontally. `[RESEARCH FINDING — standard navigation-engineering practice]`

---

# 14. Calibration + Alignment

**Startup sequence**:
1. App starts; phone presumed/prompted stationary.
2. Stationary-period gyro bias & orientation initialization:
   - **Gyroscope bias**: `b_g = mean(gyro)` (true angular rate is ≈0 at rest).
   - **Initial Pitch & Roll**: at rest, measured specific force $\mathbf{f}_m^b$ aligns with upward reaction to gravity (assuming initial accelerometer bias prior $\mathbf{b}_a \approx \mathbf{0}$) → initializes initial pitch and roll.
   - **Accelerometer bias observability principle**: A single static rest pose **cannot** independently separate accelerometer bias from gravity tilt (5 unknowns: 2 tilt angles + 3 bias axes, with only 3 accelerometer measurements). Multi-pose tumbling calibration is explicitly not required for v1. Instead, $\mathbf{b}_a$ is initialized with a nominal zero prior and dynamically estimated and refined by the ESKF during motion when GNSS fixes, NHC constraints, and ZUPT updates arrive.
3. **Yaw remains unresolved** at rest — gravity gives no horizontal-heading information; a coarse magnetometer seed may be used, explicitly downweighted, until step 4.
4. Once the vehicle moves above a minimum speed threshold on a roughly straight path, yaw is resolved by comparing sensed heading drift (gyro-integrated since step 2) against GPS track heading; the residual completes `R_b^v`.
5. `R_b^v` applied to every sample from here on, until a re-calibration trigger fires.
6. **Re-calibration trigger**: an orientation discontinuity inconsistent with plausible vehicle dynamics (phone bumped/remounted) — re-runs steps 1-5.

**Not implemented for v1**: accelerometer/gyro axis-misalignment calibration (bias dominates the error budget at this sensor grade — misalignment calibration is a P2 refinement), magnetometer hard-iron/soft-iron calibration (magnetometer is fallback-only by design, Section 9). Multi-position static tumbling is omitted in favor of dynamic in-flight ESKF bias estimation.

---

# 15. INS / Strapdown

**State propagated every IMU sample**: attitude (quaternion `q`), velocity `v`, position `p` (local ENU frame), plus the bias states tracked jointly inside the ESKF (Section 16).

```
q[k+1] = q[k] ⊗ Δq(ω_v[k]·Δt)                         # vehicle attitude propagation
g^n = [0, 0, -g]ᵀ                                     # physical gravity in ENU, g ≈ 9.80665 m/s²
a_true = R_v^n · (f_m^v − b_a^v) + g^n                # kinematic coordinate acceleration in ENU
v[k+1] = v[k] + a_true·Δt                              # velocity propagation
p[k+1] = p[k] + v[k]·Δt + ½·a_true·Δt²                 # position propagation
```
**Definitions & Direction Conventions**:
- `f_m^v`: measured specific force represented in the vehicle frame (derived from calibrated device accelerometer after applying alignment `R_b^v`; includes the ground reaction force counteracting gravity; reads `+9.81 m/s²` on `+Z` when vehicle is level on a horizontal surface).
- `b_a^v`: accelerometer bias represented in the vehicle frame.
- `R_v^n`: rotation matrix mapping vectors from vehicle frame (`v`) to local ENU navigation frame (`n`), derived from vehicle attitude quaternion `q`.
- `g^n`: physical gravitational acceleration in local ENU frame, directed downward toward Earth's center along `-Z` (`[0, 0, -g]ᵀ`).
- `a_true`: kinematic coordinate acceleration in local ENU frame.

**Stationary-Vehicle Numerical Sanity Check**:
With the vehicle at rest on a horizontal surface, facing north:
`R_v^n = I₃×₃`, `b_a^v = 0`. The support surface exerts an upward reaction force on the proof mass: `f_m^v = [0, 0, +g]ᵀ`.
Then:
`a_true = I · [0, 0, +g]ᵀ + [0, 0, -g]ᵀ = [0, 0, 0]ᵀ`.
Kinematic coordinate acceleration is identically zero; vertical velocity and position do not drift.

Bias is **not** treated as a fixed constant — it is a tracked ESKF state, continuously refined by BiasNet's pseudo-measurements (Section 16) and, implicitly, by every GNSS update's effect on the joint covariance.

---

# 16. ESKF — Definitive Design

**State Dimension & Manifold Representation**:
- **Nominal state** `x` (16 elements): `[position(3), velocity(3), attitude-quaternion(4), accel_bias(3), gyro_bias(3)]`, with unit quaternion constraint `||q|| = 1`.
- **Error state** `δx` (15 elements): `[δposition(3), δvelocity(3), δattitude(3), δaccel_bias(3), δgyro_bias(3)]`, where attitude error `δθ ∈ so(3)` is parameterized as a minimal 3-parameter rotation vector (`q = q̂ ⊗ [1, ½δθ]ᵀ`).
- **Covariance matrix** `P`: strictly **15×15**. A 16×16 covariance would be singular and overparameterized due to the quaternion unity constraint; all Kalman gain, innovation covariance, and covariance update recursions operate strictly in 15 dimensions.

**Prediction** (every IMU sample): nominal state via Section 15's equations; `P ← F·P·Fᵀ + Q`.

**Update** (whenever a measurement is available):
```
innovation y = z − h(x)
K = P·Hᵀ·(H·P·Hᵀ + R)⁻¹
x ← inject(x, K·y) ;  P ← (I − K·H)·P ;  reset δx ← 0
```

| Measurement source | `z` | `h(x)` | `H` Jacobian & Notes | `R` |
|---|---|---|---|---|
| GNSS (when trusted) | lat/lon/speed/heading, converted to local frame | predicted position/velocity | `H_gnss = [I₆×₆, 0₆×₉]` | From reported accuracy, quality-gated (Section 17) |
| VelocityNet | predicted forward speed | velocity projected onto vehicle forward axis: `e_fwdᵀ · R_n^v · v^n` | `H_vnet = [0₁×₃, e_fwdᵀ · R_n^v, e_fwdᵀ · R_n^v · [v^n]_×, 0₁×₃, 0₁×₃]` (attitude-dependent) | VelocityNet's own predicted log-variance |
| BiasNet | `current_bias_estimate + Δbias` | bias sub-state, direct selection | `H_bias = [0₆×₆, 0₆×₃, I₆×₆]`; innovation is predicted `Δb` | BiasNet's own predicted log-variance |
| NHC | 0 (lateral), 0 (vertical), vehicle frame | lateral/vertical velocity components | `H_nhc = [0₂×₃, e_lat/vertᵀ · R_n^v, e_lat/vertᵀ · R_n^v · [v^n]_×, 0₂×₃, 0₂×₃]` | Small fixed noise term, inflated/skipped during detected skid |
| Gated ZUPT | `[0, 0, 0]ᵀ` (when stationary confirmed) | velocity in navigation frame `v^n` | `H_zupt = [0₃×₃, I₃×₃, 0₃×₃, 0₃×₃, 0₃×₃]` | `σ_z² · I₃×₃` (e.g., `σ_z = 0.03 m/s`) |

**Filter choice: ESKF, not UKF.** `[OUR DESIGN DECISION, research-grounded]` UKF gives at most marginal accuracy gains over error-state EKF for MEMS-grade GPS/INS integration at meaningfully higher per-cycle cost (2n+1 sigma points vs. one linearized propagation) — unjustified against the shared 10 Hz/~100 ms budget split across two ML models, NHC, and map matching. `[RESEARCH FINDING — El-Sheimy, Shin & Niu, Inside GNSS 2006]` UKF remains a documented future option only if profiling shows spare budget and a specific measured accuracy gap ESKF can't close.

**Gating/consistency checks**: reject (skip) an update if the innovation `y` is implausibly large relative to `R` (chi-squared/Mahalanobis-style gate) — protects the filter from a single bad GNSS fix or a momentarily out-of-distribution ML output corrupting the state in one step.

**Divergence handling**: covariance reset if numerical divergence is detected.

**Decoupled Standalone Operation**: `VelocityNet + Classical ESKF + GNSS + NHC + Gated ZUPT` is fully functional and standalone without BiasNet. BiasNet is strictly an additive, gated pseudo-measurement; if disabled or unvalidated, the classical bias random walk in `Q` governs bias propagation.

---

# 17. GNSS Quality + Mode Switching

**GNSS quality state** (continuous, classical): a trust score built from reported accuracy, satellite count, fix-to-fix plausibility, and the ESKF's own innovation statistics on recent GNSS updates — **not** a discrete state, and **not** a trained classifier `[re-confirmed via Section 30's decision register]`.

**Navigation FSM mode** (discrete, three states): `GNSS_AIDED ⇄ DR_ONLY ⇄ REACQUIRING`.

**The distinction, stated explicitly** (this was a real ambiguity in earlier drafts, resolved during review): "GNSS degraded" is **not** a fourth FSM state — it lives entirely inside `GNSS_AIDED` as a falling continuous trust score that down-weights (larger `R`) rather than rejects GNSS updates. Only a *confirmed unavailable* signal (after a grace period) triggers `DR_ONLY`; only a *confirmed plausible* returning fix triggers `REACQUIRING`.

| State | Entry | Exit | Notes |
|---|---|---|---|
| `GNSS_AIDED` | Default; or REACQUIRING converges | Outage confirmed | Degraded GNSS handled internally via trust score, not a mode change |
| `DR_ONLY` | Outage confirmed (grace period elapsed) | Plausible fix returns | Uncertainty generally increases (GNSS absent), while ML, NHC, and ZUPT constrain velocity and orientation drift |
| `REACQUIRING` | Plausible fix while in `DR_ONLY` | Converged (agreement across a few consecutive fixes) or fix proves implausible | Bounded-rate blend, distinct from steady-state `GNSS_AIDED` handling because starting uncertainty is much larger |

**Hysteresis/timeouts/safety checks**: minimum dwell time per state (prevents flapping); outage-confirmation grace period; plausibility/innovation gate at every transition boundary.

**A fourth state for "long vs. short outage" was considered and rejected**: the ESKF's own covariance `P` already encodes elapsed-outage duration continuously and correctly — a discrete duration-based state would be redundant and add another hand-tuned threshold to get wrong.

**Jamming/spoofing** `[scope decision]`: treated identically to any other implausible fix via the same plausibility/innovation gate — full anti-spoofing (carrier-phase consistency, multi-constellation cross-validation) is GNSS-receiver security engineering, explicitly out of scope; the PS asks for continuity through degraded/lost GNSS, not a spoofing-resistant receiver.

---

# 18. Dynamic Constraints: NHC & Gated ZUPT

### 18.1 Non-Holonomic Constraints (NHC)
A wheeled vehicle's lateral and vertical velocity **in the vehicle frame** are approximately zero (no sideways slide, no lift-off). Fed to the ESKF as a pseudo-measurement (`z_lateral=0, z_vertical=0`, small `R`) through the identical update mechanism as any other measurement — no special-case code path.

**Applied**: every fusion cycle, after the main ESKF update, further tightening the already-corrected state.

**Relaxed/skipped when**: a genuine skid (implausibly large NHC innovation), an abnormal maneuver, or a sensor-failure condition making the vehicle-frame projection itself unreliable (e.g., gyro saturation during an extreme maneuver).

### 18.2 Classical Gated ZUPT (Zero Velocity Update)
- **Zero-ML Dependency**: Stationary detection operates entirely independently of VelocityNet or neural inference, relying purely on classical inertial signal processing:
  1. Gyroscope norm: $\|\boldsymbol{\omega}^b\| < 0.05 \text{ rad/s}$
  2. Accelerometer magnitude & variance: $|\|\mathbf{f}_m^b\| - g| < 0.25 \text{ m/s}^2$ and $\text{Var}(\|\mathbf{f}_m^b\|) < 0.015 \text{ (m/s}^2\text{)}^2$ over a sliding window $W \approx 0.8\text{ s}$ (8 frames at 10 Hz)
  3. Classical external signals (when available): GNSS speed $< 0.1 \text{ m/s}$ or CAN wheel speed $= 0$.
- **Optional ML Confirmation**: Downstream, if VelocityNet is operational, its predicted speed $< 0.15 \text{ m/s}$ can serve as an additional confirmation flag, but ZUPT remains 100% operational without ML.
- **Role**: Opportunistic stationary update applied at engine start, traffic lights, stop signs, and tunnel traffic standstills. It resets accumulated velocity error to zero and allows IMU biases to be observed. **It is NOT an always-available moving outage solution**; moving outages rely on INS propagation + VelocityNet + NHC.

---

# 19. Downstream Map Matching (OSM + HMM)

```
ESKF State Estimate → NHC & ZUPT Tightening → Fused Trajectory (lat/lon)
→ Candidate generation: nearby OSM road-graph edges within search radius
→ Emission probability: Gaussian-in-distance to each candidate edge
→ Transition probability: road-graph routing connectivity between consecutive candidates
→ Fixed-Lag Viterbi decode: online sliding window (last 5–10 epochs) finds the most probable road path
→ Trajectory Snapping: snap displayed trajectory only if match confidence is high; otherwise emit unsnapped estimate
```

**Downstream Pipeline Flow (Strictly Output-Level)**:
$$\text{ESKF} \longrightarrow \text{NHC / ZUPT} \longrightarrow \text{Map Matching} \longrightarrow \text{Output / Display}$$
Map matching is strictly a **downstream snapping module** for display, route progress, and road-corridor visualization. In v1, it **never feeds corrections back into the ESKF state vector or covariance matrix**. This architectural decoupling prevents dangerous feedback loops, corridor mis-snapping disasters (e.g., snapping onto a parallel frontage road), and unmodeled cross-correlations.

**Fallback Design**: If candidate confidence is low or OSM coverage is absent, the system seamlessly displays the raw unsnapped ESKF estimate without disruption. `[RESEARCH FINDING — Newson & Krumm, ACM SIGSPATIAL 2009]`

---

# 20. Real-Time Execution Model

| Cadence | Operations |
|---|---|
| **Native rate, up to ~200 Hz** (edge)/typical phone rate | Ingestion, calibration application, alignment application, filtering, strapdown propagation & gravity resolution |
| **~2 Hz** | VelocityNet inference + ESKF update |
| **~1 Hz** | BiasNet inference + ESKF update |
| **Event-driven** | GNSS fix arrival → quality check → ESKF update or outage/recovery evaluation |
| **10 Hz** (mobile) `[OFFICIAL ISRO REQUIREMENT]`, higher on edge | Final navigation output: NHC applied, map matching (may lag, ≤2 Hz internally, reusing last result between its own updates), lat/lon conversion, emit to UI/log |
| **30-60 fps** | UI rendering, decoupled from the 10 Hz nav-state update via interpolation |

---

# 21. Phone vs. Edge

| Stage | Identical? | Difference |
|---|---|---|
| Ingestion | Logic identical | Adapter differs: `SensorManager` callbacks vs. file/socket packet stream |
| Calibration/alignment/filtering/strapdown | **Identical math & logic** | Python reference implementation on edge, behaviorally equivalent Kotlin port on Android; native rate differs (phone vs. up to ~200 Hz edge/FOG) |
| ML inference & Deployment | **Identical models, identical ~1-2 Hz cadence** | Definitive pipeline: PyTorch training → ONNX export → ONNX Runtime (edge) and LiteRT (Android) with export/numerical parity tests. Quantization (FP32 baseline, FP16/INT8 empirical evaluation on-device) is evaluated against accuracy/latency budgets, not assumed |
| ESKF, NHC, Gated ZUPT | Identical math & logic | Python reference vs. Kotlin production implementation |
| GNSS handling | Identical if present; if absent on an edge rig, the FSM simply never leaves the DR-only-equivalent logic, unmodified | — |
| Map matching | Identical | Downstream-only (no ESKF feedback). Some edge use-cases may simply not invoke it |
| Output | **Not identical, by design** | Mobile UI vs. edge log/API |

**Sensor adapter / shared-architecture concept**: a `SensorAdapter` interface standardizes every source (phone, external MEMS, FOG) into one packet schema (`seq, t_host_ns, t_sensor_ns, accel, gyro, mag?, declared_rate_hz, sensor_id`) at the boundary; nothing downstream of that boundary ever branches on sensor type or rate — the Feature Builder's canonical-10Hz-decimation step is the single mechanism that makes "same model, any sensor" true, not a claim requiring per-sensor special-casing.

---

# 22. Software Architecture

```
SensorManager / EdgeSensorAdapter → GNSSManager/EdgeGNSSAdapter
        → TimestampSynchronizer
        → Calibration → Alignment → Preprocessor (filtering)
                ├→ ESKF Strapdown (gravity resolution & propagation)
                └→ FeatureBuilder → ModelRunner (VelocityNet, BiasNet)
        → ESKF ← GNSSManager (fix, when present)
        → NHC → MapMatcher → ModeManager (FSM)
        → Logger / UI (mobile) / OutputWriter (edge)
```

**Core interfaces**:
```
SensorAdapter:   read() -> RawSample{accel, gyro, mag?, timestamp, sensor_id}
GnssAdapter:      read() -> GnssFix{lat, lon, speed, heading, accuracy, sat_count, timestamp} | None
NavigationCore:   process(raw_sample, gnss_fix?) -> PositionEstimate{lat, lon, heading, speed, mode, uncertainty}
ModelRunner:      infer(window_tensor) -> (mean, log_variance)
MapMatcher:       match(estimate, road_graph) -> SnappedPosition | None
```
`NavigationCore.process()` defines the canonical processing contract implemented by both the Python reference engine (used by the edge CLI) and the behaviorally equivalent Kotlin production port (used by the Android app) — realizing the shared navigation architecture defined in Section 21.

**Core data structures**: `RawIMUSample`, `AlignedIMUSample`, `FeatureWindow`, `GNSSSample`, `OrientationState`, `NavigationState` (position_local, velocity_local, orientation, biases, covariance, reference_point, mode, timestamp), `MLPrediction`, `MapMatchResult` — full field definitions in the End-to-End Trace document, Part 28.

---

# 23. Configuration + Artifacts

| Artifact | Contents | Versioned how |
|---|---|---|
| Model weights | VelocityNet/BiasNet, per format | Filename includes version + dataset-hash tag |
| `model_config.json` | Normalization means/stds, filter coefficients, window/stride params, channel order | Versioned alongside its weights file — single source of truth so training and deployment can never silently desync |
| Calibration profiles | Per-device bias/scale | Stored per session/device, not globally versioned |
| Map data | OSM extract for the demo region | Bundled artifact, versioned by extract date |
| Dataset split definition | Which driver/files → train/val/test | Small JSON/CSV in `/data/splits`, versioned in git (not the raw data itself) |
| Training config | Hyperparameters, split reference | One YAML/JSON per run |
| Experiment metadata | Metrics, git commit | MLflow or CSV fallback |

---

# 24. Failure Handling

| Failure | Fallback |
|---|---|
| Missing sensor (no accel/gyro at all) | Unrecoverable by definition — hard error, not a silent bad estimate |
| Malformed data / timestamp gaps | Skip affected window for ML; classical propagation continues on remaining valid samples |
| Bad GNSS (implausible jump) | Rejected by the plausibility/innovation gate, treated as no fix |
| GNSS spoof-like jump | Same mechanism as any implausible fix (Section 17's scope decision) |
| Vibration | Attenuated by the fixed filter chain; if still out-of-range, treated as a low-confidence window |
| Phone movement mid-drive | Re-calibration trigger (Section 14, step 7) |
| ML failure (NaN, garbage output) | Model inference monitor flags it; skip that cycle's pseudo-measurement |
| ML overconfidence | Independent ESKF innovation gate rejects an implausible update regardless of the model's own stated confidence |
| Map matching failure (no coverage) | Falls back to unsnapped estimate, no crash |
| ESKF divergence | Covariance reset |
| External IMU mismatch (rate, domain gap) | Rate: solved architecturally (canonical decimation). Domain gap: acknowledged limitation (Section 28), not solved for v1 |

**Governing principle**: the ESKF never blindly trusts any single input — every measurement source is gated by both its own confidence and an independent plausibility check.

---

# 25. Performance Targets

| Metric | Official benchmark | Engineering estimate (unverified) | Measured result |
|---|---|---|---|
| Positional drift | **< 10% of distance travelled** `[OFFICIAL]` | — | Not yet measured |
| Example benchmark A | **< 5 m / 50 m / < 1 min** `[OFFICIAL]` | — | Not yet measured |
| Example benchmark B | **< 100 m / 1 km @ 60 km/h** `[OFFICIAL]` | — | Not yet measured |
| Mobile update rate | **10 Hz** `[OFFICIAL]` | — | Not yet measured |
| Edge update rate | **~200 Hz w/ FOG** `[OFFICIAL]` | — | Not yet measured |
| ML inference latency | — | < 5 ms/model `[unverified]` | Not yet measured |
| Model size (each, quantized) | — | < 500 KB `[unverified]` | Not yet measured |
| Mode-switch latency | — | < 200 ms, our interpretation of "milliseconds" `[unverified]` | Not yet measured |
| Re-convergence time | — | ~1-2 s `[unverified]` | Not yet measured |
| Speed RMSE, ATE, RTE, heading error, map-matching accuracy | — | To be established via the ablation ladder (Section 26) | Not yet measured |

**No number in this table has been measured yet — every non-official-benchmark figure is an engineering target to validate, not a claimed capability.**

---

# 26. Testing + Validation Philosophy

- **Unit tests**: calibration/alignment/NHC math against synthetic known-answer inputs.
- **Integration tests**: full IO-VNBD sequence replay end-to-end.
- **Replay/regression tests**: re-run the fixed held-out test set after every change, track metric deltas.
- **Dataset tests**: driver/file-level split integrity (no window-level leakage — Section 11).
- **Sensor/phone tests**: ≥2 physical devices, catching sensor-API/mount variance.
- **Edge tests**: parity test — identical logged window through both the LiteRT (mobile) and ONNX (edge) model, assert matching output within float tolerance.
- **Field tests**: self-collected real-drive data (Section 10), including a genuine tunnel/underpass where feasible.

**Evaluation & Ablation Framework (Three Evaluation Axes)**:

To rigorously answer *"what actually improved because of AI/ML vs. classical constraints vs. operating conditions"*, the evaluation suite explicitly separates three distinct evaluation dimensions rather than mixing them into an ambiguous linear sequence:

* **Axis A — Fusion & Component Contribution (Ablation Ladder at 60s Outage)**:
  1. `Level 1: Pure Strapdown INS` — Unconstrained dead reckoning (quaternion propagation + gravity), baseline drift rate.
  2. `Level 2: INS + Continuous GNSS` — Nominal reference trajectory and baseline tracker accuracy.
  3. `Level 3: INS + GNSS + VelocityNet` — Quantifies exact dead reckoning drift reduction from ML speed aiding.
  4. `Level 4: INS + GNSS + VelocityNet + BiasNet` — Quantifies incremental gain from ML bias residual estimation.
  5. `Level 5: Level 4 + Classical NHC` — Evaluates lateral/vertical non-holonomic velocity constraints ($v_y^v \approx 0, v_z^v \approx 0$).
  6. `Level 6: Level 5 + Gated ZUPT` — Adds classical stationary zero-velocity updates during detected vehicle stops.
  7. `Level 7: Full System (+ Downstream Map Matching)` — Adds output-level HMM road-snapping (purely downstream, zero filter feedback).

  *Key Isolations Enabled by Axis A*:
  - *Classical-only baseline* vs. *+VelocityNet* (isolates speed ML).
  - *+VelocityNet* vs. *+VelocityNet + BiasNet* (isolates bias ML).
  - *Full ML fusion* vs. *Classical kinematic constraints* (isolates NHC/ZUPT physics).
  - *ESKF state* vs. *Map-matched output* (isolates display snapping).

* **Axis B — GNSS Outage & Operating-Condition Analysis**:
  - `B1: Continuous GNSS` — Open-sky baseline tracking error (RMSE).
  - `B2: Short Synthetic Outages (10s, 30s)` — Urban canyon / overpass drift scaling.
  - `B3: Standard Benchmark Outage (60s)` — Primary SIH competition metric (<1.5% distance drift).
  - `B4: Extended Outages (120s, 300s)` — Stress testing filter divergence bounds and covariance growth.
  - `B5: Real Environmental Outages` — Field test validation (underpasses, parking structures, tunnels).
  - `B6: Reacquisition & Recovery` — Convergence time, innovation Mahalanobis gate behavior, smooth covariance collapse.

* **Axis C — Output Processing & Display Refinement**:
  - `C1: Raw ESKF State` — Metric position, velocity, attitude directly from estimator.
  - `C2: ESKF + Kinematic Constraints` — Filter output with NHC and gated ZUPT active.
  - `C3: ESKF + Downstream Map Matching` — Snapped to offline OSM road network for UI presentation.

This 3-axis methodology provides bulletproof, scientifically defensible proof of ML efficacy required by ISRO evaluators.

---

# 27. Demo Requirements

- **Normal navigation**: live drive, GNSS-aided, smooth output.
- **GNSS degradation**: a segment with genuinely weak (not absent) signal, showing the continuous trust-score behavior rather than a hard mode flip.
- **GNSS outage**: a real overpass/underpass/parking structure, or a controlled mock-location injection for repeatability.
- **Continued navigation**: visibly smooth icon through the outage, debug overlay showing mode/uncertainty climbing.
- **GNSS recovery**: visible bounded-rate re-convergence, not a snap.
- **Edge engine**: replay of an externally-sourced (or synthetic high-rate) IMU log through the CLI, output compared side-by-side with the mobile app's output on the same underlying trajectory.
- **Debug overlay**: mode, trust score, uncertainty, live — this is not cosmetic, it's the credibility mechanism for the whole live demo.
- **Plots/metrics**: raw GNSS track vs. fused+snapped track vs. reference, on at least one IO-VNBD held-out sequence; the ablation-ladder table (Section 26).

---

# 28. Security / Limitations / Honest Claims

- **No claim of sub-meter "lane-level" accuracy as a standalone sensor-only capability** — only after map-matching snap, and only where OSM coverage exists.
- **Long outages (multi-minute) are explicitly beyond the PS's own stated benchmark** (which implies ~1 minute scale) — treated as a stretch case, not a guaranteed pass.
- **Indoor/underground parking**: relative DR only; no absolute road-graph anchor typically exists there, so map matching's benefit is limited to whatever OSM coverage happens to exist.
- **External-IMU/FOG domain gap**: both models are trained exclusively on IO-VNBD's smartphone-grade MEMS noise profile; they will very likely work *directionally* on cleaner FOG data but this is an **assumption, not a verified fact** until ISRO-provided FOG data allows re-validation.
- **Generalization**: driver/vehicle/country diversity is what IO-VNBD (UK/France/Nigeria, 4 vehicles) and self-collected Indian-road data together provide — no claim of universal generalization beyond what's actually been tested.
- **Spoofing**: handled only as "any other implausible fix" (Section 17) — this is explicitly not an anti-spoofing security system.
- **Dataset limitations**: no true RTK ground truth in IO-VNBD (GPS/wheel-speed only); no tagged tunnel scenario; 10 Hz native rate. All disclosed, not hidden, in the screening proposal.

---

# 29. What NOT to Build

| Idea | Why not |
|---|---|
| Giant Transformer for any ML component | Unjustified data appetite/mobile cost for 20-timestep windows and narrowly-scoped models |
| Fully end-to-end neural positioning (no ESKF) | Fragile, undebuggable, worse live-demo reliability than the classical-backbone hybrid (Section 7's architecture comparison) |
| Cloud inference / any backend | Conflicts with offline-first requirement; nothing in scope needs one |
| Unnecessary classifiers (motion-state, GNSS-quality, as trained models) | Rule-based alternatives are adequate and more defensible live (Section 7) |
| Neural map matching | HMM is mature, cheap, explainable; no evidence justifies replacing it |
| Federated learning | No multi-user deployment exists or is needed |
| Premature C++/Rust rewrite | No demonstrated performance bottleneck at these model sizes; Python reference and Kotlin production implementations are sufficient for the hackathon timeline |
| Full HD-map/lane-graph infrastructure | OSM is sufficient and PS-aligned; HD maps are a different, much larger engineering problem |
| Ensembling/MC-dropout uncertainty | Heteroscedastic single-head regression gives comparable value at a fraction of the mobile inference cost |
| UKF as the primary filter | Marginal accuracy gain over ESKF at meaningfully higher cost, for this sensor grade (Section 16) |
| A third bias/residual model separate from BiasNet | Explicitly merged into one model; no reason found for two |

---

# 30. Final Decision Register

| Decision | Final choice | Alternatives considered | Why chosen |
|---|---|---|---|
| Number of ML models | **Two** (VelocityNet, BiasNet) | One end-to-end model; many specialized models | Matches exactly the two quantities classical methods can't supply (speed w/o OBD-II, context-dependent bias); avoids the debuggability/data-appetite cost of an end-to-end model and the redundancy of a third+ model |
| Model architecture | **GRU** for both | LSTM, TCN, Transformer, MLP | Fewer parameters than LSTM at comparable accuracy; simpler mobile-runtime support and literature precedent over TCN; unjustified cost/data appetite for Transformer; MLP lacks temporal memory |
| Fusion filter | **ESKF** | UKF, full-state EKF | UKF's accuracy gain over ESKF is marginal for MEMS-grade sensors at meaningfully higher per-cycle cost; error-state formulation specifically handles orientation more robustly than full-state EKF `[El-Sheimy/Shin/Niu 2006]` |
| BiasNet's target | **6-vector bias correction** | Velocity residual, position residual, heading residual, generic "INS state residual" | Bias has a small, physically bounded, roughly stationary range regardless of outage duration; position/heading residual scale unboundedly with elapsed outage time, making them poorly-posed regression targets |
| Canonical ML input rate | **10 Hz, decimated from any source rate** | Train separate models per source rate; run ML at native rate | Matches IO-VNBD's own native rate (avoids upsampling artifacts); makes "one model, any sensor" architecturally true rather than aspirational |
| Map matching | **HMM (Newson & Krumm)** | Learned/graph-neural map matching | Mature, cheap, explainable; no evidence found favoring a learned matcher |
| GNSS outage/quality detection | **Rule-based, continuous trust score** | Trained classifier | Simple, effective, easier to defend live; the ESKF's own innovation statistics already provide the principled signal a classifier would approximate |
| Mode FSM | **Three states** (GNSS_AIDED/DR_ONLY/REACQUIRING) | Four+ states (explicit "degraded", explicit "long outage") | Continuous trust score and covariance already encode "degraded" and "how long" without extra flapping-prone states |
| Mobile inference runtime | **LiteRT** (renamed from TensorFlow Lite) | ONNX Runtime Mobile, PyTorch Mobile, native | Best-supported current first-party Android on-device runtime; drop-in migration from the TFLite naming used in earlier drafts |
| Edge inference runtime | **ONNX Runtime** | LiteRT, native | One shared ONNX export feeds both targets; no reason to run ONNX Runtime specifically on the phone when LiteRT is better-supported there |
| Primary dataset | **IO-VNBD** | OxIOD/RIDI/RoNIN, comma2k19/KITTI/nuScenes | Only dataset matching this exact niche (phone-mounted IMU + real vehicle + usable speed/position reference); others are pedestrian-focused or camera/LiDAR-centric autonomous-driving rigs |
| Backend/cloud | **None** | Fleet dashboard/backend | PS requires offline operation; nothing in scope needs a backend |
| Uncertainty mechanism | **Heteroscedastic regression (auxiliary variance head)** | Ensembles, MC dropout | Single forward pass, principled, far cheaper on mobile than either alternative |

---

# 31. Open Questions

`[OPEN QUESTION]` — genuinely unresolved without external input, not speculative filler:

1. **Exact IO-VNBD license/redistribution terms** — not independently re-verified with certainty; check before redistributing any derived model weights publicly.
2. **What "additional datasets" ISRO will provide at screening**, and in what format — the pipeline is built dataset-agnostic (same ingestion contract) specifically to absorb this unknown as a config change, not a rewrite.
3. **Whether the finale will provide live FOG hardware or only a recorded FOG-rate log** — affects whether R11 needs live hardware integration or replay-only support.
4. **Real-world INT8 quantization accuracy loss for VelocityNet/BiasNet** — assumed negligible given model size, but not yet empirically measured.
5. **Actual on-device inference latency** on genuinely mid-range (not flagship) target hardware — all current latency figures are engineering estimates (Section 25).
6. **Whether BiasNet's optimization-derived label is stable enough to graduate from Huber to Gaussian NLL loss** — Section 8 stages this deliberately as an open, to-be-validated question, not a settled fact.
7. **Domain-gap magnitude between IO-VNBD's smartphone-MEMS training distribution and real FOG-grade edge data** — cannot be resolved without ISRO-provided or independently sourced FOG data.
8. **OSM coverage quality for the actual finale demo venue** — needs to be checked once the venue/route is known, not assumed.

---

# 32. Final One-Page Project Summary

**What we are building**: a unified navigation architecture — realized as an authoritative Python reference engine for edge deployment and a behaviorally equivalent Kotlin production app for Android — that keeps a vehicle's estimated position on the road through a GNSS outage and reacquires GNSS seamlessly when it returns, for SIH PS 26168 (ISRO).

**How it works**: raw phone/external IMU undergoes immutable quality tagging, calibration, alignment to the vehicle frame, and filtering; a classical strapdown INS rotates vehicle-frame specific force into navigation coordinates using ESKF attitude, adds physical gravity (`g^n = [0,0,-g]ᵀ`), and propagates position/velocity/attitude every sample; two small GRU models (VelocityNet, BiasNet) continuously supply a forward-speed estimate and a bias correction, both as measurements into a classical Error-State Kalman Filter that also consumes GNSS (when trusted) and a Non-Holonomic-Constraint pseudo-measurement; the result is further constrained by HMM map matching against an offline OpenStreetMap extract; a three-state mode FSM tracks whether GNSS is currently aiding, absent, or being smoothly reacquired.

**What AI does**: exactly two things — estimates forward vehicle speed without an OBD-II feed, and corrects context-dependent IMU bias beyond what static calibration captures. Nothing else in the system is learned.

**What classical navigation does**: everything else — sensor fusion math, orientation/gravity handling, vehicle-motion constraints, map matching, and all mode-switching logic.

**What happens when GNSS disappears**: nothing about the IMU propagation or the two ML models changes at all — only the GNSS branch of the fusion filter goes quiet, and the system's own position uncertainty generally increases to reflect that while VelocityNet, NHC, and ZUPT updates continue to reduce uncertainty in their observed directions.

**How it returns**: a returning fix is checked for plausibility, then blended in over roughly a second or two, never snapped instantly.

**What data we train on**: IO-VNBD (primary), supplemented by a modest amount of self-collected Indian-road data to close two specific, disclosed gaps (native sampling rate, tunnel scenario coverage) that IO-VNBD alone doesn't cover.

**What the final software consists of**: one shared navigation architecture featuring an authoritative Python reference engine (`edge/` and `navigation/`), a behaviorally equivalent Kotlin production app (`android/`), and shared trained ONNX model artifacts (executed via ONNX Runtime on edge and LiteRT on Android).

**What success looks like**: reproducing the PS's own drift-percentage benchmark on held-out IO-VNBD data with a documented, honest ablation study separating component contribution (Axis A), outage duration (Axis B), and output refinement (Axis C) — demonstrated live, on both the mobile app and the edge engine, with a visibly seamless transition through a real or realistically simulated GNSS outage.

