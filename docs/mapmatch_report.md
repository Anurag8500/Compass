# Phase 12 Engineering Report: Downstream Map Matching & Trajectory Snapping (OSM + HMM)

**Project**: Cognitive Off-grid Machine-learning Positioning And Sensor System (C.O.M.P.A.S.S.)  
**Problem Statement**: SIH 26168 (ISRO)  
**Phase**: Phase 12 — Downstream Map Matching & Trajectory Snapping  
**Validation Gate**: *"Downstream map matching validated on both synthetic and real data; fallback behavior confirmed safe."*  
**Status**: **COMPLETE AND VALIDATED**

---

## 1. Executive Summary & Core Principle

Phase 12 integrates an offline OpenStreetMap (OSM) Hidden Markov Model (HMM) road matcher strictly downstream of the dead reckoning and sensor fusion pipeline.

```
ESKF + NHC + ZUPT
        ↓
fused trajectory (NavigationState / ESKFState)
        ↓
candidate road search (segment-safe AABB spatial binning)
        ↓
emission probability (covariance-aware road-normal Gaussian)
        ↓
road-graph transition probability (Dijkstra network routing vs displacement)
        ↓
fixed-lag online Viterbi (strictly causal, 8-epoch sliding window)
        ↓
confidence evaluation & safeguards (displacement, ambiguity, connectivity)
        ↓
snapped output OR safe fallback (exact original estimator coordinate)
        ↓
display / output telemetry only
```

### The Cardinal Rule: Strictly Downstream
Map matching operates purely on the display/output tier. Snapped coordinates, edge IDs, and map azimuths **NEVER feed back into**:
- ESKF nominal position, velocity, attitude quaternion, or sensor biases ($b_a, b_g$)
- ESKF error-state covariance matrix $P$
- VelocityNet or BiasNet ML inferences
- GNSS Finite State Machine (FSM)
- Non-Holonomic Constraints (NHC) or Zero Velocity Updates (ZUPT)

Every test confirms bit-for-bit numerical identity of the filter state before and after map matching.

---

## 2. OpenStreetMap (OSM) Offline Extract Details

To eliminate all internet dependency during live vehicle operations, the road network for the evaluation region (Coventry / Warwick, UK) was extracted offline and converted into a deterministic directed graph representation.

- **Bounding Box**:
  - South: $52.395^\circ\text{N}$, North: $52.425^\circ\text{N}$
  - West: $-1.610^\circ\text{E}$, East: $-1.500^\circ\text{E}$
  - Total Span: $2.4\text{ km} \times 6.6\text{ km}$ ($15.8\text{ km}^2$)
- **Source & Extract Date**: OpenStreetMap via Overpass API (2026-09-11).
- **Driveable Highway Filter**: `motorway`, `trunk`, `primary`, `secondary`, `tertiary`, `unclassified`, `residential`, and associated link ways. Non-driveable pedestrian paths, cycleways, and footways were excluded.
- **Graph Structure**:
  - Nodes: 25,252 nodes with WGS84 coordinates `(lat, lon)`
  - Directed Edges: 9,648 edges
  - Linear Subsegments: 47,661 linear line segments
  - Directionality: Bidirectional streets create twin directed edges (`way_id_fwd` and `way_id_rev`); one-way streets create a single directed edge.
- **Session ENU Harmonization**:
  The graph is saved locally at `data/maps/coventry_s1_road_graph.json` (9.03 MB). When loaded at runtime, all node coordinates and polyline vertices are projected into the active session-local East-North-Up (ENU) Cartesian frame using the session's fixed `GeoReference`. The estimator trajectory and the road graph operate in the exact same metric Euclidean frame.

---

## 3. Segment-Safe Candidate Search

Rather than relying on segment midpoints (which fail for long highway segments), `CandidateSearch` employs Axis-Aligned Bounding Box (AABB) spatial binning:
1. **Spatial Binning Index**: All 47,661 segments are indexed into a uniform 2D grid ($100\text{ m}$ cell size) expanded by the search radius $R_{\text{search}} = 35.0\text{ m}$.
2. **Exact Orthogonal Projection**: For each candidate segment $\overline{p_0 p_1}$ and query point $q$:
   $$t = \text{clamp}\left(\frac{(q - p_0) \cdot (p_1 - p_0)}{||p_1 - p_0||^2}, 0.0, 1.0\right)$$
   $$p_{\text{proj}} = p_0 + t(p_1 - p_0)$$
   $$d_{\text{perp}} = ||q - p_{\text{proj}}||_2$$
3. **Deterministic Pruning**: For each unique directed edge, only the closest segment projection is retained. Candidates are sorted deterministically by perpendicular distance ascending, then edge ID.

---

## 4. Covariance-Aware Emission Model

The emission probability integrates the directional position covariance from the ESKF state:
1. **Road-Normal Unit Vector**: Given segment tangent $t = [t_x, t_y]^T$, the unit normal is $n = [-t_y, t_x]^T$.
2. **Projected Positional Variance**:
   $$\sigma_d^2 = n^T P_{pp} n + \sigma_{\text{road}}^2$$
   where $P_{pp} = P[0:2, 0:2]$ is the $2 \times 2$ horizontal position error covariance from the ESKF, and $\sigma_{\text{road}} = 4.0\text{ m}$ is the baseline road/lane width uncertainty.
3. **Log-Gaussian Distance Likelihood**:
   $$\ln p(z_t | c_t) = -\frac{1}{2}\ln(2\pi \sigma_d^2) - \frac{d(z_t, c_t)^2}{2\sigma_d^2}$$
4. **Heading Consistency Term**: When vehicle forward velocity exceeds $1.5\text{ m/s}$ ($5.4\text{ km/h}$), heading difference $\Delta \psi = |\text{wrap}(\psi_{\text{veh}} - \psi_{\text{edge}})|$ adds:
   $$\ln p(\psi_t | c_t) = -\frac{1}{2}\ln(2\pi \sigma_\psi^2) - \frac{\Delta \psi^2}{2\sigma_\psi^2}$$
   with $\sigma_\psi = 25^\circ$ ($0.436\text{ rad}$).

---

## 5. Road-Graph Transition Model

Between candidate $c_{t-1}$ at time $t-1$ and candidate $c_t$ at time $t$:
1. **Trajectory Displacement**: Observed metric displacement $\Delta d_{\text{traj}} = ||z_t - z_{t-1}||_2$.
2. **Network Route Distance**: Shortest path along the directed graph:
   - *Same edge*: $d_{\text{along}} = c_t.\text{dist} - c_{t-1}.\text{dist}$. Backward travel on any directed edge returns $-\infty$ to enforce directed semantics.
   - *Different edges*: $d_{\text{graph}} = d_{\text{rem}}(c_{t-1}) + \text{Dijkstra}(v_{t-1}, u_t) + d_{\text{prog}}(c_t)$.
   - *Disconnected / Exceeds physical speed limit ($v_{\max} = 45\text{ m/s}$)*: returns $-\infty$.
3. **Exponential Transition Likelihood**:
   $$\ln p(c_t | c_{t-1}) = -\ln(\beta) - \frac{|d_{\text{graph}} - \Delta d_{\text{traj}}|}{\beta}$$
   with scale parameter $\beta = 5.0\text{ m}$.

---

## 6. Strictly Causal Fixed-Lag Online Viterbi

The Viterbi decoder operates online with a strict sliding buffer of $W = 8$ epochs:
- **Strict Causality**: At time step $t$, the system receives observation $z_t$ and updates dynamic programming scores $V_t(j)$ and backpointers. It NEVER peeks ahead into samples $t+1, t+2, \dots$.
- **Maturity Commitment**: A decision for epoch $t - W$ is permanently committed only when the buffer length reaches $W + 1$.
- **Mature Epoch Metadata Isolation**: The commit carries the exact metadata from epoch $t - W$ (`candidate_count`, scores, margin), ensuring that evaluation is completely decoupled from epoch $t$.
- **Deterministic Tie-Breaking**: When paths have identical scores (within $10^{-12}$), tie-breaking selects the candidate with the lowest lexicographical edge ID.
- **Bounded Memory**: Committed epochs are popped from the buffer, maintaining $O(W \cdot M)$ memory complexity.

---

## 7. Confidence Evaluation & Safe Fallback Logic

To prevent catastrophic snaps, five anti-catastrophic-snap safeguards are enforced before emitting a snapped position:

| Check | Safeguard Condition | Fallback Action | Reason Code |
|---|---|---|---|
| 1 | Candidate set is empty ($M = 0$) | Return exact estimator coordinate | `NO_CANDIDATES` |
| 2 | Perpendicular snap distance $d > d_{\max}$ ($25.0\text{ m}$) | Return exact estimator coordinate | `LARGE_DISPLACEMENT` |
| 3 | Runner-up alternative margin $\Delta L < \Delta L_{\text{margin}}$ ($1.0\text{ nat}$) | Return exact estimator coordinate | `AMBIGUOUS_PARALLEL_ROADS` |
| 4 | Best path transition score $\le -10^8$ (disconnected graph jump) | Return exact estimator coordinate | `DISCONNECTED_TRANSITION` |
| 5 | Composite confidence score $< \gamma_{\text{conf}}$ ($0.45$) | Return exact estimator coordinate | `LOW_CONFIDENCE` |

In **EVERY** fallback event, `display_enu == estimator_enu` and `display_lat_lon == estimator_lat_lon` bit-for-bit. A map matcher is allowed to do nothing when uncertain.

---

## 8. Verification on Synthetic Known Graph

A hand-constructed 3-road deterministic synthetic graph was tested:
- Road A (Main): $(0, 0) \to (200, 0)$
- Road B (Parallel): $(0, 12) \to (200, 12)$ ($12\text{ m}$ separation)
- Road C (Turn): $(200, 0) \to (200, 150)$

### Test Results:
1. **Candidate Projection**: Accurate to $< 10^{-3}\text{ m}$.
2. **Covariance Awareness**: Confirmed that increasing North position uncertainty dynamically inflates road-normal variance $\sigma_d^2$.
3. **Connectivity & Directionality**: Forward travel along Road A scores $-\ln(5.0)$; disconnected jump from Road B to Road C returns $-\infty$.
4. **Noisy Trajectory Decoding**: 100% of trajectory points correctly selected Road A followed by the turn into Road C.
5. **Causality Verification**: Changing trajectory points at $t > t_{\text{mature}} + W$ produced 0 change in committed outputs up to $t_{\text{mature}}$.
6. **Determinism**: 100.0% bit-for-bit identical outputs across repeated runs.

---

## 9. Real-Data Benchmark Results (IO-VNBD Session S1)

The complete Phase 12 benchmark was executed on IO-VNBD Session S1 (`Categorised_S1.npz`) at $10\text{ Hz}$ sampling rate.

### SIH Problem Statement Benchmark & Target Compliance Summary

| Scenario | Outage Duration | Distance Travelled | Final Drift (m) | Drift % | Official SIH PS Benchmark (<10.0%) | Internal Stronger Target (<1.5%) |
|---|---|---|---|---|---|---|
| **Scenario B (10s Outage)** | 10.0 s | 142.2 m | 7.15 m | **5.03%** | **PASS** (<10.0%) | **FAIL** (>1.5%) |
| **Scenario B (30s Outage)** | 30.0 s | 426.6 m | 86.19 m | **20.20%** | **FAIL** (>10.0%) | **FAIL** (>1.5%) |
| **Scenario B (60s Outage)** | 60.0 s | 839.5 m | 174.33 m | **20.77%** | **FAIL** (>10.0%) | **FAIL** (>1.5%) |

### Scenario Comparison Table

| Scenario | Duration | Snap Rate (%) | Fallback Rate (%) | Median Snap Dist (m) | P95 Snap Dist (m) | Max Snap Dist (m) | Phase 11 Estimator RMSE (m) | Phase 12 Map-Matched Display RMSE (m) |
|---|---|---|---|---|---|---|---|---|
| **Scenario A: Continuous GNSS** | 60.0 s (600 epochs) | **98.5%** | 1.5% | 1.34 m | 2.58 m | 2.84 m | **1.5496 m** | 1.8153 m |
| **Scenario B: 10s GNSS Outage** | 30.0 s (300 epochs) | **95.7%** | 4.3% | 1.97 m | 3.76 m | 6.87 m | **7.15 m** (final) | **7.15 m** (final) |
| **Scenario B: 30s GNSS Outage** | 50.0 s (500 epochs) | **45.6%** | 54.4% | 2.42 m | 4.34 m | 7.39 m | **86.19 m** (final) | **86.19 m** (final) |
| **Scenario B: 60s GNSS Outage** | 80.0 s (800 epochs) | **28.5%** | 71.5% | 2.42 m | 4.34 m | 7.39 m | **174.33 m** (final) | **174.33 m** (final) |
| **Scenario C: Sharp Turn Dynamics** | 40.0 s (400 epochs) | **69.2%** | 30.8% | 2.93 m | 4.38 m | 7.45 m | **19.988 m** | 20.088 m |
| **Scenario D: Stop-and-Go** | 30.0 s (300 epochs) | **97.0%** | 3.0% | 1.51 m | 2.31 m | 5.45 m | **1.375 m** | 2.205 m |
| **Scenario E: Zero Coverage** | 5.0 s (50 epochs) | **0.0%** | 100.0% | 0.00 m | 0.00 m | 0.00 m | N/A | N/A |

### Fallback Statistics Breakdown

- **Scenario A (Continuous GNSS)**:
  - Total Epochs: 600
  - Snapped: 591 (98.5%)
  - Fallbacks: 9 (1.5%)
  - Breakdown: `{'AMBIGUOUS_PARALLEL_ROADS': 9}`
- **Scenario B (60s Outage)**:
  - Total Epochs: 800
  - Snapped: 228 (28.5%)
  - Fallbacks: 572 (71.5%)
  - Breakdown: `{'AMBIGUOUS_PARALLEL_ROADS': 34, 'LOW_CONFIDENCE': 61, 'LARGE_DISPLACEMENT': 91, 'NO_CANDIDATES': 170, 'DISCONNECTED_TRANSITION': 216}`
  - Outage Distance: 839.5 m | Drift: 174.33 m (20.77%)
- **Scenario C (Sharp Turn)**:
  - Total Epochs: 400
  - Snapped: 277 (69.2%)
  - Fallbacks: 123 (30.8%)
  - Breakdown: `{'LOW_CONFIDENCE': 48, 'AMBIGUOUS_PARALLEL_ROADS': 3, 'LARGE_DISPLACEMENT': 29, 'NO_CANDIDATES': 43}`

---

## 10. SIH Problem Statement Benchmark vs. Internal Target Status

The official SIH Problem Statement (PS 26168) Dead Reckoning benchmark requirement states:
> *"Dead Reckoning: positional drift must be LESS THAN 10% of the total distance travelled during GNSS blackout."*

### Official Compliance Analysis:
- **Scenario B (10s Outage)**:
  - Distance Travelled: **142.2 m**
  - Final Drift: **7.15 m**
  - Drift Percentage: **5.03%**
  - Status: **PASS** (5.03% is well below the official 10.0% threshold).
- **Scenario B (30s Outage)**:
  - Distance Travelled: **426.6 m**
  - Final Drift: **86.19 m**
  - Drift Percentage: **20.20%**
  - Status: **FAIL** (20.20% exceeds the 10.0% threshold).
- **Scenario B (60s Outage)**:
  - Distance Travelled: **839.5 m**
  - Final Drift: **174.33 m**
  - Drift Percentage: **20.77%**
  - Status: **FAIL** (20.77% exceeds the 10.0% threshold).

### Internal Stronger Target (<1.5%):
The internal project roadmap defines an aspirational target of $<1.5\%>$ drift. None of the extended outage scenarios currently achieve the $<1.5\%>$ internal target. This distinction is maintained transparently: the official SIH requirement is $<10\%>$, not $<1.5\%>$.

### Critical Architectural Distinction:
Map matching operates strictly downstream on the display/output tier. When the dead reckoning filter drifts past $25	ext{ m}$, the matcher safely activates `LARGE_DISPLACEMENT` and `NO_CANDIDATES` fallbacks. Map matching **MUST NOT** be used to artificially mask dead-reckoning drift or claim dead-reckoning benchmark compliance. Dead reckoning performance is evaluated on the sensor fusion pipeline in Phase 13.

---

## 11. Detailed Investigation: Stop-and-Go (Scenario D) & Centerline Effects

A detailed investigation was conducted into Scenario D (Stop-and-Go), where the estimator RMSE is **1.375 m** while the map-matched display RMSE is **2.205 m**, with 93.0% of epochs showing a positive error delta:

1. **Centerline Offset vs. Travel Lane**:
   - OpenStreetMap represents roadways as 1D linear centerlines.
   - Real vehicles drive within a specific travel lane, typically $1.5	ext{ m}$ to $2.4	ext{ m}$ offset from the centerline.
   - Ground-truth evaluation is performed against a roof-mounted VBOX antenna centered over the vehicle in its lane.
2. **High-Precision Estimator during Stop**:
   - During stationary periods, ZUPT locks the velocity to zero and position error remains $< 0.5	ext{ m}$ from true antenna position.
   - Snapping the vehicle onto the OSM centerline forcefully shifts the displayed coordinate by the lane offset ($2.39	ext{ m}$).
   - This shifts the display coordinate away from the true antenna ground truth, causing an apparent numerical degradation.
3. **Display Alignment vs. Antenna Accuracy**:
   - On navigation displays, snapping the vehicle onto the roadway ensures the user sees their vehicle on the road rather than hovering on sidewalk boundaries.
   - The estimator filter state remains uncorrupted, and the display trade-off is an expected physical consequence of centerline mapping.

---

## 12. Preserved Phase 11 Baseline Verification

To guarantee zero regression of the frozen Phase 11 baseline:
- Pre-Phase 12 Phase 11 Estimator RMSE: `1.5496224217307877 m`
- Post-Phase 12 Phase 11 Estimator RMSE: `1.5496224217307877 m`
- Numerical Delta: **$0.0000000000000000\text{ m}$** (bit-for-bit identical).
- Downstream Feedback: **STRICTLY ZERO**. ESKF state before and after map matching evaluated identical via assertion at every epoch.

---

## 13. Generated Publication Diagnostic Figures (A through O)

All 15 figures were generated automatically from the final replay output and saved to `docs/phase12_figures/`:

1. `01_full_trajectory_comparison.png`: Plot A — Full trajectory comparison with fallback markings
2. `02_position_error_timeline.png`: Plot B — Position error timeline across 60s outage
3. `03_cross_track_error_timeline.png`: Plot C — Cross-track error timeline with rejected snaps
4. `04_along_vs_cross_track_error.png`: Plot D — Along-track vs cross-track error scatter
5. `05_snap_displacement_timeline.png`: Plot E — Snap displacement distance timeline
6. `06_fallback_reason_timeline.png`: Plot F — Categorical fallback sequence across 60s outage
7. `07_confidence_timeline.png`: Plot G — Confidence score and ambiguity margin timeline
8. `08_scenario_rmse_comparison.png`: Plot H — Scenario-by-scenario RMSE comparison bar chart
9. `09_scenario_final_drift_comparison.png`: Plot I — Final drift across 10s, 30s, 60s outages
10. `10_drift_percentage_vs_outage.png`: Plot J — Dead reckoning drift % vs outage duration with both 10% and 1.5% thresholds
11. `11_snap_distance_distribution.png`: Plot K — Orthogonal snap distance distribution
12. `12_regression_audit.png`: Plot L — Point-by-point regression audit (% improved, degraded, unchanged)
13. `13_ambiguity_diagnostic.png`: Plot M — Viterbi candidate log-scores and ambiguity diagnostic
14. `14_trajectory_zooms.png`: Plot N — 4-quadrant trajectory zoom analysis
15. `15_benchmark_summary_table.png`: Plot O — Official SIH PS Benchmark (<10%) & Internal Target (<1.5%) Compliance Table

---

## 14. Final Acceptance Verdict

Phase 12 is **COMPLETE AND FROZEN**:
- Transition gate uses physically justified kinematic and projection uncertainty bounds.
- Directed edge semantics strictly enforced.
- Viterbi timestamp resolution and mature-epoch metadata isolation verified.
- Strict causality with mature window buffer ($W=8$) preserved without future lookahead.
- Anti-catastrophic-snap safeguards active with graceful fallbacks.
- Phase 11 baseline remains bit-for-bit identical ($1.5496224217307877\text{ m}$).
- Numerical outputs in `docs/phase12_mapmatch_results.json` and `docs/mapmatch_report.md` are 100% synchronized.
