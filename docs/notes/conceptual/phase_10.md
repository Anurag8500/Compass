# Phase 10 Complete Explanation: GNSS Quality, Trust Scoring, Outage Detection, 3-State FSM & Bounded-Rate Recovery

**Project**: C.O.M.P.A.S.S. (Cognitive Off-grid Machine-learning Positioning And Sensor System)  
**Problem Statement**: SIH 2026 Problem Statement 26168 — ISRO  
**Stage**: Phase 10 Complete Teaching & Reference Walkthrough  

---

## 1. Introduction & Phase Overview

### What Problem Did Phase 10 Solve?
In Phase 5 and Phase 9, we built the mathematical machinery to fuse IMU physics, GNSS updates, and deep neural networks (VelocityNet and BiasNet). However, this machinery assumed an idealized environment: either GNSS fixes arrive reliably with known Gaussian accuracy, or GNSS is completely absent.

In real-world vehicle navigation, **GNSS signals degrade continuously and unpredictably**:
- **Multipath Reflections**: In urban canyons surrounded by skyscrapers or mountain cliffs, satellite signals bounce off buildings before reaching the antenna, introducing tens of meters of pseudorange delay.
- **Canopy and Portal Attenuation**: Driving under dense tree canopies, underpasses, or highway tunnels causes satellite tracking count ($N_{\text{sats}}$) and Dilution of Precision (DOP) to deteriorate rapidly before total signal loss.
- **The Catastrophic Re-Snap Shock**: When an autonomous vehicle exits a 60-second tunnel, dead-reckoning drift might have accumulated $50\,\text{meters}$. If the Kalman filter ingests the returning GNSS fix naively, it snaps position by $50\,\text{meters}$ across a single $100\,\text{ms}$ epoch. This implies an artificial vehicle acceleration spike of $5,000\,\text{m/s}^2$ ($500\text{ G}$!), severely corrupting filter velocity and covariance estimates.

**Phase 10 solves the behavioral supervisory problem**: How does the navigation system evaluate GNSS signal trustworthiness on a continuous scale, detect genuine outages without flapping in borderline noise, govern system operational modes via an authoritative Finite State Machine (FSM), and safely blend returning GNSS fixes back into the trajectory without shock?

---

## 2. Core Concepts & Terminology

### 1. Continuous Trust Score vs. Discrete Switching
- **The Classical Mistake**: Many navigation systems use hard binary switching: if HDOP $< 2.0$, trust GPS; if HDOP $\ge 2.0$, cut off GPS. Hard thresholds cause severe jitter, erratic mode flapping, and sudden covariance steps.
- **The COMPASS Solution**: COMPASS implements a continuous scalar trust score:
  $$S_{\text{trust}} \in [0.0, 1.0]$$
  Inside the nominal operating mode, degraded GNSS is accommodated by **continuous covariance scaling**:
  $$\mathbf{R}_{\text{eff}} = \frac{1}{\max(S_{\text{trust}}, 0.05)} \mathbf{R}_{\text{base}}$$
  - When signals are pristine ($S_{\text{trust}} = 1.0$), $\mathbf{R}_{\text{eff}} = \mathbf{R}_{\text{base}}$ ($3.0\,\text{m}$ standard deviation).
  - When signals degrade in light canopy ($S_{\text{trust}} = 0.20$), $\mathbf{R}_{\text{eff}} = 5.0 \cdot \mathbf{R}_{\text{base}}$ ($6.7\,\text{m}$ standard deviation). The filter smoothly increases uncertainty, automatically downweighting GNSS in favor of internal inertial propagation.
  - Crucially, **there is NO discrete "DEGRADED" mode in COMPASS**. Degraded operation lives entirely inside `GNSS_AIDED` through continuous covariance adaptation.

### 2. Current-Fix Pre-Update Innovation NIS
To prevent circular reasoning, the trust score must not depend on post-update statistics. Phase 10 evaluates the current fix's pre-update Normalized Innovation Squared (NIS) against the unscaled baseline covariance $\mathbf{R}_{\text{base}}$:
$$\text{NIS}_{\text{pre}} = \mathbf{y}^T \left(\mathbf{H} \mathbf{P} \mathbf{H}^T + \mathbf{R}_{\text{base}}\right)^{-1} \mathbf{y}$$
This eliminates 1-fix reporting lag and ensures that unexpected spatial jumps are detected before the measurement is accepted.

---

## 3. The 3-State Finite State Machine (FSM)

COMPASS defines three authoritative operational states governed by strict transitions and anti-flapping hysteresis:

```
                  +-----------------------+
                  |      GNSS_AIDED       |
                  | Nominal navigation    |
                  | Continuous R scaling  |
                  +-----------------------+
                        /           ^
         No fixes for  /             \  3 consecutive
         T_grace=2.0s /               \  convergent fixes
                     v                 \
          +-------------------+     +--------------------+
          |      DR_ONLY      |---->|    REACQUIRING     |
          | Dead reckoning    | Fix | Bounded-rate blend |
          | ML + Kinematics   | seen| v_blend <= 2.0 m/s |
          +-------------------+     +--------------------+
                     ^                 /
                      \               / Timeout (10s) or
                       +-------------+  Divergent fix
```

### 1. `GNSS_AIDED` (Nominal Multi-Sensor Fusion)
- **Active Sensors**: IMU + GNSS (continuously scaled) + VelocityNet + BiasNet + NHC + ZUPT.
- **Entry Condition**: System initialization, or successful completion of `REACQUIRING`.
- **Exit Condition**: No valid GNSS fixes received for $T_{\text{grace}} = 2.0\,\text{seconds}$.

### 2. `DR_ONLY` (Autonomous Dead Reckoning)
- **Active Sensors**: IMU + VelocityNet + BiasNet + NHC + ZUPT. GNSS updates are zeroed out.
- **Entry Condition**: Outage confirmation after $T_{\text{grace}}$ elapsed, or reacquisition abort.
- **Behavior**: The filter relies on internal dynamics. Covariance $\mathbf{P}$ expands monotonically according to process noise $\mathbf{Q}$.
- **Exit Condition**: Arrival of a plausible GNSS fix satisfying basic position/velocity feasibility.

### 3. `REACQUIRING` (Bounded-Rate Convergence)
- **Active Sensors**: IMU + ML + Kinematics + Supervisory Bounded-Rate GNSS Blending.
- **Entry Condition**: First plausible fix arriving after a `DR_ONLY` period.
- **Exit Condition**:
  - *Promotion to `GNSS_AIDED`*: 3 consecutive fixes satisfy the convergence threshold ($\|\mathbf{y}\| \le 1.5\,\text{m}$).
  - *Abort to `DR_ONLY`*: Reacquisition timeout ($T_{\text{reacq}} = 10.0\,\text{s}$) expires, or a returning fix exhibits severe Chi-square divergence ($\text{NIS} > 11.345$).

### Anti-Flapping Hysteresis ($T_{\text{dwell}} = 2.0\,\text{s}$)
In environments with intermittent tree cover, GPS fixes may flicker on and off every few hundred milliseconds. If an FSM switches states rapidly, filter covariance resets and re-initializations cause severe numerical instability.
- COMPASS enforces a minimum dwell time: once entering `GNSS_AIDED` or `DR_ONLY`, the FSM is locked into that state for at least **$2.0\,\text{seconds}$** before any exit transition can trigger.

---

## 4. Continuous Trust Score Formulation

The composite trust score $S_{\text{trust}} \in [0.0, 1.0]$ is computed as a weighted harmonic product of four physical indicators:
$$S_{\text{trust}} = S_{\text{hdop}} \cdot S_{\text{sats}} \cdot S_{\text{plaus}} \cdot S_{\text{nis}}$$

### 1. Dilution of Precision Factor ($S_{\text{hdop}}$)
$$\sigma_h = \text{reported\_accuracy\_m} \quad \text{or} \quad \text{HDOP} \cdot \sigma_{\text{nominal}}$$
$$S_{\text{hdop}} = \exp\left(-\frac{\max(0, \sigma_h - \sigma_{\text{good}})}{\sigma_{\text{scale}}}\right)$$
Where $\sigma_{\text{good}} = 3.0\,\text{m}$ and $\sigma_{\text{scale}} = 10.0\,\text{m}$. For high-precision RTK or open-sky GPS ($\sigma_h \le 3\,\text{m}$), $S_{\text{hdop}} = 1.0$. If accuracy degrades to $15\,\text{m}$, $S_{\text{hdop}}$ drops to $0.30$.

### 2. Satellite Visibility Factor ($S_{\text{sats}}$)
A minimum of 4 satellites is required for a 3D trilateration fix. Redundant satellites ($N \ge 8$) dramatically reduce multipath vulnerability:
$$S_{\text{sats}} = \text{clamp}\left(\frac{N_{\text{sats}} - 4}{8 - 4}, \, 0.0, \, 1.0\right)$$
- $N_{\text{sats}} \le 4 \implies S_{\text{sats}} = 0.0$ (Zero trust).
- $N_{\text{sats}} \ge 8 \implies S_{\text{sats}} = 1.0$ (Full trust).

### 3. Fix-to-Fix Kinematic Plausibility ($S_{\text{plaus}}$)
Compares the apparent displacement between consecutive GNSS fixes ($\Delta \mathbf{p}_{\text{gnss}} / \Delta t$) against the vehicle's independent Doppler velocity:
$$e_v = \left\|\frac{\mathbf{p}_{\text{gnss}}[k] - \mathbf{p}_{\text{gnss}}[k-1]}{\Delta t_k} - \mathbf{v}_{\text{doppler}}[k]\right\|$$
$$S_{\text{plaus}} = \exp\left(-\frac{e_v^2}{2 \sigma_{\text{plaus}}^2}\right) \qquad (\sigma_{\text{plaus}} = 3.0\,\text{m/s})$$
If multipath causes a sudden $20\,\text{m}$ coordinate jump while Doppler velocity is $10\,\text{m/s}$, $S_{\text{plaus}} \to 0$, protecting the filter from the jump.

### 4. Innovation Consistency Factor ($S_{\text{nis}}$)
Evaluates pre-update innovation against baseline covariance using the 99% Chi-Square threshold for 3 degrees of freedom ($\chi_3^2(0.99) = 11.345$):
$$S_{\text{nis}} = \begin{cases} 1.0 & \text{if } \text{NIS}_{\text{pre}} \le 3.0 \\ \frac{11.345 - \text{NIS}_{\text{pre}}}{11.345 - 3.0} & \text{if } 3.0 < \text{NIS}_{\text{pre}} \le 11.345 \\ 0.0 & \text{if } \text{NIS}_{\text{pre}} > 11.345 \text{ (Outlier Rejected)} \end{cases}$$

---

## 5. Bounded-Rate Reacquisition & State Smoothing

When an autonomous vehicle emerges from a long tunnel into open sky, the returning GNSS fix is accurate, but the dead-reckoning position estimate may have drifted by $30\text{--}100\,\text{meters}$.

### Why Standard Kalman Updates Fail on Reacquisition
If the returning fix is processed as a standard Kalman update with small measurement covariance $R_{\text{gnss}}$:
$$K = P H^T (H P H^T + R)^{-1} \approx I$$
$$\delta\mathbf{x} = K (\mathbf{z}_{\text{gnss}} - \hat{\mathbf{p}}) \approx 50.0\,\text{meters}$$
The filter injects this entire $50\,\text{meter}$ error state across a single $100\,\text{ms}$ step. Because velocity and attitude cross-covariances are non-zero ($P_{pv} \ne 0, P_{p\theta} \ne 0$), the massive position correction forces an instantaneous velocity correction:
$$\delta\mathbf{v} = P_{vp} P_{pp}^{-1} \delta\mathbf{p} \sim 25\,\text{m/s}$$
This instantaneous velocity shock destabilizes the vehicle's control loop and can cause catastrophic loss of control.

### The Bounded-Rate Solution
In `REACQUIRING` mode, COMPASS decouples position convergence from internal covariance dynamics using a **rate-limited supervisory blender**:
1. **Maximum Velocity Slew Rate**: The position state moves toward the GNSS fix at a maximum blending velocity:
   $$v_{\text{blend}} \le 2.0\,\text{m/s} \quad (7.2\,\text{km/h})$$
2. **Maximum Single-Step Displacement**: The position adjustment at epoch $k$ is clamped:
   $$\Delta \mathbf{p}_{\text{blend}} = \text{clamp}\left(\mathbf{p}_{\text{gnss}} - \hat{\mathbf{p}}, \, \Delta p_{\max} = 3.0\,\text{meters}\right)$$
3. **Zero Covariance Shock**: The blend smoothly adjusts the nominal position coordinate $\mathbf{p}_{\text{nom}}$ without mutating the error-state covariance matrix $\mathbf{P}$.
4. **Three-Fix Convergence Lock**: Once the blended position approaches within $1.5\,\text{meters}$ of GNSS across 3 consecutive fixes, full standard ESKF updates resume and the FSM returns to `GNSS_AIDED`.

---

## 6. Verification & Automated Test Suite

Phase 10 implemented 37 dedicated unit and integration tests verifying behavioral robustness:

1. **Anti-Flapping Validation (`test_fsm_no_flapping.py`)**:
   - Synthetic borderline GNSS signal ($S_{\text{trust}}$ oscillating between $0.05$ and $0.90$ at $5\,\text{Hz}$) was injected.
   - Verified that the FSM strictly obeyed the $2.0\,\text{second}$ dwell time, experiencing **zero state oscillations**.
2. **Bounded-Rate Reacquisition Validation (`test_recovery_bounded_rate.py`)**:
   - Injected a $50\,\text{meter}$ dead-reckoning position discrepancy at the moment of GNSS re-entry.
   - Asserted that position correction velocity never exceeded $2.0\,\text{m/s}$ and single-step displacement never exceeded $3.0\,\text{m}$.
   - Verified that velocity tracking remained stable and smooth throughout reacquisition.
3. **Rigorous Test Suite Status**:
   - Full test suite: **384 tests collected, 384 passed, 0 failed in 37.24s**.

---

## 7. Phase Summary & Handoff to Phase 11

| Property | Phase 10 Behavioral Specification |
|---|---|
| **Authoritative FSM** | 3-State: `GNSS_AIDED` $\longleftrightarrow$ `DR_ONLY` $\longleftrightarrow$ `REACQUIRING` |
| **Degraded GNSS Handling** | Continuous covariance scaling ($\mathbf{R}_{\text{eff}} = \mathbf{R}_{\text{base}} / S_{\text{trust}}$); NO discrete degraded state |
| **Outage Detection** | $T_{\text{grace}} = 2.0\,\text{s}$ grace period, $3.0\,\text{s}$ timeout |
| **Anti-Flapping** | $2.0\,\text{s}$ dwell time hysteresis on all state exits |
| **Reacquisition Guard** | Slew-rate limited blending ($v_{\text{blend}} \le 2.0\,\text{m/s}$, $\Delta p_{\max} \le 3.0\,\text{m}$), 3-fix lock |
| **Outlier Threshold** | Strict Chi-Square gate $\chi_3^2(0.99) = 11.345$ |
| **Acceptance Gate Status** | **COMPLETE, VALIDATED & FROZEN** |

Now that the system intelligently governs GNSS modes and reacquisition, we address the root cause of cross-track drift identified in Phase 9. In **Phase 11**, we implement **Non-Holonomic Constraints (NHC)** and the **Simon-Chia Constrained Kalman Filter** to prevent lateral vehicle slipping.
