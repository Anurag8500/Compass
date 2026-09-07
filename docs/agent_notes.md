# COMPASS Agent Notes — Architecture & Implementation Invariants

These invariants are derived from `FINAL_MASTER_PLAN_SIH26168.md`, `FINAL_IMPLEMENTATION_PLAN_SIH26168.md`, and `EndToEnd_Trace_SIH26168.md`. Every implementation agent working on COMPASS must remember and adhere strictly to these facts across all phases.

---

### Core Architecture Invariants

1. **Exactly Two ML Models**:
   - **VelocityNet**: Predicts scalar forward speed ($v_{\text{forward}}$) and uncertainty ($\log \sigma^2$) from a $(20, 9)$ vehicle-frame motion feature tensor.
   - **BiasNet**: Predicts 6-axis IMU bias residual corrections ($\Delta \mathbf{b}_a^v, \Delta \mathbf{b}_g^v$) and uncertainty from the same $(20, 9)$ feature tensor. BiasNet is one of the two intended ML measurement sources for the system. The overall navigation system must remain functional if BiasNet is unavailable, but BiasNet itself is part of the intended architecture. No other ML models exist.

2. **ESKF as the Central Fusion Backbone**:
   - 100% classical Error-State Kalman Filter mathematics governs state estimation and covariance propagation.
   - Learned models (VelocityNet, BiasNet) provide **pseudo-measurements** to the ESKF update step — they never overwrite the state or replace the filter.
   - Strapdown INS integrates at native sensor rate; physical gravity is added in the local navigation frame using the ESKF attitude quaternion ($\mathbf{a}^n = R_v^n(\mathbf{f}_m^v - \mathbf{b}_a^v) + \mathbf{g}^n$).

3. **Canonical 10 Hz ML Input Rate**:
   - All ML feature tensors are windowed at canonical 10 Hz (20 samples = 2.0 s window, 0.5 s stride).
   - Higher-rate sensor streams (e.g. 100-200 Hz on Android or FOG on edge) are decimated to 10 Hz for ML inference, while strapdown INS propagation runs at full native rate.

4. **Three-State GNSS Finite State Machine (FSM)**:
   - `GNSS_AIDED`: High-confidence GNSS available; active measurement updates.
   - `DR_ONLY`: GNSS unavailable/rejected; dead reckoning powered by INS propagation + VelocityNet + BiasNet + NHC + ZUPT.
   - `REACQUIRING`: GNSS reappears; innovation checks and bounded-rate blend over ~1-2 s before returning to `GNSS_AIDED`. Never an instant position snap.

5. **Authoritative Dual-Platform Strategy**:
   - **Python** (`navigation/` and `edge/`): The authoritative reference implementation and behavioral oracle for all mathematics and data pipelines.
   - **Kotlin** (`android/`): Production mobile port; must match Python numerical and behavioral outputs.

6. **Non-Destructive Raw Data Policy**:
   - Raw sensor logs in `data/raw/` are strictly immutable and never altered, deleted, or overwritten.
   - Quality evaluation assigns bitmask flags (`FLAG_NAN_OR_NONFINITE`, `FLAG_INVALID_TIMESTAMP`, `FLAG_NON_MONOTONIC_TIMESTAMP`, `FLAG_DUPLICATE_TIMESTAMP`, `FLAG_EXTREME_MOTION`, `FLAG_SENSOR_DROPOUT`).
   - Extreme motion ($|f| > 4g$ or $|\omega| > 10 \text{ rad/s}$) reflects physical vehicle events (potholes, speed bumps, emergency braking) and is **preserved**.
   - Only non-finite or non-monotonic samples are excluded from the downstream validated stream.

7. **GNSS is NOT an ML Inference Input**:
   - Neither VelocityNet nor BiasNet ingests GNSS coordinates, speed, or quality metrics at runtime.
   - GNSS is used strictly as a reference during training/offline label generation and as an independent measurement source in the ESKF.

8. **Zero Navigation Mathematics in Phase 0**:
   - Phase 0 is purely discovery, scaffolding, and verification. No INS, ESKF, or ML math is implemented in Phase 0.
