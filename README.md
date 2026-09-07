# C.O.M.P.A.S.S.
## Cognitive Off-grid Machine-learning Positioning And Sensor System
### AI-ML Based Intelligent Dead Reckoning System — SIH Problem Statement 26168 (ISRO)

COMPASS is an intelligent dead-reckoning navigation system designed to maintain continuous, highly accurate vehicle positioning during GNSS outages (tunnels, underground parking, urban canyons, dense forests, jamming).

The system combines:
1. **Classical Strapdown INS**: Kinematic state propagation at native sensor rate.
2. **Error-State Kalman Filter (ESKF)**: Central fusion backbone integrating kinematic propagation, learned pseudo-measurements, GNSS updates, Non-Holonomic Constraints (NHC), and Zero Velocity Updates (ZUPT).
3. **Deep Learning Virtual Sensors**:
   - **VelocityNet**: Forward speed estimator from vehicle-frame IMU features.
   - **BiasNet**: Context-dependent IMU bias correction model.
4. **Offline Map Matching**: Hidden Markov Model (HMM) snapping trajectory to OpenStreetMap road graphs.
5. **Mode FSM**: Hysteresis-governed transitions between `GNSS_AIDED`, `DR_ONLY`, and `REACQUIRING`.

---

## Authoritative Specifications
- `FINAL_MASTER_PLAN_SIH26168.md`: Single master technical and architectural specification.
- `FINAL_IMPLEMENTATION_PLAN_SIH26168.md`: Execution roadmap across 20 phases.
- `EndToEnd_Trace_SIH26168.md`: Concrete runtime and dataflow trace from raw sensor to map display.

---

## Project Status: Phase 0 Completed
- Scaffolding and environment verified.
- Non-destructive dataset inspection tooling implemented (`scripts/inspect_iovnbd.py`).
- Raw IO-VNBD dataset discovery foundation established.
