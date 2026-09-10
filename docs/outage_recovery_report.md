# Phase 10 Outage & Recovery Report

## Overview

This report documents the GNSS outage detection and recovery behaviour validated in Phase 10. Results are split into two explicitly labelled categories — synthetic and real IO-VNBD data are never conflated.

---

## Architecture Summary

Phase 10 implements four cooperating modules:

**trust_score.py** — Continuous trust score in [MIN_TRUST_SCORE=0.05, 1.0] from up to four evidence sources with dynamic weight redistribution:

| Component | Weight | Evidence |
|---|---|---|
| Horizontal accuracy | 0.35 | GPS reported error (metres) |
| Satellite count | 0.15 | Satellites used in fix |
| Fix-to-fix plausibility | 0.25 | Implied speed between consecutive fixes |
| ESKF innovation (NIS) | 0.25 | Kalman filter residual consistency |

When an evidence source is unavailable (e.g. satellite count not reported, no prior fix), its weight is redistributed proportionally to the remaining sources. The NIS component is the key addition over naive accuracy+satellite scoring: if the Kalman filter is already disagreeing with GPS fixes (NIS > chi2_3(0.95) = 7.815), trust falls *before* the FSM grace period expires — catching urban multipath where GPS is physically present but consistently wrong.

A clean open-sky fix (accuracy ~4 m, 12 satellites, consistent with filter) scores ~0.85. A lost-fix row (accuracy 999, sat 0, frozen position) scores MIN_TRUST_SCORE = 0.05.

**outage_detection.py** — Bridges raw GNSS rows to the FSM with two detection paths:

- *Silent dropout*: if no fix arrives for > 1500 ms, a zero-trust tick is injected so the FSM grace-period timer advances during tunnel/underpass scenarios.
- *Persistent rejection*: if the ESKF innovation gate rejects 3 consecutive fixes, `record_eskf_result(accepted=False)` overrides trust to MIN_TRUST_SCORE on the next `process_fix()` call, driving the FSM toward DR_ONLY even when GPS rows are still arriving.

**fsm.py** — Three-state FSM per Master Plan Section 17:

| Transition | Condition |
|---|---|
| GNSS_AIDED → DR_ONLY | trust < 0.3 for ≥ 2000 ms |
| DR_ONLY → REACQUIRING | trust ≥ 0.3 (immediate) |
| REACQUIRING → GNSS_AIDED | trust ≥ 0.7 for ≥ 1000 ms |
| REACQUIRING → DR_ONLY | trust < 0.3 before convergence |

**recovery.py** — Bounded-rate position blend during REACQUIRING:

- Rate limit: `max_step = min(5 m, 2 m/s × dt_s)` — dt-aware, so at 10 Hz the per-cycle cap is 0.2 m, eliminating any visible position discontinuity
- Convergence threshold: 3 m (tightened from 10 m)
- Requires 3 consecutive fixes within 3 m before signalling GNSS_AIDED recovery
- Plausibility gate: rejects fixes > 2000 m from blended position (multipath artifacts)

---

## SOURCE: SYNTHETIC — Outage Detection Results

All rows in this section are programmatically constructed. No real IO-VNBD data is involved.

### Test suite: test_outage_synthetic.py

| Test | Result |
|---|---|
| Clean 10 s signal stays GNSS_AIDED | PASS |
| SHORT outage (60 s) → DR_ONLY | PASS |
| SHORT outage recovery → GNSS_AIDED | PASS |
| SHORT recovery passes through REACQUIRING (no instant snap) | PASS |
| LONG outage (60 s / 1 km @ 60 km/h) → DR_ONLY → GNSS_AIDED | PASS |
| Silent dropout via tick_no_fix → DR_ONLY | PASS |
| Short tick gap (< 1500 ms) stays GNSS_AIDED | PASS |
| Outage from session row zero → DR_ONLY | PASS |
| Two sequential outage/recovery cycles both complete correctly | PASS |

**9 / 9 tests passed.**

### PS benchmark outage durations

Both benchmark scenarios from the problem statement are covered:

- **~50 m drift / < 1 min**: modelled by the SHORT 60 s outage. FSM correctly enters DR_ONLY after the 2 s grace period and recovers through REACQUIRING to GNSS_AIDED after sustained good fixes.
- **~1 km @ 60 km/h (~60 s)**: covered by the LONG scenario. Full drift measurement requires the complete ESKF+VelocityNet+BiasNet replay pipeline, deferred to Phase 13's ablation ladder.

### Anti-flapping (test_fsm_no_flapping.py)

30 s of borderline trust scores alternating around the 0.3 threshold at 200 ms intervals.

| Check | Result |
|---|---|
| Total transitions ≤ 4 over 30 s | PASS |
| GNSS_AIDED → DR_ONLY never fires before GRACE_PERIOD_MS (2000 ms) | PASS |
| REACQUIRING → GNSS_AIDED never fires before CONVERGENCE_MS (1000 ms) | PASS |
| Pattern genuinely straddles LOW_TRUST_THRESHOLD | PASS |

**4 / 4 tests passed.** FSM does not flap under sustained borderline noise.

---

## SOURCE: SYNTHETIC — Bounded-Rate Recovery Results

### Test suite: test_recovery_bounded_rate.py

Rate limit is dt-aware: at 10 Hz (dt=0.1 s), effective cap = min(5 m, 2 m/s × 0.1 s) = **0.2 m/cycle**.

| Test | Drift | Result |
|---|---|---|
| Single-cycle correction ≤ 0.2 m (at 10 Hz) | 10 m | PASS |
| Single-cycle correction ≤ 0.2 m (at 10 Hz) | 50 m | PASS |
| Single-cycle correction ≤ 0.2 m (at 10 Hz) | 200 m | PASS |
| Single-cycle correction ≤ 0.2 m (at 10 Hz) | 1000 m | PASS |
| Converges eventually | 10 m | PASS |
| Converges eventually | 50 m | PASS |
| Converges eventually | 200 m | PASS |
| Converges eventually | 1000 m | PASS |
| Not converged on first fix | — | PASS |
| Converges exactly at CONVERGENCE_MIN_FIXES (3) | — | PASS |
| 200 m drift: first cycle ≤ 0.2 m from seed (no instant snap) | 200 m | PASS |
| Implausible fix (> 2000 m) returns None | — | PASS |
| Blended position unchanged after rejection | — | PASS |
| Blended within 3 m of raw fix at convergence | 20 m | PASS |
| Blended within 3 m of raw fix at convergence | 100 m | PASS |
| Blended within 3 m of raw fix at convergence | 300 m | PASS |
| reset() clears convergence counter | — | PASS |
| reset() seeds from new ESKF position | — | PASS |

**18 / 18 tests passed.**

### Convergence timing

At 10 Hz (dt = 0.1 s), effective rate = 2 m/s → 0.2 m/cycle:

| Drift | Cycles to close gap | + 3 confirmation | Total cycles | Wall time @ 10 Hz |
|---|---|---|---|---|
| 20 m | ~85 | 3 | ~88 | ~8.8 s |
| 50 m | ~235 | 3 | ~238 | ~23.8 s |
| 200 m | ~985 | 3 | ~988 | ~98.8 s |

This is tighter than the previous 5 m/cycle design. For a typical 60 s urban outage at ~14 m/s, accumulated drift is ~50–100 m, giving recovery in ~25–50 seconds. For the PS headline benchmark of <1.5% drift over 60 s on a highway segment, Phase 13 will measure actual end-to-end position error with NHC and ZUPT also active — recovery time here is a Phase 10 standalone characterisation only.

---

## SOURCE: REAL IO-VNBD DATA

### Phase 0 finding (from docs/dataset_notes.md)

The IO-VNBD synchronized archive contains **no pre-tagged GPS outage windows**. `has_real_outages = False` for all trips in the manifest. There is no external outage index file. This is an explicit, documented property of the dataset.

### What was tested on real data

`test_outage_real_iovnbd.py` validates stability and calibration sanity on real sensor rows:

1. **Stability** — no exceptions, no out-of-range trust scores on any real row.
2. **Calibration sanity** — open-sky trip S1 (Driver A) must spend ≥ 95% of time in GNSS_AIDED.
3. **Trust score distribution** — mean trust > 0.3, non-zero variance, on good open-sky data.

All 6 tests skipped (not failed) when raw dataset is not mounted.

```
python -m pytest tests/integration/test_outage_real_iovnbd.py -v
```

---

## Definition of Done — Phase 10

| Criterion | Status |
|---|---|
| FSM transitions correctly (all 4 transition rules) | ✅ |
| FSM does not flap under borderline noise | ✅ |
| Trust score includes ESKF innovation (NIS) component | ✅ |
| Persistent innovation-gate rejection drives DR_ONLY | ✅ |
| Recovery is dt-aware rate-limited (0.2 m/cycle at 10 Hz) | ✅ |
| Convergence threshold tightened to 3 m | ✅ |
| Synthetic outage results documented with SOURCE label | ✅ |
| Real IO-VNBD outage results documented with SOURCE label | ✅ |
| Synthetic and real results clearly distinguished | ✅ |

Full drift/RMSE benchmarking against PS targets (< 1.5% over 60 s) is deferred to Phase 13.
