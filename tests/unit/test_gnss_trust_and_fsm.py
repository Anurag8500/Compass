"""Unit tests for Phase 10 GNSS Quality/Trust, Outage Detection, and 3-State FSM.

Verifies:
1. Continuous trust score bounded strictly in [0, 1] across all conditions.
2. Monotonic scaling of trust with respect to reported accuracy and satellite count.
3. Graceful handling of missing satellite count (e.g. IO-VNBD sat_count = -1).
4. Implausible velocity jump and high filter innovation degradation.
5. Continuous measurement covariance scaling R_effective = (1 / max(trust, 0.05)) * R_base.
6. Outage detector grace period: single missed fix does NOT trigger DR_ONLY.
7. Outage detector persistent rejection tracking.
8. Authoritative 3-state FSM transitions and minimum dwell times.
9. Returning fix plausibility validation and reason codes.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from navigation.schemas.state import GNSSMode
from navigation.eskf.state import ESKFNominalState, ESKFState
from navigation.gnss.trust_score import (
    GNSSTrustScoreCalculator,
    TrustScoreConfig,
    scale_gnss_covariance,
)
from navigation.gnss.outage_detection import (
    GNSSOutageDetector,
    OutageCondition,
    OutageDetectorConfig,
)
from navigation.gnss.fsm import FSMConfig, GNSSModeFSM
from navigation.gnss.recovery import (
    GNSSRecoveryManager,
    RecoveryConfig,
    ReturningFixStatus,
)


class TestGNSSTrustScore:
    """Test suite for continuous GNSS trust score and covariance scaling."""

    def test_excellent_gnss_trust(self) -> None:
        calc = GNSSTrustScoreCalculator()
        res = calc.compute_trust(
            accuracy_m=1.5,
            sat_count=12,
            current_pos_enu=np.array([10.0, 20.0, 0.0]),
            timestamp_ns=int(1e9),
            nis=1.0,
        )
        assert res.trust_score == pytest.approx(1.0, abs=1e-3)
        assert res.accuracy_component == 1.0
        assert res.satellite_component == 1.0
        assert res.plausibility_component == 1.0
        assert res.innovation_component == 1.0

    def test_poor_accuracy_degrades_trust_monotonically(self) -> None:
        calc = GNSSTrustScoreCalculator()
        scores = []
        accuracies = [2.0, 5.0, 10.0, 15.0, 25.0, 40.0]
        for acc in accuracies:
            calc.reset()
            res = calc.compute_trust(
                accuracy_m=acc,
                sat_count=10,
                current_pos_enu=np.array([0.0, 0.0, 0.0]),
                timestamp_ns=int(1e9),
            )
            scores.append(res.trust_score)

        # Monotonically non-increasing
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], f"Trust not monotonic: {scores}"
        assert scores[-1] < 0.7

    def test_missing_satellite_count_handled_gracefully(self) -> None:
        """Simulates IO-VNBD dataset where sat_count is -1 or None."""
        calc = GNSSTrustScoreCalculator()
        res = calc.compute_trust(
            accuracy_m=2.0,
            sat_count=-1,
            current_pos_enu=np.array([0.0, 0.0, 0.0]),
            timestamp_ns=int(1e9),
        )
        assert res.satellite_component is None
        assert 0.0 <= res.trust_score <= 1.0
        assert "SATELLITE_COUNT" not in res.available_evidence
        assert "ACCURACY" in res.available_evidence

    def test_implausible_jump_degrades_trust(self) -> None:
        calc = GNSSTrustScoreCalculator(TrustScoreConfig(max_apparent_speed_mps=50.0))
        # Fix 1 at (0, 0)
        calc.compute_trust(accuracy_m=2.0, sat_count=10, current_pos_enu=np.zeros(3), timestamp_ns=0)
        # Fix 2 at 1.0s: 200m away (apparent speed 200 m/s > 50 m/s)
        res2 = calc.compute_trust(accuracy_m=2.0, sat_count=10, current_pos_enu=np.array([200.0, 0.0, 0.0]), timestamp_ns=int(1e9))
        assert res2.plausibility_component == 0.0
        assert any("IMPLAUSIBLE_JUMP" in r for r in res2.reason_codes)

    def test_high_filter_innovation_degrades_trust(self) -> None:
        calc = GNSSTrustScoreCalculator(TrustScoreConfig(nis_threshold_95=7.815))
        res_low_nis = calc.compute_trust(accuracy_m=2.0, sat_count=10, current_pos_enu=np.zeros(3), timestamp_ns=0, nis=2.0)
        calc.reset()
        res_high_nis = calc.compute_trust(accuracy_m=2.0, sat_count=10, current_pos_enu=np.zeros(3), timestamp_ns=0, nis=50.0)
        assert res_high_nis.innovation_component < 0.1
        assert res_high_nis.trust_score < res_low_nis.trust_score

    def test_scale_gnss_covariance_bounds(self) -> None:
        R_base = np.eye(3) * 4.0
        # Trust = 1.0 -> R unchanged
        R_1 = scale_gnss_covariance(R_base, trust_score=1.0)
        assert np.allclose(R_1, R_base)

        # Trust = 0.5 -> R doubled
        R_half = scale_gnss_covariance(R_base, trust_score=0.5)
        assert np.allclose(R_half, R_base * 2.0)

        # Trust = 0.0 -> clamped to min_trust (0.05 -> 20x R_base), finite and PSD
        R_zero = scale_gnss_covariance(R_base, trust_score=0.0, min_trust=0.05)
        assert np.all(np.isfinite(R_zero))
        assert np.allclose(R_zero, R_base * 20.0)
        assert np.all(np.linalg.eigvalsh(R_zero) > 0)


class TestGNSSOutageDetector:
    """Test suite for timestamp-based outage detection and persistent rejection tracking."""

    def test_nominal_arrivals(self) -> None:
        det = GNSSOutageDetector(OutageDetectorConfig(expected_interval_s=1.0, grace_period_s=2.0))
        det.reset(session_start_timestamp_ns=0)

        det.record_fix_received(int(1e9))
        det.record_update_result(int(1e9), applied=True)

        status = det.evaluate_outage(int(1.5e9))
        assert status.condition == OutageCondition.NOMINAL
        assert status.is_outage is False

    def test_grace_period_does_not_declare_outage_prematurely(self) -> None:
        """Proves that a single missed fix (elapsed 1.8s) remains in GRACE_PERIOD, not CONFIRMED_OUTAGE."""
        det = GNSSOutageDetector(OutageDetectorConfig(expected_interval_s=1.0, grace_period_s=2.0))
        det.reset(session_start_timestamp_ns=0)

        det.record_fix_received(int(1e9))
        # Evaluate at 2.8s (1.8s elapsed since fix at 1.0s)
        # 1.0s expected < 1.8s < 3.0s timeout
        status = det.evaluate_outage(int(2.8e9))
        assert status.condition == OutageCondition.GRACE_PERIOD
        assert status.is_outage is False

    def test_outage_confirmed_after_timeout(self) -> None:
        det = GNSSOutageDetector(OutageDetectorConfig(expected_interval_s=1.0, grace_period_s=2.0))
        det.reset(session_start_timestamp_ns=0)

        det.record_fix_received(int(1e9))
        # Evaluate at 4.5s (3.5s elapsed > 3.0s timeout)
        status = det.evaluate_outage(int(4.5e9))
        assert status.condition == OutageCondition.CONFIRMED_OUTAGE
        assert status.is_outage is True

    def test_persistent_rejection_tracking(self) -> None:
        det = GNSSOutageDetector(OutageDetectorConfig(persistent_rejection_threshold=3))
        det.reset(session_start_timestamp_ns=0)

        # 3 consecutive rejected fixes
        for i in range(1, 4):
            t_ns = int(i * 1e9)
            det.record_fix_received(t_ns)
            det.record_update_result(t_ns, applied=False)

        status = det.evaluate_outage(int(3.1e9))
        assert status.condition == OutageCondition.PERSISTENT_REJECTION
        assert status.is_outage is True
        assert status.consecutive_rejections == 3

        # Next fix accepted clears persistent rejection
        det.record_fix_received(int(4e9))
        det.record_update_result(int(4e9), applied=True)
        status_after = det.evaluate_outage(int(4.1e9))
        assert status_after.condition == OutageCondition.NOMINAL
        assert status_after.is_outage is False


class TestGNSSModeFSM:
    """Test suite for 3-state FSM transitions and minimum dwell time."""

    def test_fsm_nominal_to_outage_transition(self) -> None:
        fsm = GNSSModeFSM(FSMConfig(min_dwell_time_s=2.0), initial_mode=GNSSMode.GNSS_AIDED, initial_timestamp_ns=0)
        assert fsm.current_mode == GNSSMode.GNSS_AIDED

        # Transition to DR_ONLY on confirmed outage
        trans = fsm.evaluate_transition(
            current_timestamp_ns=int(3e9),
            is_outage=True,
            outage_reason="CONFIRMED_OUTAGE",
        )
        assert trans is not None
        assert trans.previous_mode == GNSSMode.GNSS_AIDED
        assert trans.new_mode == GNSSMode.DR_ONLY
        assert fsm.current_mode == GNSSMode.DR_ONLY

    def test_fsm_dr_to_reacquiring_dwell_enforcement(self) -> None:
        fsm = GNSSModeFSM(FSMConfig(min_dwell_time_s=2.0), initial_mode=GNSSMode.DR_ONLY, initial_timestamp_ns=0)

        # Valid fix arrives at 1.0s (dwell < 2.0s): transition blocked by anti-flapping dwell
        trans1 = fsm.evaluate_transition(
            current_timestamp_ns=int(1e9),
            is_outage=False,
            outage_reason="NOMINAL",
            is_returning_fix_valid=True,
        )
        assert trans1 is None
        assert fsm.current_mode == GNSSMode.DR_ONLY

        # Valid fix arrives at 2.5s (dwell >= 2.0s): transition permitted
        trans2 = fsm.evaluate_transition(
            current_timestamp_ns=int(2.5e9),
            is_outage=False,
            outage_reason="NOMINAL",
            is_returning_fix_valid=True,
        )
        assert trans2 is not None
        assert trans2.new_mode == GNSSMode.REACQUIRING
        assert fsm.current_mode == GNSSMode.REACQUIRING

    def test_fsm_reacquiring_to_aided_convergence(self) -> None:
        fsm = GNSSModeFSM(FSMConfig(min_dwell_time_s=2.0), initial_mode=GNSSMode.REACQUIRING, initial_timestamp_ns=0)

        # Convergence achieved at 3.0s (dwell >= 2.0s)
        trans = fsm.evaluate_transition(
            current_timestamp_ns=int(3e9),
            is_outage=False,
            outage_reason="NOMINAL",
            is_returning_fix_valid=True,
            is_recovery_converged=True,
        )
        assert trans is not None
        assert trans.new_mode == GNSSMode.GNSS_AIDED
        assert fsm.current_mode == GNSSMode.GNSS_AIDED

    def test_fsm_reacquiring_timeout_transitions_to_dr_only(self) -> None:
        """Authoritative FSMConfig.reacq_timeout_s: transitions REACQUIRING -> DR_ONLY if unconverged."""
        fsm = GNSSModeFSM(FSMConfig(min_dwell_time_s=2.0, reacq_timeout_s=10.0), initial_mode=GNSSMode.REACQUIRING, initial_timestamp_ns=0)

        # At 5.0s (dwell < 10.0s), not converged: remains REACQUIRING
        trans1 = fsm.evaluate_transition(
            current_timestamp_ns=int(5e9),
            is_outage=False,
            outage_reason="NOMINAL",
            is_returning_fix_valid=True,
            is_recovery_converged=False,
        )
        assert trans1 is None
        assert fsm.current_mode == GNSSMode.REACQUIRING

        # At 10.1s (dwell >= 10.0s), not converged: authoritative timeout back to DR_ONLY
        trans2 = fsm.evaluate_transition(
            current_timestamp_ns=int(10.1e9),
            is_outage=False,
            outage_reason="NOMINAL",
            is_returning_fix_valid=True,
            is_recovery_converged=False,
        )
        assert trans2 is not None
        assert trans2.previous_mode == GNSSMode.REACQUIRING
        assert trans2.new_mode == GNSSMode.DR_ONLY
        assert "REACQUISITION_TIMEOUT_EXCEEDED" in trans2.reason
        assert fsm.current_mode == GNSSMode.DR_ONLY



class TestGNSSRecoveryManager:
    """Test suite for returning-fix plausibility and bounded correction calculations."""

    @pytest.fixture
    def sample_state(self) -> ESKFState:
        nom = ESKFNominalState.from_components(
            position_enu=[100.0, 200.0, 0.0],
            velocity_enu=[10.0, 0.0, 0.0],
            timestamp_ns=int(1e9),
        )
        P = np.eye(15) * 4.0
        return ESKFState(nominal=nom, covariance=P)

    def test_validate_returning_fix_plausible(self, sample_state: ESKFState) -> None:
        mgr = GNSSRecoveryManager()
        # Fix 5m away
        gnss_pos = np.array([103.0, 204.0, 0.0])
        res = mgr.validate_returning_fix(gnss_pos, sample_state, trust_score=0.9, timestamp_ns=int(2e9))
        assert res.is_plausible is True
        assert res.status == ReturningFixStatus.RETURNED_FIX_VALID

    def test_validate_returning_fix_low_trust_rejected(self, sample_state: ESKFState) -> None:
        mgr = GNSSRecoveryManager(RecoveryConfig(min_reacq_trust=0.4))
        gnss_pos = np.array([102.0, 201.0, 0.0])
        res = mgr.validate_returning_fix(gnss_pos, sample_state, trust_score=0.2, timestamp_ns=int(2e9))
        assert res.is_plausible is False
        assert res.status == ReturningFixStatus.RETURNED_FIX_LOW_TRUST

    def test_bounded_correction_limits_maximum_step(self, sample_state: ESKFState) -> None:
        mgr = GNSSRecoveryManager(RecoveryConfig(max_displacement_rate_mps=2.0, max_single_step_m=2.0))
        # Huge 50m position discrepancy
        gnss_pos = np.array([150.0, 200.0, 0.0])
        step = mgr.compute_bounded_correction(gnss_pos, sample_state, dt_s=1.0)
        assert step.is_clamped is True
        assert step.applied_step_norm_m == pytest.approx(2.0, abs=1e-5)
        # Clamped delta must point along direction of discrepancy
        assert step.delta_p_bounded[0] == pytest.approx(2.0, abs=1e-5)
        assert step.delta_p_bounded[1] == pytest.approx(0.0, abs=1e-5)


class TestESKFInnovationToTrustWiringAndGating:
    """Test suite proving real current-fix ESKF innovation NIS evaluation before trust calculation,
    timing distinction between consecutive fixes (N1 vs N2), first-fix behavior, continuous trust
    covariance scaling, Phase 5 innovation gate preservation, and reacquisition timeout/abort.
    """

    def test_first_gnss_fix_computes_and_uses_pre_update_nis_immediately(self) -> None:
        """First GNSS fix k=0 must evaluate pre-update innovation against predicted ESKF state and base covariance.

        Proves that trust calculation for the first fix immediately incorporates ESKF innovation,
        rather than passing None or waiting for a second fix.
        """
        from navigation.core import NavigationCore, NavigationCoreConfig
        core = NavigationCore(NavigationCoreConfig(
            velocitynet_enabled=False,
            biasnet_enabled=False,
            zupt_enabled=False,
            gnss_enabled=True,
        ))
        lat0, lon0, alt0 = 52.0, -1.5, 100.0
        core.initialize(
            lat0=lat0,
            lon0=lon0,
            alt0=alt0,
            p0_enu=np.zeros(3),
            v0_enu=np.zeros(3),
            q0=np.array([1.0, 0.0, 0.0, 0.0]),
            timestamp_ns=0,
        )

        assert core._current_measurement_nis is None
        assert core._last_update_nis is None

        # Fix 1: exactly at nominal position
        core.step_gnss_fix(lat0, lon0, alt0, accuracy_h_m=2.0, timestamp_ns=int(1e9))

        # 1. Current measurement NIS was computed prior to update
        assert core._current_measurement_nis is not None
        assert core._current_measurement_nis == pytest.approx(0.0, abs=1e-3)

        # 2. Trust calculator immediately evaluated this NIS
        assert core._last_gnss_quality is not None
        assert "ESKF_INNOVATION" in core._last_gnss_quality.available_evidence
        assert core.gnss_trust_calculator.last_evaluated_nis == pytest.approx(core._current_measurement_nis, rel=1e-5)

        # 3. Authoritative post-update diagnostics and telemetry
        assert core._last_update_nis is not None
        assert core._last_gnss_nis == core._last_update_nis
        telem = core.get_gnss_telemetry()
        assert telem["current_measurement_nis"] == pytest.approx(core._current_measurement_nis, rel=1e-5)
        assert telem["last_update_nis"] == pytest.approx(core._last_update_nis, rel=1e-5)
        assert telem["last_eskf_nis"] == pytest.approx(core._last_update_nis, rel=1e-5)

    def test_current_fix_nis_timing_distinguishes_consecutive_fixes(self) -> None:
        """Fix k has NIS=N1 and fix k+1 has deliberately different NIS=N2.

        Proves that trust calculation for fix k+1 evaluates N2 (current fix),
        NOT stale N1 from fix k. Under the old 1-fix-delayed bug, fix k+1 evaluated N1.
        """
        from navigation.core import NavigationCore, NavigationCoreConfig
        core = NavigationCore(NavigationCoreConfig(
            velocitynet_enabled=False,
            biasnet_enabled=False,
            zupt_enabled=False,
            gnss_enabled=True,
        ))
        lat0, lon0, alt0 = 52.0, -1.5, 100.0
        core.initialize(
            lat0=lat0,
            lon0=lon0,
            alt0=alt0,
            p0_enu=np.zeros(3),
            v0_enu=np.zeros(3),
            q0=np.array([1.0, 0.0, 0.0, 0.0]),
            timestamp_ns=0,
        )

        # Fix k: at origin (N1 ~ 0.0)
        core.step_gnss_fix(lat0, lon0, alt0, accuracy_h_m=2.0, timestamp_ns=int(1e9))
        n1 = core._current_measurement_nis
        assert n1 is not None
        assert n1 < 0.1
        assert core.gnss_trust_calculator.last_evaluated_nis == pytest.approx(n1, rel=1e-5)

        # Fix k+1: 4.5m offset East (within 99% chi2 gate threshold 11.345, giving N2 ~ 5.0)
        d_lon = 4.5 / (6371000.0 * math.cos(math.radians(lat0))) * (180.0 / math.pi)
        core.step_gnss_fix(lat0, lon0 + d_lon, alt0, accuracy_h_m=2.0, timestamp_ns=int(2e9))

        n2 = core._current_measurement_nis
        assert n2 is not None
        assert n2 > 2.0  # Deliberately different from N1
        assert abs(n2 - n1) > 2.0

        # CRITICAL TEST: Trust calculation for fix k+1 used N2, NOT stale N1!
        # If the code had the old 1-fix delay, last_evaluated_nis would be n1 (< 0.1).
        assert core.gnss_trust_calculator.last_evaluated_nis == pytest.approx(n2, rel=1e-5)
        assert core.gnss_trust_calculator.last_evaluated_nis != pytest.approx(n1, abs=0.5)

    def test_sudden_nis_change_between_consecutive_fixes_scales_trust_immediately(self) -> None:
        """A sudden NIS jump on fix k must immediately degrade trust on fix k, not fix k+1."""
        from navigation.core import NavigationCore, NavigationCoreConfig
        core = NavigationCore(NavigationCoreConfig(
            velocitynet_enabled=False,
            biasnet_enabled=False,
            zupt_enabled=False,
            gnss_enabled=True,
        ))
        lat0, lon0, alt0 = 52.0, -1.5, 100.0
        core.initialize(
            lat0=lat0,
            lon0=lon0,
            alt0=alt0,
            p0_enu=np.zeros(3),
            v0_enu=np.zeros(3),
            q0=np.array([1.0, 0.0, 0.0, 0.0]),
            timestamp_ns=0,
        )

        # Fix 1: nominal (trust ~ 1.0)
        core.step_gnss_fix(lat0, lon0, alt0, accuracy_h_m=2.0, timestamp_ns=int(1e9))
        trust_nominal = core._last_gnss_quality.trust_score
        assert trust_nominal > 0.95

        # Fix 2: 6.8m displacement (elevated NIS ~ 9.2, past 95% threshold 7.815, below 99% gate 11.345)
        d_lon = 6.8 / (6371000.0 * math.cos(math.radians(lat0))) * (180.0 / math.pi)
        core.step_gnss_fix(lat0, lon0 + d_lon, alt0, accuracy_h_m=2.0, timestamp_ns=int(2e9))
        trust_elevated = core._last_gnss_quality.trust_score

        # Immediate degradation on fix 2 without 1-cycle lag
        assert trust_elevated < trust_nominal
        assert core._current_measurement_nis > 7.815

    def test_high_eskf_nis_lowers_trust_continuously_in_aided_mode(self) -> None:
        calc = GNSSTrustScoreCalculator(TrustScoreConfig(nis_threshold_95=7.815))
        # Nominal NIS (2.0 < 7.815)
        res_nom = calc.compute_trust(accuracy_m=2.0, sat_count=10, current_pos_enu=np.zeros(3), timestamp_ns=int(1e9), nis=2.0)
        assert res_nom.innovation_component == 1.0

        # Elevated NIS (12.0 > 7.815) -> continuous exponential decay
        calc.reset()
        res_elevated = calc.compute_trust(accuracy_m=2.0, sat_count=10, current_pos_enu=np.zeros(3), timestamp_ns=int(1e9), nis=12.0)
        assert 0.0 < res_elevated.innovation_component < 1.0
        assert res_elevated.trust_score < res_nom.trust_score
        assert 0.0 <= res_elevated.trust_score <= 1.0

        # Higher NIS (25.0) -> further decay
        calc.reset()
        res_high = calc.compute_trust(accuracy_m=2.0, sat_count=10, current_pos_enu=np.zeros(3), timestamp_ns=int(1e9), nis=25.0)
        assert res_high.innovation_component < res_elevated.innovation_component
        assert res_high.trust_score < res_elevated.trust_score

    def test_phase5_gating_preservation_rejects_outlier_normally(self) -> None:
        """Phase 5 innovation gate preservation:
        A large outlier must still be rejected even if trust score drops;
        low trust score cannot artificially inflate covariance to accept an invalid fix.
        """
        from navigation.core import NavigationCore, NavigationCoreConfig
        core = NavigationCore(NavigationCoreConfig(
            velocitynet_enabled=False,
            biasnet_enabled=False,
            zupt_enabled=False,
            gnss_enabled=True,
        ))
        lat0, lon0, alt0 = 52.0, -1.5, 100.0
        core.initialize(
            lat0=lat0,
            lon0=lon0,
            alt0=alt0,
            p0_enu=np.zeros(3),
            v0_enu=np.zeros(3),
            q0=np.array([1.0, 0.0, 0.0, 0.0]),
            timestamp_ns=0,
        )

        # Nominal fix
        core.step_gnss_fix(lat0, lon0, alt0, accuracy_h_m=2.0, timestamp_ns=int(1e9))
        pos_before = core.state.nominal.position_enu.copy()

        # Send a corrupt / outlier fix 80 meters away in GNSS_AIDED mode
        d_lon80 = 80.0 / (6371000.0 * math.cos(math.radians(lat0))) * (180.0 / math.pi)
        applied = core.step_gnss_fix(lat0, lon0 + d_lon80, alt0, accuracy_h_m=1.0, timestamp_ns=int(2e9))

        # Phase 5 Chi-Square gating must reject the update completely
        assert applied is False
        diag_p, _ = core.last_gnss_diagnostics
        assert diag_p is not None
        assert diag_p.applied is False
        assert diag_p.gating is not None
        assert diag_p.gating.accepted is False
        # State position must be strictly uncorrupted
        assert np.allclose(core.state.nominal.position_enu, pos_before)
        # Baseline innovation NIS was calculated
        assert core._current_measurement_nis > 16.266
        # Telemetry records rejection
        telem = core.get_gnss_telemetry()
        assert telem["total_fixes_rejected"] >= 1

    def test_reacquiring_aborts_on_implausible_fix(self) -> None:
        from navigation.core import NavigationCore, NavigationCoreConfig
        core = NavigationCore(NavigationCoreConfig(
            velocitynet_enabled=False,
            biasnet_enabled=False,
            zupt_enabled=False,
            gnss_enabled=True,
        ))
        lat0, lon0, alt0 = 52.0, -1.5, 100.0
        core.initialize(
            lat0=lat0,
            lon0=lon0,
            alt0=alt0,
            p0_enu=np.zeros(3),
            v0_enu=np.zeros(3),
            q0=np.array([1.0, 0.0, 0.0, 0.0]),
            timestamp_ns=0,
        )

        # Force state to REACQUIRING
        core.gnss_fsm.force_mode(GNSSMode.REACQUIRING, timestamp_ns=int(1e9), reason="RETURNING_FIX_ACCEPTED")
        core.mode = GNSSMode.REACQUIRING

        # Send an implausible fix (25m away when covariance is tiny -> NIS > 11.345)
        d_lon25 = 25.0 / (6371000.0 * math.cos(math.radians(lat0))) * (180.0 / math.pi)
        applied = core.step_gnss_fix(lat0, lon0 + d_lon25, alt0, accuracy_h_m=2.0, timestamp_ns=int(2e9))

        # Must reject and abort immediately back to DR_ONLY
        assert applied is False
        assert core.mode == GNSSMode.DR_ONLY
        assert core.gnss_recovery_manager.consecutive_valid_fixes == 0

    def test_core_reacquisition_timeout_falls_back_to_dr_only(self) -> None:
        """NavigationCore respects authoritative FSMConfig.reacq_timeout_s and resets recovery on timeout."""
        from navigation.core import NavigationCore, NavigationCoreConfig
        core = NavigationCore(NavigationCoreConfig(
            velocitynet_enabled=False,
            biasnet_enabled=False,
            zupt_enabled=False,
            gnss_enabled=True,
        ))
        lat0, lon0, alt0 = 52.0, -1.5, 100.0
        core.initialize(
            lat0=lat0,
            lon0=lon0,
            alt0=alt0,
            p0_enu=np.zeros(3),
            v0_enu=np.zeros(3),
            q0=np.array([1.0, 0.0, 0.0, 0.0]),
            timestamp_ns=0,
        )

        # Put into REACQUIRING
        core.gnss_fsm.force_mode(GNSSMode.REACQUIRING, timestamp_ns=int(1e9), reason="START_REACQ")
        core.mode = GNSSMode.REACQUIRING
        assert core.mode == GNSSMode.REACQUIRING

        # Simulate 12 seconds of IMU propagation without convergence (timeout is 10.0s)
        # Step IMU at 13.0s (dwell = 12.0s >= 10.0s)
        f_imu = np.array([0.0, 0.0, 9.80665])
        w_imu = np.zeros(3)
        core.step_imu(f_imu, w_imu, dt_s=0.01, timestamp_ns=int(13e9))

        # Mode must have fallen back to DR_ONLY due to authoritative timeout
        assert core.mode == GNSSMode.DR_ONLY
        telem = core.get_gnss_telemetry()
        assert telem["current_mode"] == "DR_ONLY"
        assert any("REACQUISITION_TIMEOUT_EXCEEDED" in t["reason"] for t in telem["transition_history"])

    def test_outage_detector_acceptance_semantics_and_smoothing_isolation(self) -> None:
        """Issue 1: Bounded recovery smoothing must NEVER masquerade as accepted GNSS update.

        Validates:
        - Unconverged recovery fix: fix_received is recorded, but update_applied is False.
        - total_fixes_accepted does NOT increment until actual ESKF measurement update occurs.
        - Converted/converged fix performs real ESKF update, applied is True, and increments total_fixes_accepted.
        """
        from navigation.core import NavigationCore, NavigationCoreConfig
        core = NavigationCore(NavigationCoreConfig(
            velocitynet_enabled=False,
            biasnet_enabled=False,
            zupt_enabled=False,
            gnss_enabled=True,
        ))
        lat0, lon0, alt0 = 52.0, -1.5, 100.0
        core.initialize(
            lat0=lat0,
            lon0=lon0,
            alt0=alt0,
            p0_enu=np.zeros(3),
            v0_enu=np.zeros(3),
            q0=np.array([1.0, 0.0, 0.0, 0.0]),
            timestamp_ns=0,
        )

        # 1. Normal accepted fix in GNSS_AIDED
        applied1 = core.step_gnss_fix(lat0, lon0, alt0, accuracy_h_m=2.0, timestamp_ns=int(1e9))
        assert applied1 is True
        telem1 = core.get_gnss_telemetry()
        assert telem1["total_fixes_received"] == 1
        assert telem1["total_fixes_accepted"] == 1
        assert telem1["total_fixes_rejected"] == 0

        # 2. Put into REACQUIRING with a 5.0m target offset
        # Target position is 5.0m away: requires multiple bounded steps to converge (tolerance 1.5m)
        core.gnss_fsm.force_mode(GNSSMode.REACQUIRING, timestamp_ns=int(2e9), reason="ENTER_REACQ")
        core.mode = GNSSMode.REACQUIRING

        d_lon5 = 5.0 / (6371000.0 * math.cos(math.radians(lat0))) * (180.0 / math.pi)
        target_lon = lon0 + d_lon5

        # Fix 1 in REACQUIRING: step bounded to <= 2.0m, remaining error ~3.0m > 1.5m tolerance -> NOT converged
        applied_reacq1 = core.step_gnss_fix(lat0, target_lon, alt0, accuracy_h_m=2.0, timestamp_ns=int(3e9))
        assert applied_reacq1 is False, "Unconverged recovery fix must NOT be reported as accepted"
        diag_p, _ = core.last_gnss_diagnostics
        assert diag_p is None, "No ESKF measurement update should execute prior to convergence"
        telem_reacq1 = core.get_gnss_telemetry()
        assert telem_reacq1["total_fixes_received"] == 2
        assert telem_reacq1["total_fixes_accepted"] == 1, "Unconverged fix must NOT increment total_fixes_accepted"
        assert telem_reacq1["total_fixes_rejected"] == 1

        # Fix 2 in REACQUIRING: another bounded step
        applied_reacq2 = core.step_gnss_fix(lat0, target_lon, alt0, accuracy_h_m=2.0, timestamp_ns=int(4e9))
        assert applied_reacq2 is False
        telem_reacq2 = core.get_gnss_telemetry()
        assert telem_reacq2["total_fixes_received"] == 3
        assert telem_reacq2["total_fixes_accepted"] == 1

        # Provide consecutive fixes at target to reach convergence (3 consecutive fixes within 1.5m)
        applied_reacq3 = core.step_gnss_fix(lat0, target_lon, alt0, accuracy_h_m=2.0, timestamp_ns=int(5e9))
        applied_reacq4 = core.step_gnss_fix(lat0, target_lon, alt0, accuracy_h_m=2.0, timestamp_ns=int(6e9))
        applied_reacq5 = core.step_gnss_fix(lat0, target_lon, alt0, accuracy_h_m=2.0, timestamp_ns=int(7e9))

        # Once converged, ESKF update is executed
        assert core.gnss_recovery_manager.consecutive_valid_fixes >= 3
        assert applied_reacq5 is True
        diag_p_final, _ = core.last_gnss_diagnostics
        assert diag_p_final is not None
        assert diag_p_final.applied is True
        telem_final = core.get_gnss_telemetry()
        assert telem_final["total_fixes_accepted"] >= 2

    def test_authority_taxonomy_ml_eskf_and_recovery_smoothing(self) -> None:
        """Issue 2: Verify architectural distinction between ML, ESKF update, and recovery smoothing.

        Distinguishes:
        1. ML prediction: fuses velocity/bias measurement via Kalman gain with covariance reduction.
        2. GNSS ESKF measurement update: fuses GNSS position via Kalman gain with covariance reduction.
        3. Bounded recovery supervisory correction: modifies nominal position by bounded step
           (<=3.0m, <=2.0m/s) without mutating error covariance P.
        """
        from navigation.core import NavigationCore, NavigationCoreConfig
        from navigation.gnss.recovery import GNSSRecoveryManager, RecoveryConfig

        rec_mgr = GNSSRecoveryManager(RecoveryConfig(max_displacement_rate_mps=2.0, max_single_step_m=3.0))

        core = NavigationCore(NavigationCoreConfig(
            velocitynet_enabled=False,
            biasnet_enabled=False,
            zupt_enabled=False,
            gnss_enabled=True,
        ))
        lat0, lon0, alt0 = 52.0, -1.5, 100.0
        core.initialize(
            lat0=lat0,
            lon0=lon0,
            alt0=alt0,
            p0_enu=np.zeros(3),
            v0_enu=np.zeros(3),
            q0=np.array([1.0, 0.0, 0.0, 0.0]),
            timestamp_ns=0,
        )

        state_before = core.state
        p_cov_before = state_before.covariance.copy()
        pos_before = state_before.nominal.position_enu.copy()

        # Compute bounded correction for an 8.0m offset target
        target_enu = np.array([8.0, 0.0, 0.0])
        step = rec_mgr.compute_bounded_correction(target_enu, state_before, dt_s=1.0)

        # Rate bound assertions
        assert step.is_clamped is True
        assert step.applied_step_norm_m <= 2.01
        assert step.delta_p_bounded[0] == pytest.approx(2.0, abs=0.01)

        # Apply bounded supervisory correction
        state_after = rec_mgr.apply_bounded_correction(state_before, step)

        # 1. Nominal position IS modified by the bounded correction
        assert not np.array_equal(state_after.nominal.position_enu, pos_before)
        assert state_after.nominal.position_enu[0] == pytest.approx(2.0, abs=0.01)

        # 2. Covariance matrix P is strictly UNTOUCHED (not mutated by smoothing step)
        assert np.array_equal(state_after.covariance, p_cov_before)

        # 3. In contrast, an actual GNSS ESKF Kalman update strictly modifies/reduces covariance P
        state_eskf, diag_p = core.gnss_model.update_position(
            state=state_after,
            lat=lat0,
            lon=lon0,
            alt=alt0,
            accuracy_h_m=2.0,
            trust_score=1.0,
            timestamp_ns=int(1e9),
        )
        assert diag_p.applied is True
        # Diagonal position variance in P must have decreased due to Kalman update
        assert state_eskf.covariance[0, 0] < p_cov_before[0, 0]


