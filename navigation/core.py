"""NavigationCore: Central Python Reference Implementation and Behavioral Oracle for C.O.M.P.A.S.S.

Coordinates sensor ingestion, strapdown INS propagation, error-state Kalman filter (ESKF)
fusion, classical ZUPT, and learned pseudo-measurements (VelocityNet v1.1 and BiasNet v1.0).

Architectural Authority Invariants:
1. The 15-state ESKF is the SOLE authoritative navigation state estimator.
2. Neural models NEVER directly overwrite state or covariance.
3. Neural predictions enter solely as uncertainty-weighted measurements through generic eskf_update.
4. Classical ZUPT remains 100% operational without ML.
5. Fallback flags allow fully independent decoupling of VelocityNet and BiasNet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from navigation.frames.local_geo import GeoReference
from navigation.eskf.gating import MahalanobisGating
from navigation.eskf.predict import ProcessNoiseConfig, predict_eskf
from navigation.eskf.state import ESKFNominalState, ESKFState
from navigation.eskf.update import UpdateDiagnostics
from navigation.eskf.scheduling import CadenceConfig, MLCadenceScheduler
from navigation.eskf.measurements.gnss import GNSSMeasurementModel, GNSSUpdateConfig
from navigation.eskf.measurements.zupt import (
    ClassicalZUPTDetector,
    ZUPTDetectorConfig,
    ZUPTDetectorDiagnostics,
    ZUPTMeasurementConfig,
    ZUPTMeasurementModel,
)
from navigation.eskf.measurements.velocitynet import (
    VelocityNetAdapterDiagnostics,
    VelocityNetConfig,
    VelocityNetMeasurementModel,
)
from navigation.eskf.measurements.biasnet import (
    BiasNetAdapterDiagnostics,
    BiasNetConfig,
    BiasNetMeasurementModel,
)
from navigation.ml.model_runner import (
    BiasNetOutput,
    ModelRunnerConfig,
    ONNXModelRunner,
    VelocityNetOutput,
)
from navigation.ml.window_buffer import CausalWindowBuffer
from navigation.schemas.state import GNSSMode, NavigationState
from navigation.gnss import (
    BoundedCorrectionStep,
    FSMConfig,
    GNSSModeFSM,
    GNSSOutageDetector,
    GNSSQualityResult,
    GNSSRecoveryManager,
    GNSSTrustScoreCalculator,
    OutageDetectorConfig,
    RecoveryConfig,
    TrustScoreConfig,
)


@dataclass(frozen=True)
class NavigationCoreConfig:
    """System-level configuration and fallback flags."""
    # Module enablement flags
    velocitynet_enabled: bool = True
    biasnet_enabled: bool = True
    zupt_enabled: bool = True
    gnss_enabled: bool = True

    # Sub-component configurations
    process_noise: ProcessNoiseConfig = field(default_factory=ProcessNoiseConfig)
    cadence: CadenceConfig = field(default_factory=CadenceConfig)
    gnss: GNSSUpdateConfig = field(default_factory=GNSSUpdateConfig)
    zupt_detector: ZUPTDetectorConfig = field(default_factory=ZUPTDetectorConfig)
    zupt_measurement: ZUPTMeasurementConfig = field(default_factory=ZUPTMeasurementConfig)
    velocitynet: VelocityNetConfig = field(default_factory=VelocityNetConfig)
    biasnet: BiasNetConfig = field(default_factory=BiasNetConfig)
    model_runner: ModelRunnerConfig = field(default_factory=ModelRunnerConfig)
    trust_score: TrustScoreConfig = field(default_factory=TrustScoreConfig)
    outage_detector: OutageDetectorConfig = field(default_factory=OutageDetectorConfig)
    fsm: FSMConfig = field(default_factory=FSMConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)


@dataclass(frozen=True)
class CovarianceHealthDiagnostics:
    """Health metrics for the 15x15 error-state covariance matrix P."""
    is_finite: bool
    is_symmetric: bool
    is_psd: bool
    min_eigenvalue: float
    max_diagonal: float
    quaternion_normalized: bool
    quaternion_norm: float


@dataclass(frozen=True)
class NavigationCoreCycleOutput:
    """Output diagnostics produced by a single IMU navigation step."""
    timestamp_ns: int
    state: ESKFState
    covariance_health: CovarianceHealthDiagnostics
    zupt_diagnostics: Optional[ZUPTDetectorDiagnostics] = None
    zupt_applied: bool = False
    velocitynet_diagnostics: Optional[VelocityNetAdapterDiagnostics] = None
    biasnet_diagnostics: Optional[BiasNetAdapterDiagnostics] = None


class NavigationCore:
    """Central navigation estimator coordinating sensor ingestion, propagation, and fusion."""

    def __init__(self, config: Optional[NavigationCoreConfig] = None) -> None:
        self.config = config or NavigationCoreConfig()
        
        # State estimation
        self.state: Optional[ESKFState] = None
        self.geo_ref: Optional[GeoReference] = None
        self.mode: GNSSMode = GNSSMode.GNSS_AIDED

        # Sub-modules
        self.window_buffer = CausalWindowBuffer(max_length=20)
        self.scheduler = MLCadenceScheduler(config=self.config.cadence)
        
        # ML Model runner
        self.model_runner = ONNXModelRunner(config=self.config.model_runner)

        # Measurement adapters
        self.zupt_detector = ClassicalZUPTDetector(config=self.config.zupt_detector)
        self.zupt_model = ZUPTMeasurementModel(config=self.config.zupt_measurement)
        self.gnss_model: Optional[GNSSMeasurementModel] = None
        self.last_gnss_diagnostics: Optional[Tuple[UpdateDiagnostics, Optional[UpdateDiagnostics]]] = None
        
        vnet_cfg = self.config.velocitynet
        if not self.config.velocitynet_enabled:
            vnet_cfg = VelocityNetConfig(enabled=False)
        self.vnet_model = VelocityNetMeasurementModel(config=vnet_cfg)

        bnet_cfg = self.config.biasnet
        if not self.config.biasnet_enabled:
            bnet_cfg = BiasNetConfig(enabled=False)
        self.bnet_model = BiasNetMeasurementModel(config=bnet_cfg)

        # GNSS Supervisory Subsystem (Phase 10)
        self.gnss_trust_calculator = GNSSTrustScoreCalculator(config=self.config.trust_score)
        self.gnss_outage_detector = GNSSOutageDetector(config=self.config.outage_detector)
        self.gnss_fsm = GNSSModeFSM(config=self.config.fsm, initial_mode=self.mode)
        self.gnss_recovery_manager = GNSSRecoveryManager(config=self.config.recovery)
        self._last_gnss_timestamp_ns: Optional[int] = None
        self._last_gnss_quality: Optional[GNSSQualityResult] = None
        self._last_recovery_step: Optional[BoundedCorrectionStep] = None
        self._current_measurement_nis: Optional[float] = None
        self._last_update_nis: Optional[float] = None
        self._last_gnss_nis: Optional[float] = None

        self._ml_telemetry: dict[str, Any] = {
            "velocitynet": {
                "scheduler_due": 0,
                "buffer_not_ready": 0,
                "inference_executed": 0,
                "update_accepted": 0,
                "update_rejected": 0,
                "rejection_reasons": {},
            },
            "biasnet": {
                "scheduler_due": 0,
                "buffer_not_ready": 0,
                "inference_executed": 0,
                "update_accepted": 0,
                "update_rejected": 0,
                "rejection_reasons": {},
            },
        }

    def initialize(
        self,
        lat0: float,
        lon0: float,
        alt0: float,
        p0_enu: np.ndarray,
        v0_enu: np.ndarray,
        q0: np.ndarray,
        gyro_bias0: Optional[np.ndarray] = None,
        accel_bias0: Optional[np.ndarray] = None,
        p0_cov: Optional[np.ndarray] = None,
        timestamp_ns: int = 0,
        mode: GNSSMode = GNSSMode.GNSS_AIDED,
    ) -> None:
        """Initialize the navigation core and local geographic tangent plane."""
        self.geo_ref = GeoReference(lat_ref=float(lat0), lon_ref=float(lon0), alt_ref=float(alt0))
        self.gnss_model = GNSSMeasurementModel(geo_reference=self.geo_ref, config=self.config.gnss)
        self.mode = mode
        self.gnss_fsm.reset(initial_mode=mode, timestamp_ns=timestamp_ns)
        self.gnss_outage_detector.reset(session_start_timestamp_ns=timestamp_ns)
        self.gnss_trust_calculator.reset()
        self.gnss_recovery_manager.reset()
        self._last_gnss_timestamp_ns = None
        self._last_gnss_quality = None
        self._last_recovery_step = None
        self._current_measurement_nis = None
        self._last_update_nis = None
        self._last_gnss_nis = None

        nom = ESKFNominalState.from_components(
            position_enu=p0_enu,
            velocity_enu=v0_enu,
            q=q0,
            accel_bias=accel_bias0,
            gyro_bias=gyro_bias0,
            timestamp_ns=int(timestamp_ns),
        )

        if p0_cov is None:
            # Conservative default initial covariance
            p0_cov = np.diag([
                1.0, 1.0, 4.0,           # Position (m^2)
                0.1, 0.1, 0.5,           # Velocity (m/s)^2
                0.01, 0.01, 0.05,        # Attitude (rad^2)
                0.05, 0.05, 0.05,        # Accel bias (m/s^2)^2
                0.005, 0.005, 0.005,     # Gyro bias (rad/s)^2
            ]) ** 2

        self.state = ESKFState(nominal=nom, covariance=p0_cov)
        self.window_buffer.reset()
        self.scheduler.reset()
        self.vnet_model.reset()
        self._reset_ml_telemetry()

    def _reset_ml_telemetry(self) -> None:
        self._ml_telemetry = {
            "velocitynet": {
                "scheduler_due": 0,
                "buffer_not_ready": 0,
                "inference_executed": 0,
                "update_accepted": 0,
                "update_rejected": 0,
                "rejection_reasons": {},
            },
            "biasnet": {
                "scheduler_due": 0,
                "buffer_not_ready": 0,
                "inference_executed": 0,
                "update_accepted": 0,
                "update_rejected": 0,
                "rejection_reasons": {},
            },
        }

    def check_covariance_health(self) -> CovarianceHealthDiagnostics:
        """Validate symmetry, positive definiteness, and bounds of covariance P."""
        if self.state is None:
            raise RuntimeError("NavigationCore not initialized")

        P = self.state.covariance
        is_finite = bool(np.all(np.isfinite(P)))
        is_sym = bool(np.allclose(P, P.T, atol=1e-7))

        try:
            eigs = np.linalg.eigvalsh(P)
            min_eig = float(np.min(eigs))
            is_psd = bool(min_eig > -1e-7)
        except Exception:
            min_eig = float("nan")
            is_psd = False

        max_diag = float(np.max(np.diag(P)))
        q_norm = float(np.linalg.norm(self.state.nominal.q))
        q_norm_ok = bool(abs(q_norm - 1.0) < 1e-3)

        return CovarianceHealthDiagnostics(
            is_finite=is_finite,
            is_symmetric=is_sym,
            is_psd=is_psd,
            min_eigenvalue=min_eig,
            max_diagonal=max_diag,
            quaternion_normalized=q_norm_ok,
            quaternion_norm=q_norm,
        )

    def step_imu(
        self,
        f_m_v: np.ndarray,
        omega_m_v: np.ndarray,
        dt_s: float,
        timestamp_ns: int,
    ) -> NavigationCoreCycleOutput:
        """Execute a single IMU strapdown propagation and scheduled measurement cycle.

        Args:
            f_m_v: (3,) vehicle-frame specific force (m/s^2).
            omega_m_v: (3,) vehicle-frame angular rate (rad/s).
            dt_s: Sample period in seconds.
            timestamp_ns: Current epoch in nanoseconds.

        Returns:
            NavigationCoreCycleOutput containing updated state and diagnostics.
        """
        if self.state is None:
            raise RuntimeError("NavigationCore must be initialized before stepping")

        t_ns = int(timestamp_ns)
        f_arr = np.asarray(f_m_v, dtype=np.float64).reshape(3)
        w_arr = np.asarray(omega_m_v, dtype=np.float64).reshape(3)

        # 1. Update causal history buffer
        self.window_buffer.push(timestamp_ns=t_ns, f_m_v=f_arr, omega_m_v=w_arr)

        # 2. ESKF strapdown INS propagation
        self.state = predict_eskf(
            state=self.state,
            f_m_v=f_arr,
            omega_m_v=w_arr,
            dt=float(dt_s),
            timestamp_ns=t_ns,
            process_noise=self.config.process_noise,
        )

        # 3. Classical Gated ZUPT check (zero-ML dependency)
        zupt_diag: Optional[ZUPTDetectorDiagnostics] = None
        zupt_applied = False
        if self.config.zupt_enabled:
            zupt_diag = self.zupt_detector.push(omega_v=w_arr, f_v=f_arr)
            if zupt_diag.is_stationary:
                self.state, d_z = self.zupt_model.update(self.state, timestamp_ns=t_ns)
                zupt_applied = d_z.applied

        # 4. Evaluate time-based cadence for neural models
        run_vnet, run_bnet = self.scheduler.evaluate_cycle(t_ns)

        vnet_diag: Optional[VelocityNetAdapterDiagnostics] = None
        if run_vnet and self.config.velocitynet_enabled:
            self._ml_telemetry["velocitynet"]["scheduler_due"] += 1
            has_win, raw_feats, win_ts, reason = self.window_buffer.get_causal_window(current_timestamp_ns=t_ns)
            if has_win and raw_feats is not None:
                self._ml_telemetry["velocitynet"]["inference_executed"] += 1
                vnet_out = self.model_runner.run_velocitynet(raw_feats, win_ts)
                self.state, vnet_diag = self.vnet_model.update(self.state, vnet_out, timestamp_ns=t_ns)
                self.scheduler.mark_velocitynet_executed(t_ns)
                if vnet_diag.applied:
                    self._ml_telemetry["velocitynet"]["update_accepted"] += 1
                else:
                    self._ml_telemetry["velocitynet"]["update_rejected"] += 1
                    rej = vnet_diag.reason or "UNKNOWN"
                    reasons = self._ml_telemetry["velocitynet"]["rejection_reasons"]
                    reasons[rej] = reasons.get(rej, 0) + 1
            else:
                self._ml_telemetry["velocitynet"]["buffer_not_ready"] += 1
                self.scheduler.mark_velocitynet_scheduled(t_ns)
                vnet_diag = VelocityNetAdapterDiagnostics(
                    applied=False,
                    reason=f"BUFFER_{reason}",
                    z_speed=0.0,
                    predicted_speed=0.0,
                    variance=0.0,
                )

        bnet_diag: Optional[BiasNetAdapterDiagnostics] = None
        if run_bnet and self.config.biasnet_enabled:
            self._ml_telemetry["biasnet"]["scheduler_due"] += 1
            has_win, raw_feats, win_ts, reason = self.window_buffer.get_causal_window(current_timestamp_ns=t_ns)
            if has_win and raw_feats is not None:
                self._ml_telemetry["biasnet"]["inference_executed"] += 1
                bnet_out = self.model_runner.run_biasnet(raw_feats, win_ts)
                self.state, bnet_diag = self.bnet_model.update(self.state, bnet_out, timestamp_ns=t_ns)
                self.scheduler.mark_biasnet_executed(t_ns)
                if bnet_diag.applied:
                    self._ml_telemetry["biasnet"]["update_accepted"] += 1
                else:
                    self._ml_telemetry["biasnet"]["update_rejected"] += 1
                    rej = bnet_diag.reason or "UNKNOWN"
                    reasons = self._ml_telemetry["biasnet"]["rejection_reasons"]
                    reasons[rej] = reasons.get(rej, 0) + 1
            else:
                self._ml_telemetry["biasnet"]["buffer_not_ready"] += 1
                self.scheduler.mark_biasnet_scheduled(t_ns)
                bnet_diag = BiasNetAdapterDiagnostics(
                    applied=False,
                    reason=f"BUFFER_{reason}",
                    delta_bias=np.zeros(6, dtype=np.float64),
                    covariance_diag=self.bnet_model.R_diag,
                )

        # 5. GNSS Outage and FSM Mode Evaluation (Phase 10)
        outage_status = self.gnss_outage_detector.evaluate_outage(t_ns)
        prev_mode = self.mode
        self.gnss_fsm.evaluate_transition(
            current_timestamp_ns=t_ns,
            is_outage=outage_status.is_outage,
            outage_reason=outage_status.condition.value,
        )
        self.mode = self.gnss_fsm.current_mode
        if prev_mode == GNSSMode.REACQUIRING and self.mode == GNSSMode.DR_ONLY:
            self.gnss_recovery_manager.reset()

        # 6. Covariance health check
        health = self.check_covariance_health()

        return NavigationCoreCycleOutput(
            timestamp_ns=t_ns,
            state=self.state,
            covariance_health=health,
            zupt_diagnostics=zupt_diag,
            zupt_applied=zupt_applied,
            velocitynet_diagnostics=vnet_diag,
            biasnet_diagnostics=bnet_diag,
        )

    def step_gnss_fix(
        self,
        lat: float,
        lon: float,
        alt: float,
        v_east: Optional[float] = None,
        v_north: Optional[float] = None,
        v_up: Optional[float] = None,
        accuracy_h_m: Optional[float] = None,
        accuracy_v_m: Optional[float] = None,
        accuracy_speed_mps: Optional[float] = None,
        sat_count: Optional[int] = None,
        trust_score: Optional[float] = None,
        timestamp_ns: Optional[int] = None,
    ) -> bool:
        """Apply an incoming GNSS position/velocity fix through Phase 10 supervisory pipeline."""
        if not self.config.gnss_enabled or self.state is None or self.gnss_model is None or self.geo_ref is None:
            return False

        t_ns = self.state.nominal.timestamp_ns if timestamp_ns is None else int(timestamp_ns)
        p_enu = np.array(self.geo_ref.geodetic_to_enu(lat, lon, alt), dtype=np.float64)

        # 1. Record fix arrival in outage detector
        self.gnss_outage_detector.record_fix_received(t_ns)

        # 2. Pre-update innovation evaluation against unscaled baseline covariance R_base
        pre_diag = self.gnss_model.evaluate_pre_update_innovation(
            state=self.state,
            lat=lat,
            lon=lon,
            alt=alt,
            accuracy_h_m=accuracy_h_m,
            accuracy_v_m=accuracy_v_m,
        )
        if pre_diag is not None and pre_diag.is_valid:
            self._current_measurement_nis = float(pre_diag.nis_base)
        else:
            self._current_measurement_nis = None

        # 3. Continuous trust evaluation using CURRENT fix pre-update NIS
        quality_res = self.gnss_trust_calculator.compute_trust(
            accuracy_m=accuracy_h_m,
            sat_count=sat_count,
            current_pos_enu=p_enu,
            timestamp_ns=t_ns,
            nis=self._current_measurement_nis,
        )
        self._last_gnss_quality = quality_res

        # If caller explicitly provided a trust score (e.g. In synthetic tests), respect it
        effective_trust = (
            float(trust_score)
            if (trust_score is not None and math.isfinite(trust_score))
            else quality_res.trust_score
        )

        applied = False

        # 4. Handle according to current authoritative FSM mode
        if self.mode == GNSSMode.DR_ONLY:
            # Validate returning fix before reacquisition
            val_res = self.gnss_recovery_manager.validate_returning_fix(
                gnss_pos_enu=p_enu,
                eskf_state=self.state,
                trust_score=effective_trust,
                timestamp_ns=t_ns,
            )
            outage_status = self.gnss_outage_detector.evaluate_outage(t_ns)
            prev_mode = self.mode
            self.gnss_fsm.evaluate_transition(
                current_timestamp_ns=t_ns,
                is_outage=outage_status.is_outage,
                outage_reason=outage_status.condition.value,
                is_returning_fix_valid=val_res.is_plausible,
                trust_score=effective_trust,
            )
            self.mode = self.gnss_fsm.current_mode

            if self.mode == GNSSMode.REACQUIRING:
                # First reacquisition fix: apply bounded rate step
                dt_s = (t_ns - self._last_gnss_timestamp_ns) * 1e-9 if self._last_gnss_timestamp_ns else 1.0
                step = self.gnss_recovery_manager.compute_bounded_correction(p_enu, self.state, dt_s)
                self.state = self.gnss_recovery_manager.apply_bounded_correction(self.state, step)
                self._last_recovery_step = step

                diag_p = None
                if step.is_converged:
                    self.state, diag_p = self.gnss_model.update_position(
                        state=self.state,
                        lat=lat,
                        lon=lon,
                        alt=alt,
                        accuracy_h_m=accuracy_h_m,
                        accuracy_v_m=accuracy_v_m,
                        trust_score=effective_trust,
                        timestamp_ns=t_ns,
                    )
                    if diag_p.gating is not None:
                        self._last_update_nis = float(diag_p.gating.mahalanobis_sq)
                        self._last_gnss_nis = self._last_update_nis
                        self.gnss_trust_calculator.update_innovation_nis(self._last_update_nis)
                self.last_gnss_diagnostics = (diag_p, None)
                update_applied = bool(diag_p is not None and diag_p.applied)
                self.gnss_outage_detector.record_update_result(t_ns, applied=update_applied)
                applied = update_applied
            else:
                # Still in DR_ONLY (e.g. fix implausible or dwell active)
                self.gnss_outage_detector.record_update_result(t_ns, applied=False)
                applied = False

        elif self.mode == GNSSMode.REACQUIRING:
            # Validate returning fix plausibility while reacquiring
            val_res = self.gnss_recovery_manager.validate_returning_fix(
                gnss_pos_enu=p_enu,
                eskf_state=self.state,
                trust_score=effective_trust,
                timestamp_ns=t_ns,
            )
            if not val_res.is_plausible:
                self.gnss_recovery_manager.reset()
                self.gnss_outage_detector.record_update_result(t_ns, applied=False)
                self.gnss_fsm.force_mode(
                    GNSSMode.DR_ONLY,
                    timestamp_ns=t_ns,
                    reason=f"REACQUISITION_FAILED_IMPLAUSIBLE_FIX_{val_res.reason}",
                )
                self.mode = self.gnss_fsm.current_mode
                self._last_gnss_timestamp_ns = t_ns
                return False

            # Continuing bounded recovery
            dt_s = (t_ns - self._last_gnss_timestamp_ns) * 1e-9 if self._last_gnss_timestamp_ns else 1.0
            step = self.gnss_recovery_manager.compute_bounded_correction(p_enu, self.state, dt_s)
            self.state = self.gnss_recovery_manager.apply_bounded_correction(self.state, step)
            self._last_recovery_step = step

            diag_p = None
            if step.is_converged:
                self.state, diag_p = self.gnss_model.update_position(
                    state=self.state,
                    lat=lat,
                    lon=lon,
                    alt=alt,
                    accuracy_h_m=accuracy_h_m,
                    accuracy_v_m=accuracy_v_m,
                    trust_score=effective_trust,
                    timestamp_ns=t_ns,
                )
                if diag_p.gating is not None:
                    self._last_update_nis = float(diag_p.gating.mahalanobis_sq)
                    self._last_gnss_nis = self._last_update_nis
                    self.gnss_trust_calculator.update_innovation_nis(self._last_update_nis)
            diag_v = None
            if v_east is not None and v_north is not None:
                v_u = 0.0 if v_up is None else float(v_up)
                self.state, diag_v = self.gnss_model.update_velocity(
                    state=self.state,
                    v_east=float(v_east),
                    v_north=float(v_north),
                    v_up=v_u,
                    accuracy_speed_mps=accuracy_speed_mps,
                    trust_score=effective_trust,
                    timestamp_ns=t_ns,
                )
            self.last_gnss_diagnostics = (diag_p, diag_v)
            update_applied = bool(diag_p is not None and diag_p.applied)
            self.gnss_outage_detector.record_update_result(t_ns, applied=update_applied)
            applied = update_applied

            # Check if convergence achieved or reacquisition timed out
            outage_status = self.gnss_outage_detector.evaluate_outage(t_ns)
            prev_mode = self.mode
            self.gnss_fsm.evaluate_transition(
                current_timestamp_ns=t_ns,
                is_outage=outage_status.is_outage,
                outage_reason=outage_status.condition.value,
                is_returning_fix_valid=True,
                is_recovery_converged=step.is_converged,
                trust_score=effective_trust,
            )
            self.mode = self.gnss_fsm.current_mode
            if prev_mode == GNSSMode.REACQUIRING and self.mode == GNSSMode.DR_ONLY:
                self.gnss_recovery_manager.reset()

        else:
            # GNSS_AIDED (nominal)
            self.state, diag_p = self.gnss_model.update_position(
                state=self.state,
                lat=lat,
                lon=lon,
                alt=alt,
                accuracy_h_m=accuracy_h_m,
                accuracy_v_m=accuracy_v_m,
                trust_score=effective_trust,
                timestamp_ns=t_ns,
            )
            diag_v = None
            if v_east is not None and v_north is not None:
                v_u = 0.0 if v_up is None else float(v_up)
                self.state, diag_v = self.gnss_model.update_velocity(
                    state=self.state,
                    v_east=float(v_east),
                    v_north=float(v_north),
                    v_up=v_u,
                    accuracy_speed_mps=accuracy_speed_mps,
                    trust_score=effective_trust,
                    timestamp_ns=t_ns,
                )
            self.last_gnss_diagnostics = (diag_p, diag_v)
            self.gnss_outage_detector.record_update_result(t_ns, applied=diag_p.applied)
            if diag_p.gating is not None:
                self._last_update_nis = float(diag_p.gating.mahalanobis_sq)
                self._last_gnss_nis = self._last_update_nis
                self.gnss_trust_calculator.update_innovation_nis(self._last_update_nis)

            # Evaluate if persistent rejection or outage occurred
            outage_status = self.gnss_outage_detector.evaluate_outage(t_ns)
            self.gnss_fsm.evaluate_transition(
                current_timestamp_ns=t_ns,
                is_outage=outage_status.is_outage,
                outage_reason=outage_status.condition.value,
                trust_score=effective_trust,
            )
            self.mode = self.gnss_fsm.current_mode
            applied = diag_p.applied

        self._last_gnss_timestamp_ns = t_ns
        return applied

    def get_navigation_state(self) -> NavigationState:
        """Serialize current state into canonical Phase 1 NavigationState schema."""
        if self.state is None or self.geo_ref is None:
            raise RuntimeError("NavigationCore not initialized")

        ref_pt = (self.geo_ref.lat_ref, self.geo_ref.lon_ref)
        return self.state.nominal.to_navigation_state(
            covariance=self.state.covariance,
            reference_point=ref_pt,
            mode=self.mode,
        )

    def get_ml_telemetry(self) -> dict[str, Any]:
        """Return standardized ML telemetry counters adhering to Part 4 taxonomy."""
        import copy
        return copy.deepcopy(self._ml_telemetry)

    def get_gnss_telemetry(self) -> dict[str, Any]:
        """Return standardized Phase 10 GNSS supervisory telemetry."""
        cur_t = self.state.nominal.timestamp_ns if self.state else 0
        outage_status = self.gnss_outage_detector.evaluate_outage(cur_t)
        return {
            "current_mode": self.mode.value,
            "time_in_current_mode_s": self.gnss_fsm.time_in_current_mode_s(cur_t),
            "transition_count": self.gnss_fsm.transition_count,
            "transition_history": [t.to_dict() for t in self.gnss_fsm.transition_history],
            "total_fixes_received": self.gnss_outage_detector.total_fixes_received,
            "total_fixes_accepted": self.gnss_outage_detector.total_fixes_accepted,
            "total_fixes_rejected": self.gnss_outage_detector.total_fixes_rejected,
            "consecutive_rejections": outage_status.consecutive_rejections,
            "outage_condition": outage_status.condition.value,
            "is_outage": outage_status.is_outage,
            "time_since_last_fix_s": outage_status.time_since_last_fix_s,
            "last_trust_score": self._last_gnss_quality.trust_score if self._last_gnss_quality else 1.0,
            "current_measurement_nis": self._current_measurement_nis,
            "last_update_nis": self._last_update_nis,
            "last_eskf_nis": self._last_update_nis,
            "covariance_scale": 1.0 / max(0.05, self._last_gnss_quality.trust_score) if self._last_gnss_quality else 1.0,
            "recovery_cycles": self.gnss_recovery_manager.recovery_cycles,
            "maximum_recovery_correction_m": self.gnss_recovery_manager.maximum_correction_applied_m,
            "consecutive_valid_recovery_fixes": self.gnss_recovery_manager.consecutive_valid_fixes,
        }
