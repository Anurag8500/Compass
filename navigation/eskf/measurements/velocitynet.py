"""VelocityNet Forward Speed ESKF Measurement Adapter (Phase 9).

Integrates VelocityNet forward speed predictions as scalar velocity measurements
projected onto the vehicle's body-forward axis in the local navigation frame (ENU).

Mathematical Formulation:
- Let R_v^n = R(q) be the current vehicle-to-navigation rotation matrix.
- The vehicle forward direction in ENU is:
      fwd_n = R_v^n[:, 0]
- Measurement:
      z_v = v_fwd_pred (m/s, smoothed via causal EMA)
- Predicted measurement:
      h_v(x) = fwd_n^T v^n
- Innovation:
      y_v = z_v - h_v(x)
- 15D Error-state Jacobian:
      H_v = [0_1x3, fwd_n^T, 0_1x3, 0_1x3, 0_1x3] in R^{1x15}
- Measurement Covariance R_v:
      R_v = clamp(exp(clamp(log_var, -10.0, 10.0)), R_v_min, R_v_max)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from navigation.eskf.gating import MahalanobisGating
from navigation.eskf.state import ESKFState
from navigation.eskf.update import UpdateDiagnostics, eskf_update
from navigation.ml.model_runner import VelocityNetOutput


class CausalEMA:
    """Causal Exponential Moving Average for smoothing neural speed predictions online."""

    def __init__(self, alpha: float = 0.2) -> None:
        self.alpha = float(alpha)
        self.state: Optional[float] = None

    def update(self, val: float) -> float:
        if self.state is None:
            self.state = float(val)
        else:
            self.state = self.alpha * float(val) + (1.0 - self.alpha) * self.state
        return self.state

    def reset(self) -> None:
        self.state = None


@dataclass(frozen=True)
class VelocityNetConfig:
    """Configuration for VelocityNet measurement adapter."""
    enabled: bool = True
    min_speed_threshold_mps: float = 0.5   # Motion gate: suppress near standstill to avoid fighting ZUPT
    use_heteroscedastic: bool = True       # Use model-predicted variance vs fixed Phase 8 baseline
    fixed_variance_mps2: float = 2.25      # (1.5 m/s)^2 Phase 8 baseline
    r_v_min_mps2: float = 1.0              # Floor: prevents neural overconfidence
    r_v_max_mps2: float = 25.0             # Ceiling: prevents degenerate singularity
    gate_threshold: float = 16.0           # Chi-square critical value for 1-DOF gate (p ~ 6e-5)
    use_causal_ema: bool = True
    ema_alpha: float = 0.2


@dataclass(frozen=True)
class VelocityNetAdapterDiagnostics:
    """Detailed diagnostics emitted from a VelocityNet measurement cycle."""
    applied: bool
    reason: Optional[str]
    z_speed: float
    predicted_speed: float
    variance: float
    update_diagnostics: Optional[UpdateDiagnostics] = None


class VelocityNetMeasurementModel:
    """Translates neural forward speed predictions into gated ESKF updates."""

    def __init__(self, config: Optional[VelocityNetConfig] = None) -> None:
        self.config = config or VelocityNetConfig()
        self.ema = CausalEMA(alpha=self.config.ema_alpha) if self.config.use_causal_ema else None
        self.gating = MahalanobisGating(threshold_override=self.config.gate_threshold)

    def reset(self) -> None:
        """Reset internal filter states at trajectory boundaries."""
        if self.ema is not None:
            self.ema.reset()

    def update(
        self,
        state: ESKFState,
        prediction: VelocityNetOutput,
        timestamp_ns: Optional[int] = None,
    ) -> Tuple[ESKFState, VelocityNetAdapterDiagnostics]:
        """Execute gated VelocityNet measurement update.

        Args:
            state: Authoritative ESKF state before update.
            prediction: Raw output from VelocityNet model runner.
            timestamp_ns: Update epoch timestamp.

        Returns:
            Tuple of (updated_or_unchanged_state, diagnostics).
        """
        if not self.config.enabled:
            return state, VelocityNetAdapterDiagnostics(
                applied=False,
                reason="MODEL_DISABLED",
                z_speed=0.0,
                predicted_speed=0.0,
                variance=0.0,
            )

        if not prediction.valid:
            return state, VelocityNetAdapterDiagnostics(
                applied=False,
                reason=f"INVALID_PREDICTION_{prediction.reason}",
                z_speed=prediction.speed_mps,
                predicted_speed=0.0,
                variance=prediction.variance,
            )

        # 1. Apply causal smoothing if enabled
        raw_speed = float(prediction.speed_mps)
        smoothed_speed = self.ema.update(raw_speed) if self.ema is not None else raw_speed

        # 2. Standstill motion gating: suppress near standstill to prevent fighting ZUPT
        if smoothed_speed < self.config.min_speed_threshold_mps:
            return state, VelocityNetAdapterDiagnostics(
                applied=False,
                reason="STANDSTILL_SUPPRESSED",
                z_speed=smoothed_speed,
                predicted_speed=0.0,
                variance=prediction.variance,
            )

        # 3. Attitude-dependent vehicle forward projection in ENU
        R_v_n = state.nominal.R_v_n
        fwd_n = R_v_n[:, 0]  # Vehicle FLU frame: +X is vehicle forward

        # 4. Measurement z and predicted measurement h(x)
        z = np.array([smoothed_speed], dtype=np.float64)
        h_val = np.array([float(np.dot(fwd_n, state.nominal.velocity_enu))], dtype=np.float64)

        # 5. 15-state error Jacobian H: selects velocity error along fwd_n
        H = np.zeros((1, 15), dtype=np.float64)
        H[0, 3:6] = fwd_n

        # 6. Covariance R
        if self.config.use_heteroscedastic:
            r_val = float(np.clip(prediction.variance, self.config.r_v_min_mps2, self.config.r_v_max_mps2))
        else:
            r_val = float(self.config.fixed_variance_mps2)
        R = np.array([[r_val]], dtype=np.float64)

        # 7. Authoritative generic ESKF update
        t = state.timestamp_ns if timestamp_ns is None else int(timestamp_ns)
        updated_state, update_diag = eskf_update(
            state=state,
            z=z,
            h_val=h_val,
            H=H,
            R=R,
            gating=self.gating,
            timestamp_ns=t,
        )

        applied = update_diag.applied
        reason = None if applied else ("GATE_REJECTED" if update_diag.gating and not update_diag.gating.accepted else "UPDATE_FAILED")

        return updated_state, VelocityNetAdapterDiagnostics(
            applied=applied,
            reason=reason,
            z_speed=smoothed_speed,
            predicted_speed=float(h_val[0]),
            variance=r_val,
            update_diagnostics=update_diag,
        )
