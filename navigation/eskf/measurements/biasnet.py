"""BiasNet Pseudo-Measurement ESKF Measurement Adapter (Phase 9).

Integrates learned IMU bias corrections as pseudo-measurements into the 15-state ESKF.

Semantic Safety Invariant:
BiasNet labels are optimization-derived pseudo-targets, NOT physical ground-truth.
BiasNet supplies a learned pseudo-measurement hypothesis about the ESKF bias state.
The ESKF determines state influence through R, innovation covariance S,
Mahalanobis gating, and Kalman gain K.

Mathematical Formulation:
- Nominal bias sub-state:
      b_nominal = [b_ax, b_ay, b_az, b_gx, b_gy, b_gz]^T in R^6
- BiasNet prediction:
      delta_b_pred = [delta_ba^T, delta_bg^T]^T in R^6
- Pseudo-measurement:
      z_b = b_nominal + delta_b_pred
- Predicted measurement:
      h_b(x) = b_nominal
- Innovation residual:
      y_b = z_b - h_b(x) = delta_b_pred
- 15D Error-State Jacobian H_b in R^{6x15}:
      Error state layout: [delta_p(3), delta_v(3), delta_theta(3), delta_ba(3), delta_bg(3)]
      H_b[0:3, 9:12]  = I_3  (maps to delta_ba in error state)
      H_b[3:6, 12:15] = I_3  (maps to delta_bg in error state)
      (Note: Nominal state has q at 6:10, so nominal ba is 10:13, but error state has delta_theta
       at 6:9, so error delta_ba is 9:12 and delta_bg is 12:15. This mapping is strictly enforced.)
- Measurement Covariance R_b:
      R_b = diag([1.21, 1.21, 1.21, 0.0025, 0.0025, 0.0025]) (Phase 8 frozen baseline)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np

from navigation.eskf.gating import MahalanobisGating
from navigation.eskf.state import ESKFState
from navigation.eskf.update import UpdateDiagnostics, eskf_update
from navigation.ml.model_runner import BiasNetOutput

# Authoritative Phase 8 frozen covariance
PHASE8_R_BIAS_DIAG: Tuple[float, ...] = (1.21, 1.21, 1.21, 0.0025, 0.0025, 0.0025)


@dataclass(frozen=True)
class BiasNetConfig:
    """Configuration for BiasNet measurement adapter."""
    enabled: bool = True
    covariance_mode: str = "fixed_phase8"  # "fixed_phase8" or "calibrated_phase9"
    fixed_r_diag: Tuple[float, ...] = PHASE8_R_BIAS_DIAG
    calibrated_r_diag: Optional[Tuple[float, ...]] = None
    gate_threshold: float = 25.0            # Chi-square critical value for 6-DOF gate (p ~ 3.5e-4)


@dataclass(frozen=True)
class BiasNetAdapterDiagnostics:
    """Detailed diagnostics emitted from a BiasNet measurement cycle."""
    applied: bool
    reason: Optional[str]
    delta_bias: np.ndarray
    covariance_diag: np.ndarray
    update_diagnostics: Optional[UpdateDiagnostics] = None


class BiasNetMeasurementModel:
    """Translates neural bias corrections into gated ESKF pseudo-measurement updates."""

    def __init__(self, config: Optional[BiasNetConfig] = None) -> None:
        self.config = config or BiasNetConfig()
        self.gating = MahalanobisGating(threshold_override=self.config.gate_threshold)
        
        # Initialize R matrix
        if self.config.covariance_mode == "calibrated_phase9" and self.config.calibrated_r_diag is not None:
            r_diag = np.asarray(self.config.calibrated_r_diag, dtype=np.float64)
        else:
            r_diag = np.asarray(self.config.fixed_r_diag, dtype=np.float64)

        if r_diag.shape != (6,):
            raise ValueError(f"R diagonal must have shape (6,), got {r_diag.shape}")
        if np.any(r_diag <= 0.0):
            raise ValueError(f"All R variances must be strictly positive, got {r_diag}")

        self.R_diag = r_diag
        self.R_mat = np.diag(r_diag)

    def update(
        self,
        state: ESKFState,
        prediction: BiasNetOutput,
        timestamp_ns: Optional[int] = None,
    ) -> Tuple[ESKFState, BiasNetAdapterDiagnostics]:
        """Execute gated BiasNet pseudo-measurement update.

        Args:
            state: Authoritative ESKF state before update.
            prediction: Raw output from BiasNet model runner.
            timestamp_ns: Update epoch timestamp.

        Returns:
            Tuple of (updated_or_unchanged_state, diagnostics).
        """
        zero_vec = np.zeros(6, dtype=np.float64)

        if not self.config.enabled:
            return state, BiasNetAdapterDiagnostics(
                applied=False,
                reason="MODEL_DISABLED",
                delta_bias=zero_vec,
                covariance_diag=self.R_diag,
            )

        if not prediction.valid:
            return state, BiasNetAdapterDiagnostics(
                applied=False,
                reason=f"INVALID_PREDICTION_{prediction.reason}",
                delta_bias=prediction.delta_bias_vector,
                covariance_diag=self.R_diag,
            )

        # 1. Extract current nominal bias components
        b_nom = np.concatenate([state.nominal.accel_bias, state.nominal.gyro_bias])  # Shape (6,)
        delta_b_pred = prediction.delta_bias_vector                                    # Shape (6,)

        # 2. Pseudo-measurement and prediction
        z = b_nom + delta_b_pred
        h_val = b_nom

        # 3. 15D Error-State Jacobian H
        # Layout: delta_p (0:3), delta_v (3:6), delta_theta (6:9), delta_ba (9:12), delta_bg (12:15)
        H = np.zeros((6, 15), dtype=np.float64)
        H[0:3, 9:12] = np.eye(3, dtype=np.float64)   # Maps accel bias error
        H[3:6, 12:15] = np.eye(3, dtype=np.float64)  # Maps gyro bias error

        # 4. Authoritative generic ESKF update
        t = state.timestamp_ns if timestamp_ns is None else int(timestamp_ns)
        updated_state, update_diag = eskf_update(
            state=state,
            z=z,
            h_val=h_val,
            H=H,
            R=self.R_mat,
            gating=self.gating,
            timestamp_ns=t,
        )

        applied = update_diag.applied
        reason = None if applied else ("GATE_REJECTED" if update_diag.gating and not update_diag.gating.accepted else "UPDATE_FAILED")

        return updated_state, BiasNetAdapterDiagnostics(
            applied=applied,
            reason=reason,
            delta_bias=delta_b_pred,
            covariance_diag=self.R_diag,
            update_diagnostics=update_diag,
        )
