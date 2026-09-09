"""Generic Gated Measurement Update for Error-State Kalman Filter (Phase 5).

Implements the central, generic measurement update function for the 15-state ESKF:
1. Computes innovation y = z - h(x) and innovation covariance S = H P H^T + R.
2. Applies Mahalanobis / Chi-Square gating. If rejected, state and covariance are unmodified.
3. Computes Kalman gain K = P H^T S^-1 using numerically stable linear solving (no matrix inversion).
4. Solves error-state correction delta_x = K y in R^15.
5. Employs the symmetric Joseph form for covariance correction:
       P = (I - K H) P (I - K H)^T + K R K^T
6. Injects delta_x into the nominal state and resets the error state to zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np

from navigation.eskf.gating import GatingDiagnostics, MahalanobisGating
from navigation.eskf.state import ESKFState, reset_covariance


@dataclass(frozen=True)
class UpdateDiagnostics:
    """Diagnostic details emitted from a measurement update.

    Attributes:
        applied: True if the measurement was accepted and applied; False if rejected.
        measurement_dim: Dimension m of the measurement vector.
        innovation: (m,) float64 innovation residual vector y = z - h(x).
        innovation_covariance: (m, m) float64 innovation covariance S = H P H^T + R.
        kalman_gain: (15, m) float64 Kalman gain matrix if applied, else None.
        delta_x: (15,) float64 error state correction if applied, else None.
        gating: GatingDiagnostics from gating check, if evaluated.
    """
    applied: bool
    measurement_dim: int
    innovation: np.ndarray
    innovation_covariance: np.ndarray
    kalman_gain: Optional[np.ndarray] = None
    delta_x: Optional[np.ndarray] = None
    gating: Optional[GatingDiagnostics] = None


def eskf_update(
    state: ESKFState,
    z: np.ndarray,
    h_val: np.ndarray,
    H: np.ndarray,
    R: np.ndarray,
    gating: Optional[MahalanobisGating] = None,
    timestamp_ns: Optional[int] = None,
) -> Tuple[ESKFState, UpdateDiagnostics]:
    """Execute generic gated ESKF measurement update.

    Args:
        state: Current joint ESKF state (nominal state + 15x15 covariance P).
        z: (m,) measurement vector.
        h_val: (m,) predicted measurement evaluated at current nominal state.
        H: (m, 15) measurement Jacobian matrix with respect to 15-element error state.
        R: (m, m) measurement noise covariance matrix.
        gating: Optional MahalanobisGating instance. If None, gating is bypassed.
        timestamp_ns: Optional timestamp to assign to the updated nominal state.

    Returns:
        Tuple[ESKFState, UpdateDiagnostics]: Updated state (or unchanged state if rejected)
        along with full diagnostic records.
    """
    z_arr = np.asarray(z, dtype=np.float64).reshape(-1)
    h_arr = np.asarray(h_val, dtype=np.float64).reshape(-1)
    m = len(z_arr)

    if len(h_arr) != m:
        raise ValueError(f"Measurement z ({m}) and prediction h_val ({len(h_arr)}) shape mismatch")

    H_mat = np.asarray(H, dtype=np.float64)
    if H_mat.shape != (m, 15):
        raise ValueError(f"Jacobian H must have shape ({m}, 15), got {H_mat.shape}")

    R_mat = np.asarray(R, dtype=np.float64)
    if R_mat.shape != (m, m):
        raise ValueError(f"Noise covariance R must have shape ({m}, {m}), got {R_mat.shape}")

    # Check finite values in inputs
    if not (np.isfinite(z_arr).all() and np.isfinite(h_arr).all() and
            np.isfinite(H_mat).all() and np.isfinite(R_mat).all()):
        # Corrupt measurement input: reject immediately
        diag = UpdateDiagnostics(
            applied=False,
            measurement_dim=m,
            innovation=np.full(m, np.nan, dtype=np.float64),
            innovation_covariance=np.full((m, m), np.nan, dtype=np.float64),
            gating=None,
        )
        return state, diag

    # Validate R structure: symmetry and strictly positive variances
    if not np.allclose(R_mat, R_mat.T, atol=1e-6) or (np.diag(R_mat) <= 0.0).any():
        diag = UpdateDiagnostics(
            applied=False,
            measurement_dim=m,
            innovation=z_arr - h_arr,
            innovation_covariance=np.full((m, m), np.nan, dtype=np.float64),
            gating=None,
        )
        return state, diag

    # 1. Innovation residual
    y = z_arr - h_arr

    # 2. Innovation covariance S = H P H^T + R
    P = state.covariance
    H_P = H_mat @ P
    S = H_P @ H_mat.T + R_mat
    # Ensure S is numerically symmetric
    S = 0.5 * (S + S.T)

    # 3. Innovation gating (Mahalanobis / Chi-Square)
    gating_diag: Optional[GatingDiagnostics] = None
    if gating is not None:
        gating_diag = gating.evaluate(y, S)
        if not gating_diag.accepted:
            # Gating rejected: State and covariance are strictly unmodified!
            diag = UpdateDiagnostics(
                applied=False,
                measurement_dim=m,
                innovation=y,
                innovation_covariance=S,
                kalman_gain=None,
                delta_x=None,
                gating=gating_diag,
            )
            return state, diag

    # 4. Numerically stable Kalman Gain K = P H^T S^-1
    # Solve S @ K^T = H @ P  -->  K = ( (H @ P)^T @ S^-1 )^T
    try:
        # np.linalg.solve(S, H_P) solves S @ X = H_P, so X = S^-1 @ H_P.
        # Then K = P @ H^T @ S^-1 = (H_P)^T @ S^-1 = X^T
        X = np.linalg.solve(S, H_P)
        K = X.T  # Shape (15, m)
    except np.linalg.LinAlgError:
        # Singular innovation covariance: abort update safely
        diag = UpdateDiagnostics(
            applied=False,
            measurement_dim=m,
            innovation=y,
            innovation_covariance=S,
            kalman_gain=None,
            delta_x=None,
            gating=gating_diag,
        )
        return state, diag

    # 5. Error state correction delta_x = K y
    delta_x = K @ y  # Shape (15,)

    # 6. Joseph form covariance update:
    # P_new = (I - K H) P (I - K H)^T + K R K^T
    I15 = np.eye(15, dtype=np.float64)
    I_KH = I15 - K @ H_mat
    P_updated = I_KH @ P @ I_KH.T + K @ R_mat @ K.T
    # Controlled numerical symmetrization
    P_updated = 0.5 * (P_updated + P_updated.T)

    # 7. Apply error-state covariance reset transformation for attitude error
    delta_theta = delta_x[6:9]
    P_reset = reset_covariance(P_updated, delta_theta)

    # 8. Inject error state into nominal state (and reset error state to 0)
    t = state.timestamp_ns if timestamp_ns is None else int(timestamp_ns)
    new_nominal = state.nominal.inject_error(delta_x, timestamp_ns=t)

    new_state = ESKFState(nominal=new_nominal, covariance=P_reset)


    diag = UpdateDiagnostics(
        applied=True,
        measurement_dim=m,
        innovation=y,
        innovation_covariance=S,
        kalman_gain=K,
        delta_x=delta_x,
        gating=gating_diag,
    )

    return new_state, diag
