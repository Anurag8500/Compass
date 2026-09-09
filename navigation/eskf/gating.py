"""Generic Mahalanobis and Chi-Square Innovation Gating (Phase 5).

Protects the Error-State Kalman Filter from corrupt, outlier, or degraded measurements
by evaluating the normalized innovation squared (Mahalanobis distance squared):
    d^2 = y^T S^-1 y
against configurable chi-square confidence thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple
import numpy as np

# Standard Chi-Square critical value distribution table {DOF: {confidence: critical_value}}
CHI2_TABLE: Dict[int, Dict[float, float]] = {
    1: {0.90: 2.706, 0.95: 3.841, 0.99: 6.635, 0.999: 10.828},
    2: {0.90: 4.605, 0.95: 5.991, 0.99: 9.210, 0.999: 13.816},
    3: {0.90: 6.251, 0.95: 7.815, 0.99: 11.345, 0.999: 16.266},
    4: {0.90: 7.779, 0.95: 9.488, 0.99: 13.277, 0.999: 18.467},
    5: {0.90: 9.236, 0.95: 11.070, 0.99: 15.086, 0.999: 20.515},
    6: {0.90: 10.645, 0.95: 12.592, 0.99: 16.812, 0.999: 22.458},
}


@dataclass(frozen=True)
class GatingDiagnostics:
    """Detailed diagnostics emitted from innovation gating evaluation.

    Attributes:
        accepted: True if measurement passed the innovation gate; False if rejected.
        mahalanobis_sq: Normalized innovation squared d^2 = y^T S^-1 y.
        threshold: Critical gating threshold applied.
        dof: Measurement dimension / degrees of freedom.
        confidence_level: Nominal statistical confidence level (e.g. 0.99 for 99%).
    """
    accepted: bool
    mahalanobis_sq: float
    threshold: float
    dof: int
    confidence_level: float


class MahalanobisGating:
    """Configurable innovation consistency gate for generic measurement updates."""

    def __init__(
        self,
        confidence_level: float = 0.99,
        threshold_override: Optional[float] = None,
    ) -> None:
        """Initialize gating test.

        Args:
            confidence_level: Chi-square cumulative probability threshold (0.90, 0.95, 0.99, 0.999).
            threshold_override: Optional explicit numeric threshold to override chi2 table.
        """
        self.confidence_level = float(confidence_level)
        self.threshold_override = None if threshold_override is None else float(threshold_override)

    def get_threshold(self, dof: int) -> float:
        """Lookup or compute chi-square threshold for given degrees of freedom."""
        if self.threshold_override is not None:
            return self.threshold_override

        if dof in CHI2_TABLE:
            conf_dict = CHI2_TABLE[dof]
            # Match closest available confidence level
            closest_conf = min(conf_dict.keys(), key=lambda c: abs(c - self.confidence_level))
            return conf_dict[closest_conf]

        # Wilson-Hilferty approximation for arbitrary DOF if not in table
        z = 2.576 if self.confidence_level >= 0.99 else 1.960
        return dof * ((1.0 - 2.0 / (9.0 * dof) + z * math.sqrt(2.0 / (9.0 * dof))) ** 3)

    def evaluate(self, innovation: np.ndarray, S: np.ndarray) -> GatingDiagnostics:
        """Evaluate Mahalanobis distance squared d^2 = y^T S^-1 y.

        Numerically stable implementation using np.linalg.solve instead of matrix inversion.

        Args:
            innovation: (m,) float64 innovation residual y = z - h(x).
            S: (m, m) float64 innovation covariance S = H P H^T + R.

        Returns:
            GatingDiagnostics: Gating decision and metric record.
        """
        y = np.asarray(innovation, dtype=np.float64).reshape(-1)
        m = len(y)
        if m == 0:
            raise ValueError("Innovation vector cannot be empty")

        S_mat = np.asarray(S, dtype=np.float64)
        if S_mat.shape != (m, m):
            raise ValueError(f"Innovation covariance must have shape ({m}, {m}), got {S_mat.shape}")

        # Check finite values
        if not np.isfinite(y).all() or not np.isfinite(S_mat).all():
            return GatingDiagnostics(
                accepted=False,
                mahalanobis_sq=float("inf"),
                threshold=self.get_threshold(m),
                dof=m,
                confidence_level=self.confidence_level,
            )

        # Enforce numerical symmetry
        S_sym = 0.5 * (S_mat + S_mat.T)

        try:
            # Check for positive definiteness
            min_eig = float(np.min(np.linalg.eigvalsh(S_sym)))
            if min_eig <= 1e-12:
                return GatingDiagnostics(
                    accepted=False,
                    mahalanobis_sq=float("nan"),
                    threshold=self.get_threshold(m),
                    dof=m,
                    confidence_level=self.confidence_level,
                )

            # Solve S @ x = y for x = S^-1 @ y
            S_inv_y = np.linalg.solve(S_sym, y)
            d_sq = float(np.dot(y, S_inv_y))
        except (np.linalg.LinAlgError, ValueError):
            # Ill-conditioned or singular innovation covariance
            return GatingDiagnostics(
                accepted=False,
                mahalanobis_sq=float("nan"),
                threshold=self.get_threshold(m),
                dof=m,
                confidence_level=self.confidence_level,
            )

        threshold = self.get_threshold(m)
        accepted = bool(0.0 <= d_sq <= threshold and math.isfinite(d_sq))

        return GatingDiagnostics(
            accepted=accepted,
            mahalanobis_sq=d_sq,
            threshold=threshold,
            dof=m,
            confidence_level=self.confidence_level,
        )

