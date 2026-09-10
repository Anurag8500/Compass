"""Continuous GNSS Quality and Trust Score Computation for COMPASS (Phase 10).

In accordance with Master Plan Section 17:
- GNSS quality state is continuous: trust ∈ [0.0, 1.0].
- "Degraded GNSS" is NOT a discrete FSM state; it lives inside GNSS_AIDED as a
  falling continuous trust score that down-weights (inflates measurement covariance R)
  rather than immediately rejecting updates.
- Evaluates reported accuracy, satellite count (when available), fix-to-fix
  kinematic plausibility, and ESKF innovation statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import List, Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class TrustScoreConfig:
    """Configuration and weighting parameters for continuous GNSS trust evaluation.

    Attributes:
        min_accuracy_m: Lower bound on horizontal accuracy (accuracy <= min yields 1.0).
        max_accuracy_m: Upper bound on horizontal accuracy (accuracy >= max yields 0.0).
        min_sat_count: Minimum satellite count for fix validity (sats <= min yields 0.0).
        nominal_sat_count: Satellite count for full confidence (sats >= nominal yields 1.0).
        max_apparent_speed_mps: Maximum plausible inter-fix vehicle speed [m/s] (~180 km/h).
        speed_excess_range_mps: Range above max speed over which plausibility decays to 0.
        nis_threshold_95: 95% chi-square threshold for 3D position innovations (chi2_3(0.95) = 7.815).
        min_trust_score: Lower bound on trust score to prevent singular covariance (default 0.05).
        weight_accuracy: Weight for reported accuracy component.
        weight_sat_count: Weight for satellite count component.
        weight_plausibility: Weight for fix-to-fix kinematic plausibility component.
        weight_innovation: Weight for filter innovation consistency component.
    """
    min_accuracy_m: float = 2.0
    max_accuracy_m: float = 25.0
    min_sat_count: int = 4
    nominal_sat_count: int = 10
    max_apparent_speed_mps: float = 50.0
    speed_excess_range_mps: float = 30.0
    nis_threshold_95: float = 7.815
    min_trust_score: float = 0.05
    weight_accuracy: float = 0.35
    weight_sat_count: float = 0.15
    weight_plausibility: float = 0.25
    weight_innovation: float = 0.25


@dataclass(frozen=True)
class GNSSQualityResult:
    """Detailed diagnostic breakdown of the continuous GNSS trust calculation.

    Attributes:
        trust_score: Final composite trust score in [0.0, 1.0].
        accuracy_component: Component score derived from horizontal accuracy [0.0, 1.0].
        satellite_component: Component score derived from satellite count, or None if unavailable.
        plausibility_component: Component score derived from kinematic displacement [0.0, 1.0].
        innovation_component: Component score derived from recent ESKF innovation [0.0, 1.0].
        available_evidence: List of evidence sources used in this evaluation.
        reason_codes: Explanatory diagnostic strings explaining quality reductions.
    """
    trust_score: float
    accuracy_component: float
    satellite_component: Optional[float]
    plausibility_component: float
    innovation_component: float
    available_evidence: List[str] = field(default_factory=list)
    reason_codes: List[str] = field(default_factory=list)


def scale_gnss_covariance(
    R_base: np.ndarray,
    trust_score: float,
    min_trust: float = 0.05,
) -> np.ndarray:
    """Scale baseline GNSS measurement covariance continuously by the trust score.

    Implements:
        R_effective = (1 / max(trust_score, min_trust)) * R_base

    Args:
        R_base: (M, M) baseline measurement covariance matrix (finite, positive definite).
        trust_score: Continuous signal trust score in [0.0, 1.0].
        min_trust: Safety lower bound on trust to cap covariance inflation factor.

    Returns:
        (M, M) scaled measurement covariance matrix.
    """
    t = float(trust_score) if math.isfinite(trust_score) else min_trust
    t = max(min_trust, min(1.0, t))
    scale = 1.0 / t
    return np.asarray(R_base, dtype=np.float64) * scale


class GNSSTrustScoreCalculator:
    """Deterministic, rule-based calculator for continuous GNSS signal trust."""

    def __init__(self, config: Optional[TrustScoreConfig] = None) -> None:
        self.config = config or TrustScoreConfig()
        self._last_pos_enu: Optional[np.ndarray] = None
        self._last_timestamp_ns: Optional[int] = None
        self._recent_nis: Optional[float] = None
        self._last_evaluated_nis: Optional[float] = None

    def reset(self) -> None:
        """Reset historical tracking states."""
        self._last_pos_enu = None
        self._last_timestamp_ns = None
        self._recent_nis = None
        self._last_evaluated_nis = None

    def update_innovation_nis(self, nis: Optional[float]) -> None:
        """Record the latest ESKF GNSS update Normalized Innovation Squared (NIS)."""
        if nis is not None and math.isfinite(nis) and nis >= 0.0:
            self._recent_nis = float(nis)
        else:
            self._recent_nis = None

    @property
    def last_evaluated_nis(self) -> Optional[float]:
        """Return the NIS value evaluated in the most recent compute_trust call."""
        return self._last_evaluated_nis

    def compute_trust(
        self,
        accuracy_m: Optional[float],
        sat_count: Optional[int],
        current_pos_enu: Optional[np.ndarray],
        timestamp_ns: int,
        nis: Optional[float] = None,
    ) -> GNSSQualityResult:
        """Compute the composite continuous trust score and return diagnostic breakdown.

        Args:
            accuracy_m: Estimated 1-sigma horizontal accuracy in meters (None if missing).
            sat_count: Number of satellites used in fix, or negative/None if unavailable.
            current_pos_enu: Current GNSS position in local ENU frame [e, n, u] (None if missing).
            timestamp_ns: Timestamp of the current GNSS fix in nanoseconds.
            nis: Optional latest 3D position Normalized Innovation Squared (NIS). If None,
                uses the stored recent NIS.

        Returns:
            GNSSQualityResult containing bounded composite trust score and component diagnostics.
        """
        available_evidence: List[str] = []
        reason_codes: List[str] = []

        # 1. Accuracy component
        acc_score, acc_reason = self._compute_accuracy_component(accuracy_m)
        if acc_reason:
            reason_codes.append(acc_reason)
        if accuracy_m is not None and math.isfinite(accuracy_m) and accuracy_m > 0:
            available_evidence.append("ACCURACY")

        # 2. Satellite count component
        sat_score, sat_reason = self._compute_satellite_component(sat_count)
        if sat_reason:
            reason_codes.append(sat_reason)
        if sat_score is not None:
            available_evidence.append("SATELLITE_COUNT")

        # 3. Kinematic plausibility component
        plaus_score, plaus_reason = self._compute_plausibility_component(current_pos_enu, timestamp_ns)
        if plaus_reason:
            reason_codes.append(plaus_reason)
        if current_pos_enu is not None:
            available_evidence.append("KINEMATIC_PLAUSIBILITY")

        # 4. Innovation component
        eval_nis = nis if nis is not None else self._recent_nis
        self._last_evaluated_nis = eval_nis
        innov_score, innov_reason = self._compute_innovation_component(eval_nis)
        if innov_reason:
            reason_codes.append(innov_reason)
        if eval_nis is not None:
            available_evidence.append("ESKF_INNOVATION")

        # Dynamic weight redistribution across available evidence
        weights = {
            "accuracy": self.config.weight_accuracy,
            "sat_count": self.config.weight_sat_count if sat_score is not None else 0.0,
            "plausibility": self.config.weight_plausibility,
            "innovation": self.config.weight_innovation,
        }

        total_weight = sum(weights.values())
        if total_weight <= 0.0:
            composite = 1.0
        else:
            w_acc = weights["accuracy"] / total_weight
            w_sat = weights["sat_count"] / total_weight
            w_plaus = weights["plausibility"] / total_weight
            w_innov = weights["innovation"] / total_weight

            effective_sat = sat_score if sat_score is not None else 1.0
            composite = (
                w_acc * acc_score
                + w_sat * effective_sat
                + w_plaus * plaus_score
                + w_innov * innov_score
            )

        # Ensure finite and strict [0.0, 1.0] bounding
        if not math.isfinite(composite):
            composite = self.config.min_trust_score
            reason_codes.append("NON_FINITE_COMPOSITE_SCORE")
        else:
            composite = float(np.clip(composite, 0.0, 1.0))

        # Update historical state for next fix
        if current_pos_enu is not None and math.isfinite(timestamp_ns):
            self._last_pos_enu = np.asarray(current_pos_enu, dtype=np.float64).copy()
            self._last_timestamp_ns = int(timestamp_ns)

        return GNSSQualityResult(
            trust_score=composite,
            accuracy_component=acc_score,
            satellite_component=sat_score,
            plausibility_component=plaus_score,
            innovation_component=innov_score,
            available_evidence=available_evidence,
            reason_codes=reason_codes,
        )

    def _compute_accuracy_component(self, accuracy_m: Optional[float]) -> Tuple[float, Optional[str]]:
        if accuracy_m is None or not math.isfinite(accuracy_m) or accuracy_m <= 0.0:
            # Neutral default if accuracy not reported
            return 0.7, "ACCURACY_UNAVAILABLE"

        acc = float(accuracy_m)
        if acc <= self.config.min_accuracy_m:
            return 1.0, None
        if acc >= self.config.max_accuracy_m:
            return 0.0, f"ACCURACY_POOR_{acc:.1f}m"

        # Smooth linear transition from 1.0 at min to 0.0 at max
        score = 1.0 - (acc - self.config.min_accuracy_m) / (self.config.max_accuracy_m - self.config.min_accuracy_m)
        reason = f"ACCURACY_DEGRADED_{acc:.1f}m" if score < 0.7 else None
        return float(np.clip(score, 0.0, 1.0)), reason

    def _compute_satellite_component(self, sat_count: Optional[int]) -> Tuple[Optional[float], Optional[str]]:
        if sat_count is None or sat_count < 0:
            # Dataset does not report satellite count (e.g. IO-VNBD Android S-file where sat_count=-1)
            return None, "SAT_COUNT_UNAVAILABLE"

        sats = int(sat_count)
        if sats >= self.config.nominal_sat_count:
            return 1.0, None
        if sats <= self.config.min_sat_count:
            return 0.0, f"SAT_COUNT_CRITICAL_{sats}"

        score = (sats - self.config.min_sat_count) / (self.config.nominal_sat_count - self.config.min_sat_count)
        return float(np.clip(score, 0.0, 1.0)), f"SAT_COUNT_LOW_{sats}"

    def _compute_plausibility_component(
        self,
        current_pos_enu: Optional[np.ndarray],
        timestamp_ns: int,
    ) -> Tuple[float, Optional[str]]:
        if current_pos_enu is None or not np.all(np.isfinite(current_pos_enu)):
            return 0.0, "POSITION_NON_FINITE"

        if self._last_pos_enu is None or self._last_timestamp_ns is None:
            # First fix is assumed plausible
            return 1.0, None

        dt_s = (timestamp_ns - self._last_timestamp_ns) * 1e-9
        if dt_s <= 0.0:
            # Duplicate or backwards timestamp
            return 0.5, "TIME_INTERVAL_NON_POSITIVE"

        displacement = float(np.linalg.norm(current_pos_enu[:2] - self._last_pos_enu[:2]))
        apparent_speed = displacement / dt_s

        if apparent_speed <= self.config.max_apparent_speed_mps:
            return 1.0, None

        excess = apparent_speed - self.config.max_apparent_speed_mps
        if excess >= self.config.speed_excess_range_mps:
            return 0.0, f"IMPLAUSIBLE_JUMP_{apparent_speed:.1f}mps"

        score = 1.0 - (excess / self.config.speed_excess_range_mps)
        return float(np.clip(score, 0.0, 1.0)), f"EXCESS_SPEED_{apparent_speed:.1f}mps"

    def _compute_innovation_component(self, nis: Optional[float]) -> Tuple[float, Optional[str]]:
        if nis is None or not math.isfinite(nis):
            return 1.0, None  # No prior filter update to evaluate against

        nis_val = max(0.0, float(nis))
        gate = self.config.nis_threshold_95
        if nis_val <= gate:
            return 1.0, None

        # Exponential decay past the 95% chi-square threshold
        excess = nis_val - gate
        score = math.exp(-0.5 * (excess / gate))
        reason = f"HIGH_INNOVATION_NIS_{nis_val:.1f}" if score < 0.6 else None
        return float(np.clip(score, 0.0, 1.0)), reason
