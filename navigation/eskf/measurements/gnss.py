"""GNSS Measurement Model and Local ENU Adapter (Phase 5).

Converts geodetic GNSS fixes (WGS84 lat, lon, alt) and velocities into local
Cartesian ENU measurement vectors, constructs error-state observation Jacobians
(H_p, H_v, H_pv), derives noise covariance R from reported accuracy telemetry,
and applies generic ESKF updates with innovation gating.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple, Union
import numpy as np

from navigation.frames.local_geo import GeoReference
from navigation.eskf.gating import GatingDiagnostics, MahalanobisGating
from navigation.eskf.state import ESKFState
from navigation.eskf.update import UpdateDiagnostics, eskf_update


@dataclass(frozen=True)
class GNSSUpdateConfig:
    """Configuration and conservative noise floors for GNSS updates.

    Attributes:
        min_horizontal_accuracy_m: Lower floor on 1-sigma horizontal position noise [m].
        min_vertical_accuracy_m: Lower floor on 1-sigma vertical position noise [m].
        default_horizontal_accuracy_m: Fallback horizontal 1-sigma noise when missing [m].
        default_vertical_accuracy_m: Fallback vertical 1-sigma noise when missing [m].
        min_speed_accuracy_mps: Lower floor on 1-sigma velocity noise [m/s].
        default_speed_accuracy_mps: Fallback velocity 1-sigma noise when missing [m/s].
        min_trust_score: Trust score below which fixes are down-weighted or rejected.
    """
    min_horizontal_accuracy_m: float = 1.5
    min_vertical_accuracy_m: float = 3.0
    default_horizontal_accuracy_m: float = 4.0
    default_vertical_accuracy_m: float = 8.0
    min_speed_accuracy_mps: float = 0.2
    default_speed_accuracy_mps: float = 0.5
    min_trust_score: float = 0.1


@dataclass(frozen=True)
class PreUpdateInnovationDiagnostics:
    """Statistical evaluation of an incoming GNSS position fix prior to trust-covariance weighting.

    Attributes:
        innovation: (3,) raw innovation residual z_p - h(x) in local ENU frame.
        innovation_covariance_base: (3, 3) innovation covariance S_base = H P H^T + R_base.
        nis_base: Normalized Innovation Squared y^T S_base^-1 y under unscaled baseline noise R_base.
        is_valid: True if measurement values and covariance are finite and non-singular.
        gating_passed: True if baseline NIS passes the authoritative innovation gate.
        gating: Optional GatingDiagnostics record from evaluating baseline innovation.
    """
    innovation: np.ndarray
    innovation_covariance_base: np.ndarray
    nis_base: float
    is_valid: bool
    gating_passed: bool = True
    gating: Optional[GatingDiagnostics] = None


def course_to_horizontal_velocity(
    speed_mps: float,
    bearing_deg: float,
) -> Tuple[float, float]:
    """Convert horizontal ground speed and bearing into 2D horizontal ENU velocity.

    Navigation Convention:
        - Bearing psi is measured in degrees clockwise from True North:
          0 deg = North, 90 deg = East, 180 deg = South, 270 deg = West.
        - In local Cartesian ENU coordinates (East, North):
          v_east  = speed * sin(psi)
          v_north = speed * cos(psi)

    Args:
        speed_mps: Horizontal ground speed in meters per second.
        bearing_deg: Course over ground in degrees clockwise from North [0, 360).

    Returns:
        Tuple of (v_east, v_north) in meters per second.
    """
    if not (math.isfinite(speed_mps) and math.isfinite(bearing_deg)):
        return float("nan"), float("nan")
    if speed_mps < 0.0:
        return float("nan"), float("nan")

    psi_rad = math.radians(bearing_deg % 360.0)
    v_east = float(speed_mps * math.sin(psi_rad))
    v_north = float(speed_mps * math.cos(psi_rad))
    return v_east, v_north


def course_to_enu_velocity(
    speed_mps: float,
    bearing_deg: float,
    v_up_mps: float = 0.0,
) -> Tuple[float, float, float]:
    """Convert horizontal ground speed and bearing into local ENU velocity.

    Navigation Convention:
        - Bearing psi is measured in degrees clockwise from True North:
          0 deg = North, 90 deg = East, 180 deg = South, 270 deg = West.
        - In local Cartesian ENU coordinates (East, North, Up):
          v_east  = speed * sin(psi)
          v_north = speed * cos(psi)
          v_up    = v_up_mps

    Note on IO-VNBD Dataset (Phase 2 S-file):
        In raw Android logger S-files (e.g. S-S1.csv), the column header is
        'GPS SPEED (Kmh)', but Android's Location.getSpeed() API natively outputs
        values in meters per second (m/s). The Phase 2 ingestion code divided
        by 3.6 per the column label, storing (true_speed_mps / 3.6).
        To recover true physical velocity without modifying frozen Phase 2 caches,
        pass (trip.s_gnss_speed_mps * 3.6) to this function.

    Args:
        speed_mps: Horizontal ground speed in meters per second.
        bearing_deg: Course over ground in degrees clockwise from North [0, 360).
        v_up_mps: Vertical velocity in meters per second (defaults to 0.0).

    Returns:
        Tuple of (v_east, v_north, v_up) in meters per second.
    """
    ve, vn = course_to_horizontal_velocity(speed_mps, bearing_deg)
    if not (math.isfinite(ve) and math.isfinite(vn) and math.isfinite(v_up_mps)):
        return float("nan"), float("nan"), float("nan")
    return ve, vn, float(v_up_mps)


class GNSSMeasurementModel:
    """Adapter transforming GNSS fixes into generic ESKF measurement updates."""

    def __init__(
        self,
        geo_reference: GeoReference,
        config: Optional[GNSSUpdateConfig] = None,
        gating: Optional[MahalanobisGating] = None,
    ) -> None:
        """Initialize GNSS measurement model with rigid session origin.

        Args:
            geo_reference: Fixed session-level GeoReference for local ENU conversion.
            config: Optional noise parameter configuration.
            gating: Optional MahalanobisGating instance (defaults to 99% chi2 gate).
        """
        self.geo_ref = geo_reference
        self.config = GNSSUpdateConfig() if config is None else config
        self.gating = MahalanobisGating(confidence_level=0.99) if gating is None else gating

    def create_position_measurement(
        self,
        lat: float,
        lon: float,
        alt: float,
        accuracy_h_m: Optional[float] = None,
        accuracy_v_m: Optional[float] = None,
        trust_score: float = 1.0,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Construct local ENU position measurement z_p, Jacobian H_p, and noise R_p.

        Returns:
            Tuple of (z_p(3,), H_p(3, 15), R_p(3, 3)) or None if input is invalid.
        """
        if not (math.isfinite(lat) and math.isfinite(lon) and math.isfinite(alt)):
            return None
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            return None

        e, n, u = self.geo_ref.geodetic_to_enu(lat, lon, alt)
        z_p = np.array([float(e), float(n), float(u)], dtype=np.float64)

        # Build Jacobian H_p (observes delta_p directly)
        H_p = np.zeros((3, 15), dtype=np.float64)
        H_p[0:3, 0:3] = np.eye(3, dtype=np.float64)

        # Build noise covariance R_p
        sigma_h = self.config.default_horizontal_accuracy_m if accuracy_h_m is None or not math.isfinite(accuracy_h_m) or accuracy_h_m <= 0 else max(accuracy_h_m, self.config.min_horizontal_accuracy_m)
        sigma_v = self.config.default_vertical_accuracy_m if accuracy_v_m is None or not math.isfinite(accuracy_v_m) or accuracy_v_m <= 0 else max(accuracy_v_m, self.config.min_vertical_accuracy_m)

        # Scale by trust score (trust in (0, 1] inflates covariance as trust falls)
        trust = max(float(trust_score), self.config.min_trust_score) if math.isfinite(trust_score) else self.config.min_trust_score
        scale = 1.0 / trust

        R_p = np.diag([
            (sigma_h ** 2) * scale,
            (sigma_h ** 2) * scale,
            (sigma_v ** 2) * scale,
        ]).astype(np.float64)

        return z_p, H_p, R_p

    def create_velocity_measurement(
        self,
        v_east: float,
        v_north: float,
        v_up: float = 0.0,
        accuracy_speed_mps: Optional[float] = None,
        trust_score: float = 1.0,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Construct local ENU velocity measurement z_v, Jacobian H_v, and noise R_v.

        Returns:
            Tuple of (z_v(3,), H_v(3, 15), R_v(3, 3)) or None if input is invalid.
        """
        if not (math.isfinite(v_east) and math.isfinite(v_north) and math.isfinite(v_up)):
            return None

        z_v = np.array([float(v_east), float(v_north), float(v_up)], dtype=np.float64)

        # Build Jacobian H_v (observes delta_v directly)
        H_v = np.zeros((3, 15), dtype=np.float64)
        H_v[0:3, 3:6] = np.eye(3, dtype=np.float64)

        # Build noise covariance R_v
        sigma_v = self.config.default_speed_accuracy_mps if accuracy_speed_mps is None or not math.isfinite(accuracy_speed_mps) or accuracy_speed_mps <= 0 else max(accuracy_speed_mps, self.config.min_speed_accuracy_mps)

        trust = max(float(trust_score), self.config.min_trust_score) if math.isfinite(trust_score) else self.config.min_trust_score
        scale = 1.0 / trust

        R_v = np.diag([
            (sigma_v ** 2) * scale,
            (sigma_v ** 2) * scale,
            ((sigma_v * 1.5) ** 2) * scale,
        ]).astype(np.float64)

        return z_v, H_v, R_v

    def evaluate_pre_update_innovation(
        self,
        state: ESKFState,
        lat: float,
        lon: float,
        alt: float,
        accuracy_h_m: Optional[float] = None,
        accuracy_v_m: Optional[float] = None,
    ) -> Optional[PreUpdateInnovationDiagnostics]:
        """Compute pre-update innovation and baseline NIS for an incoming fix before trust weighting.

        Evaluates the discrepancy between the incoming raw GNSS position and the current predicted
        ESKF nominal state against the unscaled baseline measurement noise covariance R_base.
        This provides an uncorrupted, non-circular statistical quality measure to feed into
        trust score calculation before final covariance scaling and update gating.
        """
        meas = self.create_position_measurement(lat, lon, alt, accuracy_h_m, accuracy_v_m, trust_score=1.0)
        if meas is None:
            return None

        z_p, H_p, R_base = meas
        h_val = state.position_enu
        y = z_p - h_val

        P = state.covariance
        S_base = H_p @ P @ H_p.T + R_base
        S_base = 0.5 * (S_base + S_base.T)

        try:
            S_inv = np.linalg.inv(S_base)
            nis_base = float(y.T @ S_inv @ y)
            is_valid = math.isfinite(nis_base) and nis_base >= 0.0
        except np.linalg.LinAlgError:
            nis_base = float("inf")
            is_valid = False

        gating_diag = self.gating.evaluate(y, S_base) if self.gating is not None else None
        gating_passed = gating_diag.accepted if gating_diag is not None else True

        return PreUpdateInnovationDiagnostics(
            innovation=y,
            innovation_covariance_base=S_base,
            nis_base=nis_base if is_valid else float("inf"),
            is_valid=is_valid,
            gating_passed=gating_passed,
            gating=gating_diag,
        )

    def update_position(
        self,
        state: ESKFState,
        lat: float,
        lon: float,
        alt: float,
        accuracy_h_m: Optional[float] = None,
        accuracy_v_m: Optional[float] = None,
        trust_score: float = 1.0,
        timestamp_ns: Optional[int] = None,
    ) -> Tuple[ESKFState, UpdateDiagnostics]:
        """Apply 3D position GNSS fix update to ESKF state."""
        # Authoritative Phase 5 innovation gating check against baseline covariance:
        # Prevents low trust scores (which inflate R) from admitting otherwise invalid outlier fixes.
        pre_diag = self.evaluate_pre_update_innovation(
            state=state,
            lat=lat,
            lon=lon,
            alt=alt,
            accuracy_h_m=accuracy_h_m,
            accuracy_v_m=accuracy_v_m,
        )
        if pre_diag is not None and not pre_diag.gating_passed:
            diag = UpdateDiagnostics(
                applied=False,
                measurement_dim=3,
                innovation=pre_diag.innovation,
                innovation_covariance=pre_diag.innovation_covariance_base,
                gating=pre_diag.gating,
            )
            return state, diag
        meas = self.create_position_measurement(lat, lon, alt, accuracy_h_m, accuracy_v_m, trust_score)
        if meas is None:
            # Corrupted / invalid input: emit rejected diagnostic
            diag = UpdateDiagnostics(
                applied=False,
                measurement_dim=3,
                innovation=np.full(3, np.nan, dtype=np.float64),
                innovation_covariance=np.full((3, 3), np.nan, dtype=np.float64),
            )
            return state, diag

        z_p, H_p, R_p = meas
        h_val = state.position_enu
        return eskf_update(
            state=state,
            z=z_p,
            h_val=h_val,
            H=H_p,
            R=R_p,
            gating=self.gating,
            timestamp_ns=timestamp_ns,
        )

    def update_velocity(
        self,
        state: ESKFState,
        v_east: float,
        v_north: float,
        v_up: float = 0.0,
        accuracy_speed_mps: Optional[float] = None,
        trust_score: float = 1.0,
        timestamp_ns: Optional[int] = None,
    ) -> Tuple[ESKFState, UpdateDiagnostics]:
        """Apply 3D velocity GNSS fix update to ESKF state."""
        meas = self.create_velocity_measurement(v_east, v_north, v_up, accuracy_speed_mps, trust_score)
        if meas is None:
            diag = UpdateDiagnostics(
                applied=False,
                measurement_dim=3,
                innovation=np.full(3, np.nan, dtype=np.float64),
                innovation_covariance=np.full((3, 3), np.nan, dtype=np.float64),
            )
            return state, diag

        z_v, H_v, R_v = meas
        h_val = state.velocity_enu
        return eskf_update(
            state=state,
            z=z_v,
            h_val=h_val,
            H=H_v,
            R=R_v,
            gating=self.gating,
            timestamp_ns=timestamp_ns,
        )

    def create_horizontal_velocity_measurement(
        self,
        speed_mps: float,
        bearing_deg: float,
        accuracy_speed_mps: Optional[float] = None,
        trust_score: float = 1.0,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Construct 2D horizontal ENU velocity measurement z_v, Jacobian H_v, and noise R_v.

        This models course-derived velocity (speed over ground + bearing). It observes only
        horizontal velocity components (v_East, v_North) and strictly avoids injecting any
        artificial or unmeasured vertical velocity.

        Returns:
            Tuple of (z_v(2,), H_v(2, 15), R_v(2, 2)) or None if input is invalid.
        """
        ve, vn = course_to_horizontal_velocity(speed_mps, bearing_deg)
        if not (math.isfinite(ve) and math.isfinite(vn)):
            return None

        z_v = np.array([float(ve), float(vn)], dtype=np.float64)

        # Build Jacobian H_v (observes delta_v_East and delta_v_North directly)
        # Note: delta_v_Up (index 5) is completely unobserved!
        H_v = np.zeros((2, 15), dtype=np.float64)
        H_v[0, 3] = 1.0  # delta_v_East
        H_v[1, 4] = 1.0  # delta_v_North

        # Build noise covariance R_v
        sigma_v = (
            self.config.default_speed_accuracy_mps
            if accuracy_speed_mps is None or not math.isfinite(accuracy_speed_mps) or accuracy_speed_mps <= 0
            else max(accuracy_speed_mps, self.config.min_speed_accuracy_mps)
        )

        trust = max(float(trust_score), self.config.min_trust_score) if math.isfinite(trust_score) else self.config.min_trust_score
        scale = 1.0 / trust

        R_v = np.diag([
            (sigma_v ** 2) * scale,
            (sigma_v ** 2) * scale,
        ]).astype(np.float64)

        return z_v, H_v, R_v

    def update_horizontal_velocity_from_course(
        self,
        state: ESKFState,
        speed_mps: float,
        bearing_deg: float,
        accuracy_speed_mps: Optional[float] = None,
        trust_score: float = 1.0,
        timestamp_ns: Optional[int] = None,
    ) -> Tuple[ESKFState, UpdateDiagnostics]:
        """Apply 2D horizontal velocity update computed from course (speed + bearing) to ESKF state.

        Observes only [v_East, v_North] and leaves vertical velocity unconstrained.
        """
        meas = self.create_horizontal_velocity_measurement(speed_mps, bearing_deg, accuracy_speed_mps, trust_score)
        if meas is None:
            diag = UpdateDiagnostics(
                applied=False,
                measurement_dim=2,
                innovation=np.full(2, np.nan, dtype=np.float64),
                innovation_covariance=np.full((2, 2), np.nan, dtype=np.float64),
            )
            return state, diag

        z_v, H_v, R_v = meas
        h_val = state.velocity_enu[0:2]
        return eskf_update(
            state=state,
            z=z_v,
            h_val=h_val,
            H=H_v,
            R=R_v,
            gating=self.gating,
            timestamp_ns=timestamp_ns,
        )

    def update_velocity_from_course(
        self,
        state: ESKFState,
        speed_mps: float,
        bearing_deg: float,
        v_up: Optional[float] = None,
        accuracy_speed_mps: Optional[float] = None,
        trust_score: float = 1.0,
        timestamp_ns: Optional[int] = None,
    ) -> Tuple[ESKFState, UpdateDiagnostics]:
        """Apply course-derived velocity update.

        If v_up is None, performs the physically accurate 2D horizontal velocity update.
        If v_up is provided, performs a 3D velocity update with the specified vertical velocity.
        """
        if v_up is None:
            return self.update_horizontal_velocity_from_course(
                state=state,
                speed_mps=speed_mps,
                bearing_deg=bearing_deg,
                accuracy_speed_mps=accuracy_speed_mps,
                trust_score=trust_score,
                timestamp_ns=timestamp_ns,
            )

        ve, vn, vu = course_to_enu_velocity(speed_mps, bearing_deg, v_up)
        if not (math.isfinite(ve) and math.isfinite(vn) and math.isfinite(vu)):
            diag = UpdateDiagnostics(
                applied=False,
                measurement_dim=3,
                innovation=np.full(3, np.nan, dtype=np.float64),
                innovation_covariance=np.full((3, 3), np.nan, dtype=np.float64),
            )
            return state, diag

        return self.update_velocity(
            state=state,
            v_east=ve,
            v_north=vn,
            v_up=vu,
            accuracy_speed_mps=accuracy_speed_mps,
            trust_score=trust_score,
            timestamp_ns=timestamp_ns,
        )

    def update_6d(
        self,
        state: ESKFState,
        lat: float,
        lon: float,
        alt: float,
        v_east: float,
        v_north: float,
        v_up: float = 0.0,
        accuracy_h_m: Optional[float] = None,
        accuracy_v_m: Optional[float] = None,
        accuracy_speed_mps: Optional[float] = None,
        trust_score: float = 1.0,
        timestamp_ns: Optional[int] = None,
    ) -> Tuple[ESKFState, UpdateDiagnostics]:
        """Apply joint 6D position and velocity GNSS update to ESKF state."""
        p_meas = self.create_position_measurement(lat, lon, alt, accuracy_h_m, accuracy_v_m, trust_score)
        v_meas = self.create_velocity_measurement(v_east, v_north, v_up, accuracy_speed_mps, trust_score)

        if p_meas is None or v_meas is None:
            diag = UpdateDiagnostics(
                applied=False,
                measurement_dim=6,
                innovation=np.full(6, np.nan, dtype=np.float64),
                innovation_covariance=np.full((6, 6), np.nan, dtype=np.float64),
            )
            return state, diag

        z_p, H_p, R_p = p_meas
        z_v, H_v, R_v = v_meas

        z_6d = np.concatenate([z_p, z_v])
        h_6d = np.concatenate([state.position_enu, state.velocity_enu])

        H_6d = np.zeros((6, 15), dtype=np.float64)
        H_6d[0:3, 0:3] = np.eye(3, dtype=np.float64)
        H_6d[3:6, 3:6] = np.eye(3, dtype=np.float64)

        R_6d = np.zeros((6, 6), dtype=np.float64)
        R_6d[0:3, 0:3] = R_p
        R_6d[3:6, 3:6] = R_v

        return eskf_update(
            state=state,
            z=z_6d,
            h_val=h_6d,
            H=H_6d,
            R=R_6d,
            gating=self.gating,
            timestamp_ns=timestamp_ns,
        )
