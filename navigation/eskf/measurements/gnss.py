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
from navigation.eskf.gating import MahalanobisGating
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
