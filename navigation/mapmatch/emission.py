"""Covariance-aware log-Gaussian emission probability model for map matching (Phase 12).

Implements the principled road-normal projected covariance formulation:
    sigma_d^2 = n^T P_pp n + sigma_road^2
where:
    n: road-normal 2D unit vector [-ty, tx]
    P_pp: 2x2 horizontal position error covariance from ESKF
    sigma_road: baseline road/lane width uncertainty

Log-domain likelihoods are strictly deterministic and numerically stable.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple
import numpy as np

from navigation.mapmatch.candidates import RoadCandidate


class EmissionModel:
    """Covariance-aware log-Gaussian emission probability evaluator."""

    def __init__(
        self,
        sigma_road_m: float = 4.0,
        sigma_heading_rad: float = math.radians(25.0),
        min_speed_for_heading_mps: float = 1.5,
    ) -> None:
        """Initialize emission model.

        Args:
            sigma_road_m: Baseline road/lane width standard deviation in meters (default 4.0 m).
            sigma_heading_rad: Standard deviation for heading consistency in radians (default ~25 deg).
            min_speed_for_heading_mps: Minimum speed to activate heading consistency gating.
        """
        if sigma_road_m <= 0.0:
            raise ValueError(f"sigma_road_m must be positive, got {sigma_road_m}")
        if sigma_heading_rad <= 0.0:
            raise ValueError(f"sigma_heading_rad must be positive, got {sigma_heading_rad}")

        self.sigma_road_m = float(sigma_road_m)
        self.sigma_road_sq = self.sigma_road_m ** 2
        self.sigma_heading_rad = float(sigma_heading_rad)
        self.sigma_heading_sq = self.sigma_heading_rad ** 2
        self.min_speed_for_heading_mps = float(min_speed_for_heading_mps)

    def compute_distance_variance(
        self,
        candidate: RoadCandidate,
        cov_enu_2x2: Optional[np.ndarray] = None,
    ) -> float:
        """Compute projected variance along road normal: sigma_d^2 = n^T P_pp n + sigma_road^2."""
        n = np.array(candidate.edge_normal_enu, dtype=np.float64)

        var_pos = 0.0
        if cov_enu_2x2 is not None:
            P = np.asarray(cov_enu_2x2, dtype=np.float64)
            if P.shape == (2, 2):
                # Ensure variance is positive semidefinite
                proj_var = float(n @ P @ n)
                var_pos = max(0.0, proj_var)

        total_variance = var_pos + self.sigma_road_sq
        return total_variance

    def compute_log_emission(
        self,
        candidate: RoadCandidate,
        cov_enu_2x2: Optional[np.ndarray] = None,
        vehicle_heading_rad: Optional[float] = None,
        vehicle_speed_mps: Optional[float] = None,
    ) -> float:
        """Compute total log emission probability ln p(z_t | c_t) in nats.

        Combines:
        1. Log-Gaussian perpendicular distance likelihood using road-normal covariance.
        2. Heading consistency term if vehicle speed exceeds min threshold and heading is provided.
        """
        # 1. Perpendicular distance term
        d = candidate.distance_to_road_m
        var_d = self.compute_distance_variance(candidate, cov_enu_2x2)

        # ln N(d; 0, var_d) = -0.5 * ln(2*pi*var_d) - d^2 / (2*var_d)
        log_p_dist = -0.5 * math.log(2.0 * math.pi * var_d) - (d ** 2) / (2.0 * var_d)

        # 2. Heading consistency term (optional)
        log_p_heading = 0.0
        if (
            vehicle_heading_rad is not None
            and vehicle_speed_mps is not None
            and vehicle_speed_mps >= self.min_speed_for_heading_mps
        ):
            # Compute angular difference wrapped to [-pi, pi]
            diff = (vehicle_heading_rad - candidate.edge_azimuth_rad + math.pi) % (2.0 * math.pi) - math.pi
            log_p_heading = -0.5 * math.log(2.0 * math.pi * self.sigma_heading_sq) - (diff ** 2) / (2.0 * self.sigma_heading_sq)

        return log_p_dist + log_p_heading
