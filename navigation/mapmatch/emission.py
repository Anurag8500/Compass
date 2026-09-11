"""Emission probability model for HMM map matching."""

import math
import numpy as np
from typing import Optional
from navigation.mapmatch.candidates import RoadCandidate

class EmissionModel:
    """Calculates emission log-probabilities based on distance to candidate edges."""
    
    def __init__(self, sigma_road_m: float = 4.0):
        """
        Args:
            sigma_road_m: Expected standard deviation of position measurement relative to road.
        """
        self.sigma_road_m = float(sigma_road_m)
        # Normalization constant for 1D Gaussian
        self._log_norm = -0.5 * math.log(2.0 * math.pi * (self.sigma_road_m ** 2))

    def compute_log_emission(
        self,
        candidate: RoadCandidate,
        cov_enu_2x2: Optional[np.ndarray] = None,
        vehicle_heading_rad: Optional[float] = None,
        vehicle_speed_mps: Optional[float] = None,
    ) -> float:
        """
        Compute the log-likelihood of observing the vehicle at its current estimated
        position given that it is actually on the candidate road edge.
        
        Args:
            candidate: The road edge candidate.
            cov_enu_2x2: Optional 2x2 horizontal position covariance from ESKF.
            vehicle_heading_rad: Optional estimated vehicle heading.
            vehicle_speed_mps: Optional estimated horizontal speed.
            
        Returns:
            Log-probability.
        """
        # Base emission uses the orthogonal projection distance
        dist = candidate.distance_to_road_m
        
        # Simple isotropic Gaussian log-likelihood
        # log(P(z|r)) = -0.5 * (d / sigma)^2 + log_norm
        log_p = -0.5 * ((dist / self.sigma_road_m) ** 2) + self._log_norm
        
        # Penalize heading misalignment if moving fast enough
        if vehicle_speed_mps is not None and vehicle_speed_mps > 2.0 and vehicle_heading_rad is not None:
            # Shortest angular distance between vehicle heading and road heading
            diff = (vehicle_heading_rad - candidate.edge_azimuth_rad + math.pi) % (2.0 * math.pi) - math.pi
            # Road could be traversed in either direction; assume bidirectional if not specified
            # For simplicity, penalize misalignment from both directions
            diff_bidir = min(abs(diff), abs((diff + math.pi + math.pi) % (2.0 * math.pi) - math.pi))
            
            # Additional penalty: assume 30 degree (0.5 rad) standard deviation
            sigma_theta = 0.5
            log_p -= 0.5 * ((diff_bidir / sigma_theta) ** 2)
            
        return log_p
