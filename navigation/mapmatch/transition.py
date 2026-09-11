"""Transition probability model for HMM map matching."""

import math
from typing import Tuple
from navigation.mapmatch.candidates import RoadCandidate

LOG_ZERO = -1e9

class TransitionModel:
    """Calculates transition log-probabilities between road candidates."""
    
    def __init__(self, beta_m: float = 5.0):
        """
        Args:
            beta_m: Scale parameter for the exponential distribution of routing discrepancies.
        """
        self.beta_m = float(beta_m)
        self._log_norm = -math.log(self.beta_m)

    def get_route_distance(self, c_prev: RoadCandidate, c_curr: RoadCandidate) -> float:
        """
        Calculates the routing distance between two candidates.
        For Phase 12 Part 1 synthetic tests, we approximate this using Euclidean
        distance between the projected points. Part 2 will override this with
        true OSMNX graph shortest path routing.
        """
        if c_prev.edge_id == c_curr.edge_id:
            # Same edge: distance is absolute difference in linear projection
            return abs(c_curr.distance_along_edge_m - c_prev.distance_along_edge_m)
        
        # Approximate distance between edges
        dx = c_curr.projected_point_enu[0] - c_prev.projected_point_enu[0]
        dy = c_curr.projected_point_enu[1] - c_prev.projected_point_enu[1]
        return math.hypot(dx, dy)

    def compute_log_transition(
        self,
        c_prev: RoadCandidate,
        c_curr: RoadCandidate,
        prev_traj_enu: Tuple[float, float],
        curr_traj_enu: Tuple[float, float],
        delta_t_s: float,
    ) -> float:
        """
        Compute log transition probability from c_prev to c_curr.
        
        Args:
            c_prev: Previous road candidate.
            c_curr: Current road candidate.
            prev_traj_enu: (East, North) of vehicle at previous epoch.
            curr_traj_enu: (East, North) of vehicle at current epoch.
            delta_t_s: Time elapsed between epochs.
        """
        # Distance travelled by the vehicle
        dx_v = curr_traj_enu[0] - prev_traj_enu[0]
        dy_v = curr_traj_enu[1] - prev_traj_enu[1]
        dist_v = math.hypot(dx_v, dy_v)
        
        # Route distance on the road graph
        dist_r = self.get_route_distance(c_prev, c_curr)
        
        # Discrepancy between actual travel distance and network route distance
        discrepancy = abs(dist_v - dist_r)
        
        # Exponential distribution for routing probability
        # P = (1 / beta) * exp(-discrepancy / beta)
        log_p = - (discrepancy / self.beta_m) + self._log_norm
        
        # If the gap is huge relative to time, we can penalize further (e.g. impossible speed)
        speed = dist_r / max(delta_t_s, 1e-3)
        if speed > 50.0:  # > 180 km/h
            log_p += -50.0 # heavy penalty for impossible jumps
            
        return log_p
