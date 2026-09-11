"""Dynamic Alignment and Observability Module.

Tracks the observability of the IMU mounting frame relative to the vehicle frame.
NHC updates require a highly accurate mounting alignment. Without sufficient lateral
excitation (e.g., straight highway driving), mounting yaw remains unobservable.
This module uses an Exponential Moving Average (EMA) variance filter to robustly
gate NHC updates when alignment is uncertain.
"""

from enum import Enum
import numpy as np

class AlignmentConfidence(str, Enum):
    """Observability state of the IMU mounting alignment."""
    UNKNOWN = "UNKNOWN"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    HIGH_CONFIDENCE = "HIGH_CONFIDENCE"


class DynamicAlignment:
    """Estimates mounting alignment observability using an EMA filter on lateral specific force.
    
    If lateral specific force variance is extremely low over a sustained period,
    mounting yaw is deemed unobservable.
    """
    
    def __init__(self, alpha: float = 0.05, variance_threshold: float = 0.15, min_samples: int = 100):
        """Initialize Dynamic Alignment tracker.
        
        Args:
            alpha: EMA smoothing factor.
            variance_threshold: Minimum EMA variance (m^2/s^4) to consider mounting observable.
            min_samples: Minimum samples required before transitioning out of UNKNOWN.
        """
        self.alpha = alpha
        self.variance_threshold = variance_threshold
        self.min_samples = min_samples
        
        self.samples_processed = 0
        self.ema_mean = 0.0
        self.ema_var = 0.0
        self.confidence = AlignmentConfidence.UNKNOWN

    def update(self, f_v: np.ndarray, gnss_speed: float) -> AlignmentConfidence:
        """Update observability given current IMU and GNSS data.
        
        Args:
            f_v: (3,) Vehicle-frame specific force [m/s^2].
            gnss_speed: Current GNSS speed [m/s].
            
        Returns:
            AlignmentConfidence: The current observability state.
        """
        self.samples_processed += 1
        lat_force = f_v[1]
        
        # Welford-like EMA update for variance
        diff = lat_force - self.ema_mean
        self.ema_mean += self.alpha * diff
        self.ema_var = (1 - self.alpha) * (self.ema_var + self.alpha * diff**2)
        
        if self.samples_processed < self.min_samples:
            self.confidence = AlignmentConfidence.UNKNOWN
            return self.confidence
            
        # Vehicle must be moving to observe mounting yaw reliably
        if gnss_speed < 1.0:
            return self.confidence  # Hold previous confidence if stationary
            
        if self.ema_var > self.variance_threshold:
            self.confidence = AlignmentConfidence.HIGH_CONFIDENCE
        elif self.ema_var > self.variance_threshold * 0.3:
            self.confidence = AlignmentConfidence.LOW_CONFIDENCE
        else:
            self.confidence = AlignmentConfidence.UNKNOWN
            
        return self.confidence
