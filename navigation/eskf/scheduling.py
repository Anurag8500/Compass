"""Time-Aware Cadence Scheduler for ML Measurement Updates (Phase 9).

Schedules VelocityNet (~2 Hz) and BiasNet (~1 Hz) measurement updates
based on actual elapsed timestamps rather than brittle loop-iteration counters.
Robust against realistic timestamp jitter, sampling gaps, and duplicates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class CadenceConfig:
    """Cadence configuration in seconds."""
    velocitynet_interval_s: float = 0.5   # ~2 Hz
    biasnet_interval_s: float = 1.0       # ~1 Hz
    min_inter_sample_s: float = 0.001     # Ignore samples arriving under 1 ms apart (duplicate/burst)


class MLCadenceScheduler:
    """Stateful scheduler tracking execution epochs for neural measurements."""

    def __init__(self, config: Optional[CadenceConfig] = None) -> None:
        self.config = config or CadenceConfig()
        self.last_vnet_time_ns: Optional[int] = None
        self.last_bnet_time_ns: Optional[int] = None
        self.last_sample_time_ns: Optional[int] = None

    def reset(self) -> None:
        """Reset internal timestamps at trajectory boundaries."""
        self.last_vnet_time_ns = None
        self.last_bnet_time_ns = None
        self.last_sample_time_ns = None

    def should_run_velocitynet(self, current_timestamp_ns: int) -> bool:
        """Evaluate whether VelocityNet update is due at current_timestamp_ns."""
        t_ns = int(current_timestamp_ns)
        if self.last_vnet_time_ns is None:
            return True
        elapsed_s = (t_ns - self.last_vnet_time_ns) * 1e-9
        return elapsed_s >= self.config.velocitynet_interval_s

    def should_run_biasnet(self, current_timestamp_ns: int) -> bool:
        """Evaluate whether BiasNet update is due at current_timestamp_ns."""
        t_ns = int(current_timestamp_ns)
        if self.last_bnet_time_ns is None:
            return True
        elapsed_s = (t_ns - self.last_bnet_time_ns) * 1e-9
        return elapsed_s >= self.config.biasnet_interval_s

    def record_velocitynet_execution(self, current_timestamp_ns: int) -> None:
        """Mark VelocityNet as having executed at current_timestamp_ns."""
        self.last_vnet_time_ns = int(current_timestamp_ns)

    def record_biasnet_execution(self, current_timestamp_ns: int) -> None:
        """Mark BiasNet as having executed at current_timestamp_ns."""
        self.last_bnet_time_ns = int(current_timestamp_ns)

    def evaluate_cycle(self, current_timestamp_ns: int) -> Tuple[bool, bool]:
        """Evaluate both models for the current timestamp.

        Args:
            current_timestamp_ns: Current epoch in nanoseconds.

        Returns:
            Tuple of (run_velocitynet: bool, run_biasnet: bool).
        """
        t_ns = int(current_timestamp_ns)
        if self.last_sample_time_ns is not None:
            dt_s = (t_ns - self.last_sample_time_ns) * 1e-9
            if dt_s < self.config.min_inter_sample_s:
                # Duplicate or degenerate timestamp: skip both ML evaluations
                return False, False

        self.last_sample_time_ns = t_ns

        run_vnet = self.should_run_velocitynet(t_ns)
        run_bnet = self.should_run_biasnet(t_ns)

        return run_vnet, run_bnet
