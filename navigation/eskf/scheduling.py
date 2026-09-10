"""Time-Aware Cadence Scheduler for ML Measurement Updates (Phase 9).

Schedules VelocityNet (~2 Hz, 0.5s interval) and BiasNet (~1 Hz, 1.0s interval)
measurement updates based on actual elapsed timestamps.

Key Architectural Invariants:
1. Strict Cadence Separation: Separates scheduler due-time from buffer availability.
2. No Warm-up Looping: If a model becomes due before the 20-sample causal history is ready,
   the scheduler advances its schedule rather than attempting fake executions on every
   subsequent 10 Hz IMU sample.
3. Jitter & Gap Resilient: Adapts to real-time timestamp jitter and rejects duplicate/burst samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class CadenceConfig:
    """Cadence configuration in seconds."""
    velocitynet_interval_s: float = 0.5   # ~2 Hz target
    biasnet_interval_s: float = 1.0       # ~1 Hz target
    min_inter_sample_s: float = 0.001     # Ignore samples arriving < 1 ms apart (duplicate/burst)


class MLCadenceScheduler:
    """Stateful scheduler tracking execution epochs and due times for neural measurements."""

    def __init__(self, config: Optional[CadenceConfig] = None) -> None:
        self.config = config or CadenceConfig()
        self.last_vnet_scheduled_ns: Optional[int] = None
        self.last_bnet_scheduled_ns: Optional[int] = None
        self.last_vnet_executed_ns: Optional[int] = None
        self.last_bnet_executed_ns: Optional[int] = None
        self.last_sample_time_ns: Optional[int] = None

    def reset(self) -> None:
        """Reset internal schedule at trajectory boundaries."""
        self.last_vnet_scheduled_ns = None
        self.last_bnet_scheduled_ns = None
        self.last_vnet_executed_ns = None
        self.last_bnet_executed_ns = None
        self.last_sample_time_ns = None

    def is_velocitynet_due(self, current_timestamp_ns: int) -> bool:
        """Check if VelocityNet has reached its scheduled execution epoch."""
        t_ns = int(current_timestamp_ns)
        if self.last_vnet_scheduled_ns is None:
            return True
        elapsed_s = (t_ns - self.last_vnet_scheduled_ns) * 1e-9
        return elapsed_s >= self.config.velocitynet_interval_s

    def is_biasnet_due(self, current_timestamp_ns: int) -> bool:
        """Check if BiasNet has reached its scheduled execution epoch."""
        t_ns = int(current_timestamp_ns)
        if self.last_bnet_scheduled_ns is None:
            return True
        elapsed_s = (t_ns - self.last_bnet_scheduled_ns) * 1e-9
        return elapsed_s >= self.config.biasnet_interval_s

    def mark_velocitynet_scheduled(self, current_timestamp_ns: int) -> None:
        """Record that VelocityNet was evaluated (or considered) at this epoch."""
        self.last_vnet_scheduled_ns = int(current_timestamp_ns)

    def mark_biasnet_scheduled(self, current_timestamp_ns: int) -> None:
        """Record that BiasNet was evaluated (or considered) at this epoch."""
        self.last_bnet_scheduled_ns = int(current_timestamp_ns)

    def mark_velocitynet_executed(self, current_timestamp_ns: int) -> None:
        """Record a successful model execution."""
        t_ns = int(current_timestamp_ns)
        self.last_vnet_executed_ns = t_ns
        self.last_vnet_scheduled_ns = t_ns

    def record_velocitynet_execution(self, current_timestamp_ns: int) -> None:
        """Backward-compatible alias for mark_velocitynet_executed."""
        self.mark_velocitynet_executed(current_timestamp_ns)

    def mark_biasnet_executed(self, current_timestamp_ns: int) -> None:
        """Record a successful model execution."""
        t_ns = int(current_timestamp_ns)
        self.last_bnet_executed_ns = t_ns
        self.last_bnet_scheduled_ns = t_ns

    def record_biasnet_execution(self, current_timestamp_ns: int) -> None:
        """Backward-compatible alias for mark_biasnet_executed."""
        self.mark_biasnet_executed(current_timestamp_ns)

    def evaluate_cycle(self, current_timestamp_ns: int) -> Tuple[bool, bool]:
        """Evaluate cadence eligibility for both models at the current timestamp.

        Args:
            current_timestamp_ns: Current sample timestamp in nanoseconds.

        Returns:
            Tuple of (vnet_due: bool, bnet_due: bool).
        """
        t_ns = int(current_timestamp_ns)
        if self.last_sample_time_ns is not None:
            dt_s = (t_ns - self.last_sample_time_ns) * 1e-9
            if dt_s < self.config.min_inter_sample_s:
                # Duplicate or degenerate timestamp
                return False, False

        self.last_sample_time_ns = t_ns

        vnet_due = self.is_velocitynet_due(t_ns)
        bnet_due = self.is_biasnet_due(t_ns)

        return vnet_due, bnet_due
