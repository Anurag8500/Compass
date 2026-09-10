"""Time-Aware Cadence Scheduler for ML Measurement Updates (Phase 9).

Schedules VelocityNet (~2 Hz, 0.5s interval) and BiasNet (~1 Hz, 1.0s interval)
measurement updates based on an anchored timeline with jitter resilience.

Key Architectural Invariants:
1. Strict Cadence Separation: Separates scheduler due-time from buffer availability.
2. Timeline-Anchored: Cadence epochs are anchored to a fixed timeline rather than
   drifting by resetting due time to arbitrary jittered execution timestamps.
3. Jitter-Resilient: Adapts to real-time timestamp jitter (default 20ms tolerance)
   so sample discretization does not cause cadence slippage.
4. No Burst Executions: If a data gap occurs, the scheduler advances along the
   anchored grid without firing duplicate inferences at a single timestamp.
5. No Warm-up Looping: If a model becomes due before the 20-sample causal history is ready,
   the scheduler advances its schedule rather than attempting fake executions on every
   subsequent 10 Hz IMU sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


@dataclass(frozen=True)
class CadenceConfig:
    """Cadence configuration in seconds."""
    velocitynet_interval_s: float = 0.5   # ~2 Hz target
    biasnet_interval_s: float = 1.0       # ~1 Hz target
    min_inter_sample_s: float = 0.001     # Ignore samples arriving < 1 ms apart (duplicate/burst)
    jitter_tolerance_s: float = 0.02      # 20 ms tolerance for IMU timestamp discretization/jitter


@dataclass(frozen=True)
class ExpectedCadence:
    """Expected cadence counts derived independently from a timestamp sequence."""
    due_epochs: int
    inference_executions: int
    buffer_not_ready: int


def compute_expected_cadence(
    timestamps_ns: Sequence[int],
    interval_s: float,
    warmup_samples: int = 20,
    jitter_tolerance_s: float = 0.02,
) -> ExpectedCadence:
    """Independently derive expected cadence epochs from the exact timestamp sequence.

    Simulates the scheduled due epochs along an anchored timeline across the provided
    chronological timestamp sequence, accounting for causal warm-up buffer availability.

    Args:
        timestamps_ns: Sequence of sample timestamps in nanoseconds.
        interval_s: Model execution interval in seconds (e.g. 0.5s for VelocityNet, 1.0s for BiasNet).
        warmup_samples: Number of historical samples required for causal window readiness (default 20).
        jitter_tolerance_s: Tolerance window for sample discretization and clock jitter (default 0.02s).

    Returns:
        ExpectedCadence containing (due_epochs, inference_executions, buffer_not_ready).
    """
    if len(timestamps_ns) == 0:
        return ExpectedCadence(due_epochs=0, inference_executions=0, buffer_not_ready=0)

    interval_ns = int(round(interval_s * 1e9))
    tol_ns = int(round(jitter_tolerance_s * 1e9))
    next_due_ns = int(timestamps_ns[0])

    due_count = 0
    exec_count = 0
    not_ready_count = 0

    for step_idx, t_ns in enumerate(timestamps_ns):
        t_cur = int(t_ns)
        if t_cur >= next_due_ns - tol_ns:
            due_count += 1
            if step_idx + 1 >= warmup_samples:
                exec_count += 1
            else:
                not_ready_count += 1

            next_due_ns += interval_ns
            while next_due_ns <= t_cur - tol_ns:
                next_due_ns += interval_ns

    return ExpectedCadence(
        due_epochs=due_count,
        inference_executions=exec_count,
        buffer_not_ready=not_ready_count,
    )


class MLCadenceScheduler:
    """Stateful scheduler tracking execution epochs and due times for neural measurements."""

    def __init__(self, config: Optional[CadenceConfig] = None) -> None:
        self.config = config or CadenceConfig()
        self.anchor_time_ns: Optional[int] = None
        self.next_vnet_due_ns: Optional[int] = None
        self.next_bnet_due_ns: Optional[int] = None
        self.last_vnet_scheduled_ns: Optional[int] = None
        self.last_bnet_scheduled_ns: Optional[int] = None
        self.last_vnet_executed_ns: Optional[int] = None
        self.last_bnet_executed_ns: Optional[int] = None
        self.last_sample_time_ns: Optional[int] = None

    def reset(self, anchor_time_ns: Optional[int] = None) -> None:
        """Reset internal schedule at trajectory boundaries."""
        self.anchor_time_ns = int(anchor_time_ns) if anchor_time_ns is not None else None
        self.next_vnet_due_ns = self.anchor_time_ns
        self.next_bnet_due_ns = self.anchor_time_ns
        self.last_vnet_scheduled_ns = None
        self.last_bnet_scheduled_ns = None
        self.last_vnet_executed_ns = None
        self.last_bnet_executed_ns = None
        self.last_sample_time_ns = None

    def is_velocitynet_due(self, current_timestamp_ns: int) -> bool:
        """Check if VelocityNet has reached its scheduled execution epoch."""
        t_ns = int(current_timestamp_ns)
        if self.next_vnet_due_ns is None:
            return True
        tol_ns = int(round(self.config.jitter_tolerance_s * 1e9))
        return t_ns >= self.next_vnet_due_ns - tol_ns

    def is_biasnet_due(self, current_timestamp_ns: int) -> bool:
        """Check if BiasNet has reached its scheduled execution epoch."""
        t_ns = int(current_timestamp_ns)
        if self.next_bnet_due_ns is None:
            return True
        tol_ns = int(round(self.config.jitter_tolerance_s * 1e9))
        return t_ns >= self.next_bnet_due_ns - tol_ns

    def mark_velocitynet_scheduled(self, current_timestamp_ns: int) -> None:
        """Record that VelocityNet was evaluated and advance target on anchored grid."""
        t_ns = int(current_timestamp_ns)
        self.last_vnet_scheduled_ns = t_ns
        interval_ns = int(round(self.config.velocitynet_interval_s * 1e9))
        tol_ns = int(round(self.config.jitter_tolerance_s * 1e9))
        if self.next_vnet_due_ns is None:
            self.next_vnet_due_ns = t_ns + interval_ns
        else:
            self.next_vnet_due_ns += interval_ns
            # If large gap occurred, advance past current time without burst executions
            while self.next_vnet_due_ns <= t_ns - tol_ns:
                self.next_vnet_due_ns += interval_ns

    def mark_biasnet_scheduled(self, current_timestamp_ns: int) -> None:
        """Record that BiasNet was evaluated and advance target on anchored grid."""
        t_ns = int(current_timestamp_ns)
        self.last_bnet_scheduled_ns = t_ns
        interval_ns = int(round(self.config.biasnet_interval_s * 1e9))
        tol_ns = int(round(self.config.jitter_tolerance_s * 1e9))
        if self.next_bnet_due_ns is None:
            self.next_bnet_due_ns = t_ns + interval_ns
        else:
            self.next_bnet_due_ns += interval_ns
            # If large gap occurred, advance past current time without burst executions
            while self.next_bnet_due_ns <= t_ns - tol_ns:
                self.next_bnet_due_ns += interval_ns

    def mark_velocitynet_executed(self, current_timestamp_ns: int) -> None:
        """Record a successful model execution and advance schedule."""
        t_ns = int(current_timestamp_ns)
        self.last_vnet_executed_ns = t_ns
        self.mark_velocitynet_scheduled(t_ns)

    def record_velocitynet_execution(self, current_timestamp_ns: int) -> None:
        """Backward-compatible alias for mark_velocitynet_executed."""
        self.mark_velocitynet_executed(current_timestamp_ns)

    def mark_biasnet_executed(self, current_timestamp_ns: int) -> None:
        """Record a successful model execution and advance schedule."""
        t_ns = int(current_timestamp_ns)
        self.last_bnet_executed_ns = t_ns
        self.mark_biasnet_scheduled(t_ns)

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
                # Duplicate or degenerate timestamp (< 1 ms)
                return False, False

        self.last_sample_time_ns = t_ns

        if self.anchor_time_ns is None:
            self.anchor_time_ns = t_ns
            self.next_vnet_due_ns = t_ns
            self.next_bnet_due_ns = t_ns

        vnet_due = self.is_velocitynet_due(t_ns)
        bnet_due = self.is_biasnet_due(t_ns)

        return vnet_due, bnet_due
