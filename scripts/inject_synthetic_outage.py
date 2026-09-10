"""Synthetic GNSS Outage Injection Tooling for COMPASS (Phase 10).

Provides deterministic, reproducible GNSS blackout masking for offline benchmarking
and integration testing, sized to the SIH 26168 benchmark durations:
- 5s outage (short temporary gap / underpass)
- 10s outage (moderate local tunnel)
- 30s outage (long highway tunnel / urban canyon)
- 60s outage (major blackout / subterranean roadway)

CRITICAL INVARIANTS:
1. IMU acceleration, angular rates, and evaluation ground truth remain strictly unmodified.
2. Only GNSS observation availability is masked.
3. All output records, logs, and telemetry are strictly tagged as SYNTHETIC_OUTAGE to
   prevent any conflation with real recorded outage index data.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import numpy as np


@dataclass(frozen=True)
class SyntheticOutageSpec:
    """Specification for a synthetic GNSS blackout window.

    Attributes:
        start_offset_s: Time in seconds from session start when outage begins.
        duration_s: Outage duration in seconds.
        label: Descriptive identifier (default: SYNTHETIC_OUTAGE).
    """
    start_offset_s: float
    duration_s: float
    label: str = "SYNTHETIC_OUTAGE"

    @property
    def end_offset_s(self) -> float:
        return self.start_offset_s + self.duration_s


def generate_synthetic_outage_mask(
    timestamps_ns: Sequence[int],
    outage_start_offset_s: float,
    outage_duration_s: float,
    session_start_ns: Optional[int] = None,
) -> np.ndarray:
    """Generate a boolean mask indicating GNSS availability.

    Args:
        timestamps_ns: Sequence of sample timestamps in nanoseconds.
        outage_start_offset_s: Seconds from session start when GNSS becomes unavailable.
        outage_duration_s: Total seconds of GNSS blackout.
        session_start_ns: Optional anchor timestamp (defaults to timestamps_ns[0]).

    Returns:
        bool ndarray of same length: True = GNSS AVAILABLE; False = GNSS MASKED (OUTAGE).
    """
    ts = np.asarray(timestamps_ns, dtype=np.int64)
    if len(ts) == 0:
        return np.array([], dtype=bool)

    t0 = ts[0] if session_start_ns is None else int(session_start_ns)
    elapsed_s = (ts - t0) * 1e-9

    outage_start_s = float(outage_start_offset_s)
    outage_end_s = outage_start_s + float(outage_duration_s)

    in_outage = (elapsed_s >= outage_start_s) & (elapsed_s < outage_end_s)
    return ~in_outage


def parse_outage_spec(duration_str: str, start_s: float = 20.0) -> SyntheticOutageSpec:
    """Parse standard benchmark outage durations (5s, 10s, 30s, 60s)."""
    clean_str = duration_str.strip().lower().replace("s", "")
    try:
        dur = float(clean_str)
    except ValueError as e:
        raise ValueError(f"Invalid outage duration string '{duration_str}': {e}") from e

    return SyntheticOutageSpec(
        start_offset_s=start_s,
        duration_s=dur,
        label=f"SYNTHETIC_OUTAGE_{int(dur)}S",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic GNSS outage masks.")
    parser.add_argument("--duration", type=str, default="30s", help="Outage duration (e.g. 5s, 10s, 30s, 60s)")
    parser.add_argument("--start", type=float, default=20.0, help="Outage start offset in seconds")
    parser.add_argument("--total_duration", type=float, default=100.0, help="Total simulation duration in seconds")
    parser.add_argument("--rate_hz", type=float, default=1.0, help="GNSS sample rate in Hz")

    args = parser.parse_args()
    spec = parse_outage_spec(args.duration, start_s=args.start)

    num_samples = int(args.total_duration * args.rate_hz)
    dt_ns = int((1.0 / args.rate_hz) * 1e9)
    ts = [i * dt_ns for i in range(num_samples)]

    mask = generate_synthetic_outage_mask(ts, spec.start_offset_s, spec.duration_s)
    masked_count = int(np.sum(~mask))
    available_count = int(np.sum(mask))

    print(f"[{spec.label}] Total: {len(ts)}, Available: {available_count}, Masked: {masked_count}")
    print(f"Start: {spec.start_offset_s:.1f}s, End: {spec.end_offset_s:.1f}s, Duration: {spec.duration_s:.1f}s")


if __name__ == "__main__":
    main()
