"""GPS Outage Indexing and Synthetic Window Generation for COMPASS.

Manages real and synthetic GNSS blackout periods.
In accordance with Phase 0 findings:
- The IO-VNBD synchronized archive contains NO pre-tagged GPS outage index CSV.
- This module reports the absence of real outages as an explicit machine-readable condition (has_real_outages=False).
- Provides a synthetic outage window generator strictly flagged as is_synthetic=True
  for evaluating the headline benchmarks:
  * 50 m drift / 50 m / <1 min
  * 100 m drift / 1 km @ 60 km/h (~60 s outage)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd


@dataclass(frozen=True)
class OutageWindow:
    """Represents a discrete GNSS outage interval in time."""
    start_timestamp_ns: int
    end_timestamp_ns: int
    is_synthetic: bool = False
    label: str = ""

    @property
    def duration_s(self) -> float:
        """Outage duration in seconds."""
        return max(0.0, (self.end_timestamp_ns - self.start_timestamp_ns) / 1e9)

    def contains(self, timestamp_ns: int) -> bool:
        """Check if a given timestamp falls within this outage window."""
        return self.start_timestamp_ns <= timestamp_ns <= self.end_timestamp_ns


class OutageIndex:
    """Manages real and synthetic GNSS outages per trip."""

    def __init__(self, index_file_path: Optional[Path | str] = None) -> None:
        self.index_file_path = Path(index_file_path) if index_file_path else None
        self.real_outages_by_trip: Dict[str, List[OutageWindow]] = {}
        self.has_real_outages: bool = False

        if self.index_file_path and self.index_file_path.exists():
            self._load_real_outage_file(self.index_file_path)

    def _load_real_outage_file(self, path: Path) -> None:
        """Load external CSV outage index if one exists.

        Raises:
            ValueError: If the file exists but has invalid/missing columns or malformed rows.
        """
        try:
            df = pd.read_csv(path)
        except Exception as e:
            raise ValueError(f"Failed to read outage index file at {path}: {e}") from e

        df.columns = [c.strip().lower() for c in df.columns]
        required_cols = {"trip_id", "start_timestamp_ns", "end_timestamp_ns"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Malformed outage index at {path}: missing required columns {missing}")

        for idx, row in df.iterrows():
            trip_id = str(row["trip_id"]).strip().upper()
            try:
                t_start = int(row["start_timestamp_ns"])
                t_end = int(row["end_timestamp_ns"])
            except (ValueError, TypeError) as e:
                raise ValueError(f"Malformed timestamp at row {idx} in outage index {path}: {e}") from e

            if t_end <= t_start:
                raise ValueError(f"Invalid outage time range at row {idx} in {path}: start {t_start} >= end {t_end}")

            window = OutageWindow(
                start_timestamp_ns=t_start,
                end_timestamp_ns=t_end,
                is_synthetic=False,
                label=str(row.get("label", "real_outage")),
            )
            self.real_outages_by_trip.setdefault(trip_id, []).append(window)

        # Sort windows deterministically by start and end timestamps
        for tid in self.real_outages_by_trip:
            self.real_outages_by_trip[tid].sort(key=lambda w: (w.start_timestamp_ns, w.end_timestamp_ns))

        self.has_real_outages = len(self.real_outages_by_trip) > 0

    def has_real_outages_for_trip(self, trip_id: str) -> bool:
        """Check if a specific trip has registered real outage windows."""
        outages = self.get_outages(trip_id)
        return any(not o.is_synthetic for o in outages)

    def get_outages(self, trip_id: str) -> List[OutageWindow]:
        """Get all real outages registered for a given trip ID."""
        normalized_id = trip_id.strip().upper()
        return self.real_outages_by_trip.get(normalized_id, [])

    @staticmethod
    def generate_synthetic_outage(
        trip_start_ns: int,
        trip_end_ns: int,
        offset_from_start_s: float = 60.0,
        outage_duration_s: float = 60.0,
        label: str = "synthetic_60s_outage",
    ) -> Optional[OutageWindow]:
        """Generate a synthetic outage window within a trip's time span.

        Used for offline DR drift benchmark verification (e.g. 60s blackout).
        """
        start_ns = trip_start_ns + int(offset_from_start_s * 1e9)
        end_ns = start_ns + int(outage_duration_s * 1e9)

        if end_ns > trip_end_ns:
            # Shift backwards if outage exceeds trip duration
            end_ns = trip_end_ns
            start_ns = max(trip_start_ns, end_ns - int(outage_duration_s * 1e9))

        if start_ns >= end_ns:
            return None

        return OutageWindow(
            start_timestamp_ns=start_ns,
            end_timestamp_ns=end_ns,
            is_synthetic=True,
            label=label,
        )
