#!/usr/bin/env python3
"""Inject a synthetic GPS outage into a real IO-VNBD smartphone CSV file.

For the outage window:
  - GPS SATELLITES IN RANGE -> 0
  - GPS ACCURACY            -> 999 (no fix)
  - GPS LATITUDE/LONGITUDE  -> frozen at last real value before outage

Real GPS values are restored exactly after the window ends.

Presets
-------
SHORT   : ~60 seconds (PS benchmark: "50m drift over <1 min")
LONG    : distance_km / speed_kmh * 3600 seconds  (e.g. 1 km @ 60 km/h = 60 s)
STRESS  : configurable duration >= 5 minutes for stress testing

Usage
-----
    python scripts/inject_synthetic_outage.py <input_csv> <start_row> <preset> [options]

Examples
--------
    python scripts/inject_synthetic_outage.py D:/sih/data/S-S1.txt 5000 SHORT
    python scripts/inject_synthetic_outage.py D:/sih/data/S-S1.txt 5000 LONG --distance 1.0 --speed 60
    python scripts/inject_synthetic_outage.py D:/sih/data/S-S1.txt 5000 STRESS --stress-duration 360000
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Preset durations (milliseconds)
# ---------------------------------------------------------------------------

PRESET_SHORT_MS:  float = 60_000.0    # 60 seconds
PRESET_STRESS_MS: float = 300_000.0   # 5 minutes default

# Outage injection values
OUTAGE_SAT_COUNT: str = "0"
OUTAGE_ACCURACY:  str = "999"


# ---------------------------------------------------------------------------
# Column name resolution (case-insensitive, strip-safe)
# ---------------------------------------------------------------------------

def _find_col(fieldnames: list[str], *keywords: str) -> Optional[str]:
    """Return first fieldname whose lower-case form contains all keywords."""
    for name in fieldnames:
        low = name.lower()
        if all(k in low for k in keywords):
            return name
    return None


# ---------------------------------------------------------------------------
# Core injection logic
# ---------------------------------------------------------------------------

def inject_outage(
    input_path: Path,
    start_row: int,
    duration_ms: float,
    preset_name: str,
    output_path: Path,
) -> None:
    """Read input CSV, inject outage, write output CSV, print summary."""

    # ---- Read all rows -------------------------------------------------------
    with open(input_path, encoding="latin-1") as f:
        reader = csv.DictReader(f)
        raw_fieldnames = reader.fieldnames
        if not raw_fieldnames:
            print("Error: CSV has no header row.")
            sys.exit(1)
        fieldnames = [c.strip() for c in raw_fieldnames]
        rows = []
        for row in reader:
            rows.append({k.strip(): v for k, v in row.items()})

    total_rows = len(rows)
    if start_row < 1 or start_row > total_rows:
        print(f"Error: start_row {start_row} out of range (1–{total_rows})")
        sys.exit(1)

    # ---- Locate required columns --------------------------------------------
    lat_col  = _find_col(fieldnames, "gps latitude")
    lon_col  = _find_col(fieldnames, "gps longitude")
    acc_col  = _find_col(fieldnames, "gps accuracy")
    sat_col  = _find_col(fieldnames, "gps satellites")
    time_col = _find_col(fieldnames, "time since start", "ms")

    if not all([lat_col, lon_col, acc_col, sat_col, time_col]):
        missing = [n for n, c in [("lat", lat_col), ("lon", lon_col),
                                   ("acc", acc_col), ("sat", sat_col),
                                   ("time", time_col)] if not c]
        print(f"Error: could not find columns for: {missing}")
        print(f"Available columns: {fieldnames}")
        sys.exit(1)

    # ---- Find outage window by timestamp ------------------------------------
    # start_row is 1-indexed; convert to 0-indexed
    start_idx = start_row - 1

    try:
        t_start_ms = float(rows[start_idx][time_col])
    except (ValueError, KeyError):
        print(f"Error: cannot parse timestamp at row {start_row}")
        sys.exit(1)

    t_end_ms = t_start_ms + duration_ms

    # Determine end index: first row whose timestamp >= t_end_ms
    end_idx = total_rows  # default: runs to end of file
    for i in range(start_idx, total_rows):
        try:
            t = float(rows[i][time_col])
        except (ValueError, KeyError):
            continue
        if t >= t_end_ms:
            end_idx = i
            break

    # ---- Capture frozen position (last real row before outage) --------------
    frozen_lat = rows[start_idx][lat_col]
    frozen_lon = rows[start_idx][lon_col]

    # ---- Build modified rows ------------------------------------------------
    modified = []
    for i, row in enumerate(rows):
        r = dict(row)
        if start_idx <= i < end_idx:
            r[lat_col] = frozen_lat
            r[lon_col] = frozen_lon
            r[acc_col] = OUTAGE_ACCURACY
            r[sat_col] = OUTAGE_SAT_COUNT
        modified.append(r)

    # ---- Write output -------------------------------------------------------
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(modified)

    # ---- Summary ------------------------------------------------------------
    actual_end_row = min(end_idx, total_rows)
    try:
        t_actual_end = float(rows[actual_end_row - 1][time_col])
    except (IndexError, ValueError, KeyError):
        t_actual_end = t_end_ms

    print()
    print("Synthetic outage injection summary")
    print("=" * 50)
    print(f"  Input file      : {input_path.name}")
    print(f"  Output file     : {output_path.name}")
    print(f"  Preset          : {preset_name}")
    print(f"  Duration        : {duration_ms/1000:.1f}s  ({duration_ms:.0f}ms)")
    print(f"  Original rows   : {total_rows}")
    print(f"  Outage rows     : {start_idx + 1} – {actual_end_row}  "
          f"({actual_end_row - start_idx} rows affected)")
    print(f"  Outage time     : {t_start_ms:.0f}ms – {t_end_ms:.0f}ms")
    print(f"  Frozen position : lat={frozen_lat}  lon={frozen_lon}")
    print()
    print(f"Run FSM test with:")
    print(f"  python scripts/test_fsm.py {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inject a synthetic GPS outage into an IO-VNBD CSV file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input",       help="Path to input CSV/TXT file")
    parser.add_argument("start_row",   type=int, help="1-indexed row to begin outage")
    parser.add_argument("preset",      choices=["SHORT", "LONG", "STRESS"],
                        help="Outage duration preset")
    parser.add_argument("--distance",  type=float, default=1.0,
                        help="LONG preset: distance in km (default 1.0)")
    parser.add_argument("--speed",     type=float, default=60.0,
                        help="LONG preset: speed in km/h (default 60.0)")
    parser.add_argument("--stress-duration", type=float, default=PRESET_STRESS_MS,
                        dest="stress_ms",
                        help=f"STRESS preset: duration in ms (default {PRESET_STRESS_MS:.0f})")
    parser.add_argument("--output",    default=None,
                        help="Output file path (default: <input>_outage_<preset>.csv)")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: file not found: {input_path}")
        sys.exit(1)

    # Resolve duration from preset
    if args.preset == "SHORT":
        duration_ms = PRESET_SHORT_MS
    elif args.preset == "LONG":
        if args.speed <= 0:
            print("Error: --speed must be > 0")
            sys.exit(1)
        duration_ms = (args.distance / args.speed) * 3_600_000.0  # hours -> ms
    else:  # STRESS
        duration_ms = args.stress_ms

    # Resolve output path
    if args.output:
        output_path = Path(args.output)
    else:
        stem = input_path.stem
        output_path = input_path.parent / f"{stem}_outage_{args.preset}.csv"

    inject_outage(input_path, args.start_row, duration_ms, args.preset, output_path)


if __name__ == "__main__":
    main()
