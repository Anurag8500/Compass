"""IO-VNBD dataset CSV parsing into canonical Phase 1 schemas.

Handles:
- S-files: Smartphone IMU (accel, gyro) in RAW DEVICE BODY FRAME and S-GPS fixes.
- V-files: Vehicle CAN / Racelogic VBOX ground-truth telemetry.
- Latin-1 encoding, comma-space header stripping, and missing field normalization.
- Explicit timestamp conversion to canonical nanoseconds (timestamp_ns).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from navigation.schemas.gnss import GNSSSample
from navigation.schemas.imu import FLAG_OK, RawIMUSample, SensorSource


@dataclass
class ParsedSTrip:
    """Container for parsed smartphone sensor and GPS records."""
    file_path: Path
    row_count: int
    timestamps_ns: np.ndarray               # 1D array int64 nanoseconds
    accel_raw: np.ndarray                   # Nx3 float64 [ax, ay, az] in device frame (m/s^2)
    gyro_raw: np.ndarray                    # Nx3 float64 [gx, gy, gz] in device frame (rad/s)
    gnss_lat: np.ndarray                    # 1D float64
    gnss_lon: np.ndarray                    # 1D float64
    gnss_alt: np.ndarray                    # 1D float64 (m)
    gnss_speed_mps: np.ndarray              # 1D float64 (m/s)
    gnss_bearing_deg: np.ndarray            # 1D float64 [0, 360)
    gnss_accuracy_m: np.ndarray             # 1D float64
    gnss_sat_count: np.ndarray              # 1D int32 (-1 if missing)
    has_gnss: bool

    def to_imu_samples(self, sensor_id: str = "phone_internal") -> List[RawIMUSample]:
        """Convert array representations to canonical Phase 1 RawIMUSample list."""
        samples: List[RawIMUSample] = []
        for i in range(self.row_count):
            t_ns = int(self.timestamps_ns[i])
            ax, ay, az = float(self.accel_raw[i, 0]), float(self.accel_raw[i, 1]), float(self.accel_raw[i, 2])
            gx, gy, gz = float(self.gyro_raw[i, 0]), float(self.gyro_raw[i, 1]), float(self.gyro_raw[i, 2])
            samples.append(
                RawIMUSample(
                    timestamp_ns=t_ns,
                    accel=(ax, ay, az),
                    gyro=(gx, gy, gz),
                    quality_flags=FLAG_OK,
                    source=SensorSource.PHONE,
                    sensor_id=sensor_id,
                )
            )
        return samples

    def to_gnss_samples(self) -> List[Optional[GNSSSample]]:
        """Convert valid GPS rows to canonical Phase 1 GNSSSample list."""
        samples: List[Optional[GNSSSample]] = []
        for i in range(self.row_count):
            lat = float(self.gnss_lat[i])
            lon = float(self.gnss_lon[i])
            if np.isnan(lat) or np.isnan(lon) or (lat == 0.0 and lon == 0.0):
                samples.append(None)
                continue

            spd = float(self.gnss_speed_mps[i]) if not np.isnan(self.gnss_speed_mps[i]) else None
            brg = float(self.gnss_bearing_deg[i]) if not np.isnan(self.gnss_bearing_deg[i]) else None
            acc = float(self.gnss_accuracy_m[i]) if not np.isnan(self.gnss_accuracy_m[i]) else None
            sats = int(self.gnss_sat_count[i]) if self.gnss_sat_count[i] >= 0 else None

            samples.append(
                GNSSSample(
                    timestamp_ns=int(self.timestamps_ns[i]),
                    lat=lat,
                    lon=lon,
                    alt=float(self.gnss_alt[i]),
                    speed=spd,
                    bearing=brg,
                    accuracy_m=acc,
                    sat_count=sats,
                    trust_score=1.0,
                )
            )
        return samples


@dataclass
class ParsedVTrip:
    """Container for parsed vehicle reference and CAN bus telemetry."""
    file_path: Path
    row_count: int
    timestamps_ns: np.ndarray               # 1D array int64 nanoseconds
    lat: np.ndarray                         # 1D float64 (degrees)
    lon: np.ndarray                         # 1D float64 (degrees)
    alt_m: np.ndarray                       # 1D float64 (meters)
    speed_mps: np.ndarray                   # 1D float64 (m/s)
    heading_deg: np.ndarray                 # 1D float64 [0, 360)
    yaw_rate_rads: np.ndarray               # 1D float64 (rad/s)
    wheel_speeds_rads: np.ndarray           # Nx4 float64 [FL, FR, RL, RR] (rad/s)
    can_accel_g: np.ndarray                 # Nx2 float64 [longitudinal, lateral] (g)
    steering_angle_deg: np.ndarray          # 1D float64 (degrees)
    handbrake: np.ndarray                   # 1D int32 (0 or 1)
    gear: np.ndarray                        # 1D int32 (0 to 5)


def parse_s_file(file_path: Path | str, allow_generic_gyro: bool = False) -> ParsedSTrip:
    """Parse an IO-VNBD smartphone (S-) CSV file into structured array representation.

    Column Mappings:
        Timestamp: 'TIME SINCE START (ms)' -> scaled by 1,000,000 to nanoseconds (signed int64).
        Accelerometer: 'ACCELEROMETER X (m/s²)', 'ACCELEROMETER Y (m/s²)', 'ACCELEROMETER Z (m/s²)'
        Gyroscope: Body axes mapped strictly as:
            gx (Roll)  = 'GYROSCOPE Roll (rad/s)'
            gy (Pitch) = 'GYROSCOPE Pitch (rad/s)'
            gz (Yaw)   = 'GYROSCOPE Yaw (rad/s)'
        GNSS: GPS LATITUDE, GPS LONGITUDE, GPS ALTITUDE, GPS SPEED (km/h / 3.6), GPS ACCURACY, GPS ORIENTATION

    Strictness Policy:
        - Required fields (timestamp, 3-axis accelerometer, 3-axis gyroscope):
          Must exist; otherwise raises KeyError specifying the missing semantic field.
        - Optional fields (GNSS coordinates, altitude, speed, bearing, accuracy, satellites):
          Preserve missingness using NaN (or -1 for satellite count); NEVER fabricate ground truth.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"S-file not found: {path}")

    # Read CSV with Latin-1 encoding and strip whitespace from headers
    df = pd.read_csv(path, encoding="latin-1", skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]

    row_count = len(df)
    if row_count == 0:
        raise ValueError(f"S-file is empty: {path}")

    # Resolve timestamp column: strict IO-VNBD canonical mapping first
    t_col = None
    for c in df.columns:
        if c.lower() == "time since start (ms)":
            t_col = c
            break
    if not t_col:
        candidates = [c for c in df.columns if "time since start" in c.lower() and "ms" in c.lower()]
        if len(candidates) == 1:
            t_col = candidates[0]
        else:
            raise KeyError(f"Ambiguous or missing timestamp column in S-file {path}. Columns: {list(df.columns)}")

    t_raw = pd.to_numeric(df[t_col], errors="coerce").to_numpy(dtype=float)
    # Safe signed int64 conversion preserving negative and non-finite timestamps for quality validation
    valid_t = np.isfinite(t_raw)
    timestamps_ns = np.zeros(row_count, dtype=np.int64)
    timestamps_ns[valid_t] = np.round(t_raw[valid_t] * 1_000_000.0).astype(np.int64)
    timestamps_ns[~valid_t] = -1  # Sentinel invalid timestamp for non-finite entries

    # Accelerometer: X, Y, Z (Strictly required)
    ax_col = next((c for c in df.columns if "accelerometer x" in c.lower()), None)
    ay_col = next((c for c in df.columns if "accelerometer y" in c.lower()), None)
    az_col = next((c for c in df.columns if "accelerometer z" in c.lower()), None)
    if not (ax_col and ay_col and az_col):
        missing_axes = []
        if not ax_col: missing_axes.append("ACCELEROMETER X")
        if not ay_col: missing_axes.append("ACCELEROMETER Y")
        if not az_col: missing_axes.append("ACCELEROMETER Z")
        raise KeyError(f"Missing required 3-axis accelerometer column(s) {missing_axes} in S-file {path}")

    accel_raw = np.column_stack([
        df[ax_col].to_numpy(dtype=float),
        df[ay_col].to_numpy(dtype=float),
        df[az_col].to_numpy(dtype=float),
    ])

    # Gyroscope: Explicit IO-VNBD dataset mapping
    # Categorised branch: Roll -> x, Pitch -> y, Yaw -> z
    # Uncategorised branch: Gyroscope X -> x, Gyroscope Y -> y, Gyroscope Z -> z
    g_roll_col = next((c for c in df.columns if "gyroscope roll" in c.lower()), None)
    g_pitch_col = next((c for c in df.columns if "gyroscope pitch" in c.lower()), None)
    g_yaw_col = next((c for c in df.columns if "gyroscope yaw" in c.lower()), None)

    if g_roll_col and g_pitch_col and g_yaw_col:
        gx_col, gy_col, gz_col = g_roll_col, g_pitch_col, g_yaw_col
    else:
        gx_col = next((c for c in df.columns if "gyroscope x" in c.lower()), None)
        gy_col = next((c for c in df.columns if "gyroscope y" in c.lower()), None)
        gz_col = next((c for c in df.columns if "gyroscope z" in c.lower()), None)
        if not (gx_col and gy_col and gz_col):
            if allow_generic_gyro:
                # Opt-in fallback for arbitrary non-canonical datasets (e.g. 'gyro x', 'rot x')
                gx_col = next((c for c in df.columns if "gyro" in c.lower() and "x" in c.lower()), None)
                gy_col = next((c for c in df.columns if "gyro" in c.lower() and "y" in c.lower()), None)
                gz_col = next((c for c in df.columns if "gyro" in c.lower() and "z" in c.lower()), None)
            if not (gx_col and gy_col and gz_col):
                raise KeyError(
                    f"Missing required canonical IO-VNBD 3-axis gyroscope columns in S-file {path}. "
                    f"Expected either ('GYROSCOPE Roll/Pitch/Yaw') or ('GYROSCOPE X/Y/Z'). Columns: {list(df.columns)}"
                )

    gyro_raw = np.column_stack([
        df[gx_col].to_numpy(dtype=float),
        df[gy_col].to_numpy(dtype=float),
        df[gz_col].to_numpy(dtype=float),
    ])

    # GNSS columns in S-file (Optional phone GPS telemetry: preserve missingness with NaN/-1)
    lat_col = next((c for c in df.columns if "gps latitude" in c.lower()), None)
    lon_col = next((c for c in df.columns if "gps longitude" in c.lower()), None)
    alt_col = next((c for c in df.columns if "gps altitude" in c.lower()), None)
    spd_col = next((c for c in df.columns if "gps speed" in c.lower()), None)
    brg_col = next((c for c in df.columns if "gps orientation" in c.lower() or "gps bearing" in c.lower()), None)
    acc_col = next((c for c in df.columns if "gps accuracy" in c.lower()), None)
    sat_col = next((c for c in df.columns if "gps satellites" in c.lower()), None)

    has_gnss = lat_col is not None and lon_col is not None
    gnss_lat = df[lat_col].to_numpy(dtype=float) if lat_col else np.full(row_count, np.nan, dtype=float)
    gnss_lon = df[lon_col].to_numpy(dtype=float) if lon_col else np.full(row_count, np.nan, dtype=float)
    gnss_alt = df[alt_col].to_numpy(dtype=float) if alt_col else np.full(row_count, np.nan, dtype=float)
    # Convert km/h to m/s (1 km/h = 1/3.6 m/s)
    gnss_speed_mps = (df[spd_col].to_numpy(dtype=float) / 3.6) if spd_col else np.full(row_count, np.nan, dtype=float)
    gnss_bearing_deg = df[brg_col].to_numpy(dtype=float) if brg_col else np.full(row_count, np.nan, dtype=float)
    gnss_accuracy_m = df[acc_col].to_numpy(dtype=float) if acc_col else np.full(row_count, np.nan, dtype=float)

    # Satellite count (might contain strings or NaNs)
    if sat_col:
        sat_series = pd.to_numeric(df[sat_col], errors="coerce").fillna(-1).to_numpy(dtype=int)
        gnss_sat_count = sat_series
    else:
        gnss_sat_count = np.full(row_count, -1, dtype=int)

    return ParsedSTrip(
        file_path=path,
        row_count=row_count,
        timestamps_ns=timestamps_ns,
        accel_raw=accel_raw,
        gyro_raw=gyro_raw,
        gnss_lat=gnss_lat,
        gnss_lon=gnss_lon,
        gnss_alt=gnss_alt,
        gnss_speed_mps=gnss_speed_mps,
        gnss_bearing_deg=gnss_bearing_deg,
        gnss_accuracy_m=gnss_accuracy_m,
        gnss_sat_count=gnss_sat_count,
        has_gnss=has_gnss,
    )


def parse_v_file(file_path: Path | str) -> ParsedVTrip:
    """Parse an IO-VNBD vehicle/VBOX (V-) CSV file into structured array representation.

    Column Mappings:
        Timestamp: 'Time Since Start of Day (seconds)' -> scaled by 1,000,000,000 to nanoseconds.
        Position: 'Latitude (degrees)', 'Longitude (degrees)', 'Height (km)' (km * 1000 -> m)
        Speed: 'Velocity (km/hr)' -> scaled by 1/3.6 to m/s.
        Heading: 'Heading (degrees)'
        Yaw Rate: 'Yaw Rate (deg/sec)' -> np.radians to rad/s.
        Wheel Speeds: Front Left, Front Right, Rear Left, Rear Right (rad/sec).
        CAN Acceleration: Indicated Longitudinal / Lateral (g).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"V-file not found: {path}")

    df = pd.read_csv(path, encoding="latin-1", skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]

    row_count = len(df)
    if row_count == 0:
        raise ValueError(f"V-file is empty: {path}")

    # Timestamp: strict IO-VNBD canonical mapping first
    t_col = None
    for c in df.columns:
        if c.lower() == "time since start of day (seconds)":
            t_col = c
            break
    if not t_col:
        candidates = [c for c in df.columns if "time since start of day" in c.lower() and "second" in c.lower()]
        if len(candidates) == 1:
            t_col = candidates[0]
        else:
            raise KeyError(f"Ambiguous or missing timestamp column in V-file {path}. Columns: {list(df.columns)}")

    t_sec = df[t_col].to_numpy(dtype=float)
    # Safe signed int64 conversion preserving negative and non-finite timestamps
    valid_t = np.isfinite(t_sec)
    timestamps_ns = np.zeros(row_count, dtype=np.int64)
    timestamps_ns[valid_t] = np.round(t_sec[valid_t] * 1_000_000_000.0).astype(np.int64)
    timestamps_ns[~valid_t] = -1

    # Position: Latitude, Longitude (Strictly required), Height (km -> m, Optional)
    lat_col = next((c for c in df.columns if c.lower() == "latitude (degrees)"), None)
    lon_col = next((c for c in df.columns if c.lower() == "longitude (degrees)"), None)
    if not (lat_col and lon_col):
        missing_pos = []
        if not lat_col: missing_pos.append("Latitude (degrees)")
        if not lon_col: missing_pos.append("Longitude (degrees)")
        raise KeyError(f"Missing required position column(s) {missing_pos} in V-file {path}")

    h_col = next((c for c in df.columns if "height" in c.lower()), None)

    lat = df[lat_col].to_numpy(dtype=float)
    lon = df[lon_col].to_numpy(dtype=float)
    alt_m = (df[h_col].to_numpy(dtype=float) * 1000.0) if h_col else np.full(row_count, np.nan, dtype=float)

    # Velocity: km/hr -> m/s (Strictly required)
    vel_col = next((c for c in df.columns if "velocity (km/hr)" in c.lower()), None)
    if not vel_col:
        raise KeyError(f"Missing required velocity column 'Velocity (km/hr)' in V-file {path}")
    speed_mps = df[vel_col].to_numpy(dtype=float) / 3.6

    # Heading: degrees (Strictly required)
    hdg_col = next((c for c in df.columns if "heading (degrees)" in c.lower()), None)
    if not hdg_col:
        raise KeyError(f"Missing required heading column 'Heading (degrees)' in V-file {path}")
    heading_deg = df[hdg_col].to_numpy(dtype=float)

    # Yaw Rate: deg/sec -> rad/s (Strictly required)
    yaw_col = next((c for c in df.columns if "yaw rate" in c.lower()), None)
    if not yaw_col:
        raise KeyError(f"Missing required yaw rate column 'Yaw Rate (deg/sec)' in V-file {path}")
    yaw_rate_rads = np.radians(df[yaw_col].to_numpy(dtype=float))

    # Wheel Speeds: Front Left, Front Right, Rear Left, Rear Right (Optional telemetry: preserve NaN)
    ws_fl = next((c for c in df.columns if "wheel speed front left" in c.lower()), None)
    ws_fr = next((c for c in df.columns if "wheel speed front right" in c.lower()), None)
    ws_rl = next((c for c in df.columns if "wheel speed rear left" in c.lower()), None)
    ws_rr = next((c for c in df.columns if "wheel speed rear right" in c.lower()), None)

    if ws_fl and ws_fr and ws_rl and ws_rr:
        wheel_speeds_rads = np.column_stack([
            df[ws_fl].to_numpy(dtype=float),
            df[ws_fr].to_numpy(dtype=float),
            df[ws_rl].to_numpy(dtype=float),
            df[ws_rr].to_numpy(dtype=float),
        ])
    else:
        wheel_speeds_rads = np.full((row_count, 4), np.nan, dtype=float)

    # CAN acceleration: longitudinal, lateral in g (Optional telemetry: preserve NaN)
    long_acc_col = next((c for c in df.columns if "indicated longitudinal acceleration" in c.lower()), None)
    lat_acc_col = next((c for c in df.columns if "indicated lateral acceleration" in c.lower()), None)
    col_long = df[long_acc_col].to_numpy(dtype=float) if long_acc_col else np.full(row_count, np.nan, dtype=float)
    col_lat = df[lat_acc_col].to_numpy(dtype=float) if lat_acc_col else np.full(row_count, np.nan, dtype=float)
    can_accel_g = np.column_stack([col_long, col_lat])

    # Steering angle, handbrake, gear (Optional telemetry: preserve NaN / sentinel -1)
    steer_col = next((c for c in df.columns if "steering angle" in c.lower()), None)
    hb_col = next((c for c in df.columns if "handbrake" in c.lower()), None)
    gear_col = next((c for c in df.columns if c.lower().startswith("gear")), None)

    steering_angle_deg = df[steer_col].to_numpy(dtype=float) if steer_col else np.full(row_count, np.nan, dtype=float)
    handbrake = pd.to_numeric(df[hb_col], errors="coerce").fillna(-1).to_numpy(dtype=int) if hb_col else np.full(row_count, -1, dtype=int)
    gear = pd.to_numeric(df[gear_col], errors="coerce").fillna(-1).to_numpy(dtype=int) if gear_col else np.full(row_count, -1, dtype=int)

    return ParsedVTrip(
        file_path=path,
        row_count=row_count,
        timestamps_ns=timestamps_ns,
        lat=lat,
        lon=lon,
        alt_m=alt_m,
        speed_mps=speed_mps,
        heading_deg=heading_deg,
        yaw_rate_rads=yaw_rate_rads,
        wheel_speeds_rads=wheel_speeds_rads,
        can_accel_g=can_accel_g,
        steering_angle_deg=steering_angle_deg,
        handbrake=handbrake,
        gear=gear,
    )
