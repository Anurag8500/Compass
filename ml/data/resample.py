"""Generic Resampling and Decimation to Canonical 10 Hz Timeline.

COMPASS Phase 6 — ML Dataset Construction.
Canonical ML architecture: 10 Hz decimation for all ML feature tensors.

Key Invariants:
1. Native 10 Hz IO-VNBD data operates as an identity pass-through.
2. Higher-rate future data (e.g. 50 Hz, 100 Hz, 200 Hz) is causally decimated
   onto a strict 10 Hz grid (delta_t = 100,000,000 ns).
3. Strictly causal: no future samples are accessed when producing sample at t_k.
4. No extrapolation beyond valid source timestamps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np

CANONICAL_DT_NS: int = 100_000_000  # 100 ms = 10 Hz
CANONICAL_RATE_HZ: float = 10.0


@dataclass(frozen=True)
class ResampleResult:
    """Container for resampled signals and lineage metadata."""
    timestamps_ns: np.ndarray          # (M,) int64 canonical 10 Hz timestamps
    f_m_v: np.ndarray                  # (M, 3) float64 vehicle-frame specific force (m/s^2)
    omega_m_v: np.ndarray              # (M, 3) float64 vehicle-frame angular velocity (rad/s)
    is_validated: np.ndarray           # (M,) bool validity mask
    aux_signals: Dict[str, np.ndarray] # Additional resampled 1D/2D arrays
    is_identity: bool                  # True if input was already native 10 Hz
    original_rate_hz: float            # Measured input rate
    resampled_rate_hz: float           # Canonical target rate (10.0 Hz)


def resample_to_canonical_10hz(
    timestamps_ns: np.ndarray,
    f_m_v: np.ndarray,
    omega_m_v: np.ndarray,
    is_validated: Optional[np.ndarray] = None,
    aux_signals: Optional[Dict[str, np.ndarray]] = None,
    tolerance_ms: float = 20.0,
) -> ResampleResult:
    """Resample or causally decimate vehicle motion signals to canonical 10 Hz.

    Args:
        timestamps_ns: (N,) int64 non-decreasing timestamps in nanoseconds.
        f_m_v: (N, 3) float64 vehicle specific force in m/s^2.
        omega_m_v: (N, 3) float64 vehicle angular velocity in rad/s.
        is_validated: Optional (N,) bool validity mask. Defaults to all True.
        aux_signals: Optional dict of auxiliary 1D or 2D signals aligned to timestamps_ns.
        tolerance_ms: Maximum jitter tolerance in milliseconds to consider native 10 Hz.

    Returns:
        ResampleResult containing 10 Hz aligned arrays and decimation metadata.
    """
    n_samples = len(timestamps_ns)
    if n_samples < 2:
        raise ValueError(f"At least 2 samples required for resampling, got {n_samples}")
    if f_m_v.shape[0] != n_samples or f_m_v.shape[1] != 3:
        raise ValueError(f"f_m_v shape must be ({n_samples}, 3), got {f_m_v.shape}")
    if omega_m_v.shape[0] != n_samples or omega_m_v.shape[1] != 3:
        raise ValueError(f"omega_m_v shape must be ({n_samples}, 3), got {omega_m_v.shape}")

    if is_validated is None:
        val_mask = np.ones(n_samples, dtype=bool)
    else:
        if len(is_validated) != n_samples:
            raise ValueError(f"is_validated length ({len(is_validated)}) != timestamps ({n_samples})")
        val_mask = np.asarray(is_validated, dtype=bool)

    # Compute empirical sampling rate
    dts_s = np.diff(timestamps_ns) * 1e-9
    valid_dts = dts_s[dts_s > 0]
    median_dt = float(np.median(valid_dts)) if len(valid_dts) > 0 else 0.1
    measured_rate = 1.0 / median_dt if median_dt > 0 else 10.0

    # Check if data is already approximately native 10 Hz
    tol_s = tolerance_ms * 1e-3
    is_native_10hz = (
        abs(median_dt - 0.1) <= tol_s
        and np.all(np.abs(valid_dts - 0.1) <= 0.05)  # Max jitter within 50ms
    )

    if is_native_10hz:
        # Pass-through identity
        res_aux = {}
        if aux_signals:
            for k, arr in aux_signals.items():
                res_aux[k] = np.copy(arr)
        return ResampleResult(
            timestamps_ns=np.copy(timestamps_ns),
            f_m_v=np.copy(f_m_v),
            omega_m_v=np.copy(omega_m_v),
            is_validated=np.copy(val_mask),
            aux_signals=res_aux,
            is_identity=True,
            original_rate_hz=measured_rate,
            resampled_rate_hz=CANONICAL_RATE_HZ,
        )

    # Higher-rate data: construct causal 10 Hz decimation grid
    t_start = int(timestamps_ns[0])
    t_end = int(timestamps_ns[-1])

    grid_times = np.arange(t_start, t_end + 1, CANONICAL_DT_NS, dtype=np.int64)
    m_grid = len(grid_times)

    if m_grid < 2:
        # Duration shorter than canonical dt, return boundary points
        grid_times = np.array([t_start], dtype=np.int64)
        m_grid = 1

    out_f = np.zeros((m_grid, 3), dtype=np.float64)
    out_omega = np.zeros((m_grid, 3), dtype=np.float64)
    out_valid = np.zeros(m_grid, dtype=bool)

    out_aux: Dict[str, np.ndarray] = {}
    if aux_signals:
        for k, arr in aux_signals.items():
            if arr.ndim == 1:
                out_aux[k] = np.zeros(m_grid, dtype=arr.dtype)
            else:
                out_aux[k] = np.zeros((m_grid, *arr.shape[1:]), dtype=arr.dtype)

    # For k=0: take value at t_0
    out_f[0] = f_m_v[0]
    out_omega[0] = omega_m_v[0]
    out_valid[0] = val_mask[0]
    if aux_signals:
        for k, arr in aux_signals.items():
            out_aux[k][0] = arr[0]

    # For k > 0: strictly causal pooling over source samples in (t_{k-1}, t_k]
    # searchsorted finds indices in timestamps_ns
    for k in range(1, m_grid):
        t_prev = grid_times[k - 1]
        t_curr = grid_times[k]

        # strictly causal interval: past sample time < t <= current sample time
        idx_start = np.searchsorted(timestamps_ns, t_prev, side="right")
        idx_end = np.searchsorted(timestamps_ns, t_curr, side="right")

        if idx_start < idx_end:
            # Pool source samples falling within the causal interval
            sub_f = f_m_v[idx_start:idx_end]
            sub_omega = omega_m_v[idx_start:idx_end]
            sub_val = val_mask[idx_start:idx_end]

            # Mean of samples in this causal bin
            out_f[k] = np.mean(sub_f, axis=0)
            out_omega[k] = np.mean(sub_omega, axis=0)
            out_valid[k] = bool(np.all(sub_val))

            if aux_signals:
                for name, arr in aux_signals.items():
                    sub_arr = arr[idx_start:idx_end]
                    if np.issubdtype(arr.dtype, np.floating):
                        out_aux[name][k] = np.mean(sub_arr, axis=0)
                    else:
                        # Discrete / categorical: take latest causal sample
                        out_aux[name][k] = sub_arr[-1]
        else:
            # No sample arrived in this discrete bin: causal zero-order hold from previous bin
            out_f[k] = out_f[k - 1]
            out_omega[k] = out_omega[k - 1]
            out_valid[k] = False  # Mark dropout / gap as unvalidated

            if aux_signals:
                for name, arr in aux_signals.items():
                    out_aux[name][k] = out_aux[name][k - 1]

    return ResampleResult(
        timestamps_ns=grid_times,
        f_m_v=out_f,
        omega_m_v=out_omega,
        is_validated=out_valid,
        aux_signals=out_aux,
        is_identity=False,
        original_rate_hz=measured_rate,
        resampled_rate_hz=CANONICAL_RATE_HZ,
    )
