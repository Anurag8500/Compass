"""Methodology audit for BiasNet label generation across horizons and conditioning thresholds.

Runs on representative real driving data (Driver E train, Driver B validation) to:
1. Evaluate horizon duration: 0.5s vs 1.0s vs 2.0s.
2. Inspect singular value and condition number distributions.
3. Quantify bound activation and residual reduction.
4. Establish documented identifiability gating rules.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Sequence
import numpy as np

from data.pipeline.sync import SynchronizedTrip
from data.pipeline.stationary_detect import StationaryDetector
from navigation.preprocessing.pipeline import PreprocessingPipeline
from ml.data.features import compute_canonical_features
from ml.data.resample import resample_to_canonical_10hz
from ml.data.windowing import extract_causal_windows
from ml.data.biasnet_labels import (
    BiasNetOptimizationConfig,
    solve_window_bias_correction,
)


def run_horizon_audit_on_trip(
    trip_path: str | Path,
    driver_id: str,
    max_windows: int = 400,
    horizons: Sequence[float] = (0.5, 1.0, 2.0),
) -> Dict[str, Dict[str, Any]]:
    """Run audit across multiple horizons on a single trip."""
    trip = SynchronizedTrip.load_npz(trip_path)
    detector = StationaryDetector()
    _, stat_mask = detector.detect(trip.timestamps_ns, trip.accel_raw, trip.gyro_raw)
    pipeline = PreprocessingPipeline(sampling_rate_hz=10.0, filter_cutoff_hz=3.0, median_window_size=3)
    preprocessed = pipeline.process_trip(trip, stationary_mask=stat_mask)

    aux = {
        "v_ref_speed_mps": trip.v_ref_speed_mps,
        "v_ref_lat": trip.v_ref_lat,
        "v_ref_lon": trip.v_ref_lon,
        "v_ref_alt_m": trip.v_ref_alt_m / 1000.0,
        "v_ref_heading_deg": trip.v_ref_heading_deg,
    }

    res = resample_to_canonical_10hz(
        timestamps_ns=preprocessed.timestamps_ns,
        f_m_v=preprocessed.f_m_v,
        omega_m_v=preprocessed.omega_m_v,
        is_validated=preprocessed.is_validated,
        aux_signals=aux,
    )

    f_10hz = res.f_m_v
    w_10hz = res.omega_m_v
    ts_10hz = res.timestamps_ns
    val_10hz = res.is_validated

    lat_10hz = res.aux_signals["v_ref_lat"]
    lon_10hz = res.aux_signals["v_ref_lon"]
    alt_10hz = res.aux_signals["v_ref_alt_m"]
    spd_10hz = res.aux_signals["v_ref_speed_mps"]
    hdg_10hz = res.aux_signals["v_ref_heading_deg"]

    features_9ch = compute_canonical_features(ts_10hz, f_10hz, w_10hz)
    windows = extract_causal_windows(
        features=features_9ch,
        timestamps_ns=ts_10hz,
        is_validated=val_10hz,
        source_file_id=str(trip_path),
        driver_id=driver_id,
        window_size=20,
        stride_samples=5,
    )

    valid_windows = [w for w in windows if w.is_valid][:max_windows]
    print(f"[{Path(trip_path).stem}] Auditing {len(valid_windows)} valid windows across horizons {list(horizons)}...")

    results_by_horizon: Dict[str, Dict[str, Any]] = {}

    if not valid_windows:
        for H in horizons:
            results_by_horizon[f"H_{H}s"] = {
                "horizon_s": H,
                "window_count": 0,
                "elapsed_s": 0.0,
                "ms_per_window": 0.0,
                "convergence_rate": 0.0,
                "bounds_active_rate": 0.0,
                "median_condition_number": 0.0,
                "p90_condition_number": 0.0,
                "p95_condition_number": 0.0,
                "median_residual_reduction": 0.0,
                "median_effective_rank": 0.0,
                "mean_singular_values": [],
                "dba_norm_median": 0.0,
                "dbg_norm_median": 0.0,
                "dba_std": [0.0, 0.0, 0.0],
                "dbg_std": [0.0, 0.0, 0.0],
            }
        return results_by_horizon

    for H in horizons:
        cfg = BiasNetOptimizationConfig(horizon_s=H, max_condition_number=1e6)
        cond_nums = []
        eff_ranks = []
        red_ratios = []
        converged_list = []
        bounds_list = []
        dba_all = []
        dbg_all = []
        sigmas_all = []

        t0 = time.time()
        for w in valid_windows:
            start_i = w.start_idx
            end_i = w.end_idx + 1

            opt_res = solve_window_bias_correction(
                f_m_v_win=f_10hz[start_i:end_i],
                omega_m_v_win=w_10hz[start_i:end_i],
                timestamps_ns_win=ts_10hz[start_i:end_i],
                ref_lat_win=lat_10hz[start_i:end_i],
                ref_lon_win=lon_10hz[start_i:end_i],
                ref_alt_m_win=alt_10hz[start_i:end_i],
                ref_speed_mps_win=spd_10hz[start_i:end_i],
                ref_heading_deg_win=hdg_10hz[start_i:end_i],
                config=cfg,
            )

            cond_nums.append(opt_res.condition_number)
            eff_ranks.append(opt_res.effective_rank)
            red_ratios.append(opt_res.residual_reduction_ratio)
            converged_list.append(opt_res.converged)
            bounds_list.append(opt_res.bounds_active)
            dba_all.append(opt_res.delta_b_unconstrained[0:3])
            dbg_all.append(opt_res.delta_b_unconstrained[3:6])
            sigmas_all.append(opt_res.singular_values)

        elapsed = time.time() - t0
        cond_arr = np.array(cond_nums)
        red_arr = np.array(red_ratios)
        dba_arr = np.array(dba_all)
        dbg_arr = np.array(dbg_all)
        sig_arr = np.array(sigmas_all)

        results_by_horizon[f"H_{H}s"] = {
            "horizon_s": H,
            "window_count": len(valid_windows),
            "elapsed_s": float(elapsed),
            "ms_per_window": float((elapsed / len(valid_windows)) * 1000.0),
            "convergence_rate": float(np.mean(converged_list)),
            "bounds_active_rate": float(np.mean(bounds_list)),
            "median_condition_number": float(np.nanmedian(cond_arr)),
            "p90_condition_number": float(np.nanpercentile(cond_arr, 90)),
            "p95_condition_number": float(np.nanpercentile(cond_arr, 95)),
            "median_residual_reduction": float(np.nanmedian(red_arr)),
            "median_effective_rank": float(np.nanmedian(eff_ranks)),
            "mean_singular_values": np.mean(sig_arr, axis=0).tolist(),
            "dba_norm_median": float(np.nanmedian(np.linalg.norm(dba_arr, axis=1))),
            "dbg_norm_median": float(np.nanmedian(np.linalg.norm(dbg_arr, axis=1))),
            "dba_std": np.nanstd(dba_arr, axis=0).tolist(),
            "dbg_std": np.nanstd(dbg_arr, axis=0).tolist(),
        }
        print(f"  H={H}s ({elapsed:.1f}s): Median Cond={np.nanmedian(cond_arr):.1f}, BoundsActive={np.mean(bounds_list)*100:.1f}%, Conv={np.mean(converged_list)*100:.1f}%")

    return results_by_horizon


if __name__ == "__main__":
    root = Path(".")
    train_trip = root / "data" / "cache" / "iovnbd" / "Categorised_Vfa01.npz"
    val_trip = root / "data" / "cache" / "iovnbd" / "Categorised_M.npz"

    print("=== Methodology Audit: Driver E (Train) ===")
    res_e = run_horizon_audit_on_trip(train_trip, driver_id="Driver E", max_windows=500)

    print("\n=== Methodology Audit: Driver B (Val) ===")
    res_b = run_horizon_audit_on_trip(val_trip, driver_id="Driver B", max_windows=500)

    audit_summary = {
        "driver_e": res_e,
        "driver_b": res_b,
    }

    out_file = Path("docs/biasnet_horizon_audit.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(audit_summary, f, indent=2)
    print(f"\nSaved audit summary to {out_file}")
