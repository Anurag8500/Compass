"""Held-out Driver A direct label-space evaluation for BiasNet v1 (Phase 8).

Executed strictly ONCE after freezing all model, horizon, and training choices.
Evaluates BiasNet against Zero Baseline and Train Mean Baseline on Driver A.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import torch

from data.pipeline.sync import SynchronizedTrip
from data.pipeline.stationary_detect import StationaryDetector
from navigation.preprocessing.pipeline import PreprocessingPipeline
from ml.data.features import compute_canonical_features
from ml.data.normalization import FeatureNormalizer
from ml.data.resample import resample_to_canonical_10hz
from ml.data.split import DriverFileSplit
from ml.data.windowing import extract_causal_windows
from ml.data.biasnet_labels import (
    BiasNetOptimizationConfig,
    solve_window_bias_correction,
)
from ml.models.biasnet import BiasNet
from ml.training.train_biasnet import compute_component_metrics


def evaluate_driver_a() -> Dict[str, Any]:
    root = Path(".")
    split_path = root / "data" / "splits" / "split_v1.json"
    split = DriverFileSplit.from_json(split_path)

    test_files = split.test_files[:5]  # Representative multi-trip evaluation on Driver A

    cfg = BiasNetOptimizationConfig(
        horizon_s=1.0,
        bound_accel_mps2=2.0,
        bound_gyro_rads=0.15,
        max_condition_number=50.0,
        min_residual_reduction=1.20,
    )

    pipeline = PreprocessingPipeline(sampling_rate_hz=10.0, filter_cutoff_hz=3.0, median_window_size=3)
    detector = StationaryDetector()
    norm_path = root / "data" / "ml_dataset_v1" / "normalization.json"
    normalizer = FeatureNormalizer.load_json(norm_path)

    # Load frozen model
    model_pt = root / "models" / "biasnet_v1_best.pt"
    model = BiasNet(input_dim=9, hidden_dim=48, num_layers=2, dense_dim=24)
    model.load_state_dict(torch.load(model_pt, map_location="cpu", weights_only=True))
    model.eval()

    # Load train mean for baseline
    train_data = np.load(root / "data" / "ml_dataset_biasnet_v1" / "bias_train.npz")
    train_mean = np.mean(train_data["labels_constrained"][train_data["is_eligible"]], axis=0)

    print(f"=== Held-Out Test Evaluation: Driver A ({len(test_files)} trips) ===")
    all_X = []
    all_y_true = []

    t0 = time.time()
    for rel_f in test_files:
        full_path = root / rel_f
        trip = SynchronizedTrip.load_npz(full_path)
        _, stat_mask = detector.detect(trip.timestamps_ns, trip.accel_raw, trip.gyro_raw)
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

        features = compute_canonical_features(res.timestamps_ns, res.f_m_v, res.omega_m_v)
        windows = extract_causal_windows(
            features=features,
            timestamps_ns=res.timestamps_ns,
            is_validated=res.is_validated,
            source_file_id=rel_f,
            driver_id="Driver A",
            window_size=20,
            stride_samples=5,
        )

        valid_windows = [w for w in windows if w.is_valid][:300]
        print(f"  [{Path(rel_f).stem}] Solving targets on {len(valid_windows)} windows...")

        for w in valid_windows:
            s_i = w.start_idx
            e_i = w.end_idx + 1

            opt_res = solve_window_bias_correction(
                f_m_v_win=res.f_m_v[s_i:e_i],
                omega_m_v_win=res.omega_m_v[s_i:e_i],
                timestamps_ns_win=res.timestamps_ns[s_i:e_i],
                ref_lat_win=res.aux_signals["v_ref_lat"][s_i:e_i],
                ref_lon_win=res.aux_signals["v_ref_lon"][s_i:e_i],
                ref_alt_m_win=res.aux_signals["v_ref_alt_m"][s_i:e_i],
                ref_speed_mps_win=res.aux_signals["v_ref_speed_mps"][s_i:e_i],
                ref_heading_deg_win=res.aux_signals["v_ref_heading_deg"][s_i:e_i],
                config=cfg,
            )

            if opt_res.is_eligible:
                all_X.append(w.window)
                all_y_true.append(opt_res.delta_b_constrained)

    all_X_arr = np.array(all_X, dtype=np.float32)
    all_y_arr = np.array(all_y_true, dtype=np.float32)
    print(f"Total eligible Driver A test windows: {len(all_X_arr)} in {time.time() - t0:.1f}s")

    # Run BiasNet inference
    norm_X = normalizer.transform(all_X_arr).astype(np.float32)
    with torch.no_grad():
        biasnet_pred = model(torch.tensor(norm_X)).cpu().numpy()

    zero_pred = np.zeros_like(all_y_arr)
    mean_pred = np.tile(train_mean, (len(all_y_arr), 1))

    biasnet_metrics = compute_component_metrics(biasnet_pred, all_y_arr)
    zero_metrics = compute_component_metrics(zero_pred, all_y_arr)
    mean_metrics = compute_component_metrics(mean_pred, all_y_arr)

    print("\n=== Held-Out Test Direct Metrics on Driver A ===")
    print(f"{'Metric':<25} | {'Zero Baseline':<15} | {'Train Mean':<15} | {'BiasNet v1':<15}")
    print("-" * 75)
    print(f"{'Total Vector RMSE':<25} | {zero_metrics['summary']['total_vector_rmse']:<15.4f} | {mean_metrics['summary']['total_vector_rmse']:<15.4f} | {biasnet_metrics['summary']['total_vector_rmse']:<15.4f}")
    print(f"{'Accel Vector RMSE':<25} | {zero_metrics['summary']['accel_vector_rmse']:<15.4f} | {mean_metrics['summary']['accel_vector_rmse']:<15.4f} | {biasnet_metrics['summary']['accel_vector_rmse']:<15.4f}")
    print(f"{'Gyro Vector RMSE':<25} | {zero_metrics['summary']['gyro_vector_rmse']:<15.4f} | {mean_metrics['summary']['gyro_vector_rmse']:<15.4f} | {biasnet_metrics['summary']['gyro_vector_rmse']:<15.4f}")
    print("-" * 75)

    results = {
        "test_driver": "Driver A (Held-out)",
        "evaluated_trips": [str(f) for f in test_files],
        "window_count": len(all_X_arr),
        "biasnet": biasnet_metrics,
        "zero_baseline": zero_metrics,
        "train_mean_baseline": mean_metrics,
    }

    out_file = root / "docs" / "biasnet_driver_a_direct_metrics.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved Driver A evaluation to {out_file}")

    return results


if __name__ == "__main__":
    evaluate_driver_a()
