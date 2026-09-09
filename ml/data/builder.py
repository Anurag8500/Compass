"""End-to-End ML Dataset Builder.

COMPASS Phase 6 — ML Dataset Construction.
Executes the unified transform chain from raw Phase 2 synchronized trips to
leakage-audited, normalized (20, 9) feature tensors and causal labels.

Data Flow:
    Phase 2 Synchronized Trip
               |
    Phase 3 Classical Preprocessing (Calibration, Mounting Alignment, Filtering)
               |
    Canonical 10 Hz Resampling / Decimation
               |
    9-Channel Vehicle-Frame Feature Engine (No gravity removal)
               |
    Strictly Causal 20-Sample Windowing (2.0s window, 0.5s stride)
               |
    Driver/File-Level Split (Driver E -> Train, Driver B -> Val, Driver A -> Test)
               |
    Training-Only Normalization (Zero Val/Test leakage)
               |
    VelocityNet Window-End Causal Speed Labels
               |
    Shared BiasNet Outage-Exclusion Plumbing
               |
    Versioned Serialization (.npz + ModelConfig + Dataset Manifest)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from data.pipeline.stationary_detect import StationaryDetector
from data.pipeline.sync import SynchronizedTrip
from navigation.preprocessing.pipeline import PreprocessingPipeline
from ml.data.exclusion_rules import evaluate_biasnet_eligibility
from ml.data.features import compute_canonical_features
from ml.data.normalization import FeatureNormalizer
from ml.data.resample import resample_to_canonical_10hz
from ml.data.split import DriverFileSplit
from ml.data.velocitynet_labels import extract_velocitynet_labels
from ml.data.windowing import extract_causal_windows, ExtractedWindow


@dataclass
class TripProcessingResult:
    """Extracted feature windows and aligned labels for a single trip."""
    file_id: str
    driver_id: str
    split: str
    windows: np.ndarray             # (M, 20, 9) float32 unnormalized features
    labels_speed: np.ndarray        # (M,) float32 causally smoothed speed
    labels_speed_raw: np.ndarray    # (M,) float32 raw reference speed
    timestamps_end_ns: np.ndarray   # (M,) int64 window end timestamps
    timestamps_start_ns: np.ndarray # (M,) int64 window start timestamps
    is_valid_mask: np.ndarray       # (M,) bool overall window validity
    biasnet_eligible: np.ndarray    # (M,) bool BiasNet eligibility
    biasnet_reasons: List[str]      # (M,) explanatory string codes


def process_trip_to_windows(
    trip: SynchronizedTrip,
    file_id: str,
    driver_id: str,
    split_name: str,
    pipeline: PreprocessingPipeline,
    detector: StationaryDetector,
) -> TripProcessingResult:
    """Run Phase 3 preprocessing, feature extraction, causal windowing, and labeling on one trip."""
    # 1. Phase 3 Preprocessing
    _, stat_mask = detector.detect(trip.timestamps_ns, trip.accel_raw, trip.gyro_raw)
    preprocessed = pipeline.process_trip(trip, stationary_mask=stat_mask)

    # 2. Resample / decimate to canonical 10 Hz
    # Pass v_ref_speed_mps as an auxiliary signal so it is synchronized identically
    aux = {"v_ref_speed_mps": trip.v_ref_speed_mps}
    res_result = resample_to_canonical_10hz(
        timestamps_ns=preprocessed.timestamps_ns,
        f_m_v=preprocessed.f_m_v,
        omega_m_v=preprocessed.omega_m_v,
        is_validated=preprocessed.is_validated,
        aux_signals=aux,
    )

    ts_10hz = res_result.timestamps_ns
    f_10hz = res_result.f_m_v
    omega_10hz = res_result.omega_m_v
    val_10hz = res_result.is_validated
    speed_10hz = res_result.aux_signals["v_ref_speed_mps"]

    # 3. 9-Channel Feature Computation
    features_9ch = compute_canonical_features(ts_10hz, f_10hz, omega_10hz)

    # 4. Strictly Causal Windowing (20 samples, 0.5s stride = 5 samples)
    extracted_windows = extract_causal_windows(
        features=features_9ch,
        timestamps_ns=ts_10hz,
        is_validated=val_10hz,
        source_file_id=file_id,
        driver_id=driver_id,
        window_size=20,
        stride_samples=5,
    )

    m_windows = len(extracted_windows)
    if m_windows == 0:
        return TripProcessingResult(
            file_id=file_id,
            driver_id=driver_id,
            split=split_name,
            windows=np.zeros((0, 20, 9), dtype=np.float32),
            labels_speed=np.zeros(0, dtype=np.float32),
            labels_speed_raw=np.zeros(0, dtype=np.float32),
            timestamps_end_ns=np.zeros(0, dtype=np.int64),
            timestamps_start_ns=np.zeros(0, dtype=np.int64),
            is_valid_mask=np.zeros(0, dtype=bool),
            biasnet_eligible=np.zeros(0, dtype=bool),
            biasnet_reasons=[],
        )

    end_indices = [w.end_idx for w in extracted_windows]

    # 5. VelocityNet Causal Labels at Window Ends
    label_batch = extract_velocitynet_labels(
        timestamps_ns=ts_10hz,
        v_ref_speed_mps=speed_10hz,
        window_end_indices=end_indices,
        is_validated=val_10hz,
        smoothing_window_size=3,
    )

    # 6. Window Arrays & BiasNet Eligibility
    window_arr = np.zeros((m_windows, 20, 9), dtype=np.float32)
    ts_end_arr = np.zeros(m_windows, dtype=np.int64)
    ts_start_arr = np.zeros(m_windows, dtype=np.int64)
    valid_arr = np.zeros(m_windows, dtype=bool)
    bn_eligible_arr = np.zeros(m_windows, dtype=bool)
    bn_reasons: List[str] = []

    for i, w in enumerate(extracted_windows):
        window_arr[i] = w.window.astype(np.float32)
        ts_end_arr[i] = w.end_timestamp_ns
        ts_start_arr[i] = w.start_timestamp_ns

        # Window is valid if both sensor window and ground-truth label are valid
        overall_valid = w.is_valid and bool(label_batch.is_valid_label[i])
        valid_arr[i] = overall_valid

        bn_eval = evaluate_biasnet_eligibility(
            window_start_timestamp_ns=w.start_timestamp_ns,
            window_end_timestamp_ns=w.end_timestamp_ns,
            is_window_valid=overall_valid,
            real_outage_windows=None,  # Verified absent in IO-VNBD
        )
        bn_eligible_arr[i] = bn_eval.eligible
        bn_reasons.append(bn_eval.reason)

    return TripProcessingResult(
        file_id=file_id,
        driver_id=driver_id,
        split=split_name,
        windows=window_arr,
        labels_speed=label_batch.speed_smoothed_mps,
        labels_speed_raw=label_batch.speed_raw_mps,
        timestamps_end_ns=ts_end_arr,
        timestamps_start_ns=ts_start_arr,
        is_valid_mask=valid_arr,
        biasnet_eligible=bn_eligible_arr,
        biasnet_reasons=bn_reasons,
    )


def build_full_dataset(
    project_root: Path | str = ".",
    split_json_path: Path | str = "data/splits/split_v1.json",
    output_dir: Path | str = "data/ml_dataset_v1",
) -> Dict[str, Any]:
    """Execute end-to-end dataset generation over all assigned IO-VNBD files."""
    root = Path(project_root).resolve()
    split = DriverFileSplit.from_json(root / split_json_path)
    out_path = root / output_dir
    out_path.mkdir(parents=True, exist_ok=True)

    pipeline = PreprocessingPipeline(sampling_rate_hz=10.0, filter_cutoff_hz=3.0, median_window_size=3)
    detector = StationaryDetector()

    split_results: Dict[str, List[TripProcessingResult]] = {
        "train": [],
        "validation": [],
        "test": [],
    }

    total_files = len(split.train_files) + len(split.validation_files) + len(split.test_files)
    processed_count = 0
    start_time = time.time()

    print(f"[DatasetBuilder] Starting build for {total_files} files across 3 splits...")

    # Process all files by split
    for split_name in ["train", "validation", "test"]:
        files = getattr(split, f"{split_name}_files")
        print(f"[DatasetBuilder] Processing {len(files)} files for split: {split_name.upper()}")

        for rel_file in files:
            npz_path = root / rel_file
            if not npz_path.exists():
                raise FileNotFoundError(f"Cached trip file not found at {npz_path}")

            trip = SynchronizedTrip.load_npz(npz_path)
            driver = split.driver_mapping.get(rel_file, split.driver_mapping.get(npz_path.stem, "Unknown"))

            res = process_trip_to_windows(
                trip=trip,
                file_id=rel_file,
                driver_id=driver,
                split_name=split_name,
                pipeline=pipeline,
                detector=detector,
            )
            split_results[split_name].append(res)
            processed_count += 1
            if processed_count % 20 == 0 or processed_count == total_files:
                elapsed = time.time() - start_time
                print(f"  Processed {processed_count}/{total_files} files ({elapsed:.1f}s)")

    # Consolidate arrays per split
    consolidated: Dict[str, Dict[str, np.ndarray]] = {}
    for split_name in ["train", "validation", "test"]:
        t_res = split_results[split_name]
        if not t_res:
            continue

        c_X = np.concatenate([r.windows for r in t_res], axis=0)
        c_y_speed = np.concatenate([r.labels_speed for r in t_res], axis=0)
        c_y_speed_raw = np.concatenate([r.labels_speed_raw for r in t_res], axis=0)
        c_ts_end = np.concatenate([r.timestamps_end_ns for r in t_res], axis=0)
        c_ts_start = np.concatenate([r.timestamps_start_ns for r in t_res], axis=0)
        c_valid = np.concatenate([r.is_valid_mask for r in t_res], axis=0)
        c_bn_elig = np.concatenate([r.biasnet_eligible for r in t_res], axis=0)

        # String arrays
        c_files = []
        c_drivers = []
        for r in t_res:
            c_files.extend([r.file_id] * len(r.windows))
            c_drivers.extend([r.driver_id] * len(r.windows))

        consolidated[split_name] = {
            "X": c_X,
            "y_speed": c_y_speed,
            "y_speed_raw": c_y_speed_raw,
            "timestamps_end_ns": c_ts_end,
            "timestamps_start_ns": c_ts_start,
            "is_valid": c_valid,
            "biasnet_eligible": c_bn_elig,
            "source_file_ids": np.array(c_files, dtype=object),
            "driver_ids": np.array(c_drivers, dtype=object),
        }

    # 7. Compute Normalization Parameters on TRAIN SPLIT ONLY
    train_valid_mask = consolidated["train"]["is_valid"]
    train_valid_X = consolidated["train"]["X"][train_valid_mask]

    print(f"[DatasetBuilder] Fitting FeatureNormalizer strictly on {len(train_valid_X)} valid TRAIN windows...")
    normalizer = FeatureNormalizer.fit(train_valid_X)
    normalizer.save_json(out_path / "normalization.json")

    # Also save as ModelConfig for Phase 7/ONNX/LiteRT compatibility
    model_cfg = normalizer.to_model_config(model_name="VelocityNet")
    model_cfg.to_file(out_path / "model_config.json")

    # Save normalized tensors and serialized splits
    for split_name in ["train", "validation", "test"]:
        d = consolidated[split_name]
        raw_X = d["X"]
        norm_X = normalizer.transform(raw_X).astype(np.float32)

        npz_target = out_path / f"{split_name}.npz"
        np.savez_compressed(
            npz_target,
            X=norm_X,
            X_raw=raw_X,
            y_speed=d["y_speed"],
            y_speed_raw=d["y_speed_raw"],
            timestamps_end_ns=d["timestamps_end_ns"],
            timestamps_start_ns=d["timestamps_start_ns"],
            is_valid=d["is_valid"],
            biasnet_eligible=d["biasnet_eligible"],
            source_file_ids=d["source_file_ids"],
            driver_ids=d["driver_ids"],
        )
        print(f"[DatasetBuilder] Saved {split_name}.npz: {len(norm_X)} windows (Valid: {np.sum(d['is_valid'])})")

    # Build dataset manifest
    manifest_data = {
        "dataset_version": "v1.0",
        "split_version": split.version,
        "sample_rate_hz": 10.0,
        "window_duration_s": 2.0,
        "window_size_samples": 20,
        "stride_duration_s": 0.5,
        "stride_samples": 5,
        "channel_count": 9,
        "channel_order": list(normalizer.channel_names),
        "total_source_files_processed": total_files,
        "files_per_split": {
            "train": len(split.train_files),
            "validation": len(split.validation_files),
            "test": len(split.test_files),
        },
        "drivers_per_split": {
            "train": list({split.driver_mapping[f] for f in split.train_files}),
            "validation": list({split.driver_mapping[f] for f in split.validation_files}),
            "test": list({split.driver_mapping[f] for f in split.test_files}),
        },
        "windows_per_split": {
            s: int(len(consolidated[s]["X"])) for s in ["train", "validation", "test"]
        },
        "valid_windows_per_split": {
            s: int(np.sum(consolidated[s]["is_valid"])) for s in ["train", "validation", "test"]
        },
        "normalization": normalizer.to_dict(),
        "outage_status": "NO_REAL_OUTAGES_IN_DATASET",
        "biasnet_eligible_windows": {
            s: int(np.sum(consolidated[s]["biasnet_eligible"])) for s in ["train", "validation", "test"]
        },
    }

    with open(out_path / "dataset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"[DatasetBuilder] Complete! Manifest saved to {out_path / 'dataset_manifest.json'}")
    return manifest_data
