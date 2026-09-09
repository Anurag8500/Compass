"""Export pipeline for VelocityNet: PyTorch -> ONNX -> LiteRT.

Generates:
1. models/velocitynet_v1_<hash>.onnx (and models/velocitynet_v1.onnx)
2. models/velocitynet_v1_<hash>.tflite (and models/velocitynet_v1.tflite)
3. Verified numerical parity between PyTorch, ONNX Runtime, and LiteRT.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Dict, Tuple

from ai_edge_litert.interpreter import Interpreter
import numpy as np
import onnx
import onnx2tf
import onnxruntime as ort
import torch

from ml.models.velocitynet import VelocityNet
from ml.training.dataset import VelocityNetDataset


def compute_file_sha256(filepath: Path | str) -> str:
    """Compute SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def export_to_onnx(
    model: VelocityNet,
    output_path: Path,
    batch_size: int = 1,
) -> Path:
    """Export PyTorch VelocityNet to standard ONNX format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    dummy_input = torch.randn(batch_size, 20, 9, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["features"],
        output_names=["speed", "log_variance"],
        opset_version=17,
        dynamo=False,
    )

    # Validate ONNX graph
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    return output_path


def convert_onnx_to_litert(
    onnx_path: Path,
    output_tflite_path: Path,
) -> Path:
    """Convert exported ONNX model to LiteRT (TFLite) using onnx2tf."""
    output_tflite_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = output_tflite_path.parent / "_temp_tflite_export"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        onnx2tf.convert(
            input_onnx_file_path=str(onnx_path),
            output_folder_path=str(temp_dir),
            keep_shape_absolutely_input_names=["features"],
            copy_onnx_input_output_names_to_tflite=True,
            non_verbose=True,
        )

        # Locate the generated float32 tflite file
        candidates = list(temp_dir.glob("*_float32.tflite"))
        if not candidates:
            candidates = list(temp_dir.glob("*.tflite"))
        if not candidates:
            raise FileNotFoundError(f"No .tflite files generated in {temp_dir}")

        shutil.copyfile(candidates[0], output_tflite_path)
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

    return output_tflite_path


def verify_export_parity(
    model: VelocityNet,
    onnx_path: Path,
    tflite_path: Path,
    test_batch: np.ndarray,
    onnx_tol: float = 1e-4,
    litert_tol: float = 1e-3,
) -> Dict[str, object]:
    """Verify numerical parity across PyTorch, ONNX Runtime, and LiteRT."""
    model.eval()
    num_samples = len(test_batch)

    # 1. PyTorch inference
    with torch.no_grad():
        pt_in = torch.from_numpy(test_batch).float()
        pt_speeds, pt_log_vars = [], []
        # Run in single-window mode to match batch=1 LiteRT
        for i in range(num_samples):
            sp, lv = model(pt_in[i : i + 1])
            pt_speeds.append(sp.item())
            pt_log_vars.append(lv.item())
    pt_speeds_arr = np.array(pt_speeds, dtype=np.float32)
    pt_log_vars_arr = np.array(pt_log_vars, dtype=np.float32)

    # 2. ONNX Runtime inference
    ort_session = ort.InferenceSession(str(onnx_path))
    ort_speeds, ort_log_vars = [], []
    for i in range(num_samples):
        out = ort_session.run(None, {"features": test_batch[i : i + 1].astype(np.float32)})
        ort_speeds.append(float(out[0].flatten()[0]))
        ort_log_vars.append(float(out[1].flatten()[0]))
    ort_speeds_arr = np.array(ort_speeds, dtype=np.float32)
    ort_log_vars_arr = np.array(ort_log_vars, dtype=np.float32)

    onnx_speed_max_err = float(np.max(np.abs(pt_speeds_arr - ort_speeds_arr)))
    onnx_speed_mean_err = float(np.mean(np.abs(pt_speeds_arr - ort_speeds_arr)))
    onnx_lv_max_err = float(np.max(np.abs(pt_log_vars_arr - ort_log_vars_arr)))
    onnx_lv_mean_err = float(np.mean(np.abs(pt_log_vars_arr - ort_log_vars_arr)))

    # 3. LiteRT inference
    interp = Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_dets = interp.get_output_details()

    litert_speeds, litert_log_vars = [], []
    for i in range(num_samples):
        interp.set_tensor(in_det["index"], test_batch[i : i + 1].astype(np.float32))
        interp.invoke()
        outs = {o["name"]: interp.get_tensor(o["index"]).flatten()[0] for o in out_dets}
        litert_speeds.append(float(outs["speed"]))
        litert_log_vars.append(float(outs["log_variance"]))
    litert_speeds_arr = np.array(litert_speeds, dtype=np.float32)
    litert_log_vars_arr = np.array(litert_log_vars, dtype=np.float32)

    litert_speed_max_err = float(np.max(np.abs(pt_speeds_arr - litert_speeds_arr)))
    litert_speed_mean_err = float(np.mean(np.abs(pt_speeds_arr - litert_speeds_arr)))
    litert_lv_max_err = float(np.max(np.abs(pt_log_vars_arr - litert_log_vars_arr)))
    litert_lv_mean_err = float(np.mean(np.abs(pt_log_vars_arr - litert_log_vars_arr)))

    onnx_pass = (onnx_speed_max_err <= onnx_tol) and (onnx_lv_max_err <= onnx_tol)
    litert_pass = (litert_speed_max_err <= litert_tol) and (litert_lv_max_err <= litert_tol)

    parity_report = {
        "num_test_samples": num_samples,
        "onnx_parity": {
            "speed_max_abs_err": onnx_speed_max_err,
            "speed_mean_abs_err": onnx_speed_mean_err,
            "log_var_max_abs_err": onnx_lv_max_err,
            "log_var_mean_abs_err": onnx_lv_mean_err,
            "tolerance": onnx_tol,
            "passed": onnx_pass,
        },
        "litert_parity": {
            "speed_max_abs_err": litert_speed_max_err,
            "speed_mean_abs_err": litert_speed_mean_err,
            "log_var_max_abs_err": litert_lv_max_err,
            "log_var_mean_abs_err": litert_lv_mean_err,
            "tolerance": litert_tol,
            "passed": litert_pass,
        },
        "all_passed": onnx_pass and litert_pass,
    }

    if not onnx_pass:
        raise AssertionError(f"ONNX parity failed: max error {onnx_speed_max_err} exceeds tolerance {onnx_tol}")
    if not litert_pass:
        raise AssertionError(f"LiteRT parity failed: max error {litert_speed_max_err} exceeds tolerance {litert_tol}")

    return parity_report


def run_full_export(
    checkpoint_path: str = "models/velocitynet_v1_best.pt",
) -> Dict[str, object]:
    """Execute complete export pipeline and save versioned artifacts."""
    project_root = Path(__file__).resolve().parents[2]
    ckpt_file = project_root / checkpoint_path
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

    print(f"[*] Loading trained checkpoint: {ckpt_file}...")
    checkpoint = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("config", {})

    model = VelocityNet(
        input_dim=cfg.get("input_dim", 9),
        hidden_dim=cfg.get("hidden_dim", 64),
        num_layers=cfg.get("num_layers", 2),
        dense_dim=cfg.get("dense_dim", 32),
        dropout=cfg.get("dropout", 0.2),
        min_log_var=cfg.get("min_log_var", -10.0),
        max_log_var=cfg.get("max_log_var", 10.0),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Determine dataset hash
    manifest_path = project_root / "data" / "ml_dataset_v1" / "dataset_manifest.json"
    data_hash = compute_file_sha256(manifest_path)[:8]
    print(f"[*] Dataset Manifest SHA-256 tag: {data_hash}")

    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Export ONNX
    onnx_versioned = models_dir / f"velocitynet_v1_{data_hash}.onnx"
    onnx_canonical = models_dir / "velocitynet_v1.onnx"
    print(f"[*] Exporting to ONNX: {onnx_versioned}...")
    export_to_onnx(model, onnx_versioned, batch_size=1)
    shutil.copyfile(onnx_versioned, onnx_canonical)

    # 2. Convert to LiteRT
    tflite_versioned = models_dir / f"velocitynet_v1_{data_hash}.tflite"
    tflite_canonical = models_dir / "velocitynet_v1.tflite"
    print(f"[*] Converting to LiteRT (TFLite): {tflite_versioned}...")
    convert_onnx_to_litert(onnx_versioned, tflite_versioned)
    shutil.copyfile(tflite_versioned, tflite_canonical)

    # 3. Verify parity on held-out test windows
    test_path = project_root / "data" / "ml_dataset_v1" / "test.npz"
    print(f"[*] Verifying parity on 100 held-out test windows...")
    test_ds = VelocityNetDataset(test_path, split_name="test")
    test_batch = test_ds.features[:100]

    parity_report = verify_export_parity(
        model=model,
        onnx_path=onnx_versioned,
        tflite_path=tflite_versioned,
        test_batch=test_batch,
        onnx_tol=1e-4,
        litert_tol=1e-3,
    )
    print("    --> ONNX Parity: PASSED (Max Diff: "
          f"{parity_report['onnx_parity']['speed_max_abs_err']:.2e} m/s)")
    print("    --> LiteRT Parity: PASSED (Max Diff: "
          f"{parity_report['litert_parity']['speed_max_abs_err']:.2e} m/s)")

    # 4. Save model configuration artifact
    norm_path = project_root / "data" / "ml_dataset_v1" / "normalization.json"
    norm_hash = compute_file_sha256(norm_path)

    model_config = {
        "model_name": "VelocityNet",
        "version": "v1.0",
        "dataset_manifest_hash": data_hash,
        "normalization_sha256": norm_hash,
        "input_contract": {
            "tensor_shape": [1, 20, 9],
            "batch_mode": "Fixed single-window deployment (B=1) for real-time edge streaming; training supports arbitrary batch sizes",
            "sequence_length": 20,
            "feature_dim": 9,
            "sample_rate_hz": 10.0,
            "window_duration_s": 2.0,
            "stride_s": 0.5,
            "coordinate_frame": "vehicle_FLU",
            "gravity_preserved": True,
            "feature_order": [
                "f_x_v", "f_y_v", "f_z_v",
                "omega_x_v", "omega_y_v", "omega_z_v",
                "norm_f_v", "norm_f_dot_v", "norm_omega_v"
            ],
        },
        "output_contract": {
            "outputs": [
                {"name": "speed", "unit": "m/s", "meaning": "Predicted forward driving speed mu_v"},
                {"name": "log_variance", "unit": "unitless", "meaning": "Predicted log variance log(sigma_v^2)"},
            ],
            "clamping": {"min_log_var": -10.0, "max_log_var": 10.0},
        },
        "architecture": {
            "type": "GRU",
            "input_size": 9,
            "hidden_size": 64,
            "num_layers": 2,
            "dense_dim": 32,
            "output_dim": 2,
            "parameters": sum(p.numel() for p in model.parameters()),
        },
        "artifacts": {
            "pytorch_checkpoint": str(ckpt_file.name),
            "onnx_model": onnx_versioned.name,
            "litert_model": tflite_versioned.name,
            "onnx_sha256": compute_file_sha256(onnx_versioned),
            "litert_sha256": compute_file_sha256(tflite_versioned),
        },
        "parity_summary": parity_report,
    }

    model_config_path = models_dir / "model_config_velocitynet_v1.json"
    with open(model_config_path, "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2)
    print(f"[*] Saved frozen model config to {model_config_path}")

    parity_json_path = models_dir / "velocitynet_v1_export_parity.json"
    with open(parity_json_path, "w", encoding="utf-8") as f:
        json.dump(parity_report, f, indent=2)

    return model_config


if __name__ == "__main__":
    run_full_export()
