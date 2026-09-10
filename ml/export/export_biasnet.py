"""Export pipeline for BiasNet: PyTorch -> ONNX -> LiteRT (Phase 8).

Generates:
1. models/biasnet_v1.onnx
2. models/biasnet_v1.tflite
3. Verified numerical parity between PyTorch, ONNX Runtime, and LiteRT on real driving windows.
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

from ml.models.biasnet import BiasNet


def compute_file_sha256(filepath: Path | str) -> str:
    """Compute SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def export_to_onnx(
    model: BiasNet,
    output_path: Path,
    batch_size: int = 1,
) -> Path:
    """Export PyTorch BiasNet to standard ONNX format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    dummy_input = torch.randn(batch_size, 20, 9, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["features"],
        output_names=["bias_correction"],
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
    temp_dir = output_tflite_path.parent / "_temp_tflite_biasnet_export"
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
    model: BiasNet,
    onnx_path: Path,
    tflite_path: Path,
    test_batch: np.ndarray,
    onnx_tol: float = 1e-4,
    litert_tol: float = 1e-3,
) -> Dict[str, object]:
    """Verify numerical parity across PyTorch, ONNX Runtime, and LiteRT on real windows."""
    model.eval()
    num_samples = len(test_batch)

    # 1. PyTorch inference
    with torch.no_grad():
        pt_in = torch.from_numpy(test_batch).float()
        pt_outs = []
        for i in range(num_samples):
            pred = model(pt_in[i : i + 1])
            pt_outs.append(pred.cpu().numpy().flatten())
    pt_arr = np.array(pt_outs, dtype=np.float32)  # (N, 6)

    # 2. ONNX Runtime inference
    ort_session = ort.InferenceSession(str(onnx_path))
    ort_outs = []
    for i in range(num_samples):
        out = ort_session.run(None, {"features": test_batch[i : i + 1].astype(np.float32)})
        ort_outs.append(out[0].flatten())
    ort_arr = np.array(ort_outs, dtype=np.float32)

    onnx_max_err = float(np.max(np.abs(pt_arr - ort_arr)))
    onnx_mean_err = float(np.mean(np.abs(pt_arr - ort_arr)))

    # Per-component errors
    dim_names = ["dba_x", "dba_y", "dba_z", "dbg_x", "dbg_y", "dbg_z"]
    onnx_comp_err = {
        dim_names[j]: float(np.max(np.abs(pt_arr[:, j] - ort_arr[:, j]))) for j in range(6)
    }

    # 3. LiteRT inference
    interp = Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_dets = interp.get_output_details()[0]

    litert_outs = []
    for i in range(num_samples):
        interp.set_tensor(in_det["index"], test_batch[i : i + 1].astype(np.float32))
        interp.invoke()
        tfl_out = interp.get_tensor(out_dets["index"]).flatten()
        litert_outs.append(tfl_out)
    litert_arr = np.array(litert_outs, dtype=np.float32)

    litert_max_err = float(np.max(np.abs(pt_arr - litert_arr)))
    litert_mean_err = float(np.mean(np.abs(pt_arr - litert_arr)))
    litert_comp_err = {
        dim_names[j]: float(np.max(np.abs(pt_arr[:, j] - litert_arr[:, j]))) for j in range(6)
    }

    onnx_pass = onnx_max_err <= onnx_tol
    litert_pass = litert_max_err <= litert_tol

    parity_report = {
        "num_test_samples": num_samples,
        "onnx_parity": {
            "max_abs_err": onnx_max_err,
            "mean_abs_err": onnx_mean_err,
            "component_max_err": onnx_comp_err,
            "tolerance": onnx_tol,
            "passed": onnx_pass,
        },
        "litert_parity": {
            "max_abs_err": litert_max_err,
            "mean_abs_err": litert_mean_err,
            "component_max_err": litert_comp_err,
            "tolerance": litert_tol,
            "passed": litert_pass,
        },
        "overall_passed": bool(onnx_pass and litert_pass),
    }

    return parity_report


def export_biasnet_pipeline(
    checkpoint_path: Path | str = "models/biasnet_v1_best.pt",
    output_dir: Path | str = "models",
    val_dataset_path: Path | str = "data/ml_dataset_biasnet_v1/bias_validation.npz",
    num_eval_samples: int = 500,
) -> Dict[str, object]:
    """Execute end-to-end export and validation."""
    root = Path(".")
    ckpt = root / checkpoint_path
    out = root / output_dir
    val_p = root / val_dataset_path

    print("=== Exporting BiasNet: PyTorch -> ONNX -> LiteRT ===")
    model = BiasNet(input_dim=9, hidden_dim=48, num_layers=2, dense_dim=24)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval()

    onnx_path = out / "biasnet_v1.onnx"
    tflite_path = out / "biasnet_v1.tflite"

    # Export ONNX
    print(f"Exporting to ONNX: {onnx_path}")
    export_to_onnx(model, onnx_path, batch_size=1)
    onnx_hash = compute_file_sha256(onnx_path)
    print(f"  ONNX SHA-256: {onnx_hash}")

    # Convert to LiteRT
    print(f"Converting to LiteRT: {tflite_path}")
    convert_onnx_to_litert(onnx_path, tflite_path)
    tflite_hash = compute_file_sha256(tflite_path)
    print(f"  LiteRT SHA-256: {tflite_hash}")

    # Verify parity on 500 real windows
    val_d = np.load(val_p)
    real_windows = val_d["X"][:num_eval_samples].astype(np.float32)
    print(f"Verifying numerical parity across {len(real_windows)} real windows...")

    parity = verify_export_parity(model, onnx_path, tflite_path, real_windows)
    print(f"  ONNX Parity: Max Error = {parity['onnx_parity']['max_abs_err']:.2e}, Passed = {parity['onnx_parity']['passed']}")
    print(f"  LiteRT Parity: Max Error = {parity['litert_parity']['max_abs_err']:.2e}, Passed = {parity['litert_parity']['passed']}")

    summary = {
        "model": "BiasNet v1",
        "onnx_path": str(onnx_path),
        "onnx_sha256": onnx_hash,
        "tflite_path": str(tflite_path),
        "tflite_sha256": tflite_hash,
        "parity": parity,
    }

    parity_out = out / "biasnet_v1_export_parity.json"
    with open(parity_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved parity report to {parity_out}")

    return summary


if __name__ == "__main__":
    export_biasnet_pipeline()
