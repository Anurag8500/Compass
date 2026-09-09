"""Integration tests for VelocityNet ONNX & LiteRT export and numerical parity."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import numpy as np
import pytest
import torch

from ai_edge_litert.interpreter import Interpreter
import onnx
import onnxruntime as ort

from ml.models.velocitynet import VelocityNet
from ml.export.export_velocitynet import (
    export_to_onnx,
    convert_onnx_to_litert,
    verify_export_parity,
    compute_file_sha256,
)


@pytest.fixture
def dummy_model() -> VelocityNet:
    """Create a deterministic VelocityNet instance for parity testing."""
    torch.manual_seed(42)
    model = VelocityNet(
        input_dim=9,
        hidden_dim=32,  # Compact for quick test execution
        num_layers=2,
        dense_dim=16,
        dropout=0.0,
    )
    model.eval()
    return model


@pytest.fixture
def sample_inputs() -> np.ndarray:
    """Create reproducible sample input tensors (N, 20, 9)."""
    np.random.seed(42)
    return np.random.randn(5, 20, 9).astype(np.float32)


def test_onnx_export_and_validity(dummy_model: VelocityNet, sample_inputs: np.ndarray):
    """Test exporting VelocityNet to ONNX and verifying graph validity."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        onnx_path = Path(tmp_dir) / "test_velocitynet.onnx"
        export_to_onnx(dummy_model, onnx_path, batch_size=1)

        assert onnx_path.exists(), "ONNX file was not created"
        assert onnx_path.stat().st_size > 1000, "ONNX file unexpectedly small"

        # Check onnx graph validity
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)

        # Verify ONNX Runtime produces matching predictions
        session = ort.InferenceSession(str(onnx_path))
        for i in range(len(sample_inputs)):
            x_np = sample_inputs[i : i + 1]
            ort_out = session.run(None, {"features": x_np})
            assert len(ort_out) == 2, "Expected 2 outputs: speed and log_variance"

            # PyTorch reference
            with torch.no_grad():
                pt_speed, pt_log_var = dummy_model(torch.from_numpy(x_np))

            speed_diff = float(np.abs(ort_out[0].item() - pt_speed.item()))
            lv_diff = float(np.abs(ort_out[1].item() - pt_log_var.item()))

            assert speed_diff < 1e-4, f"ONNX speed mismatch: {speed_diff}"
            assert lv_diff < 1e-4, f"ONNX log_var mismatch: {lv_diff}"


def test_litert_conversion_and_parity(dummy_model: VelocityNet, sample_inputs: np.ndarray):
    """Test full conversion to LiteRT and numerical parity verification."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        onnx_path = Path(tmp_dir) / "test_model.onnx"
        tflite_path = Path(tmp_dir) / "test_model.tflite"

        export_to_onnx(dummy_model, onnx_path, batch_size=1)
        convert_onnx_to_litert(onnx_path, tflite_path)

        assert tflite_path.exists(), "LiteRT .tflite file was not created"
        assert tflite_path.stat().st_size > 1000, "LiteRT file unexpectedly small"

        parity_report = verify_export_parity(
            model=dummy_model,
            onnx_path=onnx_path,
            tflite_path=tflite_path,
            test_batch=sample_inputs,
            onnx_tol=1e-4,
            litert_tol=1e-3,
        )

        assert parity_report["onnx_parity"]["passed"] is True, f"ONNX parity failed: {parity_report['onnx_parity']}"
        assert parity_report["onnx_parity"]["speed_max_abs_err"] <= 1e-4
        assert parity_report["onnx_parity"]["log_var_max_abs_err"] <= 1e-4
        assert parity_report["litert_parity"]["passed"] is True, f"LiteRT parity failed: {parity_report['litert_parity']}"
        assert parity_report["litert_parity"]["speed_max_abs_err"] <= 1e-3
        assert parity_report["litert_parity"]["log_var_max_abs_err"] <= 1e-3
        assert parity_report["all_passed"] is True


def test_sha256_utility(tmp_path: Path):
    """Test SHA256 digest function."""
    test_file = tmp_path / "hello.bin"
    test_file.write_bytes(b"COMPASS_PHASE7_VELOCITYNET")
    digest = compute_file_sha256(test_file)
    assert len(digest) == 64
    assert isinstance(digest, str)


def test_export_parity_500_windows_acceptance():
    """Acceptance test verifying numerical parity across 500 real held-out test windows.
    
    Enforces production tolerances:
    - ONNX: max absolute error <= 1e-4 m/s
    - LiteRT: max absolute error <= 1e-3 m/s
    """
    project_root = Path(__file__).resolve().parents[2]
    test_npz = project_root / "data" / "ml_dataset_v1" / "test.npz"
    onnx_path = project_root / "models" / "velocitynet_v1.onnx"
    tflite_path = project_root / "models" / "velocitynet_v1.tflite"
    ckpt_path = project_root / "models" / "velocitynet_v1_best.pt"

    if not (test_npz.exists() and onnx_path.exists() and tflite_path.exists() and ckpt_path.exists()):
        pytest.skip("Required model artifacts or test dataset not found for 500-window acceptance test")

    data = np.load(test_npz, allow_pickle=True)
    val_mask = data["is_valid"].astype(bool)
    X_test = data["X"][val_mask][:500].astype(np.float32)
    assert len(X_test) == 500, f"Expected 500 test windows, got {len(X_test)}"

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("config", {})
    model = VelocityNet(
        input_dim=cfg.get("input_dim", 9),
        hidden_dim=cfg.get("hidden_dim", 64),
        num_layers=cfg.get("num_layers", 2),
        dense_dim=cfg.get("dense_dim", 32),
        dropout=cfg.get("dropout", 0.2),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    parity_report = verify_export_parity(
        model=model,
        onnx_path=onnx_path,
        tflite_path=tflite_path,
        test_batch=X_test,
        onnx_tol=1e-4,
        litert_tol=1e-3,
    )

    assert parity_report["onnx_parity"]["passed"] is True, f"500-window ONNX parity failed: {parity_report['onnx_parity']}"
    assert parity_report["onnx_parity"]["speed_max_abs_err"] <= 1e-4
    assert parity_report["onnx_parity"]["log_var_max_abs_err"] <= 1e-4

    assert parity_report["litert_parity"]["passed"] is True, f"500-window LiteRT parity failed: {parity_report['litert_parity']}"
    assert parity_report["litert_parity"]["speed_max_abs_err"] <= 1e-3
    assert parity_report["litert_parity"]["log_var_max_abs_err"] <= 1e-3
    assert parity_report["all_passed"] is True
