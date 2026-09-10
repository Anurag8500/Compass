"""Integration test verifying BiasNet ONNX and LiteRT export parity (Phase 8).

Verifies:
1. Exported artifacts exist: models/biasnet_v1.onnx and models/biasnet_v1.tflite.
2. ONNX graph parses cleanly and passes onnx.checker.
3. LiteRT interpreter initializes and runs inference on production shape (1, 20, 9).
4. Numerical parity report models/biasnet_v1_export_parity.json shows max abs error <= 1e-4 for ONNX
   and <= 1e-3 for LiteRT across at least 500 real windows.
5. In-flight numerical parity passes live on test samples.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import torch
import pytest
from ai_edge_litert.interpreter import Interpreter

from ml.models.biasnet import BiasNet


class TestBiasNetExportParity:
    """Integration test suite verifying BiasNet export artifacts and numerical parity."""

    @classmethod
    def setup_class(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]
        cls.onnx_path = cls.root / "models" / "biasnet_v1.onnx"
        cls.tflite_path = cls.root / "models" / "biasnet_v1.tflite"
        cls.pt_path = cls.root / "models" / "biasnet_v1_best.pt"
        cls.parity_json = cls.root / "models" / "biasnet_v1_export_parity.json"

        if not cls.onnx_path.exists() or not cls.tflite_path.exists() or not cls.pt_path.exists():
            pytest.skip("Export artifacts not present")

        cls.model = BiasNet(input_dim=9, hidden_dim=48, num_layers=2, dense_dim=24)
        cls.model.load_state_dict(torch.load(cls.pt_path, map_location="cpu", weights_only=True))
        cls.model.eval()

    def test_artifacts_exist_and_non_empty(self) -> None:
        """Exported ONNX and LiteRT models must exist and have non-zero size."""
        assert self.onnx_path.exists()
        assert self.onnx_path.stat().st_size > 10_000

        assert self.tflite_path.exists()
        assert self.tflite_path.stat().st_size > 10_000

    def test_onnx_model_valid(self) -> None:
        """ONNX graph must pass structural verification."""
        onnx_model = onnx.load(str(self.onnx_path))
        onnx.checker.check_model(onnx_model)

    def test_litert_interpreter_runs(self) -> None:
        """LiteRT interpreter must invoke cleanly with (1, 20, 9) input."""
        interp = Interpreter(model_path=str(self.tflite_path))
        interp.allocate_tensors()

        in_det = interp.get_input_details()[0]
        out_det = interp.get_output_details()[0]

        dummy_x = np.random.randn(1, 20, 9).astype(np.float32)
        interp.set_tensor(in_det["index"], dummy_x)
        interp.invoke()

        out = interp.get_tensor(out_det["index"])
        assert out.shape == (1, 6)
        assert np.isfinite(out).all()

    def test_saved_parity_report(self) -> None:
        """Saved parity JSON must show both ONNX and LiteRT passed within tolerances."""
        assert self.parity_json.exists(), f"Missing {self.parity_json}"
        with open(self.parity_json) as f:
            p = json.load(f)

        assert p["parity"]["overall_passed"]
        assert p["parity"]["onnx_parity"]["passed"]
        assert p["parity"]["litert_parity"]["passed"]
        assert p["parity"]["num_test_samples"] >= 500
        assert p["parity"]["onnx_parity"]["max_abs_err"] <= 1e-4
        assert p["parity"]["litert_parity"]["max_abs_err"] <= 1e-3

    def test_live_numerical_parity(self) -> None:
        """Live execution of PyTorch vs ONNX vs LiteRT on random test sample."""
        x = np.random.randn(1, 20, 9).astype(np.float32)

        # PyTorch
        with torch.no_grad():
            pt_out = self.model(torch.from_numpy(x)).cpu().numpy().flatten()

        # ONNX
        ort_sess = ort.InferenceSession(str(self.onnx_path))
        ort_out = ort_sess.run(None, {"features": x})[0].flatten()

        # LiteRT
        interp = Interpreter(model_path=str(self.tflite_path))
        interp.allocate_tensors()
        in_idx = interp.get_input_details()[0]["index"]
        out_idx = interp.get_output_details()[0]["index"]
        interp.set_tensor(in_idx, x)
        interp.invoke()
        tfl_out = interp.get_tensor(out_idx).flatten()

        np.testing.assert_allclose(pt_out, ort_out, atol=1e-4)
        np.testing.assert_allclose(pt_out, tfl_out, atol=1e-3)
