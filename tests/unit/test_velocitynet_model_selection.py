"""Unit tests for VelocityNet model selection provenance and causal EMA logic."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from ml.experiments.final_model_selection import apply_causal_ema_to_split
from ml.training.dataset import VelocityNetDataset


class TestVelocityNetModelSelection:
    """Test suite for model selection record integrity and causal post-processing."""

    @pytest.fixture
    def selection_file(self):
        root = Path(__file__).resolve().parents[2]
        sel_path = root / "models" / "velocitynet_model_selection.json"
        if not sel_path.exists():
            pytest.skip("models/velocitynet_model_selection.json not found")
        return sel_path

    def test_selection_json_schema_and_provenance(self, selection_file):
        """Verify model selection record contains required provenance fields without NaNs/Infs."""
        content = selection_file.read_text(encoding="utf-8")
        assert "NaN" not in content, "JSON must not contain NaN"
        assert "Infinity" not in content, "JSON must not contain Infinity"

        data = json.loads(content)
        assert data["evaluation_split"] == "Driver B (21,080 windows)"
        assert "candidates_comparison" in data
        assert len(data["candidates_comparison"]) >= 3
        assert "selected_candidate" in data
        assert "selected_ema_alpha" in data
        assert "selection_rationale" in data

        # Check candidate fields
        for cand in data["candidates_comparison"]:
            assert "candidate" in cand
            assert "params" in cand
            assert "val_nll" in cand
            assert "val_rmse" in cand
            assert "high_speed_rmse" in cand
            assert "val_rmse_with_ema" in cand

    def test_causal_ema_resets_at_trip_boundaries(self):
        """Verify that causal EMA strictly resets its internal state across source file boundaries."""
        class MockDataset:
            def __init__(self):
                # 2 different trips of 5 samples each
                self.source_file_ids = np.array(["trip_1"] * 5 + ["trip_2"] * 5)
                self.timestamps_end_ns = np.array(list(range(0, 500, 100)) + list(range(0, 500, 100)))

        mock_ds = MockDataset()
        # Trip 1 has constant speed 10, Trip 2 has constant speed 50
        raw_preds = np.array([10.0] * 5 + [50.0] * 5, dtype=np.float32)
        alpha = 0.5

        filtered = apply_causal_ema_to_split(mock_ds, raw_preds, alpha=alpha)

        # At the start of Trip 2 (index 5), filtered value must immediately equal raw_preds[5] (50.0)
        # without lagging or bleeding from Trip 1's value (10.0)
        assert filtered[0] == 10.0, "Trip 1 start must reset"
        assert filtered[5] == 50.0, "Trip 2 start must strictly reset, with no leakage from Trip 1"

        # Within Trip 2, verify forward EMA formula:
        # If speed suddenly changed to 60 at index 6:
        raw_preds_dynamic = np.array([10.0] * 5 + [50.0, 60.0, 60.0, 60.0, 60.0], dtype=np.float32)
        filtered_dynamic = apply_causal_ema_to_split(mock_ds, raw_preds_dynamic, alpha=alpha)
        assert filtered_dynamic[5] == 50.0
        expected_t1 = 0.5 * 60.0 + 0.5 * 50.0  # 55.0
        assert np.isclose(filtered_dynamic[6], expected_t1), f"Expected {expected_t1}, got {filtered_dynamic[6]}"
