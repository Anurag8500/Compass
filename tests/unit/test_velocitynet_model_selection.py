"""Unit tests for VelocityNet model selection provenance, hierarchical selection policy, and causal EMA logic."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pytest

from ml.experiments.final_model_selection import (
    apply_causal_ema_to_split,
    evaluate_candidate_ranking,
)
from ml.training.dataset import VelocityNetDataset


class TestVelocityNetModelSelection:
    """Test suite for model selection record integrity, hierarchical policy, and causal post-processing."""

    @pytest.fixture
    def selection_file(self):
        root = Path(__file__).resolve().parents[2]
        sel_path = root / "models" / "velocitynet_model_selection.json"
        if not sel_path.exists():
            pytest.skip("models/velocitynet_model_selection.json not found")
        return sel_path

    @pytest.fixture
    def v1_1_config_file(self):
        root = Path(__file__).resolve().parents[2]
        cfg_path = root / "models" / "model_config_velocitynet_v1_1.json"
        if not cfg_path.exists():
            pytest.skip("models/model_config_velocitynet_v1_1.json not found")
        return cfg_path

    def test_selection_json_schema_and_provenance(self, selection_file):
        """Verify model selection record contains required provenance fields without NaNs/Infs."""
        content = selection_file.read_text(encoding="utf-8")
        assert "NaN" not in content, "JSON must not contain NaN"
        assert "Infinity" not in content, "JSON must not contain Infinity"

        data = json.loads(content)
        assert data["evaluation_split"] == "Driver B (21,080 windows)"
        assert "selection_policy" in data
        assert data["selection_primary_metric"] == "val_rmse"
        assert "ordered_candidate_ranking" in data
        assert len(data["ordered_candidate_ranking"]) == 3
        assert data["selected_candidate"] == "Candidate B (1D-CNN)"
        assert data["selected_ema_alpha"] == 0.2
        assert "selection_rationale" in data

        # Check that top-ranked candidate is the selected candidate
        assert data["ordered_candidate_ranking"][0]["candidate"] == data["selected_candidate"]
        assert data["ordered_candidate_ranking"][0]["rank"] == 1

        # Check candidate fields
        for cand in data["candidates_comparison"]:
            assert "candidate" in cand
            assert "params" in cand
            assert "val_nll" in cand
            assert "val_rmse" in cand
            assert "high_speed_rmse" in cand
            assert "val_rmse_with_ema" in cand

    def test_hierarchical_selection_policy_generic_and_deterministic(self):
        """Verify explicit hierarchical policy evaluates all candidates, treats RMSE as primary, and breaks ties."""
        synthetic_candidates: List[Dict[str, Any]] = [
            {
                "candidate": "Candidate Alpha",
                "val_rmse": 4.60,
                "val_mae": 3.50,
                "val_nll": 2.80,  # Lower NLL than Beta
                "high_speed_rmse": 6.0,
                "latency_p50_ms": 1.2,
                "params": 15000,
            },
            {
                "candidate": "Candidate Beta",
                "val_rmse": 4.40,  # Lowest RMSE -> MUST WIN
                "val_mae": 3.30,
                "val_nll": 2.92,
                "high_speed_rmse": 5.6,
                "latency_p50_ms": 0.25,
                "params": 25000,
            },
            {
                "candidate": "Candidate Gamma",
                "val_rmse": 5.00,
                "val_mae": 3.90,
                "val_nll": 3.00,
                "high_speed_rmse": 5.7,
                "latency_p50_ms": 0.8,
                "params": 40000,
            },
        ]

        ranked, policy = evaluate_candidate_ranking(synthetic_candidates)
        assert len(ranked) == 3
        assert policy["primary_metric"] == "val_rmse"
        # Candidate Beta must win on primary metric RMSE even though Alpha has lower NLL
        assert ranked[0]["candidate"] == "Candidate Beta"
        assert ranked[0]["rank"] == 1
        assert ranked[1]["candidate"] == "Candidate Alpha"
        assert ranked[1]["rank"] == 2
        assert ranked[2]["candidate"] == "Candidate Gamma"
        assert ranked[2]["rank"] == 3

        # Test tie-breaking: identical RMSE breaks on secondary MAE, then NLL
        tied_candidates: List[Dict[str, Any]] = [
            {
                "candidate": "Model X",
                "val_rmse": 4.50,
                "val_mae": 3.50,
                "val_nll": 2.90,
                "high_speed_rmse": 5.5,
                "latency_p50_ms": 0.5,
                "params": 20000,
            },
            {
                "candidate": "Model Y",
                "val_rmse": 4.50,
                "val_mae": 3.40,  # Lower MAE wins tie
                "val_nll": 2.95,
                "high_speed_rmse": 5.5,
                "latency_p50_ms": 0.5,
                "params": 20000,
            },
        ]
        ranked_tied, _ = evaluate_candidate_ranking(tied_candidates)
        assert ranked_tied[0]["candidate"] == "Model Y"
        assert ranked_tied[1]["candidate"] == "Model X"

    def test_training_config_matches_actual_run(self, v1_1_config_file):
        """Verify model_config_velocitynet_v1_1.json matches the actual training parameters and architecture."""
        cfg = json.loads(v1_1_config_file.read_text(encoding="utf-8"))
        tc = cfg["training_configuration"]

        assert tc["max_epochs"] == 15
        assert tc["patience"] == 5
        assert tc["scheduler"] == "CosineAnnealingLR"
        assert tc["scheduler_t_max"] == 15
        assert tc["scheduler_eta_min"] == 1e-6
        assert tc["batch_size"] == 256
        assert tc["learning_rate"] == 0.001
        assert tc["weight_decay"] == 1e-5
        assert tc["seed"] == 42
        assert tc["gradient_clipping"] == 5.0

        # Verify exact architecture details
        arch = cfg["architecture_details"]
        assert arch["architecture_class"] == "CNN1DVelocityBaseline"
        assert arch["input_dim"] == 9
        assert arch["conv_channels"] == [48, 64, 64]
        assert arch["kernel_size"] == 3
        assert arch["padding"] == 1
        assert arch["dense_dim"] == 32
        assert arch["dropout"] == 0.2

        # Verify artifacts
        assert cfg["artifacts"]["checkpoint"] == "velocitynet_v1_1_best.pt"
        assert cfg["artifacts"]["onnx_model"] == "velocitynet_v1_1.onnx"
        assert cfg["artifacts"]["tflite_model"] == "velocitynet_v1_1.tflite"

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
        raw_preds_dynamic = np.array([10.0] * 5 + [50.0, 60.0, 60.0, 60.0, 60.0], dtype=np.float32)
        filtered_dynamic = apply_causal_ema_to_split(mock_ds, raw_preds_dynamic, alpha=alpha)
        assert filtered_dynamic[5] == 50.0
        expected_t1 = 0.5 * 60.0 + 0.5 * 50.0  # 55.0
        assert np.isclose(filtered_dynamic[6], expected_t1), f"Expected {expected_t1}, got {filtered_dynamic[6]}"
