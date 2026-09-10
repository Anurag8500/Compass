"""Unit tests for BiasNet neural network architecture (Phase 8).

Verifies:
1. Input contract: accepts (B, 20, 9) tensor.
2. Output contract: outputs (B, 6) tensor for Stage A.
3. In-graph physical clamping: extreme activations are clamped within configured bounds.
4. Parameter count: matches expected ~24k parameters.
5. Deterministic eval inference: identical outputs on identical inputs.
6. Optional uncertainty head (Stage B): outputs (B, 6) means and (B, 6) log-variances.
"""

from __future__ import annotations

import torch
import pytest

from ml.models.biasnet import BiasNet


class TestBiasNetModel:
    """Test suite verifying BiasNet neural network mechanics."""

    def test_forward_shape_stage_a(self) -> None:
        """Stage A forward pass produces (B, 6) tensor."""
        model = BiasNet(input_dim=9, hidden_dim=48, num_layers=2, dense_dim=24)
        model.eval()

        batch_size = 4
        seq_len = 20
        x = torch.randn(batch_size, seq_len, 9)

        out = model(x)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (batch_size, 6)
        assert torch.isfinite(out).all()

    def test_in_graph_physical_clamps(self) -> None:
        """Outputs must never exceed bound_accel_mps2 and bound_gyro_rads even under huge inputs."""
        bound_a = 1.5
        bound_g = 0.10
        model = BiasNet(
            input_dim=9,
            hidden_dim=48,
            num_layers=2,
            bound_accel_mps2=bound_a,
            bound_gyro_rads=bound_g,
        )
        model.eval()

        # Input huge values to saturate network
        x_huge = torch.randn(8, 20, 9) * 1000.0
        out = model(x_huge)

        dba = out[:, 0:3]
        dbg = out[:, 3:6]

        assert (torch.abs(dba) <= bound_a + 1e-6).all()
        assert (torch.abs(dbg) <= bound_g + 1e-6).all()

    def test_parameter_count(self) -> None:
        """Verify parameter count is compact (~23,934 parameters)."""
        model = BiasNet(input_dim=9, hidden_dim=48, num_layers=2, dense_dim=24)
        total_params = sum(p.numel() for p in model.parameters())

        assert 20_000 <= total_params <= 30_000
        assert total_params == 23_934

    def test_deterministic_eval_mode(self) -> None:
        """Evaluation mode must be completely deterministic."""
        model = BiasNet(input_dim=9, hidden_dim=48, num_layers=2, dense_dim=24)
        model.eval()

        x = torch.randn(2, 20, 9)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)

        torch.testing.assert_close(out1, out2)

    def test_stage_b_uncertainty_head(self) -> None:
        """When predict_uncertainty=True, model outputs 6 means and 6 log-variances."""
        model = BiasNet(
            input_dim=9,
            hidden_dim=48,
            num_layers=2,
            dense_dim=24,
            predict_uncertainty=True,
        )
        model.eval()

        x = torch.randn(3, 20, 9)
        means, log_vars = model(x)

        assert means.shape == (3, 6)
        assert log_vars.shape == (3, 6)
        assert torch.isfinite(means).all()
        assert torch.isfinite(log_vars).all()
