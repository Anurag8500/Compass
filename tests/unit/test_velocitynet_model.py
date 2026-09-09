"""Unit tests for VelocityNet model architecture and invariants."""

import torch
import pytest
from ml.models.velocitynet import VelocityNet


class TestVelocityNetModel:
    """Test suite for VelocityNet forward pass, shapes, and behavior."""

    def test_model_initialization_and_layer_structure(self) -> None:
        """Verify model components match GRU(9->64, 2L) -> Dense(64->32) -> Dense(32->2)."""
        model = VelocityNet(input_dim=9, hidden_dim=64, num_layers=2, dense_dim=32)
        assert model.gru.input_size == 9
        assert model.gru.hidden_size == 64
        assert model.gru.num_layers == 2
        assert model.fc1.in_features == 64
        assert model.fc1.out_features == 32
        assert model.fc_out.in_features == 32
        assert model.fc_out.out_features == 2

    def test_forward_output_shapes_and_types(self) -> None:
        """Verify batched forward pass produces (batch,) speed and log_variance."""
        model = VelocityNet()
        batch_size = 8
        dummy_input = torch.randn(batch_size, 20, 9)

        speed, log_var = model(dummy_input)

        assert speed.shape == (batch_size,)
        assert log_var.shape == (batch_size,)
        assert speed.dtype == torch.float32
        assert log_var.dtype == torch.float32

    def test_single_window_forward(self) -> None:
        """Verify single window (1, 20, 9) inference works correctly."""
        model = VelocityNet()
        x = torch.randn(1, 20, 9)
        speed, log_var = model(x)
        assert speed.shape == (1,)
        assert log_var.shape == (1,)
        assert torch.isfinite(speed).item()
        assert torch.isfinite(log_var).item()

    def test_invalid_input_shape_raises_value_error(self) -> None:
        """Verify invalid input dimensions are rejected immediately."""
        model = VelocityNet()
        # Wrong sequence length (10 instead of 20)
        with pytest.raises(ValueError, match="Expected input shape"):
            model(torch.randn(4, 10, 9))

        # Wrong channel count (8 instead of 9)
        with pytest.raises(ValueError, match="Expected input shape"):
            model(torch.randn(4, 20, 8))

        # Wrong dimensionality (2D instead of 3D)
        with pytest.raises(ValueError, match="Expected 3D input"):
            model(torch.randn(20, 9))

    def test_eval_determinism(self) -> None:
        """Verify model in eval mode produces bit-identical outputs for identical inputs."""
        model = VelocityNet()
        model.eval()
        x = torch.randn(16, 20, 9)

        with torch.no_grad():
            s1, v1 = model(x)
            s2, v2 = model(x)

        assert torch.all(s1 == s2)
        assert torch.all(v1 == v2)

    def test_log_variance_clamping(self) -> None:
        """Verify predicted log-variance remains clamped within [-10, 10]."""
        model = VelocityNet(min_log_var=-10.0, max_log_var=10.0)
        # Force weights to produce very large outputs
        with torch.no_grad():
            model.fc_out.bias.fill_(100.0)
            _, log_var_high = model(torch.randn(4, 20, 9))
            assert torch.all(log_var_high <= 10.0)

            model.fc_out.bias.fill_(-100.0)
            _, log_var_low = model(torch.randn(4, 20, 9))
            assert torch.all(log_var_low >= -10.0)

    def test_parameter_counts_exact(self) -> None:
        """Verify exact parameter counts for VelocityNet and lightweight 1D-CNN baseline."""
        from ml.models.baselines.cnn1d_velocity import CNN1DVelocityBaseline

        model_gru = VelocityNet(input_dim=9, hidden_dim=64, num_layers=2, dense_dim=32)
        total_gru_params = sum(p.numel() for p in model_gru.parameters())
        assert total_gru_params == 41506, f"Expected 41,506 params, got {total_gru_params}"

        model_cnn = CNN1DVelocityBaseline(input_dim=9, channels=(48, 64, 64), kernel_size=3, dense_dim=32)
        total_cnn_params = sum(p.numel() for p in model_cnn.parameters())
        assert total_cnn_params == 25474, f"Expected 25,474 params, got {total_cnn_params}"

    def test_cnn1d_baseline_forward_shapes(self) -> None:
        """Verify lightweight 1D-CNN baseline forward pass produces valid shapes."""
        from ml.models.baselines.cnn1d_velocity import CNN1DVelocityBaseline

        model = CNN1DVelocityBaseline()
        x = torch.randn(4, 20, 9)
        speed, log_var = model(x)
        assert speed.shape == (4,)
        assert log_var.shape == (4,)
        assert torch.isfinite(speed).all()
        assert torch.isfinite(log_var).all()

