"""Unit tests for Gaussian Negative Log-Likelihood (NLL) loss function."""

import torch
from ml.models.velocitynet import GaussianNLLLoss


class TestGaussianNLLLoss:
    """Test suite for GaussianNLLLoss properties and numerical stability."""

    def test_loss_finite_on_standard_inputs(self) -> None:
        """Verify loss is finite on typical predictions and targets."""
        loss_fn = GaussianNLLLoss()
        pred_speed = torch.tensor([10.0, 15.0, 20.0])
        pred_log_var = torch.tensor([0.0, 0.5, -0.5])
        target_speed = torch.tensor([10.2, 14.8, 19.5])

        loss = loss_fn(pred_speed, pred_log_var, target_speed)
        assert torch.isfinite(loss).item()
        assert loss.item() > 0.0

    def test_lower_loss_for_better_prediction(self) -> None:
        """Verify that accurate predictions yield lower loss than inaccurate predictions."""
        loss_fn = GaussianNLLLoss()
        target = torch.tensor([20.0, 20.0])
        log_var = torch.tensor([0.0, 0.0])

        good_pred = torch.tensor([20.01, 19.99])
        poor_pred = torch.tensor([15.0, 25.0])

        good_loss = loss_fn(good_pred, log_var, target)
        poor_loss = loss_fn(poor_pred, log_var, target)

        assert good_loss.item() < poor_loss.item()

    def test_gradients_flow_to_both_outputs(self) -> None:
        """Verify non-zero gradients backpropagate into both speed and log_variance."""
        loss_fn = GaussianNLLLoss()
        pred_speed = torch.tensor([10.0, 15.0], requires_grad=True)
        pred_log_var = torch.tensor([1.0, 1.0], requires_grad=True)
        target = torch.tensor([12.0, 14.0])

        loss = loss_fn(pred_speed, pred_log_var, target)
        loss.backward()

        assert pred_speed.grad is not None
        assert pred_log_var.grad is not None
        assert torch.all(torch.isfinite(pred_speed.grad))
        assert torch.all(torch.isfinite(pred_log_var.grad))
        assert not torch.all(pred_speed.grad == 0.0)
        assert not torch.all(pred_log_var.grad == 0.0)

    def test_numerical_stability_under_extreme_inputs(self) -> None:
        """Verify no NaN or Inf occurs under large residuals or extreme log-variances."""
        loss_fn = GaussianNLLLoss()
        pred_speed = torch.tensor([1e4, -1e4])
        pred_log_var = torch.tensor([100.0, -100.0])  # extreme log-variances
        target = torch.tensor([0.0, 0.0])

        loss = loss_fn(pred_speed, pred_log_var, target)
        assert torch.isfinite(loss).item()
