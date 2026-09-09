"""VelocityNet: Deep Recurrent Neural Network for Forward Speed & Uncertainty Estimation.

Architecture (authoritative spec per FINAL_MASTER_PLAN_SIH26168.md):
- Input: (batch_size, 20, 9) causal vehicle-frame motion tensor at 10 Hz
- Recurrent Backbone: 2-layer GRU with 64 hidden units (dropout=0.2 between layers)
- Dense Intermediate: 64 -> 32 with ReLU activation
- Output Head: 32 -> 2 producing:
    1. forward_speed (mu_v, in m/s)
    2. log_variance (log sigma_v^2, unitless log-variance)
- Loss: Heteroscedastic Gaussian Negative Log-Likelihood (NLL)

Note: VelocityNet is a forward-speed measurement source; it does NOT directly output
positions, quaternions, or complete velocity vectors.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn


class VelocityNet(nn.Module):
    """Production VelocityNet architecture for forward vehicle speed and uncertainty estimation."""

    def __init__(
        self,
        input_dim: int = 9,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dense_dim: int = 32,
        dropout: float = 0.2,
        seq_len: Optional[int] = 20,
        min_log_var: float = -10.0,
        max_log_var: float = 10.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dense_dim = dense_dim
        self.seq_len = seq_len
        self.min_log_var = min_log_var
        self.max_log_var = max_log_var

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc1 = nn.Linear(hidden_dim, dense_dim)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(dense_dim, 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 20, 9) representing 2.0 s history
               of vehicle-frame calibrated motion features.

        Returns:
            Tuple of:
                - speed: (batch_size,) predicted forward speed in m/s
                - log_var: (batch_size,) predicted log variance log(sigma_v^2)
        """
        if x.ndim != 3:
            raise ValueError(f"Expected 3D input (batch_size, seq_len, features), got shape {tuple(x.shape)}")
        if (self.seq_len is not None and x.shape[1] != self.seq_len) or x.shape[2] != self.input_dim:
            expected_seq = self.seq_len if self.seq_len is not None else "any"
            raise ValueError(
                f"Expected input shape (batch_size, {expected_seq}, {self.input_dim}), got (batch_size, {x.shape[1]}, {x.shape[2]})"
            )

        # GRU forward pass: out shape (batch_size, 20, 64)
        out, _ = self.gru(x)

        # Strictly causal: select the last timestep feature vector at t=T (window end)
        last_step = out[:, -1, :]  # shape: (batch_size, 64)

        # Dense projection
        h = self.relu(self.fc1(last_step))  # shape: (batch_size, 32)
        preds = self.fc_out(h)  # shape: (batch_size, 2)

        speed = preds[:, 0]
        log_var = torch.clamp(preds[:, 1], min=self.min_log_var, max=self.max_log_var)

        return speed, log_var


class GaussianNLLLoss(nn.Module):
    """Numerically stable Gaussian Negative Log-Likelihood loss for heteroscedastic regression.

    Given ground-truth target y, predicted mean mu, and predicted log-variance s = log(sigma^2):
        NLL = 0.5 * (log(2 * pi) + s + (y - mu)^2 / exp(s))
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.log_2pi = math.log(2.0 * math.pi)

    def forward(
        self,
        pred_speed: torch.Tensor,
        pred_log_var: torch.Tensor,
        target_speed: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mean Gaussian NLL loss across the batch.

        Args:
            pred_speed: (batch_size,) predicted forward speed mu
            pred_log_var: (batch_size,) predicted log variance s = log(sigma^2)
            target_speed: (batch_size,) ground-truth reference speed y

        Returns:
            Scalar tensor representing mean Gaussian NLL loss.
        """
        # Clamp log_var for extreme numerical stability
        s = torch.clamp(pred_log_var, min=-10.0, max=10.0)
        inv_var = torch.exp(-s)
        sq_err = (target_speed - pred_speed) ** 2

        # 0.5 * (log(2*pi) + s + (y - mu)^2 * exp(-s))
        loss = 0.5 * (self.log_2pi + s + sq_err * inv_var)
        return torch.mean(loss)
