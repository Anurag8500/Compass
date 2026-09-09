"""1D-CNN Baseline Model for Velocity Estimation (Experimental Benchmark).

IMPORTANT:
This model is strictly an experimental comparative baseline used to evaluate whether
recurrent dynamics (GRU) outperform temporal convolutions on this dataset.
It is NOT part of the production navigation architecture.
"""

from __future__ import annotations

from typing import Tuple
import torch
import torch.nn as nn


class CNN1DVelocityBaseline(nn.Module):
    """Lightweight 1D Temporal Convolutional experimental baseline for velocity estimation."""

    def __init__(
        self,
        input_dim: int = 9,
        channels: Tuple[int, ...] = (48, 64, 64),
        kernel_size: int = 3,
        dense_dim: int = 32,
        dropout: float = 0.2,
        min_log_var: float = -10.0,
        max_log_var: float = 10.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.min_log_var = min_log_var
        self.max_log_var = max_log_var

        # Input to conv is (batch, channels=9, length=20)
        layers = []
        in_c = input_dim
        for out_c in channels:
            layers.extend([
                nn.Conv1d(in_c, out_c, kernel_size=kernel_size, padding=1),
                nn.BatchNorm1d(out_c),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_c = out_c
        self.conv_net = nn.Sequential(*layers)

        # Global average pooling across time dimension
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(channels[-1], dense_dim)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(dense_dim, 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 20, 9).

        Returns:
            Tuple of (speed, log_var)
        """
        if x.ndim != 3:
            raise ValueError(f"Expected 3D input (batch_size, seq_len, features), got shape {tuple(x.shape)}")
        if x.shape[1] != 20 or x.shape[2] != self.input_dim:
            raise ValueError(
                f"Expected input shape (batch_size, 20, {self.input_dim}), got (batch_size, {x.shape[1]}, {x.shape[2]})"
            )

        # Transpose from (batch, 20, 9) to (batch, 9, 20) for Conv1d
        x_conv = x.transpose(1, 2)
        feats = self.conv_net(x_conv)  # (batch, 64, 20)
        pooled = self.global_pool(feats).squeeze(-1)  # (batch, 64)

        h = self.relu(self.fc1(pooled))  # (batch, 32)
        preds = self.fc_out(h)  # (batch, 2)

        speed = preds[:, 0]
        log_var = torch.clamp(preds[:, 1], min=self.min_log_var, max=self.max_log_var)

        return speed, log_var
